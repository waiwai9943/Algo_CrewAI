"""
backtest_tool.py
----------------
CrewAI custom tool: vectorised intraday backtest engine for VWAP+RSI strategies.

Takes indicator-enriched bar data (JSON from IndicatorTool) and user-defined
entry/exit thresholds, then simulates trades and computes performance metrics.

Assumptions:
  - Fills at the OPEN of the bar AFTER the signal bar (next-bar execution).
  - No partial fills. One contract per trade.
  - Commissions: $4.10 per round-trip (IB rate ~$2.05/side for NQ/CL).
  - Slippage: configurable in ticks.
  - One position at a time (no pyramiding).
  - Mandatory EOD flat before session close.
"""

from __future__ import annotations

import json
import math
from typing import Type

import numpy as np
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field


# ─────────────────────────────────────────────
# Input Schema
# ─────────────────────────────────────────────

class BacktestInput(BaseModel):
    """Input schema for BacktestTool."""

    input_file: str = Field(
        default="indicator_data.json",
        description="Local JSON file containing indicator-enriched bar data (output from IndicatorTool).",
    )
    output_file: str = Field(
        default="backtest_results.json",
        description="Local JSON file where detailed trade-by-trade logs and metrics will be saved.",
    )
    # ── Long entry ──────────────────────────────────────────────
    long_vwap_zscore_threshold: float = Field(
        default=-1.5,
        description=(
            "VWAP z-score must be BELOW this value to trigger a long entry. "
            "Negative values mean price is below VWAP. Default: -1.5."
        ),
    )
    long_rsi_threshold: float = Field(
        default=35.0,
        description="RSI must be BELOW this value to confirm long entry. Default: 35.",
    )
    # ── Short entry ─────────────────────────────────────────────
    short_vwap_zscore_threshold: float = Field(
        default=1.5,
        description=(
            "VWAP z-score must be ABOVE this value to trigger a short entry. Default: 1.5."
        ),
    )
    short_rsi_threshold: float = Field(
        default=65.0,
        description="RSI must be ABOVE this value to confirm short entry. Default: 65.",
    )
    # ── Exit ────────────────────────────────────────────────────
    take_profit_ticks: int = Field(
        default=20,
        description=(
            "Take-profit distance in ticks from entry price. "
            "NQ tick = $5 (0.25 pts). CL tick = $10 (0.01 pts)."
        ),
    )
    stop_loss_ticks: int = Field(
        default=12,
        description="Stop-loss distance in ticks from entry price.",
    )
    # ── Instrument spec ─────────────────────────────────────────
    instrument: str = Field(
        default="NQ",
        description="'NQ' or 'CL'. Determines tick size and dollar value.",
    )
    slippage_ticks: int = Field(
        default=1,
        description="Assumed slippage per side in ticks. Default: 1 tick.",
    )
    commission_per_rt: float = Field(
        default=4.10,
        description="Round-trip commission per contract in USD. IB default: $4.10.",
    )
    eod_exit_bar_from_end: int = Field(
        default=3,
        description=(
            "Bars before session end to force-close any open position. "
            "Default: 3 bars before EOD = 15 minutes before close."
        ),
    )


# ─────────────────────────────────────────────
# Instrument Specs
# ─────────────────────────────────────────────

INSTRUMENT_SPECS: dict[str, dict] = {
    "NQ": {
        "tick_size": 0.25,          # 0.25 index points
        "tick_value_usd": 5.0,      # $5 per tick
        "point_value_usd": 20.0,    # $20 per full point
    },
    "CL": {
        "tick_size": 0.01,          # $0.01 per barrel
        "tick_value_usd": 10.0,     # $10 per tick (1000 barrels)
        "point_value_usd": 1000.0,  # $1000 per $1 move
    },
}


# ─────────────────────────────────────────────
# Backtest Engine
# ─────────────────────────────────────────────

def run_backtest(
    df: pd.DataFrame,
    long_z_thresh: float,
    long_rsi_thresh: float,
    short_z_thresh: float,
    short_rsi_thresh: float,
    tp_ticks: int,
    sl_ticks: int,
    instrument: str,
    slippage_ticks: int,
    commission_rt: float,
    eod_exit_bar_from_end: int,
) -> dict:
    """
    Vectorised single-contract backtest of a VWAP+RSI mean-reversion strategy.

    Entry logic (next-bar open execution):
        LONG : vwap_zscore < long_z_thresh AND rsi < long_rsi_thresh
        SHORT: vwap_zscore > short_z_thresh AND rsi > short_rsi_thresh

    Exit logic (whichever comes first):
        - Take-profit: price moves tp_ticks in favour
        - Stop-loss:   price moves sl_ticks against
        - EOD:         last `eod_exit_bar_from_end` bars of each session

    Returns:
        Dict with trade log and aggregated performance metrics.
    """
    spec = INSTRUMENT_SPECS.get(instrument.upper(), INSTRUMENT_SPECS["NQ"])
    tick_size = spec["tick_size"]
    tick_val = spec["tick_value_usd"]

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Build session-end mask
    session_dates = df["session_date"].unique() if "session_date" in df.columns else [None]
    eod_mask = pd.Series(False, index=df.index)
    for date in session_dates:
        if date is None:
            continue
        day_bars = df[df["session_date"] == date]
        if len(day_bars) <= eod_exit_bar_from_end:
            eod_mask.loc[day_bars.index] = True
        else:
            eod_mask.loc[day_bars.index[-eod_exit_bar_from_end:]] = True

    # Entry signals on current bar; execute on NEXT bar's open
    long_signal = (df["vwap_zscore"] < long_z_thresh) & (df["rsi"] < long_rsi_thresh)
    short_signal = (df["vwap_zscore"] > short_z_thresh) & (df["rsi"] > short_rsi_thresh)

    trades: list[dict] = []
    position: str | None = None   # 'long', 'short', or None
    entry_price: float = 0.0
    entry_bar_idx: int = 0
    tp_price: float = 0.0
    sl_price: float = 0.0

    bars = list(df.itertuples())
    n = len(bars)

    for i, bar in enumerate(bars):
        # ── Check exit conditions for open position ──────────────
        if position is not None:
            is_eod = eod_mask.iloc[i]
            high = float(bar.high)
            low = float(bar.low)
            exit_price: float | None = None
            exit_reason: str = ""

            if position == "long":
                if high >= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                elif low <= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                elif is_eod:
                    exit_price = float(bar.open)
                    exit_reason = "eod_flat"
            else:  # short
                if low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                elif high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                elif is_eod:
                    exit_price = float(bar.open)
                    exit_reason = "eod_flat"

            if exit_price is not None:
                raw_pnl_pts = (
                    (exit_price - entry_price) if position == "long"
                    else (entry_price - exit_price)
                )
                # Apply slippage (adverse on both legs)
                slip_cost = slippage_ticks * tick_val * 2
                pnl_usd = (raw_pnl_pts / tick_size) * tick_val - commission_rt - slip_cost

                trades.append({
                    "direction": position,
                    "entry_bar": str(bars[entry_bar_idx].Index),
                    "exit_bar": str(bar.Index),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "exit_reason": exit_reason,
                    "pnl_usd": round(pnl_usd, 2),
                    "pnl_ticks": round(raw_pnl_pts / tick_size, 1),
                })
                position = None

        # ── Check entry signals for new position (next bar execution) ──
        if position is None and i < n - 1 and not eod_mask.iloc[i]:
            next_bar = bars[i + 1]
            exec_price = float(next_bar.open)

            if long_signal.iloc[i]:
                position = "long"
                entry_price = exec_price + slippage_ticks * tick_size  # adverse slip
                tp_price = entry_price + tp_ticks * tick_size
                sl_price = entry_price - sl_ticks * tick_size
                entry_bar_idx = i + 1

            elif short_signal.iloc[i]:
                position = "short"
                entry_price = exec_price - slippage_ticks * tick_size  # adverse slip
                tp_price = entry_price - tp_ticks * tick_size
                sl_price = entry_price + sl_ticks * tick_size
                entry_bar_idx = i + 1

    if not trades:
        return {
            "error": "No trades generated. Check signal thresholds — they may be too restrictive.",
            "params": {
                "long_z_thresh": long_z_thresh,
                "long_rsi_thresh": long_rsi_thresh,
                "short_z_thresh": short_z_thresh,
                "short_rsi_thresh": short_rsi_thresh,
                "tp_ticks": tp_ticks,
                "sl_ticks": sl_ticks,
            },
        }

    # ── Aggregate metrics ─────────────────────────────────────
    pnls = [t["pnl_usd"] for t in trades]
    n_trades = len(trades)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    n_win = len(winners)
    n_loss = len(losers)
    win_rate = n_win / n_trades

    gross_profit = sum(winners) if winners else 0.0
    gross_loss = abs(sum(losers)) if losers else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    avg_win = np.mean(winners) if winners else 0.0
    avg_loss = np.mean(losers) if losers else 0.0
    avg_rr = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

    # Equity curve and max drawdown
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_drawdown = float(np.min(drawdown))

    # Annualised Sharpe (assuming ~252 trading days, ~78 bars/day for 5-min RTH)
    pnl_arr = np.array(pnls)
    daily_pnl: list[float] = []
    trade_df = pd.DataFrame(trades)
    trade_df["exit_bar"] = pd.to_datetime(trade_df["exit_bar"])
    trade_df["exit_date"] = trade_df["exit_bar"].dt.date
    for date, grp in trade_df.groupby("exit_date"):
        daily_pnl.append(grp["pnl_usd"].sum())

    if len(daily_pnl) > 1:
        daily_arr = np.array(daily_pnl)
        sharpe = (np.mean(daily_arr) / (np.std(daily_arr) + 1e-9)) * np.sqrt(252)
    else:
        sharpe = float("nan")

    by_reason = trade_df.groupby("exit_reason")["pnl_usd"].agg(["count", "sum", "mean"])

    return {
        "instrument": instrument,
        "params": {
            "long_vwap_zscore_threshold": long_z_thresh,
            "long_rsi_threshold": long_rsi_thresh,
            "short_vwap_zscore_threshold": short_z_thresh,
            "short_rsi_threshold": short_rsi_thresh,
            "take_profit_ticks": tp_ticks,
            "take_profit_usd": round(tp_ticks * tick_val, 2),
            "stop_loss_ticks": sl_ticks,
            "stop_loss_usd": round(sl_ticks * tick_val, 2),
            "slippage_ticks": slippage_ticks,
            "commission_per_rt_usd": commission_rt,
        },
        "performance": {
            "total_trades": n_trades,
            "winning_trades": n_win,
            "losing_trades": n_loss,
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win_usd": round(float(avg_win), 2),
            "avg_loss_usd": round(float(avg_loss), 2),
            "avg_reward_risk_ratio": round(float(avg_rr), 3),
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "net_pnl_usd": round(sum(pnls), 2),
            "profit_factor": round(profit_factor, 3) if not math.isinf(profit_factor) else "inf",
            "max_drawdown_usd": round(max_drawdown, 2),
            "sharpe_ratio_annualised": round(float(sharpe), 3) if not math.isnan(sharpe) else "N/A",
        },
        "exit_reason_breakdown": by_reason.to_dict(),
        "trade_log": trades,     # full trade-by-trade log
        "equity_curve": [round(float(e), 2) for e in equity.tolist()],
    }


# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────

class BacktestTool(BaseTool):
    """
    Runs a vectorised intraday backtest of a VWAP+RSI mean-reversion strategy.

    Requires indicator-enriched bar data saved in a local JSON file.
    Simulates next-bar execution and saves detailed results to a local file.
    """

    name: str = "Strategy Backtester"
    description: str = (
        "Runs a vectorised intraday backtest of a VWAP+RSI strategy using "
        "historical bar data from a local indicator file. "
        "Accepts entry thresholds, take-profit/stop-loss in ticks, and instrument. "
        "Saves the detailed results and full trade log to a local file, "
        "and returns a summary JSON with key performance metrics. "
        "Use this to validate strategy parameters."
    )
    args_schema: Type[BaseModel] = BacktestInput

    def _run(
        self,
        input_file: str = "indicator_data.json",
        output_file: str = "backtest_results.json",
        long_vwap_zscore_threshold: float = -1.5,
        long_rsi_threshold: float = 35.0,
        short_vwap_zscore_threshold: float = 1.5,
        short_rsi_threshold: float = 65.0,
        take_profit_ticks: int = 20,
        stop_loss_ticks: int = 12,
        instrument: str = "NQ",
        slippage_ticks: int = 1,
        commission_per_rt: float = 4.10,
        eod_exit_bar_from_end: int = 3,
    ) -> str:
        """
        Execute backtest, save to file, and return summary JSON.
        """
        try:
            with open(input_file, "r") as f:
                payload = json.load(f)
        except Exception as e:
            return json.dumps({"error": f"Failed to read indicator data file {input_file}: {e}"})

        if "error" in payload:
            return json.dumps(payload)

        records: list[dict] = payload.get("full_data", [])
        if not records:
            return json.dumps({"error": "'full_data' in indicator file is empty."})

        df = pd.DataFrame(records)
        dt_col = "datetime" if "datetime" in df.columns else df.columns[0]
        df[dt_col] = pd.to_datetime(df[dt_col])
        df = df.set_index(dt_col).sort_index()

        required_cols = ["open", "high", "low", "close", "vwap_zscore", "rsi"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps(
                {"error": f"Missing columns in indicator data: {missing}. "
                          "Ensure IndicatorTool was called first."}
            )

        for col in required_cols + ["volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["vwap_zscore", "rsi", "open", "high", "low", "close"], inplace=True)

        if len(df) < 50:
            return json.dumps({"error": "Insufficient bars for backtesting (need ≥ 50)."})

        results = run_backtest(
            df=df,
            long_z_thresh=long_vwap_zscore_threshold,
            long_rsi_thresh=long_rsi_threshold,
            short_z_thresh=short_vwap_zscore_threshold,
            short_rsi_thresh=short_rsi_threshold,
            tp_ticks=take_profit_ticks,
            sl_ticks=stop_loss_ticks,
            instrument=instrument,
            slippage_ticks=slippage_ticks,
            commission_rt=commission_per_rt,
            eod_exit_bar_from_end=eod_exit_bar_from_end,
        )

        if "error" in results:
            return json.dumps(results)

        try:
            with open(output_file, "w") as f:
                json.dump(results, f, default=str)
        except Exception as e:
            return json.dumps({"error": f"Failed to save backtest results to {output_file}: {e}"})

        # Return a small summary to avoid LLM context overflow
        return_summary = {
            "instrument": results.get("instrument"),
            "params": results.get("params"),
            "performance": results.get("performance"),
            "exit_reason_breakdown": results.get("exit_reason_breakdown"),
            "saved_to": output_file,
            "total_trades_saved": len(results.get("trade_log", [])),
        }

        return json.dumps(return_summary, default=str)
