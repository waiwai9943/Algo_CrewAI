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
    donchian_period: int = Field(
        default=20,
        description="Donchian Channel lookback period. Default: 20.",
    )
    ema_period: int = Field(
        default=200,
        description="Exponential Moving Average period for trend filtering. Default: 200.",
    )
    atr_period: int = Field(
        default=14,
        description="Average True Range (ATR) period. Default: 14.",
    )
    sl_atr_multiplier: float = Field(
        default=2.0,
        description="ATR multiplier for Stop Loss. Default: 2.0.",
    )
    tp_atr_multiplier: float = Field(
        default=4.0,
        description="ATR multiplier for Take Profit. Default: 4.0.",
    )
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
        default=0,
        description=(
            "Bars before session end to force-close any open position. "
            "Set to 0 to disable EOD flat (enable overnight swing trading). Default: 0."
        ),
    )
    run_walk_forward: bool = Field(
        default=True,
        description="If True, runs a 10-year Walk-Forward Optimization (WFO) to avoid overfitting. Default: True.",
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
    donchian_period: int,
    ema_period: int,
    sl_atr_mult: float,
    tp_atr_mult: float,
    instrument: str,
    slippage_ticks: int,
    commission_rt: float,
    eod_exit_bar_from_end: int,
) -> dict:
    """
    Vectorised backtest of a Donchian Breakout + EMA strategy.

    Entry logic (next-bar open execution):
        LONG : close > ema AND close > donchian_high.shift(1)
        SHORT: close < ema AND close < donchian_low.shift(1)

    Exit logic (whichever comes first):
        - Take-profit: price moves tp_atr_mult * ATR in favour
        - Stop-loss:   price moves sl_atr_mult * ATR against
        - EOD (optional): last `eod_exit_bar_from_end` bars of each session
    """
    spec = INSTRUMENT_SPECS.get(instrument.upper(), INSTRUMENT_SPECS["NQ"])
    tick_size = spec["tick_size"]
    tick_val = spec["tick_value_usd"]
    point_val = spec["point_value_usd"]

    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Shift donchian channel boundaries to avoid lookahead bias
    df["donchian_high_prev"] = df["donchian_high"].shift(1)
    df["donchian_low_prev"] = df["donchian_low"].shift(1)

    # Build session-end mask if eod_exit is active
    eod_mask = pd.Series(False, index=df.index)
    if eod_exit_bar_from_end > 0:
        session_dates = df["session_date"].unique() if "session_date" in df.columns else [None]
        for date in session_dates:
            if date is None:
                continue
            day_bars = df[df["session_date"] == date]
            if len(day_bars) <= eod_exit_bar_from_end:
                eod_mask.loc[day_bars.index] = True
            else:
                eod_mask.loc[day_bars.index[-eod_exit_bar_from_end:]] = True

    # Signal definitions
    long_signal = (df["close"] > df["ema"]) & (df["close"] > df["donchian_high_prev"])
    short_signal = (df["close"] < df["ema"]) & (df["close"] < df["donchian_low_prev"])

    # Convert to numpy arrays for maximum performance
    opens = df["open"].values
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    atrs = df["atr"].values
    eod_mask_vals = eod_mask.values
    long_signal_vals = long_signal.values
    short_signal_vals = short_signal.values

    trades: list[dict] = []
    position: str | None = None   # 'long', 'short', or None
    entry_price: float = 0.0
    entry_bar_idx: int = 0
    tp_price: float = 0.0
    sl_price: float = 0.0

    n = len(df)

    for i in range(n):
        # ── Check exit conditions for open position ──────────────
        if position is not None:
            is_eod = eod_mask_vals[i]
            high = highs[i]
            low = lows[i]
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
                    exit_price = opens[i]
                    exit_reason = "eod_flat"
            else:  # short
                if low <= tp_price:
                    exit_price = tp_price
                    exit_reason = "take_profit"
                elif high >= sl_price:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                elif is_eod:
                    exit_price = opens[i]
                    exit_reason = "eod_flat"

            if exit_price is not None:
                raw_pnl_pts = (
                    (exit_price - entry_price) if position == "long"
                    else (entry_price - exit_price)
                )
                slip_cost = slippage_ticks * tick_val * 2
                pnl_usd = (raw_pnl_pts * point_val) - commission_rt - slip_cost

                trades.append({
                    "direction": position,
                    "entry_bar": str(df.index[entry_bar_idx]),
                    "exit_bar": str(df.index[i]),
                    "entry_price": round(entry_price, 4),
                    "exit_price": round(exit_price, 4),
                    "exit_reason": exit_reason,
                    "pnl_usd": round(pnl_usd, 2),
                    "pnl_points": round(raw_pnl_pts, 4),
                })
                position = None

        # ── Check entry signals for new position (next bar execution) ──
        if position is None and i < n - 1 and not eod_mask_vals[i]:
            exec_price = opens[i + 1]
            atr_val = atrs[i]

            if long_signal_vals[i] and atr_val > 0 and not np.isnan(atr_val):
                position = "long"
                # Slippage added to entry price
                entry_price = exec_price + slippage_ticks * tick_size
                tp_price = entry_price + tp_atr_mult * atr_val
                sl_price = entry_price - sl_atr_mult * atr_val
                entry_bar_idx = i + 1

            elif short_signal_vals[i] and atr_val > 0 and not np.isnan(atr_val):
                position = "short"
                # Slippage subtracted from entry price
                entry_price = exec_price - slippage_ticks * tick_size
                tp_price = entry_price - tp_atr_mult * atr_val
                sl_price = entry_price + sl_atr_mult * atr_val
                entry_bar_idx = i + 1

    if not trades:
        return {"error": "No trades generated."}

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

    # Annualised Sharpe ratio based on daily returns in pure Python
    daily_totals: dict[str, float] = {}
    for t in trades:
        # Extract YYYY-MM-DD date string from exit_bar string
        date_str = t["exit_bar"][:10]
        daily_totals[date_str] = daily_totals.get(date_str, 0.0) + t["pnl_usd"]

    daily_pnl = list(daily_totals.values())
    if len(daily_pnl) > 1:
        daily_arr = np.array(daily_pnl)
        sharpe = (np.mean(daily_arr) / (np.std(daily_arr) + 1e-9)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "instrument": instrument,
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
            "sharpe_ratio_annualised": round(float(sharpe), 3) if not math.isnan(sharpe) else 0.0,
        },
        "trade_log": trades,
        "equity_curve": [round(float(e), 2) for e in equity.tolist()],
    }


# ─────────────────────────────────────────────
# Walk-Forward Optimization
# ─────────────────────────────────────────────

def run_wfo_optimization(
    df: pd.DataFrame,
    instrument: str,
    slippage_ticks: int,
    commission_rt: float,
    eod_exit_bar_from_end: int,
) -> dict:
    """
    Executes Walk-Forward Optimization (WFO) over a 10-year period (2016-2026).
    """
    df = df.copy()
    df.index = pd.to_datetime(df.index)

    # Walk-forward settings: utilize the 10-year period 2016 to 2026.
    # We train on 2-year In-Sample windows and walk forward in 1-year steps.
    start_test_date = pd.to_datetime("2016-01-01")
    end_date = df.index.max()
    
    is_window = pd.DateOffset(years=2)
    oos_window = pd.DateOffset(years=1)
    
    current_date = start_test_date
    
    # Larger Donchian period grid search space [40, 60, 80, 100] to reduce trade frequency
    # and fit a robust intraday/swing breakout trading style.
    donchian_grid = [40, 60, 80, 100]
    sl_grid = [2.5, 3.5, 4.5]
    tp_grid = [5.0, 7.5, 10.0]
    
    all_oos_trades = []
    wfo_steps = []
    
    while current_date < end_date:
        train_start = current_date - is_window
        train_end = current_date
        test_end = current_date + oos_window
        
        df_is = df[(df.index >= train_start) & (df.index < train_end)]
        df_oos = df[(df.index >= train_end) & (df.index < test_end)]
        
        if len(df_is) < 200 or len(df_oos) < 50:
            current_date = test_end
            continue
            
        best_sharpe = -float("inf")
        best_params = (40, 2.0, 4.0)
        
        # Optimize on In-Sample
        for dp in donchian_grid:
            df_is_temp = df_is.copy()
            df_is_temp["donchian_high"] = df_is_temp["high"].rolling(window=dp).max()
            df_is_temp["donchian_low"] = df_is_temp["low"].rolling(window=dp).min()
            
            for sl in sl_grid:
                for tp in tp_grid:
                    res = run_backtest(
                        df=df_is_temp,
                        donchian_period=dp,
                        ema_period=200,
                        sl_atr_mult=sl,
                        tp_atr_mult=tp,
                        instrument=instrument,
                        slippage_ticks=slippage_ticks,
                        commission_rt=commission_rt,
                        eod_exit_bar_from_end=eod_exit_bar_from_end,
                    )
                    if "error" not in res:
                        sharpe = res["performance"]["sharpe_ratio_annualised"]
                        if sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_params = (dp, sl, tp)
                            
        # Run optimal parameters on Out-of-Sample
        dp, sl, tp = best_params
        df_oos_temp = df_oos.copy()
        df_oos_temp["donchian_high"] = df_oos_temp["high"].rolling(window=dp).max()
        df_oos_temp["donchian_low"] = df_oos_temp["low"].rolling(window=dp).min()
        
        oos_res = run_backtest(
            df=df_oos_temp,
            donchian_period=dp,
            ema_period=200,
            sl_atr_mult=sl,
            tp_atr_mult=tp,
            instrument=instrument,
            slippage_ticks=slippage_ticks,
            commission_rt=commission_rt,
            eod_exit_bar_from_end=eod_exit_bar_from_end,
        )
        
        step_trades = oos_res.get("trade_log", [])
        all_oos_trades.extend(step_trades)
        
        wfo_steps.append({
            "is_start": str(train_start.date()),
            "is_end": str(train_end.date()),
            "oos_start": str(train_end.date()),
            "oos_end": str(min(test_end, end_date).date()),
            "optimal_params": {
                "donchian_period": dp,
                "sl_atr_multiplier": sl,
                "tp_atr_multiplier": tp,
            },
            "is_sharpe": round(best_sharpe, 3),
            "oos_trades_count": len(step_trades),
            "oos_net_pnl": round(sum(t["pnl_usd"] for t in step_trades), 2) if step_trades else 0.0
        })
        
        current_date = test_end

    if not all_oos_trades:
        return {"error": "WFO yielded no trades in out-of-sample periods."}

    # Stitch metrics
    pnls = [t["pnl_usd"] for t in all_oos_trades]
    n_trades = len(all_oos_trades)
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

    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_drawdown = float(np.min(drawdown))

    # Annualised Sharpe ratio based on daily returns in pure Python
    daily_totals: dict[str, float] = {}
    for t in all_oos_trades:
        date_str = t["exit_bar"][:10]
        daily_totals[date_str] = daily_totals.get(date_str, 0.0) + t["pnl_usd"]

    daily_pnl = list(daily_totals.values())
    if len(daily_pnl) > 1:
        daily_arr = np.array(daily_pnl)
        sharpe = (np.mean(daily_arr) / (np.std(daily_arr) + 1e-9)) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        "instrument": instrument,
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
            "sharpe_ratio_annualised": round(float(sharpe), 3),
        },
        "wfo_steps": wfo_steps,
        "trade_log": all_oos_trades,
        "equity_curve": [round(float(e), 2) for e in equity.tolist()],
    }


# ─────────────────────────────────────────────
# Tool
# ─────────────────────────────────────────────

class BacktestTool(BaseTool):
    """
    Runs a vectorised backtest or a 10-year Walk-Forward Optimization (WFO)
    of the Donchian Breakout + EMA trend strategy.
    """

    name: str = "Strategy Backtester"
    description: str = (
        "Runs a vectorised backtest or 10-year Walk-Forward Optimization (WFO) of "
        "the Donchian Breakout + EMA strategy. Accepts parameter ranges, "
        "instrument, and optimization flags. Saves detailed logs and returns a summary JSON."
    )
    args_schema: Type[BaseModel] = BacktestInput

    def _run(
        self,
        input_file: str = "indicator_data.json",
        output_file: str = "backtest_results.json",
        donchian_period: int = 20,
        ema_period: int = 200,
        atr_period: int = 14,
        sl_atr_multiplier: float = 2.0,
        tp_atr_multiplier: float = 4.0,
        instrument: str = "NQ",
        slippage_ticks: int = 1,
        commission_per_rt: float = 4.10,
        eod_exit_bar_from_end: int = 0,
        run_walk_forward: bool = True,
    ) -> str:
        """
        Execute backtest/WFO, save to file, and return summary JSON.
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

        required_cols = ["open", "high", "low", "close", "donchian_high", "donchian_low", "ema", "atr"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            return json.dumps(
                {"error": f"Missing columns in indicator data: {missing}. "
                          "Ensure IndicatorTool was called first with breakout settings."}
            )

        for col in required_cols + ["volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        df.dropna(subset=["donchian_high", "donchian_low", "ema", "atr", "open", "high", "low", "close"], inplace=True)

        if len(df) < 100:
            return json.dumps({"error": "Insufficient bars for backtesting."})

        if run_walk_forward:
            results = run_wfo_optimization(
                df=df,
                instrument=instrument,
                slippage_ticks=slippage_ticks,
                commission_rt=commission_per_rt,
                eod_exit_bar_from_end=eod_exit_bar_from_end,
            )
        else:
            results = run_backtest(
                df=df,
                donchian_period=donchian_period,
                ema_period=ema_period,
                sl_atr_mult=sl_atr_multiplier,
                tp_atr_mult=tp_atr_multiplier,
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

        return_summary = {
            "instrument": results.get("instrument"),
            "performance": results.get("performance"),
            "saved_to": output_file,
            "total_trades_saved": len(results.get("trade_log", [])),
        }
        if "wfo_steps" in results:
            return_summary["wfo_steps_count"] = len(results.get("wfo_steps", []))
            return_summary["wfo_steps"] = results.get("wfo_steps")

        return json.dumps(return_summary, default=str)
