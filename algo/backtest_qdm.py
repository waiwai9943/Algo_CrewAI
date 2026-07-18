"""
backtest_qdm.py
---------------
Run VWAP+RSI mean-reversion backtest directly on QuantDataManager CSV export.

Data: C:\QuantDataManager125\export\2026.6.7LIGHTCMDUSD-M5-No Session.csv
      (Light Crude Oil USD, 5-minute bars, Jan 2013 – Jun 2026)

Strategy: VWAP z-score + RSI mean-reversion
  - LONG  when vwap_zscore < -1.5 AND RSI < 35
  - SHORT when vwap_zscore >  1.5 AND RSI > 65
  - TP = 80 ticks ($800 for CL, $10/tick)
  - SL = 40 ticks ($400)
  - EOD flat 3 bars before session end
  - Slippage: 1 tick / side
  - Commission: $4.10 RT

CL tick = $0.01/barrel = $10 per tick (1000 barrels/contract)
"""

from __future__ import annotations

import json
import sys
import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────
# Load & parse QDM CSV
# ─────────────────────────────────────────────────────────────────

CSV_PATH = r"data/2026.6.7LIGHTCMDUSD-M5-No Session.csv"

print(f"Loading CSV: {CSV_PATH}")
raw = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(raw):,}  Cols: {raw.columns.tolist()}")

# Parse datetime
raw["datetime"] = pd.to_datetime(raw["Date"] + " " + raw["Time"], dayfirst=False)
raw = raw.set_index("datetime").sort_index()
raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)

# Use only last 2 years for reasonable backtest scope (still huge dataset)
cutoff = raw.index.max() - pd.DateOffset(years=2)
df = raw[raw.index >= cutoff].copy()
print(f"  Using last 2 years: {df.index[0]} to {df.index[-1]}")
print(f"  Bars: {len(df):,}")

# ─────────────────────────────────────────────────────────────────
# Indicator Computation
# ─────────────────────────────────────────────────────────────────

print("\nComputing indicators...")

# Session date (trading day)
df["session_date"] = df.index.date

# Typical price
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0

# Session-anchored VWAP
def compute_session_vwap(df: pd.DataFrame) -> pd.Series:
    vwap_values = np.empty(len(df))
    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_date = None
    for i, (idx, row) in enumerate(df.iterrows()):
        current_date = row["session_date"]
        if current_date != prev_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_date = current_date
        cum_tp_vol += float(row["typical_price"]) * float(row["volume"])
        cum_vol += float(row["volume"])
        vwap_values[i] = cum_tp_vol / cum_vol if cum_vol > 0 else float(row["typical_price"])
    return pd.Series(vwap_values, index=df.index, name="vwap")

df["vwap"] = compute_session_vwap(df)

# VWAP z-score (20-bar rolling std)
VWAP_STD_WINDOW = 20
deviation = df["close"] - df["vwap"]
rolling_std = deviation.rolling(window=VWAP_STD_WINDOW, min_periods=VWAP_STD_WINDOW).std()
df["vwap_zscore"] = deviation / rolling_std.replace(0, np.nan)

# RSI (14-period, Wilder smoothing)
RSI_PERIOD = 14
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD, adjust=False).mean()
avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

df.dropna(subset=["vwap_zscore", "rsi", "open", "high", "low", "close"], inplace=True)

valid = df.dropna(subset=["rsi", "vwap_zscore"])
print(f"  Valid bars: {len(valid):,}")
print(f"  RSI mean={valid['rsi'].mean():.1f}  pct<30={( valid['rsi']<30 ).mean()*100:.1f}%  pct>70={( valid['rsi']>70 ).mean()*100:.1f}%")
print(f"  VWAP-Z mean={valid['vwap_zscore'].mean():.3f}  pct<-1.5={(valid['vwap_zscore']<-1.5).mean()*100:.2f}%  pct>+1.5={(valid['vwap_zscore']>1.5).mean()*100:.2f}%")

# ─────────────────────────────────────────────────────────────────
# Backtest Engine (CL instrument)
# ─────────────────────────────────────────────────────────────────

INSTRUMENT = "CL"
TICK_SIZE = 0.01         # $0.01/barrel
TICK_VAL = 10.0          # $10/tick
COMMISSION_RT = 4.10
SLIPPAGE_TICKS = 1

def run_backtest(df, long_z, long_rsi, short_z, short_rsi, tp_ticks, sl_ticks, label=""):
    """Run single backtest and return metrics dict."""
    tick_size = TICK_SIZE
    tick_val = TICK_VAL

    df = df.copy()

    # Build EOD mask (last 3 bars of each session)
    eod_exit_bar_from_end = 3
    eod_mask = pd.Series(False, index=df.index)
    for date in df["session_date"].unique():
        day_bars = df[df["session_date"] == date]
        if len(day_bars) <= eod_exit_bar_from_end:
            eod_mask.loc[day_bars.index] = True
        else:
            eod_mask.loc[day_bars.index[-eod_exit_bar_from_end:]] = True

    long_signal = (df["vwap_zscore"] < long_z) & (df["rsi"] < long_rsi)
    short_signal = (df["vwap_zscore"] > short_z) & (df["rsi"] > short_rsi)

    trades = []
    position = None
    entry_price = 0.0
    entry_bar_idx = 0
    tp_price = 0.0
    sl_price = 0.0

    bars = list(df.itertuples())
    n = len(bars)

    for i, bar in enumerate(bars):
        if position is not None:
            is_eod = eod_mask.iloc[i]
            high = float(bar.high)
            low = float(bar.low)
            exit_price = None
            exit_reason = ""

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
            else:
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
                raw_pnl_pts = (exit_price - entry_price) if position == "long" else (entry_price - exit_price)
                slip_cost = SLIPPAGE_TICKS * tick_val * 2
                pnl_usd = (raw_pnl_pts / tick_size) * tick_val - COMMISSION_RT - slip_cost
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

        if position is None and i < n - 1 and not eod_mask.iloc[i]:
            next_bar = bars[i + 1]
            exec_price = float(next_bar.open)
            if long_signal.iloc[i]:
                position = "long"
                entry_price = exec_price + SLIPPAGE_TICKS * tick_size
                tp_price = entry_price + tp_ticks * tick_size
                sl_price = entry_price - sl_ticks * tick_size
                entry_bar_idx = i + 1
            elif short_signal.iloc[i]:
                position = "short"
                entry_price = exec_price - SLIPPAGE_TICKS * tick_size
                tp_price = entry_price - tp_ticks * tick_size
                sl_price = entry_price + sl_ticks * tick_size
                entry_bar_idx = i + 1

    if not trades:
        return {"label": label, "error": "No trades generated"}

    pnls = [t["pnl_usd"] for t in trades]
    n_trades = len(trades)
    winners = [p for p in pnls if p > 0]
    losers = [p for p in pnls if p <= 0]
    n_win = len(winners)
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

    # Daily Sharpe
    trade_df = pd.DataFrame(trades)
    trade_df["exit_bar"] = pd.to_datetime(trade_df["exit_bar"])
    trade_df["exit_date"] = trade_df["exit_bar"].dt.date
    daily_pnl = [grp["pnl_usd"].sum() for _, grp in trade_df.groupby("exit_date")]
    if len(daily_pnl) > 1:
        daily_arr = np.array(daily_pnl)
        sharpe = (np.mean(daily_arr) / (np.std(daily_arr) + 1e-9)) * np.sqrt(252)
    else:
        sharpe = float("nan")

    return {
        "label": label,
        "params": {
            "long_z": long_z, "long_rsi": long_rsi,
            "short_z": short_z, "short_rsi": short_rsi,
            "tp_ticks": tp_ticks, "sl_ticks": sl_ticks,
            "tp_usd": tp_ticks * TICK_VAL,
            "sl_usd": sl_ticks * TICK_VAL,
        },
        "performance": {
            "total_trades": n_trades,
            "win_rate_pct": round(win_rate * 100, 2),
            "avg_win_usd": round(float(avg_win), 2),
            "avg_loss_usd": round(float(avg_loss), 2),
            "avg_rr": round(float(avg_rr), 3),
            "net_pnl_usd": round(sum(pnls), 2),
            "gross_profit_usd": round(gross_profit, 2),
            "gross_loss_usd": round(gross_loss, 2),
            "profit_factor": round(profit_factor, 3) if profit_factor != float("inf") else "inf",
            "max_drawdown_usd": round(max_drawdown, 2),
            "sharpe_annualised": round(float(sharpe), 3) if not np.isnan(sharpe) else "N/A",
        },
        "trade_log": trades,
        "equity_curve": [round(float(e), 2) for e in equity.tolist()],
    }

# ─────────────────────────────────────────────────────────────────
# Run parameter variants
# ─────────────────────────────────────────────────────────────────

print("\nRunning backtests on QDM data...\n")

variants = [
    # label,           long_z, long_rsi, short_z, short_rsi, tp_ticks, sl_ticks
    ("Base (z=-1.5, RSI 35/65, TP80 SL40)",   -1.5, 35.0, 1.5, 65.0, 80, 40),
    ("Tight (z=-1.8, RSI 30/70, TP60 SL30)",  -1.8, 30.0, 1.8, 70.0, 60, 30),
    ("Loose (z=-1.2, RSI 40/60, TP100 SL50)", -1.2, 40.0, 1.2, 60.0, 100, 50),
    ("Aggressive (z=-1.0, RSI 40/60, TP120 SL60)", -1.0, 40.0, 1.0, 60.0, 120, 60),
]

all_results = []
for label, lz, lr, sz, sr, tp, sl in variants:
    print(f"Running: {label}...")
    res = run_backtest(df, lz, lr, sz, sr, tp, sl, label=label)
    all_results.append(res)
    if "error" in res:
        print(f"  ERROR: {res['error']}")
    else:
        p = res["performance"]
        print(f"  Trades={p['total_trades']}  WinRate={p['win_rate_pct']}%  NetPnL=${p['net_pnl_usd']:,.2f}  MaxDD=${p['max_drawdown_usd']:,.2f}  Sharpe={p['sharpe_annualised']}  PF={p['profit_factor']}")
    print()

# Save all results
with open("backtest_results_qdm.json", "w") as f:
    json.dump(all_results, f, default=str, indent=2)
print("\nAll results saved to backtest_results_qdm.json")
