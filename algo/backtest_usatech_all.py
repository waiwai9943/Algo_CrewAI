import pandas as pd
import numpy as np

CSV_PATH = r"data/2026.6.7USATECHIDXUSD-M5-No Session.csv"
print(f"Loading CSV: {CSV_PATH}")
raw = pd.read_csv(CSV_PATH)

# Parse datetime
raw["datetime"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str))
raw = raw.set_index("datetime").sort_index()
raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
df = raw.copy()
print(f"Bars: {len(df):,}")

# Indicators (Vectorized)
df["session_date"] = df.index.date
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
df["tp_vol"] = df["typical_price"] * df["volume"]

# Vectorized VWAP grouped by session_date
grouped = df.groupby("session_date")
df["cum_tp_vol"] = grouped["tp_vol"].cumsum()
df["cum_vol"] = grouped["volume"].cumsum()
df["vwap"] = np.where(df["cum_vol"] > 0, df["cum_tp_vol"] / df["cum_vol"], df["typical_price"])

# VWAP Z-Score
VWAP_STD_WINDOW = 20
deviation = df["close"] - df["vwap"]
rolling_std = deviation.rolling(window=VWAP_STD_WINDOW, min_periods=VWAP_STD_WINDOW).std()
df["vwap_zscore"] = deviation / rolling_std.replace(0, np.nan)

# RSI
RSI_PERIOD = 14
delta = df["close"].diff()
gain = delta.clip(lower=0)
loss = (-delta).clip(lower=0)
avg_gain = gain.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD, adjust=False).mean()
avg_loss = loss.ewm(com=RSI_PERIOD - 1, min_periods=RSI_PERIOD, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

# ADX Calculation
ADX_PERIOD = 14
df['prev_close'] = df['close'].shift(1)
df['tr'] = np.maximum(df['high'] - df['low'], 
                      np.maximum(abs(df['high'] - df['prev_close']), 
                                 abs(df['low'] - df['prev_close'])))

df['up_move'] = df['high'] - df['high'].shift(1)
df['down_move'] = df['low'].shift(1) - df['low']

df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

# Smoothed TR and DM using Wilder's Smoothing (alpha=1/14)
df['atr'] = df['tr'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df['atr'])
df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df['atr'])

df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, np.nan))
df['adx'] = df['dx'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

df.dropna(subset=["vwap_zscore", "rsi", "adx", "open", "high", "low", "close"], inplace=True)

# Backtest Engine
POINT_VALUE = 20.0
COMMISSION_RT = 4.10
SLIPPAGE_POINTS = 0.25

def run_backtest(df, long_z, long_rsi, short_z, short_rsi, tp_points, sl_points, adx_threshold=25, label=""):
    # EOD mask
    eod_mask = pd.Series(False, index=df.index)
    eod_mask.loc[df.groupby("session_date").tail(3).index] = True

    # REGIME FILTER: Only trade when ADX < adx_threshold
    long_signal = (df["vwap_zscore"] < long_z) & (df["rsi"] < long_rsi) & (df["adx"] < adx_threshold)
    short_signal = (df["vwap_zscore"] > short_z) & (df["rsi"] > short_rsi) & (df["adx"] < adx_threshold)

    trades = []
    position = None
    entry_price = 0.0
    
    bars = list(df.itertuples())
    n = len(bars)
    
    # We still need to iterate for precise trade management, but iteration over 800k rows is manageable if we don't do complex stuff inside
    for i, bar in enumerate(bars):
        if position is not None:
            is_eod = eod_mask.iloc[i]
            high = float(bar.high)
            low = float(bar.low)
            exit_price = None
            if position == "long":
                if high >= tp_price: exit_price = tp_price
                elif low <= sl_price: exit_price = sl_price
                elif is_eod: exit_price = float(bar.open)
            else:
                if low <= tp_price: exit_price = tp_price
                elif high >= sl_price: exit_price = sl_price
                elif is_eod: exit_price = float(bar.open)

            if exit_price is not None:
                raw_pnl_pts = (exit_price - entry_price) if position == "long" else (entry_price - exit_price)
                slip_cost_usd = SLIPPAGE_POINTS * POINT_VALUE * 2
                pnl_usd = (raw_pnl_pts * POINT_VALUE) - COMMISSION_RT - slip_cost_usd
                trades.append({"pnl_usd": round(pnl_usd, 2), "exit_bar": str(bar.Index)})
                position = None

        if position is None and i < n - 1 and not eod_mask.iloc[i]:
            if long_signal.iloc[i]:
                position = "long"
                entry_price = float(bars[i + 1].open) + SLIPPAGE_POINTS
                tp_price = entry_price + tp_points
                sl_price = entry_price - sl_points
            elif short_signal.iloc[i]:
                position = "short"
                entry_price = float(bars[i + 1].open) - SLIPPAGE_POINTS
                tp_price = entry_price - tp_points
                sl_price = entry_price + sl_points

    if not trades: return print("No trades generated")
    
    pnls = [t["pnl_usd"] for t in trades]
    n_win = len([p for p in pnls if p > 0])
    win_rate = n_win / len(trades)
    gross_profit = sum([p for p in pnls if p > 0])
    gross_loss = abs(sum([p for p in pnls if p <= 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    equity = np.cumsum(pnls)
    max_drawdown = float(np.min(equity - np.maximum.accumulate(equity)))
    trade_df = pd.DataFrame(trades)
    trade_df["exit_bar"] = pd.to_datetime(trade_df["exit_bar"])
    trade_df["exit_date"] = trade_df["exit_bar"].dt.date
    daily_pnl = [grp["pnl_usd"].sum() for _, grp in trade_df.groupby("exit_date")]
    sharpe = (np.mean(daily_pnl) / (np.std(daily_pnl) + 1e-9)) * np.sqrt(252) if len(daily_pnl) > 1 else float("nan")
    
    print(f"\n[{label}]")
    print(f"Total Trades: {len(trades):,}")
    print(f"Win Rate:     {win_rate*100:.2f}%")
    print(f"Net P&L:      ${sum(pnls):,.2f}")
    print(f"Max Drawdown: ${max_drawdown:,.2f}")
    print(f"Sharpe Ratio: {sharpe:.3f}")
    print(f"Profit Factor:{profit_factor:.3f}\n")

run_backtest(df, -2.0, 30.0, 2.0, 70.0, 20.0, 10.0, adx_threshold=25, label="Tight Variant (ADX Filtered) - ALL 15 YEARS DATA (2011-2026)")
