import pandas as pd
import numpy as np
from datetime import timedelta
from backtesting import Backtest, Strategy

CSV_PATH = r"data/2026.6.7USATECHIDXUSD-M5-No Session.csv"
print(f"Loading CSV: {CSV_PATH}")
raw = pd.read_csv(CSV_PATH)

# Parse datetime
raw["datetime"] = pd.to_datetime(raw["Date"].astype(str) + " " + raw["Time"].astype(str))
raw = raw.set_index("datetime").sort_index()
raw.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
df = raw.copy()

print(f"Calculating Indicators...")
# Indicators (Vectorized)
df["session_date"] = df.index.date
df["typical_price"] = (df["high"] + df["low"] + df["close"]) / 3.0
df["tp_vol"] = df["typical_price"] * df["volume"]

# VWAP
grouped = df.groupby("session_date")
df["cum_tp_vol"] = grouped["tp_vol"].cumsum()
df["cum_vol"] = grouped["volume"].cumsum()
df["vwap"] = np.where(df["cum_vol"] > 0, df["cum_tp_vol"] / df["cum_vol"], df["typical_price"])

# Z-Score
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

# ADX
ADX_PERIOD = 14
df['prev_close'] = df['close'].shift(1)
df['tr'] = np.maximum(df['high'] - df['low'], 
                      np.maximum(abs(df['high'] - df['prev_close']), 
                                 abs(df['low'] - df['prev_close'])))
df['up_move'] = df['high'] - df['high'].shift(1)
df['down_move'] = df['low'].shift(1) - df['low']
df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
df['atr'] = df['tr'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()
df['plus_di'] = 100 * (df['plus_dm'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df['atr'])
df['minus_di'] = 100 * (df['minus_dm'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean() / df['atr'])
df['dx'] = 100 * (abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di']).replace(0, np.nan))
df['adx'] = df['dx'].ewm(alpha=1/ADX_PERIOD, adjust=False).mean()

df.dropna(subset=["vwap_zscore", "rsi", "adx", "open", "high", "low", "close"], inplace=True)

POINT_VALUE = 20.0
COMMISSION_RT = 4.10
SLIPPAGE_POINTS = 0.25

# EOD Mask
eod_mask = pd.Series(False, index=df.index)
eod_mask.loc[df.groupby("session_date").tail(3).index] = True
df['is_eod'] = eod_mask

def get_signal_array(df_slice, long_z, short_z, rsi_thresh, adx_thresh):
    long_signal = (df_slice["vwap_zscore"] < long_z) & (df_slice["rsi"] < rsi_thresh) & (df_slice["adx"] < adx_thresh)
    short_signal = (df_slice["vwap_zscore"] > short_z) & (df_slice["rsi"] > (100 - rsi_thresh)) & (df_slice["adx"] < adx_thresh)
    
    # 1 for long, -1 for short, 0 for neutral
    signals = np.zeros(len(df_slice))
    signals[long_signal] = 1
    signals[short_signal] = -1
    return signals

def run_backtest_fast(df_slice, long_z, short_z, rsi_thresh, adx_thresh, tp_points, sl_points):
    """Fast backtest function for parameter grid search"""
    signals = get_signal_array(df_slice, long_z, short_z, rsi_thresh, adx_thresh)
    
    trades = []
    position = None
    entry_price = 0.0
    
    bars = list(df_slice.itertuples())
    n = len(bars)
    
    for i, bar in enumerate(bars):
        if position is not None:
            is_eod = bar.is_eod
            high, low = float(bar.high), float(bar.low)
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
                pnl_usd = (raw_pnl_pts * POINT_VALUE) - COMMISSION_RT - (SLIPPAGE_POINTS * POINT_VALUE * 2)
                trades.append(pnl_usd)
                position = None

        if position is None and i < n - 1 and not bar.is_eod:
            sig = signals[i]
            if sig == 1:
                position = "long"
                entry_price = float(bars[i + 1].open) + SLIPPAGE_POINTS
                tp_price, sl_price = entry_price + tp_points, entry_price - sl_points
            elif sig == -1:
                position = "short"
                entry_price = float(bars[i + 1].open) - SLIPPAGE_POINTS
                tp_price, sl_price = entry_price - tp_points, entry_price + sl_points
                
    if not trades: return -999999
    return sum(trades)

print("Running Walk Forward Optimization...")
# Grid Search space
z_scores = [-2.0, -2.5, -3.0]
rsi_threshes = [20, 25, 30]
adx_threshes = [20, 25, 30]

start_date = df.index.min()
end_date = df.index.max()
IS_YEARS = 2
OOS_MONTHS = 6

current_date = start_date + pd.DateOffset(years=IS_YEARS)

# Pre-populate empty master signal arrays
df['wfo_signal'] = 0

while current_date < end_date:
    train_start = current_date - pd.DateOffset(years=IS_YEARS)
    train_end = current_date
    test_end = current_date + pd.DateOffset(months=OOS_MONTHS)
    
    print(f"Optimizing IS: {train_start.date()} to {train_end.date()} | Testing OOS: {train_end.date()} to {min(test_end, end_date).date()}")
    
    df_is = df[(df.index >= train_start) & (df.index < train_end)]
    oos_mask = (df.index >= train_end) & (df.index < test_end)
    df_oos = df[oos_mask]
    
    if len(df_is) == 0 or len(df_oos) == 0:
        current_date = test_end
        continue
        
    best_pnl = -float('inf')
    best_params = None
    
    for z in z_scores:
        for rsi in rsi_threshes:
            for adx in adx_threshes:
                pnl = run_backtest_fast(df_is, z, abs(z), rsi, adx, 20.0, 10.0)
                if pnl > best_pnl:
                    best_pnl = pnl
                    best_params = (z, abs(z), rsi, adx)
                    
    if best_params:
        print(f"  Best IS Params: Z={best_params[0]}, RSI={best_params[2]}, ADX={best_params[3]} -> IS PnL: ${best_pnl:,.2f}")
        # Apply out of sample signals to master dataframe
        oos_signals = get_signal_array(df_oos, *best_params)
        df.loc[oos_mask, 'wfo_signal'] = oos_signals
    
    current_date = test_end

print("\nRunning Master Stitched Backtest...")

class WFOStitchedStrategy(Strategy):
    tp_points = 20.0
    sl_points = 10.0
    
    def init(self):
        self.wfo_signal = self.I(lambda x: x, self.data.wfo_signal, name='WFO Signal')
        self.is_eod = self.I(lambda x: x, self.data.is_eod, name='Is EOD')
        
        self.vwap_zscore = self.I(lambda x: x, self.data.vwap_zscore, name='Z-Score')
        self.adx = self.I(lambda x: x, self.data.adx, name='ADX')

    def next(self):
        if self.position:
            if self.is_eod[-1]:
                self.position.close()
            return
            
        if self.is_eod[-1]:
            return
            
        sig = self.wfo_signal[-1]
        
        if sig == 1:
            self.buy(size=20, sl=self.data.Close[-1] - self.sl_points, tp=self.data.Close[-1] + self.tp_points)
        elif sig == -1:
            self.sell(size=20, sl=self.data.Close[-1] + self.sl_points, tp=self.data.Close[-1] - self.tp_points)

df_bt = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})
# We must use raw pandas dataframe logic for Open/High/Low/Close
bt = Backtest(df_bt, WFOStitchedStrategy, cash=1000000, margin=1.0, trade_on_close=False, exclusive_orders=True)
stats = bt.run()
print(stats)

print("Generating backtesting.py HTML plot (resampled)...")
bt.plot(filename='wfo_backtest.html', resample=True, open_browser=False)
print("Finished!")
