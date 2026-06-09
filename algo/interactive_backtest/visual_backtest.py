"""
visual_backtest.py
------------------
Interactive HTML Backtest for the VWAP+RSI Strategy using Backtesting.py.
We run this on the US Tech Index (NQ equivalent) M5 data.
"""

from backtesting import Strategy, Backtest
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────────────
# Indicators
# ─────────────────────────────────────────────────────────────────

def compute_vwap(high, low, close, volume, dates):
    """Compute session-anchored VWAP."""
    tp = (high + low + close) / 3.0
    vwap = np.empty_like(tp)
    cum_tp_vol = 0.0
    cum_vol = 0.0
    prev_date = None
    
    for i in range(len(tp)):
        current_date = dates[i].date()
        if current_date != prev_date:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            prev_date = current_date
            
        cum_tp_vol += tp[i] * volume[i]
        cum_vol += volume[i]
        vwap[i] = cum_tp_vol / cum_vol if cum_vol > 0 else tp[i]
        
    return vwap

def compute_vwap_zscore(close, vwap, window=20):
    """Compute VWAP Z-Score."""
    dev = close - vwap
    dev_series = pd.Series(dev)
    rolling_std = dev_series.rolling(window=window, min_periods=1).std()
    rolling_std = rolling_std.replace(0, np.nan).bfill()
    zscore = dev / rolling_std.values
    return zscore

def compute_rsi(close, period=14):
    """Compute Wilder's RSI."""
    delta = np.diff(close, prepend=close[0])
    gain = np.clip(delta, a_min=0, a_max=None)
    loss = -np.clip(delta, a_min=None, a_max=0)
    
    avg_gain = pd.Series(gain).ewm(com=period - 1, min_periods=1, adjust=False).mean()
    avg_loss = pd.Series(loss).ewm(com=period - 1, min_periods=1, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.values

# ─────────────────────────────────────────────────────────────────
# Strategy
# ─────────────────────────────────────────────────────────────────

class VwapRsiMeanReversion(Strategy):
    z_long = -2.0
    z_short = 2.0
    rsi_long = 30.0
    rsi_short = 70.0
    
    tp_pts = 20.0
    sl_pts = 10.0
    
    def init(self):
        dates = self.data.index
        
        self.vwap = self.I(compute_vwap, self.data.High, self.data.Low, self.data.Close, self.data.Volume, dates, name="VWAP", overlay=True)
        self.zscore = self.I(compute_vwap_zscore, self.data.Close, self.vwap, name="VWAP Z-Score")
        self.rsi = self.I(compute_rsi, self.data.Close, name="RSI")
        
        self.dates = dates

    def next(self):
        # EOD Flat logic: exit if within last 3 bars of the day
        current_time = self.data.index[-1].time()
        # For this dataset, end of session is approx 23:55 or before midnight. Let's just exit at 23:50.
        if current_time.hour == 23 and current_time.minute >= 45:
            if self.position:
                self.position.close()
            return
            
        if not self.position:
            # Entry rules
            if self.zscore[-1] < self.z_long and self.rsi[-1] < self.rsi_long:
                self.buy(sl=self.data.Close[-1] - self.sl_pts, tp=self.data.Close[-1] + self.tp_pts)
            elif self.zscore[-1] > self.z_short and self.rsi[-1] > self.rsi_short:
                self.sell(sl=self.data.Close[-1] + self.sl_pts, tp=self.data.Close[-1] - self.tp_pts)

# ─────────────────────────────────────────────────────────────────
# Execution
# ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    CSV_PATH = r"C:\QuantDataManager125\export\2026.6.7USATECHIDXUSD-M5-No Session.csv"
    print(f"Loading data from {CSV_PATH}...")
    
    df = pd.read_csv(CSV_PATH)
    df["datetime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))
    df = df.set_index("datetime").sort_index()
    
    # Capitalize for backtesting.py
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    
    # We use only a smaller chunk (e.g., 2 months) so the HTML plot doesn't crash the browser
    cutoff = df.index.max() - pd.DateOffset(months=2)
    df_plot = df[df.index >= cutoff].copy()
    
    print(f"Running backtest on {len(df_plot)} bars...")
    
    # NQ metrics: 1 full point = $20, but we will trade 1 unit where 1 point = $1, so we set margin=1. 
    # To accurately simulate $20 per point, we set commission and run it.
    bt = Backtest(df_plot, VwapRsiMeanReversion, cash=50000, margin=1.0, trade_on_close=False)
    stats = bt.run()
    
    print("\n--- BACKTEST RESULTS ---")
    print(stats)
    
    print("\nGenerating HTML plot...")
    # Generate the plot and save it to an interactive HTML file
    bt.plot(filename='usatech_strategy_plot.html', open_browser=False)
    print("Plot saved to usatech_strategy_plot.html!")
