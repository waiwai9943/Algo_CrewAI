import json

files = {
    "primary": "backtest_results.json",
    "v1_tight": "backtest_results_v1.json",
    "v2_loose": "backtest_results_v2.json",
    "v3": "backtest_results_v3.json",
    "aggressive": "backtest_results_aggressive.json",
    "conservative": "backtest_results_conservative.json",
    "base": "backtest_results_base.json",
    "tight": "backtest_results_tight.json",
    "loose": "backtest_results_loose.json",
}

for label, fname in files.items():
    try:
        with open(fname) as f:
            d = json.load(f)
        p = d.get("params", {})
        perf = d.get("performance", {})
        print(f"=== {label} ({fname}) ===")
        print(f"  long_z={p.get('long_vwap_zscore_threshold')}  long_rsi={p.get('long_rsi_threshold')}  short_z={p.get('short_vwap_zscore_threshold')}  short_rsi={p.get('short_rsi_threshold')}")
        print(f"  TP={p.get('take_profit_ticks')}tks  SL={p.get('stop_loss_ticks')}tks  slip={p.get('slippage_ticks')}tks  comm=${p.get('commission_per_rt_usd')}")
        print(f"  Trades={perf.get('total_trades')}  WinRate={perf.get('win_rate_pct')}%  NetPnL=${perf.get('net_pnl_usd')}  MaxDD=${perf.get('max_drawdown_usd')}  Sharpe={perf.get('sharpe_ratio_annualised')}  PF={perf.get('profit_factor')}")
        er = d.get("exit_reason_breakdown", {})
        counts = er.get("count", {})
        sums = er.get("sum", {})
        print(f"  TP exits={counts.get('take_profit',0)} (${round(sums.get('take_profit',0),2)})  SL exits={counts.get('stop_loss',0)} (${round(sums.get('stop_loss',0),2)})  EOD exits={counts.get('eod_flat',0)} (${round(sums.get('eod_flat',0),2)})")
        print()
    except Exception as e:
        print(f"{label}: ERROR - {e}")
        print()
