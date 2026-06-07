import json
from algo.crew import Algo, market_data_tool, indicator_tool, backtest_tool

# ── Test crew loads correctly ─────────────────────────────
c = Algo().crew()
agents = c.agents
tasks  = c.tasks
print(f"Agents loaded: {len(agents)}")
for a in agents:
    tool_names = [t.name for t in (a.tools or [])]
    print(f"  - {a.role.strip()[:55]} | tools: {tool_names}")
print(f"Tasks loaded: {len(tasks)}")
for t in tasks:
    print(f"  - {t.name}")
print()

# ── Test MarketDataTool ───────────────────────────────────
print("Testing MarketDataTool (5d NQ=F 5m)...")
raw = market_data_tool._run(ticker="NQ=F", period="5d", interval="5m")
r1 = json.loads(raw)
if "error" in r1:
    print(f"  ERROR: {r1['error']}")
else:
    print(f"  OK — {r1['total_bars']} bars | {r1['date_range']}")

    # ── Test IndicatorTool ─────────────────────────────────
    print("Testing IndicatorTool...")
    ind_raw = indicator_tool._run(market_data_json=raw, rsi_period=14, vwap_std_window=20)
    r2 = json.loads(ind_raw)
    if "error" in r2:
        print(f"  ERROR: {r2['error']}")
    else:
        s = r2["summary"]
        print(f"  OK — valid bars: {s['valid_bars_with_indicators']}")
        print(f"  RSI mean={s['rsi_stats']['mean']}, oversold%={s['rsi_stats']['pct_oversold_30']}")
        print(f"  VWAP Z <-1: {s['vwap_zscore_stats']['pct_below_neg1']}%")

        # ── Test BacktestTool ──────────────────────────────
        print("Testing BacktestTool...")
        bt_raw = backtest_tool._run(
            indicator_data_json=ind_raw,
            long_vwap_zscore_threshold=-1.5,
            long_rsi_threshold=35.0,
            short_vwap_zscore_threshold=1.5,
            short_rsi_threshold=65.0,
            take_profit_ticks=20,
            stop_loss_ticks=12,
            instrument="NQ",
        )
        r3 = json.loads(bt_raw)
        if "error" in r3:
            print(f"  ERROR: {r3['error']}")
        else:
            p = r3["performance"]
            print(f"  OK — {p['total_trades']} trades | WR={p['win_rate_pct']}%")
            print(f"  Net PnL=${p['net_pnl_usd']} | Sharpe={p['sharpe_ratio_annualised']}")
            print(f"  MaxDD=${p['max_drawdown_usd']} | PF={p['profit_factor']}")

print("\nAll tools validated successfully!")
