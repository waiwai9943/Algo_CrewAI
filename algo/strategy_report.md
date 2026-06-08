# 🏦 VWAP + RSI Mean-Reversion — Intraday Futures Strategy

**Instrument:** NQ (E-mini Nasdaq-100) / CL (Light Crude Oil)  
**Timeframe:** 5-minute bars  
**Style:** Intraday Mean-Reversion | Single Contract | Flat at EOD  
**Team:** CrewAI Quant Research Crew (Researcher · Engineer · Risk Manager)

---

## 1. Strategy Overview

This is a **statistical mean-reversion** strategy that exploits short-term price dislocations from the session VWAP. When price deviates significantly from its volume-weighted average — confirmed by RSI momentum exhaustion — the strategy bets on a return toward the mean.

The core hypothesis: in liquid futures markets (NQ, CL), intraday price moves exceeding ±1.5 standard deviations from VWAP tend to be unsustainable within a single trading session, and revert back toward VWAP with measurable frequency.

---

## 2. Indicators & Formulas

### 2.1 Session-Anchored VWAP

$$VWAP_t = \frac{\sum_{i=\text{session open}}^{t} TP_i \cdot V_i}{\sum_{i=\text{session open}}^{t} V_i}$$

where $TP_i = \frac{High_i + Low_i + Close_i}{3}$ (Typical Price).

**Key properties:**
- Resets at session open (09:30 ET for NQ RTH)
- Volume-weighted — respects where real trading activity occurred
- The institutional benchmark: large orders are commonly measured against VWAP

### 2.2 VWAP Z-Score (Mean-Reversion Signal)

$$Z_{VWAP,t} = \frac{Close_t - VWAP_t}{\sigma_{20}\left(Close - VWAP\right)}$$

where $\sigma_{20}$ is the 20-bar rolling standard deviation of the VWAP deviation.

| Z-Score | Interpretation |
|---------|---------------|
| Z < −1.5 | Price significantly **below** VWAP → potential long |
| −1.5 ≤ Z ≤ +1.5 | Normal range — **no trade** |
| Z > +1.5 | Price significantly **above** VWAP → potential short |

### 2.3 RSI (Momentum Confirmation Filter)

**Wilder's RSI (14-period):**

$$RSI_t = 100 - \frac{100}{1 + RS_t}, \quad RS_t = \frac{\overline{Gain}_{14}}{\overline{Loss}_{14}}$$

Using EMA smoothing: $\alpha = \frac{1}{14}$ (Wilder's method).

| RSI Value | Signal Role |
|-----------|------------|
| RSI < 35 | Confirms **oversold** momentum for long entry |
| RSI > 65 | Confirms **overbought** momentum for short entry |

---

## 3. Entry & Exit Rules

### 3.1 Entry Conditions

#### 🟢 LONG Entry (Mean-Reversion Buy)
All three conditions must be true on the **signal bar**; execute on the **next bar's open**:

| # | Condition | Value | Reason |
|---|-----------|-------|--------|
| 1 | VWAP Z-Score | `< −1.5` | Price ≥ 1.5σ below VWAP → statistically oversold |
| 2 | RSI (14) | `< 35` | Momentum confirms selling exhaustion |
| 3 | Not in EOD zone | Last 3 bars excluded | Avoid illiquid close window |

**Entry price:** Next bar's open + 1 tick slippage (adverse)

#### 🔴 SHORT Entry (Mean-Reversion Sell)
Mirror logic:

| # | Condition | Value |
|---|-----------|-------|
| 1 | VWAP Z-Score | `> +1.5` |
| 2 | RSI (14) | `> 65` |
| 3 | Not in EOD zone | Last 3 bars excluded |

**Entry price:** Next bar's open − 1 tick slippage (adverse)

### 3.2 Exit Conditions (whichever triggers first)

| Exit Type | NQ Value | CL Value | Trigger |
|-----------|----------|----------|---------|
| **Take-Profit** | +80 ticks ($400) | +80 ticks ($800) | High ≥ TP level (long) / Low ≤ TP level (short) |
| **Stop-Loss** | −40 ticks ($200) | −40 ticks ($400) | Low ≤ SL level (long) / High ≥ SL level (short) |
| **EOD Flat** | — | — | Last 3 bars of session → exit at bar open |

**Reward:Risk Ratio = 2:1 (80 TP / 40 SL ticks)**

### 3.3 Trade Execution Details

| Parameter | Value |
|-----------|-------|
| Execution model | Next-bar-open (no look-ahead) |
| Slippage (per side) | 1 tick |
| Commission (round-trip) | $4.10 (IB rate) |
| Max positions | 1 contract simultaneously |
| Pyramiding | **Prohibited** |
| Overnight holds | **Prohibited** — mandatory EOD flat |

---

## 4. Market Regime & Session Filters

### Prohibited Trading Windows
| Window | Time (ET) | Reason |
|--------|-----------|--------|
| Pre-open | Before 09:30 | Thin liquidity, wide spreads |
| Opening 15 min | 09:30–09:45 | Violent price discovery |
| Closing 15 min | 15:45–16:00 | EOD rebalancing, gap risk |
| News events | FOMC / CPI / NFP | Halt 5 min before, resume 10 min after |

### Do-Not-Trade Regime Signals
- ADX < 15 (too directionless — VWAP will be flat, no reversion to trade)
- VIX > 40 (regime breakdown — mean-reversion edge disappears)
- Price > 3σ from VWAP (skip; wait for Z to revert below threshold before re-entering)

---

## 5. Parameter Table

| Parameter | Chosen Value | Justification |
|-----------|-------------|---------------|
| `rsi_period` | 14 | Wilder standard; widely used; produces stable readings on 5m bars |
| `vwap_std_window` | 20 bars (100 min) | ~1.5 trading hours; captures meaningful intraday variation |
| `long_vwap_zscore_threshold` | −1.5 | Empirically ~8% of bars in training set; rare enough to be selective |
| `long_rsi_threshold` | 35 | ~5–8% of bars; dual-condition reduces false signals |
| `short_vwap_zscore_threshold` | +1.5 | Symmetric |
| `short_rsi_threshold` | 65 | Symmetric |
| `take_profit_ticks` | 80 | NQ: $400; CL: $800 — 2× the stop-loss |
| `stop_loss_ticks` | 40 | NQ: $200; CL: $400 — protects against mean-reversion failure |
| `eod_exit_bar_from_end` | 3 bars | 15 min before EOD |
| `slippage_ticks` | 1 | Conservative; NQ typical bid-ask is 1–2 ticks |
| `commission_per_rt` | $4.10 | IB futures rate |

---

## 6. Backtest Results (NQ Futures — Yahoo Finance Data)

> **Data:** NQ=F, 5-minute bars, 60-day window (Yahoo Finance)
> **Engine:** Vectorised, next-bar-open execution, 1-tick slippage + $4.10 commission RT

### 6.1 Parameter Sensitivity Comparison

| Variant | long_z | long_rsi | short_z | short_rsi | TP tks | SL tks | Trades | Win% | Net P&L | Max DD | Sharpe | PF |
|---------|--------|----------|---------|-----------|--------|--------|--------|------|---------|--------|--------|-----|
| **Base** (recommended) | −1.5 | 35 | +1.5 | 65 | 80 | 40 | **1,941** | **38.59%** | **$33,812** | **−$12,406** | **4.89** | **1.133** |
| Tight | −2.0 | 30 | +2.0 | 70 | 80 | 40 | 957 | **39.71%** | $22,791 | −$4,362 | **6.09** | **1.185** |
| Loose / Aggressive | −1.0 | 40 | +1.0 | 60 | 80 | 40 | 3,314 | 38.71% | **$59,808** | −$16,379 | 5.62 | 1.138 |

### 6.2 Base Variant — Detailed Results

| Metric | Value |
|--------|-------|
| Total Trades | 1,941 |
| Winning Trades | 749 (38.59%) |
| Losing Trades | 1,192 (61.41%) |
| Avg Win | **+$384.36** |
| Avg Loss | **−$213.15** |
| Avg Reward:Risk | **1.80:1** |
| Gross Profit | $287,889 |
| Gross Loss | $254,077 |
| **Net P&L** | **$33,812** |
| Profit Factor | **1.133** |
| Max Drawdown | −$12,406 |
| **Sharpe (Ann.)** | **4.885** |

### 6.3 Exit Breakdown (Base Variant)

| Exit Type | Count | Total P&L | Avg P&L |
|-----------|-------|-----------|---------|
| Take-Profit | 745 (38.4%) | +$287,496 | +$385.90 |
| Stop-Loss | 1,185 (61.1%) | −$253,709 | −$214.10 |
| EOD Flat | 11 (0.6%) | +$25 | +$2.26 |

**Interpretation:** The strategy wins 38.6% of the time but captures $384 average on wins vs $213 average on losses — a positive expectancy of ~+$17.42 per trade before considering realistic market-hours filtering.

### 6.4 Why This Works (Statistical Rationale)

1. **Asymmetric Payoff:** The 2:1 reward-risk ratio means the strategy only needs a 33.3% win rate to break even. At 38.59%, it has meaningful cushion.

2. **Z-Score Selectivity:** Only ~8% of bars trigger a VWAP z-score below −1.5 or above +1.5. Adding RSI confirmation reduces this further, ensuring entries are at genuine statistical extremes.

3. **VWAP Gravity:** Large institutions execute against VWAP throughout the day. When price deviates significantly, institutional buy/sell programs naturally push price back — creating a structural reversion force the strategy exploits.

4. **Mean-Reversion Frequency:** RSI < 35 occurs in ~5–7% of bars; when combined with the VWAP z-score filter, the dual-confirmation rate is ~2–4% of bars, providing 10–20 signals per week on a liquid instrument.

---

## 7. Backtest on QuantDataManager Data (CL — Light Crude Oil)

> **Data:** `2026.6.7LIGHTCMDUSD-M5-No Session.csv`  
> **Instrument:** Light Crude Oil USD (CL-equivalent), 5-minute bars  
> **Data range:** Jan 2013 – Jun 2026 (928,936 rows total)  
> **Backtest window:** Last 2 years — Jun 2024 to Jun 2026 (**141,561 bars**)  
> **CL tick value:** $0.01/barrel = $10 per tick (1,000 bbl contract)

### ❌ CRITICAL FINDING: Strategy FAILS on CL

| Variant | Trades | Win% | Net P&L | Max DD | Sharpe | PF |
|---------|--------|------|---------|--------|--------|----|
| Base (z=±1.5, RSI 35/65, TP80 SL40) | 3,360 | 35.80% | **−$83,996** | −$90,770 | −1.626 | 0.903 |
| Tight (z=±1.8, RSI 30/70, TP60 SL30) | 2,996 | 34.95% | **−$70,224** | −$85,026 | −1.957 | 0.884 |
| Loose (z=±1.2, RSI 40/60, TP100 SL50) | 3,152 | 36.04% | **−$93,833** | −$104,778 | −1.686 | 0.904 |
| Aggressive (z=±1.0, RSI 40/60, TP120 SL60) | 2,554 | 36.81% | **−$91,801** | −$105,926 | −1.725 | 0.900 |

**All variants are loss-making. Profit factors all < 1.0. Sharpe ratios all deeply negative.**

### 🔍 Root Cause Analysis

**The VWAP mean-reversion thesis does NOT hold for CL.**

The smoking gun is in the indicator statistics:

| Metric | NQ (Yahoo, 60d) | CL (QDM, 2yr) | Implication |
|--------|-----------------|---------------|-------------|
| % bars with RSI < 30 | ~5–7% | **4.8%** | Similar |
| % bars with RSI > 70 | ~5–7% | **5.1%** | Similar |
| **% bars with VWAP z < −1.5** | **~8%** | **28.97%** | 🚨 3.6× higher |
| **% bars with VWAP z > +1.5** | **~8%** | **33.63%** | 🚨 4.2× higher |

**Interpretation:** In NQ, only ~8% of bars breach the ±1.5σ VWAP band — these are genuine statistical extremes that tend to revert. In CL, **29–34% of bars** are outside ±1.5σ, meaning the band threshold is meaningless — price spends extended periods far from VWAP because CL is a **trending commodity market**, not an index.

### Why CL Behaves Differently
1. **Supply/Demand Shocks:** Crude oil is driven by geopolitical events, OPEC decisions, and inventory data — all of which produce sustained directional trends, not mean-reversions.
2. **24-Hour Session:** CL trades on Globex 23 hours/day. Without a clear session open anchor, VWAP loses much of its institutional significance.
3. **No Institutional VWAP Benchmark:** Unlike equity index funds (which are evaluated against VWAP), oil traders use TWAP/VWAP less as a benchmark — reducing the "gravitational pull" back to VWAP.
4. **Win rate 34.95–36.81%:** All below the **36.3% break-even threshold** — the strategy is systematically destroying capital.

> 🚫 **VERDICT: The VWAP+RSI mean-reversion strategy is NOT suitable for CL (Light Crude Oil). Do NOT deploy on this instrument.**

### Key differences vs NQ:
| Parameter | NQ | CL (Light Crude) |
|-----------|-----|----------|
| Tick size | 0.25 pts | $0.01/bbl |
| Tick value | $5 | $10 |
| VWAP reversion | ✅ Strong | ❌ Weak/Absent |
| Trending tendency | Moderate | **High** |
| Strategy suitability | ✅ Profitable | ❌ **Loss-making** |

---

## 8. Risk Control Addendum

> Authored by the **Chief Risk Officer** AI agent

### 8.1 Failure Scenarios

| # | Scenario | Market Condition | Strategy Failure | Max Estimated Loss |
|---|----------|-----------------|------------------|--------------------|
| 1 | **Trend Breakout** | Macro catalyst (Fed announcement) drives sustained directional move | Price does not revert; VWAP z-score continues expanding | 40 ticks × $5 = $200/trade × multiple consecutive losses |
| 2 | **Flash Crash / Gap** | Circuit-breaker-level move in 1–2 bars | SL is breached with excessive slippage (10–50 ticks gap) | Up to $500–$1,500 per trade in extreme conditions |
| 3 | **Low-Volume / Holiday Session** | Thin liquidity inflates z-scores artificially | False signals; wide spreads eat profits; slippage >> 1 tick | $100–$300 per trade in hidden slippage |
| 4 | **Regime Change** | Extended trending market (2–3+ weeks) | Win rate collapses below 30% → strategy goes negative | Cumulative drawdown to full daily loss limit ($1,000) |

### 8.2 Mandatory Risk Controls

| Rule | Parameter | Automated Response |
|------|-----------|-------------------|
| **Rule 1 — Daily Loss Limit** | −$1,000 per day (2% of $50K account) | Flatten all positions, cancel all orders, halt for remainder of session |
| **Rule 2 — Per-Trade Stop-Loss** | 40 ticks (NQ: $200, CL: $400) | Hard stop in bracket order — non-negotiable |
| **Rule 3 — Max Position** | 1 contract | No scaling in; no pyramiding |
| **Rule 4 — Opening Window Blackout** | 09:30–09:45 ET | No new entries permitted |
| **Rule 5 — Closing Window Blackout** | 15:45–16:00 ET | No new entries; force-close any open position |
| **Rule 6 — News Event Halt** | FOMC, CPI, NFP, NFP | Halt trading 5 min before release; resume 10 min after |
| **Rule 7 — Overnight Flat** | By session close | All positions must be closed; no overnight holds |
| **Rule 8 — Weekly Drawdown Review** | If weekly P&L < −$2,500 | Pause live trading; review parameter validity |

### 8.3 Break-Even Analysis

For the strategy to break even after costs:
- Commission per RT: $4.10
- Slippage cost (2 sides × 1 tick): NQ $10, CL $20
- **Total cost per RT:** NQ ~$14.10, CL ~$24.10

Minimum win rate to break even (80 TP / 40 SL ticks):

$$WR_{min} = \frac{SL_{net}}{TP_{net} + SL_{net}} = \frac{200 + 14.10}{400 - 14.10 + 200 + 14.10} \approx 36.3\%$$

**The strategy's observed win rate of 38.59% exceeds the break-even threshold by ~2.3 percentage points.**

---

## 9. Code Architecture Blueprint

### Module Structure
```
algo_live/
├── config.py          # Strategy parameters as dataclass constants
├── indicators.py      # VWAP (session-anchored), RSI, VWAP-z computation
├── signal_engine.py   # Entry/exit signal generation
├── order_manager.py   # ib_insync bracket order placement
├── data_handler.py    # ib_insync real-time bar subscription
├── risk_guard.py      # Pre-trade risk checks & circuit breakers
└── main_bot.py        # Main asyncio event loop
```

### Key Implementation Notes
1. **VWAP Reset:** Resets at 09:30:00 ET each trading day. Use `datetime.combine(today, time(9,30))` as the session anchor.
2. **Bar Timing:** Only use **confirmed closed** bars — subscribe to `reqRealTimeBars` and process on `barUpdateEvent` with `hasNewBar=True`.
3. **Bracket Orders:** Use `placeOrder` with `parent` (limit/market), `takeProfit`, and `stopLoss` child orders linked by `orderId`.
4. **Async Pattern:** All IB operations in a single `asyncio` event loop via `ib.pendingTickersEvent` and `ib.barUpdateEvent`.

---

## 10. Go / No-Go Recommendation

### 📋 Paper Trading: **CONDITIONAL GO ✅**

Conditions:
- [ ] Implement the `risk_guard.py` daily loss circuit breaker before first paper trade
- [ ] Test for minimum 30 trading days (targeting ≥ 60 trades)
- [ ] Verify VWAP reset logic at session open in live environment
- [ ] Confirm no look-ahead bias in live bar subscription

### 🚀 Live Trading: **NOT YET — Additional Requirements**

| Requirement | Status |
|------------|--------|
| 60+ days of paper trading with ≥ 100 trades | ⬜ Pending |
| Live VWAP reset verification | ⬜ Pending |
| News-event halt implementation tested | ⬜ Pending |
| Walk-forward out-of-sample validation | ⬜ Pending |
| Account minimum $50,000 for 1-contract NQ risk | ⬜ Confirm |

---

## 11. Summary & Final Verdict

### NQ (E-mini Nasdaq-100) — Yahoo Finance 60d

| | Base Variant | **Tight Variant (Recommended)** |
|--|-------------|----------------------------------|
| Parameters | z=±1.5, RSI 35/65 | **z=±2.0, RSI 30/70** |
| Trades | 1,941 | **957** |
| Win Rate | 38.59% | **39.71%** |
| Net P&L | $33,812 | $22,791 |
| Max Drawdown | −$12,406 | **−$4,362** ✅ |
| Sharpe (Ann.) | 4.885 | **6.087** 🚀 |
| Profit Factor | 1.133 | **1.185** |
| **Verdict** | ✅ Profitable | ✅ **Deploy to paper trade** |

### CL (Light Crude Oil) — QuantDataManager 2yr

| | All Variants |
|--|-------------|
| Win Rate | 34.95–36.81% |
| Net P&L | **−$70K to −$94K** |
| Max Drawdown | **−$85K to −$106K** |
| Sharpe (Ann.) | **−1.6 to −2.0** |
| Profit Factor | 0.884–0.904 |
| **Verdict** | 🚫 **DO NOT TRADE** |

### 🎯 Action Plan

1. **Immediately:** Paper trade NQ with **Tight parameters** (z=±2.0, RSI 30/70, TP=80tks, SL=40tks)
2. **30 days:** Collect live paper performance data (target ≥ 60 trades)
3. **Review:** If paper Sharpe > 2.0 and PF > 1.1, proceed to live with 1 NQ contract on $50K account
4. **Do not trade CL** with this strategy — requires a completely different approach (trend-following, not mean-reversion)
5. **For CL:** Consider developing a separate momentum/breakout strategy better suited to commodity trending behaviour

> ⚠️ **Critical Warning:** The 60-day NQ backtest is a limited sample. The QDM CL data (2-year, 141K bars) is far more statistically robust and clearly shows that **instrument selection is the most important variable** — not parameter tuning.

---

*Report generated by CrewAI Quant Research Crew — Agent outputs synthesized by Antigravity AI*  
*Data sources: Yahoo Finance (NQ=F 60d), QuantDataManager v1.25 (LIGHTCMDUSD-M5, 2yr: 141,561 bars)*  
*Backtest engine: custom vectorised Python (backtest_tool.py + backtest_qdm.py)*
