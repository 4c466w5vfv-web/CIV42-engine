# ARK-42 Trading System Master v1.0

## Mission
Build one evidence-based trading system that combines:
1. Bear-market shock mean reversion
2. SMA200 trend transition
3. ATR-based trend capture
4. Evidence-earned pyramiding
5. Cross-asset rotation/persistence
6. Portfolio risk controls
7. Daily Sharpe/Sortino/Calmar validation

The system must not trade because a chart 'looks good'. Capital is deployed only when both the **system edge** and the **current setup** are supported by evidence.

---

## 1. Master State Machine

Bear Trend
→ Important D1/W1 Zone
→ H4 Shock / Absorption Candidate
→ H1 Bottom Construction
→ Reclaim
→ Higher Low
→ Swing-High Close Break
→ Acceptance / Pullback Hold
→ Mean-Reversion Entry
→ Partial / Runner
→ D1 SMA200 Reclaim
→ SMA200 Acceptance
→ Trend Mode
→ New HL / Breakout
→ Evidence-Earned Pyramid
→ 2ATR Ratcheting Trail
→ Exit
→ Review / Runtime Update

Cycle R is defined as:

Cycle R = Mean-Reversion R + Trend R + Pyramid R

One cycle can contain multiple H1 entries/re-entries/tranches.

---

## 2. Timeframe Roles

### W1 / D1 — Context and Location
- Major close-based support/resistance zones
- Prior confirmed swing zones
- Regime: price vs D1 SMA200
- SMA200 slope
- Major structural state

### H4 — Shock / Absorption Context
- Abnormal relative volume where volume is meaningful
- Shock cluster
- Downside efficiency failure
- Reclaim of structural zone

### H1 — Execution
- Reclaim
- Higher Low
- Swing-high close break
- Acceptance
- Tactical failure
- Re-entry
- Pyramid add

Preferred architecture:

D1/W1 Location → H4 Shock → H1 Execution

---

## 3. Important Zone

Use close-based structural zones rather than wick-only levels.

Initial mechanical candidates:
- prior confirmed D1/W1 pivot-low support
- previous breakout/retest area
- major close congestion
- long-term mean area if structurally relevant

No-lookahead rule:
- pivot is usable only after the right-side confirmation bars exist

Zone validity ends after repeated structural failure:
- LL → LH → LL → LH
- plus failure to reclaim the original zone on a completed close

---

## 4. Shock / Absorption Evidence

RVOL is context, not entry.

Candidate shock thresholds to test:
- Strong: RVOL >= 2.0
- Medium: 1.5 <= RVOL < 2.0
- Weak: < 1.5

For H1/H4, session-normalized or same-time-slot relative volume is preferred where possible.

Absorption proxy requires:
1. Important location
2. Abnormal volume
3. Small further downside extension relative to ATR
4. Reclaim
5. Higher Low / structure break

High volume + continued downside expansion = initiative selling, not absorption.

---

## 5. Mean-Reversion Entry

Do not buy the shock candle or the first bounce.

Required evidence sequence:

Reclaim → HL → Swing-High Close Break → Acceptance

Risk is earned by evidence:
- Shock / observation only: 0R
- Reclaim + HL: up to 0.25R
- Swing-high close break: cumulative up to 0.60R
- Acceptance / pullback hold: cumulative up to 1.00R

1R convention for research and sizing:
- 1R = 1% of account equity

No averaging down.
No stop widening.
No market chase after material extension.

---

## 6. Mean-Reversion Exit / Transition

Initial safety stop:

Initial Stop = Entry - 2 × Wilder ATR14

Tactical exit:
- completed H1 close back below accepted/reclaimed structure

Partial-exit configuration to test:
- 50% at +1R
- 50% runner

SMA200 is a regime gate, not a fixed take-profit.

If D1 closes above SMA200 and acceptance follows:

Mean Reversion → Trend Following

---

## 7. Trend Mode

Trend entry/add must not be triggered by SMA200 alone.

Required sequence:

D1 Close > SMA200
→ SMA200 slope non-negative/positive
→ Acceptance / Pullback Hold
→ Higher Low
→ Swing-High Close Break
→ Trend Entry / Add

Existing runner may remain open.
New trend entries are separate tranches with their own initial risk.

---

## 8. Trend Exit

Use Wilder ATR14.

Trailing Stop = Highest High Since Entry - 2 × current ATR14

Rules:
- stop only ratchets upward
- completed bar sets the stop for the next bar
- no same-bar lookahead
- stop never loosens

---

## 9. Evidence-Earned Pyramiding

Pyramiding is allowed only when information improves and open risk has fallen.

Conditions for a new add:
1. Existing position is not being averaged down
2. Existing stop has risen
3. Current open downside risk has been reduced
4. New HL exists
5. New breakout / acceptance exists
6. Portfolio and cluster risk remain within limits

Illustrative tranche sequence to test:
- SMA200 reclaim + HL: +0.30R
- breakout acceptance: cumulative +0.60R
- next HH → HL continuation: additional risk only from released open-risk budget

Core constraint:

Sum(Current Downside Risk of All Open Tranches) <= Cycle / Portfolio Risk Cap

Pyramiding is retained only if Sharpe/Sortino or Calmar improve after realistic costs, not merely because total R rises.

---

## 10. Re-entry

One failed reversal does not automatically kill the cycle.

One re-entry allowed if:
- original zone remains valid
- low is defended
- fresh HL forms
- fresh swing-high close break occurs
- acceptance follows

Discard old setup after:
- two LL/LH failure cycles
- and failure to reclaim original zone on close

A genuinely new shock/zone becomes a new Setup ID.

---

## 11. Cross-Asset Rotation / Persistence

Do not allocate to the biggest immediate bounce.

Candidate scoring inputs:
- RS 20D / 60D
- SMA200 position and slope
- HH/HL persistence
- breakout hold/failure
- price progress per ATR
- volume accompanying progress where valid
- session-normalized RVOL

Candidate score:

Capital Rotation Score = Shock Quality × Persistence × Relative Strength

Select independent risk sources, not duplicate tickers.
Example: NAS100, SPX500 and US30 belong largely to one equity-beta cluster.

---

## 12. Portfolio Risk

Trade risk is not the same as portfolio risk.

Required controls:
- per-setup risk cap
- current open-risk cap
- cluster/correlation cap
- total portfolio risk cap
- no averaging down
- no stop widening

Research convention:
- 1R = 1% account equity

Portfolio-level hard governance should be validated against actual prop drawdown rules before live use.

---

## 13. System Evidence Score (SES)

Sharpe is a strategy/allocation metric, not an entry signal.

Required evidence:
1. Daily marked-to-market Sharpe
2. Sortino
3. Calmar
4. PF
5. Expectancy in R
6. MDD
7. Losing-streak distribution
8. Cost sensitivity
9. Stability across assets/years/regimes
10. Concentration test

Provisional governance gates:
- RED: OOS Sharpe <= 0 or PF <= 1.0 → no capital
- AMBER: OOS Sharpe 0–0.75 or PF 1.0–1.3 → research/paper only
- GREEN: OOS Sharpe >= 0.75, PF >= 1.3, positive expectancy after 0.10ATR costs, acceptable MDD → small live-risk candidate
- STRONG: OOS Sharpe >= 1.25 with cross-asset/year stability and no concentration failure → normal strategy-risk candidate

These are operating gates, not universal laws or future-return claims.

---

## 14. Setup Evidence Score (SET)

Mean-reversion evidence units:
- +1 Important D1/W1 zone
- +1 Abnormal volume/shock where volume is meaningful
- +1 Failed downside / low additional downside efficiency
- +1 Reclaim
- +1 Higher Low
- +1 Prior swing-high close break
- +1 Acceptance / pullback hold

Trend evidence units:
- +1 D1 close above SMA200
- +1 SMA200 slope non-negative/positive
- +1 Pullback holds regime area
- +1 Higher Low
- +1 Prior swing-high close break
- +1 Acceptance
- +1 Relative-strength / persistence confirmation

Hard vetoes override score:
- averaging down
- stop widening
- market chase
- cluster/portfolio risk breach
- unconfirmed close-dependent signal
- stale/broken zone

If evidence is incomplete, position size = 0.

---

## 15. Sharpe / Validation Architecture

Formal Sharpe:

Sharpe = mean(daily portfolio excess return) / std(daily portfolio excess return) × sqrt(252)

Must be computed from daily marked-to-market NAV, not a handful of trade R values.

Compare at minimum:
1. MR only
2. MR + SMA200 Trend
3. MR + Trend + Pyramid
4. Rotation OFF
5. Rotation ON
6. Portfolio combined

Promotion rule:

A module is promoted only if net return improves without materially degrading OOS Sharpe/Sortino or violating drawdown/concentration limits.

---

## 16. Current Verified Trend Baseline Evidence

Dataset: TradingView daily NDX / SPX, tested from 2023-01-03 after full SMA200 warmup to 2026-05-28.
Entry baseline: prior close > SMA200, positive 20D SMA200 slope, next-day open entry, Wilder ATR14, 2ATR stop/trail, 1% equity risk.

### NDX
0 cost:
- 70 closed trades
- win rate 42.86%
- expectancy +0.30R
- PF 1.96
- sum +21.33R
- daily Sharpe 1.22
- Sortino 1.80
- daily MDD -6.87%
- CAGR 6.32%
- Calmar 0.92

0.10 ATR round-trip cost:
- expectancy +0.25R
- PF 1.73
- sum +17.83R
- daily Sharpe 1.00
- Sortino 1.44
- daily MDD -7.66%
- CAGR 5.23%
- Calmar 0.68

### SPX
0 cost:
- 72 closed trades
- win rate 45.83%
- expectancy +0.20R
- PF 1.65
- sum +14.72R
- daily Sharpe 0.90
- Sortino 1.30
- daily MDD -5.95%
- CAGR 4.29%
- Calmar 0.72

0.10 ATR round-trip cost:
- expectancy +0.15R
- PF 1.45
- sum +11.12R
- daily Sharpe 0.67
- Sortino 0.94
- daily MDD -6.42%
- CAGR 3.19%
- Calmar 0.50

Boundary:
This is only a trend-control baseline. It is not the performance of the full MR + Trend + Pyramid system.

---

## 17. Required Full Backtest

Period:
- 2018–2026

Asset groups:
- Precious Metals: XAU, XAG, XPT/XPD where data exists
- Equity Indices: NDX/NAS100, SPX500, US30/DJI, UK100/RUT where data exists
- Energy: WTI, Brent
- FX: EURUSD, GBPUSD, USDJPY, AUDUSD, USDCAD, others where reliable
- Crypto: BTC, ETH
- Other commodities where reliable

Execution architecture:
- W1/D1 zones
- H4 shock / absorption context
- H1 execution where data permits

Cost sensitivity:
- 0
- 0.05 ATR round trip
- 0.10 ATR round trip

Must use:
- no-lookahead pivots
- next-bar execution after close-confirmed signals
- prior-bar stop logic
- common parameters before optimization

Metrics:
- trade count
- cycle count
- Net R
- Win Rate
- Expectancy
- PF
- MFE / MAE
- Sharpe
- Sortino
- Calmar
- MDD
- worst year
- losing streak
- MR contribution
- Trend contribution
- Pyramid contribution
- rotation contribution
- concentration by year/asset
- correlation-adjusted exposure
- prop-rule survival probability

---

## 18. Daily Execution Order

1. Discipline Gate
2. System Evidence / SES
3. Portfolio Risk
4. Big View
5. Rotation / Persistence
6. Important Zone
7. Shock / Absorption Context
8. Setup Evidence / SET
9. Confirmation
10. Position Size
11. Execution
12. 2ATR Management / Tactical Exit
13. Review

Core rule:

**검증된 시스템만 거래하고, 증명된 셋업에만 리스크를 건다.**

If evidence is incomplete, position size is zero.
