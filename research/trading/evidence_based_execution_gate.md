# Evidence-Based Execution Gate v0.1

Purpose: use evidence quality to decide **whether to trade and how much risk to deploy**. Sharpe is a system/allocation metric, not a single-entry trigger.

## 1. Two independent scores

### A. System Evidence Score (SES)
Used to decide whether a strategy module deserves capital.

Required evidence:
1. Out-of-sample daily Sharpe from marked-to-market portfolio returns.
2. Sortino and Calmar.
3. Profit Factor and Expectancy in R.
4. Max Drawdown and losing-streak distribution.
5. Cost sensitivity: 0 / 0.05 ATR / 0.10 ATR round trip.
6. Stability across assets, years and regimes.
7. No single asset/year contributes a dominant share of total P&L.

Provisional allocation gate (must be validated, not treated as a universal law):
- SES RED: OOS Sharpe <= 0 or PF <= 1.0 -> no capital.
- SES AMBER: OOS Sharpe 0–0.75 or PF 1.0–1.3 -> research / paper only.
- SES GREEN: OOS Sharpe >= 0.75, PF >= 1.3, positive expectancy after 0.10 ATR costs, acceptable MDD -> eligible for small live risk.
- SES STRONG: OOS Sharpe >= 1.25 with cross-asset/year stability and no concentration failure -> eligible for normal strategy risk subject to portfolio caps.

These thresholds are governance gates, not claims of future performance.

### B. Setup Evidence Score (SET)
Used for the individual trade.

Mean-reversion evidence units:
- +1 Important D1/W1 close-based zone reached.
- +1 Abnormal volume / shock in a data source where volume is meaningful.
- +1 Failed downside / low additional downside efficiency.
- +1 Reclaim on completed bar.
- +1 Higher Low.
- +1 Prior swing-high close break.
- +1 Acceptance / pullback hold.

Trend evidence units after D1 SMA200 reclaim:
- +1 D1 close > SMA200.
- +1 SMA200 slope non-negative/positive.
- +1 Pullback holds above reclaimed regime area.
- +1 Higher Low.
- +1 Prior swing-high close break.
- +1 Acceptance after breakout.
- +1 Relative-strength / persistence confirmation versus the cross-asset universe.

Hard vetoes override the score:
- averaging down
- stop widening
- market chase after material extension
- portfolio/cluster risk cap breach
- unconfirmed close-dependent signal
- stale/broken Important Zone

## 2. Risk translation

Risk is earned by evidence; it is not chosen first.

Mean-reversion cycle:
- Observation / shock only: 0R
- Reclaim + HL probe: up to 0.25R
- Swing-high close-break: cumulative up to 0.60R
- Acceptance / pullback hold: cumulative up to 1.00R

Trend cycle above SMA200:
- SMA200 reclaim alone: 0R new add
- Reclaim accepted + HL: up to 0.30R add
- Breakout acceptance: cumulative up to 0.60R
- New HH -> HL -> continuation break with released open risk: additional tranches only while total current downside risk remains inside the cycle/portfolio cap.

Never increase risk because of P&L alone. Add only because evidence increased and existing risk has been reduced by stop ratcheting.

## 3. Exit evidence

Mean-reversion phase:
- Tactical exit when accepted/reclaimed structure fails on a completed execution-timeframe close.
- Hard safety stop remains 2ATR unless a tighter structural stop is already valid.

Trend phase:
- Highest High Since Entry - 2 x Wilder ATR14.
- Stop ratchets only upward.
- Completed bar sets the stop for the next bar; no same-bar lookahead.

## 4. Sharpe usage

Sharpe must be calculated from **daily marked-to-market equity returns**, not a handful of trade R values.

Use it at three levels:
1. Module: Mean Reversion only vs MR + Trend vs MR + Trend + Pyramid.
2. Asset: XAU, NDX, SPX, US30, oil, FX, crypto, etc.
3. Portfolio: correlation/cluster-adjusted combined book.

Decision rule:
> A module is promoted only if adding it improves net return without materially degrading OOS Sharpe/Sortino or violating drawdown and concentration limits.

For pyramiding specifically:
> Keep pyramiding only if Sharpe/Sortino or Calmar improve after realistic costs, not merely because total R increases.

## 5. Daily execution order

Discipline Gate -> System Evidence (SES) -> Portfolio Risk -> Big View -> Rotation/Persistence -> Important Zone -> Setup Evidence (SET) -> Confirmation -> Position Size -> Execution -> Review.

If evidence is incomplete, position size is zero.
