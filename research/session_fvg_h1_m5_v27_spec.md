# Session High/Low → H1 FVG → M5 Execution v2.7

Status: **RESEARCH SPEC — NOT VALIDATED**

## User trading principle translated to a causal test

1. Read the big picture first.
2. Record the high and low of each major session.
3. Look for a Fair Value Gap in the direction of the larger trend.
4. Use the session high/low as the liquidity/reference level.
5. Require confirmation on H1.
6. Use M5 only for execution.
7. Prop risk is evaluated in R; nominal operating candidate is fixed 1% per setup.
8. A day-trade may be promoted to a short swing only if structure remains valid.

## Initial instruments

- EURUSD — Dukascopy `eurusd`
- GOLD — Dukascopy `xauusd`
- WTI — Dukascopy `lightcmdusd`
- BRENT — Dukascopy `brentcmdusd`

Dukascopy is a research/reference feed. CFD execution prices, spreads, swaps and session rules may differ from FXIFY.

## Mechanical definitions

### Big-picture trend

Frozen baseline: Daily close relative to causal 200-day SMA with SMA slope.

- LONG regime: close > SMA200 and SMA200 > SMA200[20]
- SHORT regime: close < SMA200 and SMA200 < SMA200[20]
- otherwise: NO TRADE

This is a baseline proxy for the user's discretionary HTF trend reading, not a claim that SMA200 is optimal.

### Session windows

Session windows are defined in their local clocks and converted to UTC so DST is handled explicitly.

- ASIA: 00:00–08:00 UTC
- LONDON: 08:00–17:00 Europe/London
- NEW_YORK: 08:00–17:00 America/New_York

The completed session high/low becomes usable only after that session ends. No future session high/low is visible intraday.

### Session sweep

After a session completes, its high/low is a reference for up to 18 hours.

- LONG candidate: later price trades below the completed session low, then an H1 close reclaims above that low.
- SHORT candidate: later price trades above the completed session high, then an H1 close closes back below that high.

### H1 Fair Value Gap

Standard three-candle causal definition:

- Bullish FVG at H1 bar t: `low[t] > high[t-2]`
- Bearish FVG at H1 bar t: `high[t] < low[t-2]`

Minimum gap baseline: 0.10 × H1 ATR(14).

For v2.7, H1 confirmation requires, after the session sweep/reclaim, a same-direction H1 FVG and directional close. The H1 signal becomes available only at that H1 bar close.

### M5 execution

M5 cannot create the thesis.

After the causal H1 signal, within 6 hours:

- LONG: first bullish M5 close above the previous 4 completed M5 highs.
- SHORT: first bearish M5 close below the previous 4 completed M5 lows.

Entry is the **next M5 open strictly after the confirming M5 bar closes** to avoid same-bar execution lookahead.

Structural stop baseline:

- LONG: below the sweep extreme / H1 FVG lower boundary, whichever is lower.
- SHORT: above the sweep extreme / H1 FVG upper boundary, whichever is higher.

### Exit ablations

Run all candidates through:

A. Fixed 2R
B. Fixed 3R
C. Fixed 5R
D. 50% at 2R + 50% H1 structural runner

Runner rule: after TP1, trail only on completed H1 structure in the favorable direction. Never widen stop.

## Cost and risk treatment

- Raw strategy metrics are recorded in R.
- Baseline execution cost is an explicit research parameter, not a broker fact.
- Fixed 1% prop risk is applied only in the risk layer after signal generation.
- 1% risk must not be confused with lot size; structural stop first, sizing second.

## Required outputs

For each asset and session:

- Trades
- Win rate
- After-cost expectancy R
- Profit factor
- Total R
- Max drawdown R
- Avg win / Avg loss
- MAE / MFE where available
- Exit-model comparison
- DEV / VAL / OOS splits

Portfolio/risk layer:

- fixed 1% per setup
- simultaneous open-risk ceiling 3%
- 2 consecutive full-SL daily stop as a separate ablation
- correlated Energy positions (WTI/Brent) must later be stress-tested as one cluster

## Validation policy

- DEV: through 2022
- VAL: 2023–2024
- OOS: 2025 onward
- Do not tune parameters using OOS.
- If OOS is inspected repeatedly it becomes contaminated and must be replaced with a fresh forward holdout.
- No claim of edge until after-cost expectancy and drawdown survive VAL plus a fresh holdout.

## Known limitations

- Historical session definitions are research conventions, not broker-specific trading hours.
- Dukascopy volume/price is a proxy where applicable.
- Standard FVG detection is a mechanical proxy for the user's visual FVG selection.
- SMA200 trend is a baseline proxy for discretionary HTF context.
- News, exact FXIFY spread/slippage/swap, floating equity DD and broker liquidation mechanics are not yet modeled.
- True CVD/aggressor flow is not part of v2.7.
