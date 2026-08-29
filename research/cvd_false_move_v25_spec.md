# v2.5 True-CVD False-Move Experiment — Frozen Spec

Status: DESIGN FROZEN / DATA NOT YET CONNECTED

## Scientific claim
Test whether exchange aggressor-side order flow adds incremental information for rejecting false structural breaks.

CVD(t) = cumulative(AskVolume - BidVolume).

Do NOT substitute CFD tick volume, OHLCV candle direction, or TradingView-style lower-timeframe volume estimation and call it true CVD.

## Venue mapping
- NASDAQ CFD execution -> CME NQ/MNQ signal
- S&P 500 CFD -> CME ES/MES
- Dow CFD -> CBOT YM/MYM
- Russell CFD -> CME RTY/M2K
- Gold CFD -> COMEX GC/MGC
- WTI CFD -> NYMEX CL/MCL
- Natural Gas CFD -> NYMEX NG
- Brent CFD -> ICE Brent futures preferred; do not silently substitute WTI

Exact feed/symbol entitlement must be recorded in provenance before a market is enabled.

## Timeframe architecture
1. D1/H4: regime/context only.
2. H1: pre-defined trusted structural zone / prior swing boundary.
3. M15: CVD false-break detection.
4. M5: price acceptance failure + failed retest confirmation.
5. M1: execution only; first executable open strictly after M5 signal availability.

M1 CVD is NOT required in baseline v2.5.

## Session rule
Baseline CVD resets at the primary futures session/trading-day boundary supplied by the exchange/feed. Session definition and timezone must be stored. Continuous/non-reset CVD is a later ablation, not mixed into baseline.

## False-break candidate
Bearish candidate:
- price exceeds a pre-existing swing high / zone boundary;
- M15 CVD fails to confirm the corresponding prior CVD extreme (price HH, CVD non-HH);
- price subsequently closes/accepts back below the broken boundary;
- M5 retest of the boundary fails.

Bullish candidate is symmetric: price LL, CVD non-LL, reclaim, M5 failed breakdown retest.

CVD divergence alone is never an entry.

## Effort/result feature
Measure aggressive flow relative to achieved price progress rather than assigning a hidden actor:
- signed_delta = AskVolume - BidVolume
- abs_delta = abs(signed_delta)
- price_progress = directional excursion / ATR
- delta_efficiency = directional price progress / max(abs_delta, epsilon)

Large delta with unusually small same-direction price progress is labelled EFFORT_RESULT_FAILURE. It is NOT labelled institutional absorption without additional evidence.

Thresholds must be estimated on DEV only, frozen, then checked on VAL. OOS cannot tune thresholds.

## Baseline gates
LOCATION -> BREAK -> CVD_NON_CONFIRMATION -> ACCEPTANCE_FAILURE -> M5_FAILED_RETEST -> M1_EXECUTION

A trade is rejected if any gate is absent.

## Ablations
Compare, on the identical structural setup population:
A. Structure only (no CVD)
B. Structure + CVD non-confirmation
C. Structure + effort/result failure
D. Structure + CVD non-confirmation + effort/result failure

Primary question: does CVD add incremental predictive value, not merely reduce trade count?

## Metrics
- n trades
- expectancy R after costs
- profit factor
- win rate (secondary)
- max drawdown
- MAE/MFE
- false-break rejection precision
- recall of profitable structural reversals
- Brier/calibration if converted to probability
- bootstrap CI for expectancy difference versus Structure-only
- per-market and per-regime decomposition
- execution-cost sensitivity 1x/2x/3x/5x

No claim of improved precision unless the incremental difference survives VAL and a fresh untouched forward holdout with uncertainty intervals.

## Risk-sizing experiment
After signal validity is established, compare separately:
- FIXED_050: 0.5% after every trade
- WIN_100_RESET_050: start 0.5%; after profitable trade use 1.0%; remain 1.0% while profitable; after losing trade reset immediately to 0.5%.

Signal selection and risk sizing must not be optimized simultaneously.

## Causality requirements
- Every bar timestamp means information-availability time, not bar-start time.
- No future-window min/max may be used to validate an earlier signal.
- CVD extreme comparison may use only data available at that timestamp.
- M5 confirmation must close before execution.
- Entry is first M1 executable open strictly after confirmation availability.
- No OOS parameter selection.

## Required input schema
trade_time, price, size, aggressor_side OR bid_volume/ask_volume bars, venue, contract, timezone, session_id.

If aggressor_side is vendor-classified, record vendor/method. If only bid/ask quotes exist without executed-trade classification, mark CVD unavailable rather than infer true CVD silently.

## Decision states
GO: incremental edge survives VAL + fresh forward holdout, costs, and market/regime decomposition.
CONTINUE: directionally useful but CI/sample insufficient.
NO-GO: no incremental edge, excessive cost sensitivity, or signal depends on contaminated/uncausal data.
INVALID: data cannot support true aggressor-side CVD or causality is violated.
