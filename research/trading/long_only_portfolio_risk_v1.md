# ARK-42 Long-Only Trading + Portfolio Risk Architecture v1

Status: FROZEN RESEARCH SPEC
Branch: backtest/ndx-trend-pilot

## Objective
Operate only long exposure across two strategy sleeves:
1. Long Momentum / Trend Following
2. Long Mean Reversion
3. Cash / No Trade as a valid state

Short strategies remain Research Queue only.

## Initial Index Universe
Primary index universe for long-only research and execution design:
- NAS100
- SPX500
- US30

All three are treated as one Equity Risk Cluster until empirical dependence analysis justifies a different allocation rule. A valid signal on all three does not create three independent risk bets.

## Core State Machine

CASH -> MEAN_REVERSION -> LONG_MOMENTUM -> CASH

### CASH
Use when:
- strong bear trend persists,
- no validated mean-reversion reversal structure exists,
- no validated long-momentum setup exists.

### MEAN_REVERSION LONG
Current research definition:
- Important D1/W1 zone
- bearish trend deceleration
- shock / abnormal downside move
- post-shock volume contraction
- downside efficiency deterioration
- VCP / volatility contraction candidate
- pivot structure
- reclaim / higher-low / swing-high close break / acceptance

Timeframe remains a research variable. Do not optimize thresholds by repeatedly looking at outcomes.

### LONG MOMENTUM
Current baseline:
- prior D1 close > SMA200
- SMA200 slope > 0
- breakout / acceptance / pullback / higher-low continuation
- initial stop based on completed-bar ATR
- ratcheting ATR trailing stop

## Evidence Hierarchy

Production priority:
1. Long Momentum: strongest current positive evidence
2. Long Mean Reversion: research candidate only until fresh holdout/forward validation

No strategy receives production risk merely because one in-sample subgroup has attractive PF/expectancy.

## Hard Portfolio Risk Rules

These are risk ceilings, not targets.

- Base risk per trade: 0.50% of current equity
- Single position open stop risk: <= 0.50% unless explicitly reduced by portfolio engine
- Equity cluster gross open-stop risk: <= 1.50%
- Single strategy sleeve open-stop risk: <= 2.00%
- Total portfolio open-stop risk: <= 4.00% HARD CAP
- No averaging down
- No stop widening
- No loss-chasing / recovery size increase
- No pyramiding in Prop / CFD sleeve
- No trade is an acceptable allocation

NAS100 / SPX500 / US30 are treated as the same Equity Risk Cluster unless empirical dependence analysis justifies otherwise.

## Drawdown Throttle

Risk multiplier applied before any new order:
- DD < 2.0%: 1.00x
- 2.0% <= DD < 3.0%: 0.75x
- 3.0% <= DD < 4.0%: 0.50x
- DD >= 4.0%: 0.00x, block new trades

With a 0.50% base trade risk this produces:
- normal: 0.50%
- DD 2-3%: 0.375%
- DD 3-4%: 0.25%
- DD >=4%: 0%

## Portfolio Risk Stack

Every candidate order must be checked against:
1. Per-trade stop risk
2. Single-asset risk
3. Cluster gross risk
4. Strategy sleeve risk
5. Total portfolio open-stop risk
6. Correlation / covariance-adjusted risk
7. VaR
8. Expected Shortfall
9. Drawdown throttle
10. Evidence grade

VaR is a diagnostic layer, not the sole risk limit. Hard stop-risk limits remain binding.

## Position Sizing

Risk dollars:
Risk$ = Equity * AllowedRiskPct

Position size:
Size = Risk$ / StopLossDollarValuePerUnit

Exact lot sizing requires the broker/prop symbol contract specification.

## Evidence Gate

Do not promote a rule because of one profitable sample.

Research sequence:
1. Mechanism hypothesis
2. Predeclared rule
3. Discovery sample
4. Freeze rule
5. Temporal holdout
6. Cross-asset holdout
7. Cost stress
8. Parameter-plateau check
9. Bootstrap / Monte Carlo sequence risk
10. Forward sample

Minimum diagnostics:
- N
- win rate
- mean R
- median R
- net R
- PF
- max sequential DD
- daily Sharpe where daily MTM series exists
- Sortino
- Calmar
- cost sensitivity
- asset/year/regime concentration
- confidence interval / bootstrap where feasible

## Anti-Overfitting Rule

Use coarse parameter buckets rather than single best values. Prefer stable plateaus over peak performance.

A parameter that works only at one narrow threshold is rejected or retained as research-only.

## Current Mean-Reversion Boundary

Generic mechanical MR remains negative in the expanded sample.
The strongest discovered candidate is trend-deceleration + post-shock volume contraction, but it remains in-sample discovery and sparse on cross-asset holdout.

VCP and pivot features are to be tested without changing the core location/shock/exhaustion mechanism.

## Execution Philosophy

Signal Engine proposes.
Evidence Gate qualifies.
Portfolio Risk Engine sizes or rejects.
Execution Engine sends the order.
Outcome DB records the result.
Reality can override prior rules.
