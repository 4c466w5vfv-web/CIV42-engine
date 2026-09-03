# Trading OS Mobile v1

A zero-typing, mobile-first trading operating console for a swing/prop workflow.

## Core operating rules

- Default thesis risk: 0.75%
- Maximum total open risk: 1.50%
- Maximum correlated cluster risk: 0.75%
- Three-part entry is one thesis, not three independent risks: 40% / 30% / 30% of the 0.75% risk budget
- Protected/BE positions recycle risk budget but remain visible as exposure
- New signals are classified ENTER / WAIT / BLOCK
- Same-sector capacity is checked before a new entry is allowed
- Journal events are generated from actions instead of requiring narrative typing

## Screens

1. HOME — Equity, open/available risk, drawdown, buffer, active positions, portfolio clusters, next signal
2. TRADE — Signal checks and three-entry slot execution
3. PORTFOLIO — Position and cluster risk allocation
4. JOURNAL — Auto-generated execution/exit events and rule compliance

## Integration path

Current version is a deterministic front-end prototype with local sample state. Production data adapters should feed the same state model:

TradingView alert -> Trading Guard API -> broker/MT5 adapter -> position state -> journal events -> dashboard.

For MT5 mobile-only operation, the production execution bridge should run outside the phone (broker API or MT5 EA on a VPS). The phone remains the monitoring/approval surface.

## Run locally

Serve this folder with any static HTTP server and open `index.html`. It has no build step.
