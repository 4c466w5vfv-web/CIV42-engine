# Trading OS Mobile v1

A zero-typing, mobile-first trading operating console for a swing/prop workflow.

## Core operating rules

- Default thesis risk: 0.75%
- Maximum total open risk: 1.50%
- Maximum correlated cluster risk: 0.75%
- One symbol, one thesis, one entry
- No scale-in, averaging down, chasing, or discretionary add-ons
- Structural stop is defined before position size
- Protected/BE positions can release risk budget but remain visible as exposure
- New signals are classified ENTER / WAIT / BLOCK
- Same-sector capacity and total portfolio capacity are checked before entry
- Journal events are generated from execution and exit actions instead of narrative typing

## Screens

1. HOME — Equity, open/available risk, drawdown, buffer, active positions, portfolio clusters, next signal
2. TRADE — BigView/HTF/zone/trigger checks, structural stop, risk amount, single-entry position sizing
3. PORTFOLIO — Position and correlated-cluster risk allocation
4. JOURNAL — Auto-generated entry/BE/exit events and rule compliance

## Integration path

Current version is a deterministic front-end prototype with local sample state. Production adapters should feed the same state model:

TradingView alert -> Trading Guard API -> broker/MT5 adapter -> position state -> journal events -> dashboard.

For MT5 mobile-only operation, the execution bridge should run outside the phone via broker API or an MT5 EA on a VPS. The phone remains the monitoring/approval surface.

## Position sizing

The browser prototype displays a simple unit calculation from equity, thesis risk, entry, and stop distance. Production CFD sizing must use broker-specific contract size, tick size, tick value, quote currency conversion, spread, commission, and slippage reserve before producing the final MT5 lot size.

## Risk recycling

A winning position may release risk capacity only after its stop has been structurally advanced to BE/profit according to the strategy. Released risk may be assigned to a new independent thesis, subject to the 1.50% portfolio cap and 0.75% correlated-cluster cap. Same-symbol discretionary re-entry is not treated as recycling.

## Run locally

Serve this folder with any static HTTP server and open `index.html`. It has no build step.
