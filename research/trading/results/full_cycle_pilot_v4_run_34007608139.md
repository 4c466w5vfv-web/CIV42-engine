# ARK-42 Integrated MR + Trend + Pyramid Pilot v4

Workflow run: `34007608139`
Job: `101417588461`
Conclusion: `success`

## Mechanical rules in this pilot
- D1 close-based pivot support / bear regime.
- H4 same-slot relative-volume shock and absorption proxy.
- H1 reclaim -> confirmed Higher Low -> prior swing-high close break -> acceptance.
- Mean-reversion staged risk: 0.25R + 0.35R + 0.40R, subject to open-risk cap.
- 50% MR partial at +1R.
- Two consecutive D1 closes above SMA200 with non-negative SMA200 slope permit Trend Mode.
- Structure-confirmed trend add, then continuation pyramids.
- No averaging down.
- 2x H4 Wilder ATR14 initial/trailing stop; stop only ratchets upward.
- 0.10 ATR round-trip cost.
- 1R = 1% equity.
- Daily marked-to-market Sharpe/Sortino.

## Results

| Asset | Shocks | MR schedules | Cycles | Tranches | Return | Sharpe | Sortino | MDD | CAGR | Calmar | PF approx | MR PnL | Trend PnL | Pyramid PnL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| XAUUSD | 37 | 26 | 5 | 55 | -0.350% | -0.051 | -0.015 | -2.012% | -0.040% | -0.020 | 0.893 | +0.436% | +0.055% | -0.841% |
| NAS100 | 2 | 1 | 1 | 58 | -0.670% | -0.092 | -0.030 | -2.588% | -0.078% | -0.030 | 0.856 | -0.026% | -0.072% | -0.571% |
| SPX500 | 4 | 1 | 1 | 25 | -0.009% | 0.0001 | 0.00003 | -2.194% | -0.001% | -0.0005 | 0.867 | +0.174% | +0.607% | -0.789% |

Daily sample printed by the pilot: 2018-01-02 through 2026-09-02.

## Key diagnostic
The v4 state-machine fix successfully allowed non-zero Trend and Pyramid PnL. The first integrated mechanical implementation is therefore actually exercising all three modules.

The most important diagnostic is that Pyramid PnL is negative in all three tested assets and overwhelms positive MR/Trend contributions in XAUUSD and SPX500. Under this implementation, pyramiding does not improve risk-adjusted performance.

## Interpretation boundary / audit still required
This is a first integrated mechanical pilot, **not** the final production backtest and not sufficient evidence for live capital allocation.

Before calling the Sharpe formal, the implementation must remove any potential higher-timeframe lookahead by ensuring intraday decisions use only the strictly previous completed D1 bar and only the strictly previous completed H4 bar. The test also needs more assets, out-of-sample separation, portfolio correlation/cluster risk, and Pyramid OFF vs ON comparison under identical signals/costs.
