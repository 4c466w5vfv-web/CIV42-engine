# NDX/SPX SMA200 + 2ATR Trend Pilot — Run 34003711663

## Scope

This is a **trend-module pilot**, not the full ARK-42 mean-reversion + trend-following system.

Rules executed:
- Signal uses only the completed prior daily bar.
- Trend regime: prior close > SMA200 and 20-day SMA200 slope > 0.
- Entry: next-day open while flat.
- ATR: Wilder ATR14.
- Initial stop: Entry - 2× prior ATR14.
- Trailing stop: Highest High Since Entry - 2× current ATR14.
- Trailing stop only ratchets upward and a completed bar updates the stop for the next bar.
- Risk normalization: 1R = 1% of equity per trade for compounded-equity figures.
- Volume ignored because the TradingView index files contain zero volume.
- Round-trip cost sensitivity: 0, 0.05 ATR, 0.10 ATR.

Workflow run: `34003711663`  
Job: `101407008870`  
Conclusion: `success`

## NDX

Source range: 1985-01-31 14:30:00 → 2026-05-28 13:30:00  
First testable date after SMA200 warm-up: 2023-01-03 14:30:00  
Buy & hold from first testable close: +178.24%

| Round-trip cost | Closed trades | Win rate | Expectancy | Median | Profit factor | Sum R | Compounded equity @ 1% R | MDD @ 1% R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 70 | 42.86% | +0.30R | -0.17R | 1.96 | +21.33R | +23.13% | -5.91% |
| 0.05 ATR | 70 | 42.86% | +0.28R | -0.19R | 1.84 | +19.58R | +21.00% | -6.29% |
| 0.10 ATR | 70 | 42.86% | +0.25R | -0.22R | 1.73 | +17.83R | +18.90% | -6.67% |

Open trade at dataset end (excluded from closed-trade statistics):  
Entry 2026-05-19 13:30 @ 28795.02; last 2026-05-28 close 30223.89; active stop 29398.56; mark-to-market +1.61R.

Selected closed trades:
- 2023-03-28 → 2023-04-25: +0.32R
- 2023-04-26 → 2023-05-24: +1.75R
- 2023-05-25 → 2023-06-21: +2.79R
- 2023-10-27 → 2023-12-04: +3.05R
- 2024-04-22 → 2024-05-30: +2.48R
- 2024-08-05 → 2024-09-03: +2.24R
- 2025-05-19 → 2025-08-01: +2.01R
- 2026-04-09 → 2026-05-18: +4.09R

## SPX

Source range: 1871-02-01 14:26:02 → 2026-05-28 13:30:00  
First testable date after SMA200 warm-up: 2023-01-03 14:30:00  
Buy & hold from first testable close: +97.79%

| Round-trip cost | Closed trades | Win rate | Expectancy | Median | Profit factor | Sum R | Compounded equity @ 1% R | MDD @ 1% R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 72 | 45.83% | +0.20R | -0.18R | 1.65 | +14.72R | +15.34% | -4.98% |
| 0.05 ATR | 72 | 43.06% | +0.18R | -0.21R | 1.55 | +12.92R | +13.28% | -5.20% |
| 0.10 ATR | 72 | 43.06% | +0.15R | -0.23R | 1.45 | +11.12R | +11.26% | -5.41% |

Open trade at dataset end (excluded from closed-trade statistics):  
Entry 2026-05-19 13:30 @ 7375.75; last 2026-05-28 close 7563.62; active stop 7425.98; mark-to-market +1.21R.

Selected closed trades:
- 2023-05-25 → 2023-06-22: +2.32R
- 2023-06-23 → 2023-08-02: +2.10R
- 2023-11-03 → 2024-01-03: +3.39R
- 2024-05-31 → 2024-07-18: +3.70R
- 2024-09-09 → 2024-10-23: +2.17R
- 2025-05-27 → 2025-08-01: +2.29R
- 2026-04-09 → 2026-05-18: +2.77R

## Interpretation boundary

Both instruments remain positive under the tested cost assumptions, but this baseline immediately permits re-entry whenever the SMA200 regime remains valid after a stop. That creates repeated whipsaws and is intentionally simpler than the intended ARK-42 trend-entry rule.

The intended comparison version should require a post-SMA200 structural confirmation sequence such as reclaim/acceptance → Higher Low → prior swing-high close break before re-entry. The current pilot is therefore a control baseline for the next structure-confirmed test.
