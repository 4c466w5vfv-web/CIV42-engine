from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import topview_sector_stock_10y_v33 as v33
from prop_pass_speed_v34 import simulate_fixed

OUT = Path("research/results/buffer_adaptive_risk_v36")
OUT.mkdir(parents=True, exist_ok=True)

MODEL = "TREND_ONLY"
TARGET_R = 1.5
MC_SIMS = 20000
HORIZONS = [30, 60, 90]
TRADES_PER_DAY = 2
MAX_TRADES = max(HORIZONS) * TRADES_PER_DAY
SEED = 3642

# Funded-account proxy: start from 0%, hard floor at -10% from initial balance.
HARD_FLOOR = -10.0
PAYOUT_TRIGGER = 5.0

# Risk policies are % of initial balance per 1R.
POLICIES = {
    "STATIC_050": [(float("-inf"), 0.50)],
    "STATIC_075": [(float("-inf"), 0.75)],
    "STATIC_100": [(float("-inf"), 1.00)],
    "STATIC_125": [(float("-inf"), 1.25)],
    "ADAPT_025_050_075": [(5.0, 0.75), (2.0, 0.50), (float("-inf"), 0.25)],
    "ADAPT_050_075_100": [(5.0, 1.00), (2.0, 0.75), (float("-inf"), 0.50)],
    "ADAPT_050_075_100_125": [(8.0, 1.25), (5.0, 1.00), (2.0, 0.75), (float("-inf"), 0.50)],
}


def risk_for(policy, equity_pct):
    for threshold, risk in POLICIES[policy]:
        if equity_pct >= threshold:
            return risk
    raise RuntimeError("policy resolution failed")


def simulate_path(rvals, policy, rng):
    sample = rng.choice(rvals, size=MAX_TRADES, replace=True)
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    breached = False
    first_payout_trade = None
    payout_hits = 0
    next_payout = PAYOUT_TRIGGER
    records = {}

    for i, r in enumerate(sample, start=1):
        rp = risk_for(policy, equity)
        equity += r * rp
        peak = max(peak, equity)
        max_dd = min(max_dd, equity - peak)

        if first_payout_trade is None and equity >= PAYOUT_TRIGGER:
            first_payout_trade = i

        while equity >= next_payout:
            payout_hits += 1
            next_payout += PAYOUT_TRIGGER

        if equity <= HARD_FLOOR:
            breached = True

        if i in {h * TRADES_PER_DAY for h in HORIZONS}:
            day = i // TRADES_PER_DAY
            records[day] = {
                "equity_pct": equity,
                "survived": not breached,
                "buffer_retained_2": equity >= 2.0,
                "buffer_retained_5": equity >= 5.0,
                "payout_reached": first_payout_trade is not None and first_payout_trade <= i,
            }

        if breached:
            # Fill remaining horizons with breached state.
            for h in HORIZONS:
                if h * TRADES_PER_DAY >= i and h not in records:
                    records[h] = {
                        "equity_pct": equity,
                        "survived": False,
                        "buffer_retained_2": False,
                        "buffer_retained_5": False,
                        "payout_reached": first_payout_trade is not None and first_payout_trade <= i,
                    }
            break

    return {
        "records": records,
        "breached": breached,
        "max_dd_pct": max_dd,
        "first_payout_trade": first_payout_trade,
        "payout_hits": payout_hits,
        "ending_equity_pct": equity,
    }


def main():
    data = v33.download()
    feat = v33.add_features(data)
    smap = v33.stock_sector_map()
    stocks = [s for s in smap if s in feat and v33.SECTORS[smap[s]]["etf"] in feat]

    trades = []
    for s in stocks:
        trades.extend(simulate_fixed(feat, s, MODEL, TARGET_R))
    df = pd.DataFrame(trades).dropna(subset=["period"])
    oos = df[df.period == "OOS"].sort_values("entry_date")
    rvals = oos.r.to_numpy(float)

    rng = np.random.default_rng(SEED)
    rows = []
    path_rows = []

    for policy in POLICIES:
        sims = [simulate_path(rvals, policy, rng) for _ in range(MC_SIMS)]
        maxdds = np.array([x["max_dd_pct"] for x in sims], dtype=float)
        endeq = np.array([x["ending_equity_pct"] for x in sims], dtype=float)
        firstp = np.array([x["first_payout_trade"] if x["first_payout_trade"] is not None else np.nan for x in sims], dtype=float)
        hits = np.array([x["payout_hits"] for x in sims], dtype=float)

        for h in HORIZONS:
            recs = [x["records"][h] for x in sims]
            rows.append({
                "policy": policy,
                "horizon_days": h,
                "survival_rate": float(np.mean([r["survived"] for r in recs])),
                "breach_rate": float(1 - np.mean([r["survived"] for r in recs])),
                "payout_reach_rate": float(np.mean([r["payout_reached"] for r in recs])),
                "buffer_2_retention_rate": float(np.mean([r["buffer_retained_2"] for r in recs])),
                "buffer_5_retention_rate": float(np.mean([r["buffer_retained_5"] for r in recs])),
                "median_equity_pct": float(np.median([r["equity_pct"] for r in recs])),
                "p05_equity_pct": float(np.quantile([r["equity_pct"] for r in recs], .05)),
                "p95_equity_pct": float(np.quantile([r["equity_pct"] for r in recs], .95)),
            })

        path_rows.append({
            "policy": policy,
            "overall_breach_rate_90d": float(np.mean([x["breached"] for x in sims])),
            "median_first_payout_trade": float(np.nanmedian(firstp)) if np.isfinite(firstp).any() else np.nan,
            "median_first_payout_days": float(np.nanmedian(firstp) / TRADES_PER_DAY) if np.isfinite(firstp).any() else np.nan,
            "payout_reach_rate_90d": float(np.mean(np.isfinite(firstp))),
            "median_payout_hits": float(np.median(hits)),
            "median_end_equity_pct": float(np.median(endeq)),
            "p05_end_equity_pct": float(np.quantile(endeq, .05)),
            "p95_end_equity_pct": float(np.quantile(endeq, .95)),
            "p05_max_dd_pct": float(np.quantile(maxdds, .05)),
            "median_max_dd_pct": float(np.median(maxdds)),
        })

    horizon = pd.DataFrame(rows)
    overall = pd.DataFrame(path_rows)

    # Decision score: payout speed and reach, strongly penalize breach and deep tail DD.
    overall["funded_score"] = (
        overall.payout_reach_rate_90d
        - 3.0 * overall.overall_breach_rate_90d
        - 0.01 * overall.median_first_payout_days.fillna(90)
        + 0.02 * overall.p05_max_dd_pct
    )

    horizon.to_csv(OUT / "horizon_metrics.csv", index=False)
    overall.sort_values("funded_score", ascending=False).to_csv(OUT / "policy_comparison.csv", index=False)
    oos.to_csv(OUT / "oos_trades_1p5r.csv", index=False)

    meta = {
        "purpose": "funded-account buffer-adaptive risk proxy",
        "model": MODEL,
        "target_r": TARGET_R,
        "oos_trades": int(len(oos)),
        "oos_win_rate": float((oos.r > 0).mean()),
        "oos_avg_r": float(oos.r.mean()),
        "oos_pf": float(oos.loc[oos.r > 0, 'r'].sum() / abs(oos.loc[oos.r < 0, 'r'].sum())),
        "mc_sims": MC_SIMS,
        "horizons_days": HORIZONS,
        "trades_per_day_proxy": TRADES_PER_DAY,
        "hard_floor_pct": HARD_FLOOR,
        "payout_trigger_pct": PAYOUT_TRIGGER,
        "policies": POLICIES,
        "limitations": [
            "daily US-stock proxy, not intraday CFD fills",
            "iid bootstrap ignores serial dependence, correlation and simultaneous exposure",
            "hard floor is static initial-balance proxy; firm-specific trailing/daily/floating-equity rules are not modeled",
            "payout trigger is a research threshold, not a broker or prop-firm rule",
            "survivorship bias remains in current-stock universe",
        ],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("=== BUFFER ADAPTIVE RISK v3.6 ===")
    print(json.dumps(meta, indent=2, default=str))
    print("\nPOLICY COMPARISON")
    print(overall.sort_values("funded_score", ascending=False).to_string(index=False))
    print("\nHORIZON METRICS")
    print(horizon.sort_values(["horizon_days", "survival_rate"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
