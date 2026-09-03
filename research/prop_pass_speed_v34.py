from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import topview_sector_stock_10y_v33 as v33

OUT = Path("research/results/prop_pass_speed_v34")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["TREND_ONLY", "FULL_TOPVIEW"]
TARGET_RS = [1.0, 1.5, 2.0, 3.0]
RISK_PCTS = [0.50, 0.75, 1.00, 1.50]
MC_SIMS = 20000
MC_HORIZON = 200
PASS_TARGET_PCT = 10.0
FAIL_FLOOR_PCT = -10.0
SEED = 4242


def simulate_fixed(feat, symbol, model, target_r):
    d = feat[symbol]
    sec = v33.stock_sector_map()[symbol]
    sig = v33.signal_for(feat, symbol, model)
    trades = []
    i, n = 210, len(d)
    while i < n - 2:
        if not bool(sig.iloc[i]):
            i += 1
            continue
        ei = i + 1
        entry = float(d["Open"].iloc[ei])
        av = float(d["atr14"].iloc[i])
        if not np.isfinite(av) or av <= 0:
            i += 1
            continue
        swing = float(d["Low"].iloc[max(0, i - 9):i + 1].min())
        stop = min(swing, entry - av)
        risk = entry - stop
        if risk <= 0 or risk / entry > 0.12:
            i += 1
            continue
        xi = None
        xp = None
        mh, ml = entry, entry
        j = ei
        tp = entry + target_r * risk
        while j < n:
            hi = float(d["High"].iloc[j])
            lo = float(d["Low"].iloc[j])
            cl = float(d["Close"].iloc[j])
            mh, ml = max(mh, hi), min(ml, lo)
            # Conservative same-bar assumption: stop first.
            if lo <= stop:
                xp, xi = stop, j
                break
            if hi >= tp:
                xp, xi = tp, j
                break
            if j - ei >= 60:
                xp, xi = cl, j
                break
            j += 1
        if xi is None:
            break
        gross = (xp - entry) / risk
        cost = (2 * v33.COST_BPS_PER_SIDE / 10000.0) * entry / risk
        rr = gross - cost
        trades.append({
            "symbol": symbol, "sector": sec, "model": model, "target_r": target_r,
            "signal_date": d.index[i], "entry_date": d.index[ei], "exit_date": d.index[xi],
            "r": float(rr), "mfe_r": float((mh-entry)/risk), "mae_r": float((ml-entry)/risk),
            "holding_days": int(xi-ei), "period": v33.period(d.index[ei]),
        })
        i = max(xi + 1, i + 1)
    return trades


def stats(g):
    wins = g.loc[g.r > 0, "r"]
    losses = g.loc[g.r < 0, "r"]
    eq = g.r.cumsum()
    dd = eq - eq.cummax()
    return {
        "n": int(len(g)),
        "win_rate": float((g.r > 0).mean()),
        "avg_r": float(g.r.mean()),
        "median_r": float(g.r.median()),
        "pf": float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.nan,
        "max_dd_r": float(dd.min()),
        "r_std": float(g.r.std(ddof=1)),
        "avg_hold_days": float(g.holding_days.mean()),
        "total_r": float(g.r.sum()),
    }


def max_loss_streak(arr):
    best = cur = 0
    for x in arr:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def mc(rvals, risk_pct, rng):
    rvals = np.asarray(rvals, dtype=float)
    pass_n = fail_n = unfinished = 0
    pass_steps = []
    max_dds = []
    streaks = []
    for _ in range(MC_SIMS):
        sample = rng.choice(rvals, size=MC_HORIZON, replace=True)
        pnl = 0.0
        peak = 0.0
        maxdd = 0.0
        outcome = None
        for k, r in enumerate(sample, start=1):
            pnl += r * risk_pct
            peak = max(peak, pnl)
            maxdd = min(maxdd, pnl - peak)
            if pnl >= PASS_TARGET_PCT:
                outcome = "pass"
                pass_steps.append(k)
                pass_n += 1
                break
            if pnl <= FAIL_FLOOR_PCT:
                outcome = "fail"
                fail_n += 1
                break
        if outcome is None:
            unfinished += 1
        max_dds.append(maxdd)
        streaks.append(max_loss_streak(sample))
    return {
        "risk_pct": risk_pct,
        "pass_rate": pass_n/MC_SIMS,
        "fail_rate": fail_n/MC_SIMS,
        "unfinished_rate": unfinished/MC_SIMS,
        "median_trades_to_pass": float(np.median(pass_steps)) if pass_steps else np.nan,
        "p75_trades_to_pass": float(np.quantile(pass_steps, .75)) if pass_steps else np.nan,
        "p05_max_dd_pct": float(np.quantile(max_dds, .05)),
        "median_max_loss_streak": float(np.median(streaks)),
        "p95_max_loss_streak": float(np.quantile(streaks, .95)),
    }


def main():
    data = v33.download()
    feat = v33.add_features(data)
    smap = v33.stock_sector_map()
    stocks = [s for s in smap if s in feat and v33.SECTORS[smap[s]]["etf"] in feat]

    alltr = []
    for model in MODELS:
        for target_r in TARGET_RS:
            for s in stocks:
                alltr.extend(simulate_fixed(feat, s, model, target_r))

    df = pd.DataFrame(alltr).dropna(subset=["period"])
    rows = []
    for (m, t, p), g in df.groupby(["model", "target_r", "period"]):
        rows.append({"model": m, "target_r": t, "period": p, **stats(g.sort_values("entry_date"))})
    summary = pd.DataFrame(rows).sort_values(["period", "avg_r"], ascending=[True, False])

    # Validation-only selection among positive DEV+VAL with >=100 trades each.
    candidates = []
    for model in MODELS:
        for t in TARGET_RS:
            a = summary[(summary.model==model)&(summary.target_r==t)&(summary.period=="DEV")]
            b = summary[(summary.model==model)&(summary.target_r==t)&(summary.period=="VAL")]
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                if a.n >= 100 and b.n >= 100 and a.avg_r > 0 and b.avg_r > 0:
                    # Objective = positive expectancy + consistency; prefer higher VAL avg_r / volatility.
                    score = float(b.avg_r / b.r_std) if b.r_std > 0 else -np.inf
                    candidates.append((score, model, t))
    candidates.sort(reverse=True)
    selected = candidates[0][1:] if candidates else None

    rng = np.random.default_rng(SEED)
    mcrows = []
    for model in MODELS:
        for t in TARGET_RS:
            g = df[(df.model==model)&(df.target_r==t)&(df.period=="OOS")].sort_values("entry_date")
            if len(g) < 100:
                continue
            for rp in RISK_PCTS:
                mcrows.append({"model":model,"target_r":t,**mc(g.r.values, rp, rng)})
    mcdf = pd.DataFrame(mcrows)

    # Explicit speed/consistency score: high pass rate, low failure, shorter median pass, shallower tail DD.
    if len(mcdf):
        mcdf["speed_consistency_score"] = (
            mcdf.pass_rate - 2.0*mcdf.fail_rate
            - 0.002*mcdf.median_trades_to_pass.fillna(MC_HORIZON)
            + 0.02*mcdf.p05_max_dd_pct
        )

    df.to_csv(OUT/"trades.csv", index=False)
    summary.to_csv(OUT/"summary.csv", index=False)
    mcdf.to_csv(OUT/"monte_carlo.csv", index=False)
    meta = {
        "start": v33.START, "end": v33.END, "stocks": len(stocks), "trades": len(df),
        "models": MODELS, "targets_r": TARGET_RS, "risk_pcts": RISK_PCTS,
        "pass_target_pct": PASS_TARGET_PCT, "fail_floor_pct": FAIL_FLOOR_PCT,
        "mc_sims": MC_SIMS, "mc_horizon_trades": MC_HORIZON,
        "selection": "VAL only; OOS untouched; selection score=VAL avgR/std",
        "selected": selected,
        "limitations": [
            "daily-bar proxy, not intraday execution",
            "survivorship bias in current-stock universe",
            "iid bootstrap ignores cross-trade correlation, simultaneous exposure, floating DD, and exact prop daily-loss mechanics",
            "BigView proprietary indicators are not used; public-data proxy only"
        ]
    }
    (OUT/"meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("=== PROP PASS SPEED + CONSISTENCY v3.4 ===")
    print(json.dumps(meta, indent=2, default=str))
    print("\nSUMMARY")
    print(summary.to_string(index=False))
    print("\nOOS MONTE CARLO")
    print(mcdf.sort_values("speed_consistency_score", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
