"""MTF v1.7 exit-only experiment.

Entry/candidate generation is inherited unchanged from causal-fixed v1.6.1.
This experiment compares fixed TP against two no-fixed-TP trailing exits.
No OOS-based parameter selection is performed here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal

OUT = Path("research/results/mtf_v17_exit")
OUT.mkdir(parents=True, exist_ok=True)
COST_R = base.COST_R
MAX_HOLD = pd.Timedelta(days=20)


def add_swings(m1):
    x = m1.copy()
    # Causal trailing references: only completed prior bars.
    x["swing15_lo"] = x.low.shift(1).rolling(15, min_periods=15).min()
    x["swing15_hi"] = x.high.shift(1).rolling(15, min_periods=15).max()
    x["swing60_lo"] = x.low.shift(1).rolling(60, min_periods=60).min()
    x["swing60_hi"] = x.high.shift(1).rolling(60, min_periods=60).max()
    return x


def fixed_tp(c, m1, tp):
    tr = base.simulate(c, m1, tp)
    if not tr.empty:
        tr["exit_model"] = f"FIXED_{int(tp)}R"
    return tr


def trailing(c, m1, model):
    x = add_swings(m1)
    rows = []
    busy_until = None
    for _, q in c.iterrows():
        et = pd.Timestamp(q.entry_time)
        if busy_until is not None and et <= busy_until:
            continue
        side = int(q.side); entry = float(q.entry); initial_stop = float(q.stop); risk = float(q.risk)
        if risk <= 0:
            continue
        stop = initial_stop
        max_r = 0.0
        w = x[(x.index > et) & (x.index <= et + MAX_HOLD)]
        if w.empty:
            continue
        xp = float(w.close.iloc[-1]); xt = w.index[-1]; reason = "timeout"
        for t, r in w.iterrows():
            # Conservative ordering: test stop with the stop known before this bar,
            # then update trailing stop from information available at this bar close.
            stop_hit = (r.low <= stop) if side == 1 else (r.high >= stop)
            if stop_hit:
                xp = stop; xt = t; reason = "trail_stop" if stop != initial_stop else "initial_stop"; break

            favorable = ((r.high-entry)/risk) if side == 1 else ((entry-r.low)/risk)
            if np.isfinite(favorable):
                max_r = max(max_r, float(favorable))

            new_stop = stop
            if model == "PURE_SWING":
                # No fixed TP. Once +2R has been seen, trail the completed 15m-equivalent
                # structure (15 prior 1m bars). Initial risk remains intact before +2R.
                if max_r >= 2.0:
                    ref = r.swing15_lo if side == 1 else r.swing15_hi
                    if np.isfinite(ref):
                        new_stop = max(stop, float(ref)) if side == 1 else min(stop, float(ref))
            elif model == "ADAPTIVE_TRAIL":
                # No fixed TP. +2R activates 15m structure; +5R deliberately widens
                # the structural horizon to 60 prior 1m bars to let large trends breathe.
                if max_r >= 5.0:
                    ref = r.swing60_lo if side == 1 else r.swing60_hi
                    if np.isfinite(ref):
                        # Never loosen an already-ratcheted monetary stop.
                        new_stop = max(stop, float(ref)) if side == 1 else min(stop, float(ref))
                elif max_r >= 2.0:
                    ref = r.swing15_lo if side == 1 else r.swing15_hi
                    if np.isfinite(ref):
                        new_stop = max(stop, float(ref)) if side == 1 else min(stop, float(ref))
            else:
                raise ValueError(model)
            stop = new_stop

        gross = side * (xp-entry) / risk
        rows.append({**q.to_dict(), "exit_time": xt, "gross_R": gross,
                     "net_R": gross-COST_R, "reason": reason,
                     "max_favorable_R": max_r, "exit_model": model})
        busy_until = xt
    return pd.DataFrame(rows)


def metrics_ext(tr):
    m = base.metrics(tr)
    if tr.empty:
        return m
    r = tr.net_R.astype(float)
    m.update({
        "median_R": float(r.median()),
        "p90_R": float(r.quantile(.90)),
        "p95_R": float(r.quantile(.95)),
        "p99_R": float(r.quantile(.99)),
        "max_R": float(r.max()),
        "share_gt_5R": float((r >= 5).mean()),
        "share_gt_10R": float((r >= 10).mean()),
    })
    return m


def main():
    name = base.os.environ["ASSET"]
    m15, m1, daily, prov = causal.load_asset_causal(name)
    c = causal.candidates_causal(m15, m1, daily)
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    c.to_csv(out / "candidates.csv", index=False)

    models = [("FIXED_2R", lambda: fixed_tp(c,m1,2.0)),
              ("FIXED_3R", lambda: fixed_tp(c,m1,3.0)),
              ("FIXED_5R", lambda: fixed_tp(c,m1,5.0)),
              ("PURE_SWING", lambda: trailing(c,m1,"PURE_SWING")),
              ("ADAPTIVE_TRAIL", lambda: trailing(c,m1,"ADAPTIVE_TRAIL"))]
    rows=[]
    for label, fn in models:
        tr=fn()
        tr.to_csv(out / f"trades_{label}.csv", index=False)
        if tr.empty:
            rows.append({"asset":name,"exit_model":label,"split":"ALL","trades":0}); continue
        tr["split"] = tr.entry_time.map(base.split)
        for sp,g in [("ALL",tr), *list(tr.groupby("split"))]:
            m=metrics_ext(g); m.update({"asset":name,"exit_model":label,"split":sp}); rows.append(m)
    metrics=pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "asset":name,"provenance":prov,"candidate_count":len(c),
        "experiment":"EXIT ONLY; causal v1.6.1 entries unchanged",
        "baseline":["FIXED_2R","FIXED_3R","FIXED_5R"],
        "challengers":{
            "PURE_SWING":"initial structural stop; no TP; after observed +2R trail prior-15m structure",
            "ADAPTIVE_TRAIL":"initial structural stop; no TP; +2R prior-15m trail; +5R prior-60m horizon; stop never loosens"
        },
        "cost_R":COST_R,"max_hold":"20d",
        "selection_rule":"Do not select using OOS. Compare DEV/VAL first; OOS is verification only."
    }, indent=2, default=str))
    print(metrics.to_markdown(index=False))

if __name__ == "__main__":
    main()
