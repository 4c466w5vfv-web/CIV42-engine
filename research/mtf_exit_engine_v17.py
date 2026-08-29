"""MTF v1.7 exit-only experiment.

Entry/candidate generation is inherited unchanged from causal-fixed v1.6.1.
This experiment compares fixed TP, pure/adaptive trailing, and a partial-profit
breakeven runner. No OOS-based parameter selection is performed here.
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
                if max_r >= 2.0:
                    ref = r.swing15_lo if side == 1 else r.swing15_hi
                    if np.isfinite(ref):
                        new_stop = max(stop, float(ref)) if side == 1 else min(stop, float(ref))
            elif model == "ADAPTIVE_TRAIL":
                if max_r >= 5.0:
                    ref = r.swing60_lo if side == 1 else r.swing60_hi
                    if np.isfinite(ref):
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


def half_profit_be_runner(c, m1, partial_r=2.0, fraction=0.5):
    """Take half at +2R, move the remainder to breakeven, then let it run.

    Runner management:
      - before partial: original structural stop, no hedge, no scale-in.
      - at +2R: realize 50% and move remaining 50% stop to entry.
      - after +2R: runner uses causal 15-minute structure.
      - after +5R MFE: runner references 60-minute structure, but monetary stop never loosens.

    Same-bar ambiguity is handled conservatively. The pre-existing stop is checked
    first. If +2R and the new breakeven level are both touched within the same 1m
    bar, the partial is credited but the runner is assumed to exit at breakeven.
    """
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
        partial_done = False
        realized_r = 0.0
        remaining = 1.0
        max_r = 0.0
        w = x[(x.index > et) & (x.index <= et + MAX_HOLD)]
        if w.empty:
            continue
        runner_xp = float(w.close.iloc[-1]); xt = w.index[-1]; reason = "timeout"

        for t, r in w.iterrows():
            # 1) Stop known before this bar gets priority.
            stop_hit = (r.low <= stop) if side == 1 else (r.high >= stop)
            if stop_hit:
                runner_xp = stop; xt = t
                reason = "runner_stop" if partial_done else "initial_stop"
                break

            favorable = ((r.high-entry)/risk) if side == 1 else ((entry-r.low)/risk)
            if np.isfinite(favorable):
                max_r = max(max_r, float(favorable))

            # 2) First +2R touch: realize half and arm BE on the runner.
            if not partial_done and max_r >= partial_r:
                partial_done = True
                realized_r += fraction * partial_r
                remaining = 1.0 - fraction
                stop = entry

                # New BE stop did not exist before the target touch. If the same
                # minute also spans entry, order is unknowable; assume runner exits
                # at BE to avoid optimistic intrabar sequencing.
                same_bar_be = (r.low <= entry) if side == 1 else (r.high >= entry)
                if same_bar_be:
                    runner_xp = entry; xt = t; reason = "partial_then_same_bar_BE"
                    break

            # 3) Runner trail only after partial. References are prior completed bars.
            if partial_done:
                new_stop = stop
                if max_r >= 5.0:
                    ref = r.swing60_lo if side == 1 else r.swing60_hi
                else:
                    ref = r.swing15_lo if side == 1 else r.swing15_hi
                if np.isfinite(ref):
                    new_stop = max(stop, float(ref)) if side == 1 else min(stop, float(ref))
                stop = new_stop

        runner_r = side * (runner_xp-entry) / risk
        gross = realized_r + remaining * runner_r
        rows.append({**q.to_dict(), "exit_time": xt, "gross_R": gross,
                     "net_R": gross-COST_R, "reason": reason,
                     "max_favorable_R": max_r, "exit_model": "HALF_2R_BE_RUNNER",
                     "partial_done": partial_done, "partial_R": partial_r,
                     "partial_fraction": fraction, "runner_R": runner_r})
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
    if "partial_done" in tr.columns:
        m["partial_hit_rate"] = float(tr.partial_done.fillna(False).mean())
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
              ("ADAPTIVE_TRAIL", lambda: trailing(c,m1,"ADAPTIVE_TRAIL")),
              ("HALF_2R_BE_RUNNER", lambda: half_profit_be_runner(c,m1,2.0,0.5))]
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
            "ADAPTIVE_TRAIL":"initial structural stop; no TP; +2R prior-15m trail; +5R prior-60m horizon; stop never loosens",
            "HALF_2R_BE_RUNNER":"no hedge; close 50% at +2R; move 50% runner stop to breakeven; causal 15m trail, then 60m reference after +5R; stop never loosens"
        },
        "cost_R":COST_R,"max_hold":"20d",
        "selection_rule":"Do not select using OOS. Compare DEV/VAL first; OOS is verification only."
    }, indent=2, default=str))
    print(metrics.to_markdown(index=False))

if __name__ == "__main__":
    main()
