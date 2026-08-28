"""v1.8 Classical Pattern Strategy (causal, deterministic).

Adds a separate entry family for traditional chart patterns without contaminating
MTF v1.7 exit experiment. Patterns are detected only from confirmed pivots and
entries occur only after breakout confirmation.

Patterns:
- Symmetric triangle
- Head and shoulders
- Inverse head and shoulders

Exit models reuse v1.7 convex exit engine: fixed 2R/3R/5R, pure swing, adaptive trail.
No OOS-based parameter selection.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal
import mtf_exit_engine_v17 as exits

OUT = Path("research/results/pattern_v18")
OUT.mkdir(parents=True, exist_ok=True)

PIVOT_L = 3
PIVOT_R = 3
BREAK_VOL = 1.2
MAX_PATTERN_BARS = 160
MIN_SEP = 4


def confirmed_pivots(df):
    """Return pivots timestamped at their CONFIRMATION time, not pivot-center time."""
    x = df.copy()
    highs = x.high.to_numpy(float); lows = x.low.to_numpy(float)
    times = x.index
    piv=[]
    for i in range(PIVOT_L, len(x)-PIVOT_R):
        wh = highs[i-PIVOT_L:i+PIVOT_R+1]
        wl = lows[i-PIVOT_L:i+PIVOT_R+1]
        ct = times[i+PIVOT_R]
        if np.isfinite(highs[i]) and highs[i] == np.nanmax(wh) and (wh == highs[i]).sum() == 1:
            piv.append({"confirm_time":ct,"pivot_time":times[i],"kind":"H","price":highs[i],"idx":i})
        if np.isfinite(lows[i]) and lows[i] == np.nanmin(wl) and (wl == lows[i]).sum() == 1:
            piv.append({"confirm_time":ct,"pivot_time":times[i],"kind":"L","price":lows[i],"idx":i})
    return pd.DataFrame(piv).sort_values(["confirm_time","pivot_time"]).reset_index(drop=True)


def line_value(t0,p0,t1,p1,t):
    dt=(t1-t0).total_seconds()
    if dt <= 0: return np.nan
    return p0 + (p1-p0)*((t-t0).total_seconds()/dt)


def detect_hs(p, inverse=False):
    """Detect H-L-H-L-H (or L-H-L-H-L) from last 5 alternating confirmed pivots."""
    want = ["L","H","L","H","L"] if inverse else ["H","L","H","L","H"]
    if len(p) < 5: return None
    q=p.iloc[-5:]
    if q.kind.tolist()!=want: return None
    a,b,c,d,e=[q.iloc[i] for i in range(5)]
    if min(c.idx-a.idx, e.idx-c.idx) < MIN_SEP: return None
    span=e.idx-a.idx
    if span > MAX_PATTERN_BARS: return None
    if inverse:
        # head lower than both shoulders; shoulders reasonably similar
        if not (c.price < a.price and c.price < e.price): return None
        shoulder_gap=abs(a.price-e.price)/max(abs(c.price),1e-12)
        if shoulder_gap > 0.03: return None
        neckline=(b,d)
        side=1
        stop=min(float(c.price), float(e.price))
        name="INV_HS"
    else:
        if not (c.price > a.price and c.price > e.price): return None
        shoulder_gap=abs(a.price-e.price)/max(abs(c.price),1e-12)
        if shoulder_gap > 0.03: return None
        neckline=(b,d)
        side=-1
        stop=max(float(c.price), float(e.price))
        name="HS"
    return {"pattern":name,"side":side,"neck0":neckline[0],"neck1":neckline[1],"stop":stop,
            "pattern_start":a.pivot_time,"pattern_end":e.confirm_time}


def detect_triangle(p):
    """Symmetric triangle: last two confirmed highs descend and last two lows ascend."""
    if len(p) < 6: return None
    q=p.iloc[-8:]
    hs=q[q.kind=="H"].tail(2); ls=q[q.kind=="L"].tail(2)
    if len(hs)<2 or len(ls)<2: return None
    h0,h1=hs.iloc[0],hs.iloc[1]; l0,l1=ls.iloc[0],ls.iloc[1]
    if not (h1.price < h0.price and l1.price > l0.price): return None
    start=min(h0.idx,l0.idx); end=max(h1.idx,l1.idx)
    if end-start > MAX_PATTERN_BARS or min(h1.idx-h0.idx,l1.idx-l0.idx)<MIN_SEP: return None
    return {"pattern":"TRIANGLE","hi0":h0,"hi1":h1,"lo0":l0,"lo1":l1,
            "pattern_start":min(h0.pivot_time,l0.pivot_time),"pattern_end":max(h1.confirm_time,l1.confirm_time)}


def build_candidates(m15):
    x=m15.copy()
    x["vol_base"]=x.volume.shift(1).rolling(20,min_periods=20).mean()
    x["vol_ratio"]=x.volume/x.vol_base.replace(0,np.nan)
    piv=confirmed_pivots(x)
    out=[]; used=set()
    for t,r in x.iterrows():
        known=piv[piv.confirm_time <= t]
        if len(known)<5 or not np.isfinite(r.vol_ratio) or r.vol_ratio < BREAK_VOL:
            continue

        # H&S families
        for inv in (False,True):
            pat=detect_hs(known,inverse=inv)
            if not pat or t <= pat["pattern_end"]: continue
            n0,n1=pat["neck0"],pat["neck1"]
            neck=line_value(n0.pivot_time,float(n0.price),n1.pivot_time,float(n1.price),t)
            if not np.isfinite(neck): continue
            side=pat["side"]
            broke=(r.close>neck and r.open<=neck) if side==1 else (r.close<neck and r.open>=neck)
            key=(pat["pattern"],pat["pattern_start"],pat["pattern_end"])
            if broke and key not in used:
                entry=float(r.close); stop=float(pat["stop"]); risk=abs(entry-stop)
                if risk>0 and ((side==1 and stop<entry) or (side==-1 and stop>entry)):
                    out.append({"entry_time":t,"signal_available_time":t,"side":side,"entry":entry,"stop":stop,
                                "risk":risk,"pattern":pat["pattern"],"pattern_start":pat["pattern_start"],
                                "pattern_end":pat["pattern_end"],"break_level":neck,"vol_ratio":float(r.vol_ratio)})
                    used.add(key)

        # Symmetric triangle can break either way
        tri=detect_triangle(known)
        if tri and t > tri["pattern_end"]:
            upper=line_value(tri["hi0"].pivot_time,float(tri["hi0"].price),tri["hi1"].pivot_time,float(tri["hi1"].price),t)
            lower=line_value(tri["lo0"].pivot_time,float(tri["lo0"].price),tri["lo1"].pivot_time,float(tri["lo1"].price),t)
            key=("TRIANGLE",tri["pattern_start"],tri["pattern_end"])
            if np.isfinite(upper) and np.isfinite(lower) and key not in used:
                if r.close>upper and r.open<=upper:
                    side=1; stop=float(tri["lo1"].price); level=upper
                elif r.close<lower and r.open>=lower:
                    side=-1; stop=float(tri["hi1"].price); level=lower
                else:
                    side=0
                if side:
                    entry=float(r.close); risk=abs(entry-stop)
                    if risk>0 and ((side==1 and stop<entry) or (side==-1 and stop>entry)):
                        out.append({"entry_time":t,"signal_available_time":t,"side":side,"entry":entry,"stop":stop,
                                    "risk":risk,"pattern":"TRIANGLE","pattern_start":tri["pattern_start"],
                                    "pattern_end":tri["pattern_end"],"break_level":level,"vol_ratio":float(r.vol_ratio)})
                        used.add(key)
    c=pd.DataFrame(out)
    if not c.empty: c=c.sort_values("entry_time").drop_duplicates(["entry_time","pattern"])
    return c


def main():
    name=base.os.environ["ASSET"]
    m15,m1,daily,prov=causal.load_asset_causal(name)
    c=build_candidates(m15)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    c.to_csv(out/"candidates.csv",index=False)
    rows=[]
    models=[("FIXED_2R",lambda: exits.fixed_tp(c,m1,2.0)),
            ("FIXED_3R",lambda: exits.fixed_tp(c,m1,3.0)),
            ("FIXED_5R",lambda: exits.fixed_tp(c,m1,5.0)),
            ("PURE_SWING",lambda: exits.trailing(c,m1,"PURE_SWING")),
            ("ADAPTIVE_TRAIL",lambda: exits.trailing(c,m1,"ADAPTIVE_TRAIL"))]
    for label,fn in models:
        tr=fn(); tr.to_csv(out/f"trades_{label}.csv",index=False)
        if tr.empty:
            rows.append({"asset":name,"exit_model":label,"split":"ALL","trades":0}); continue
        tr["split"]=tr.entry_time.map(base.split)
        for sp,g in [("ALL",tr),*list(tr.groupby("split"))]:
            m=exits.metrics_ext(g); m.update({"asset":name,"exit_model":label,"split":sp}); rows.append(m)
    metrics=pd.DataFrame(rows); metrics.to_csv(out/"metrics.csv",index=False)
    (out/"meta.json").write_text(json.dumps({
        "asset":name,"provenance":prov,"candidate_count":len(c),
        "patterns":["TRIANGLE","HS","INV_HS"],
        "pivot_rule":f"causal confirmed pivots L={PIVOT_L}, R={PIVOT_R}",
        "break_volume_min":BREAK_VOL,
        "principle":"pattern must be fully confirmed before breakout; no repainting pivots",
        "selection_rule":"DEV/VAL for selection; OOS verification only"
    },indent=2,default=str))
    print(metrics.to_markdown(index=False))

if __name__=="__main__": main()
