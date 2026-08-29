"""v1.9 Diagonal Structure Strategy (causal, no ZigZag).

Separate entry-family experiment. Trendlines are built only from confirmed 15m
pivots. No chart-angle degrees are used; slope is normalized by ATR and time.
Entry requires a causal trendline break, then a retest/reclaim within a fixed
window. Exit models are inherited unchanged from v1.7.

Research protocol: DEV/VAL for selection; OOS verification only.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal
import mtf_exit_engine_v17 as exits

OUT = Path("research/results/diagonal_v19")
OUT.mkdir(parents=True, exist_ok=True)

PIVOT_L = 3
PIVOT_R = 3
ATR_N = 14
VOL_N = 20
BREAK_VOL = 1.2
MIN_PIVOT_SEP = 4
MAX_LINE_BARS = 160
RETEST_BARS = 8
RETEST_TOL_ATR = 0.20
MIN_NORM_SLOPE = 0.01
MAX_NORM_SLOPE = 0.50


def enrich(m15):
    x = m15.copy()
    prev = x.close.shift(1)
    tr = pd.concat([(x.high-x.low).abs(), (x.high-prev).abs(), (x.low-prev).abs()], axis=1).max(axis=1)
    x["atr"] = tr.shift(1).rolling(ATR_N, min_periods=ATR_N).mean()
    x["vol_base"] = x.volume.shift(1).rolling(VOL_N, min_periods=VOL_N).mean()
    x["vol_ratio"] = x.volume / x.vol_base.replace(0, np.nan)
    return x


def confirmed_pivots(x):
    """Pivot becomes usable only at center + PIVOT_R bars; no ZigZag/repainting."""
    h=x.high.to_numpy(float); l=x.low.to_numpy(float); times=x.index
    rows=[]
    for i in range(PIVOT_L, len(x)-PIVOT_R):
        wh=h[i-PIVOT_L:i+PIVOT_R+1]; wl=l[i-PIVOT_L:i+PIVOT_R+1]
        ct=times[i+PIVOT_R]
        if np.isfinite(h[i]) and h[i] == np.nanmax(wh) and (wh == h[i]).sum() == 1:
            rows.append({"confirm_time":ct,"pivot_time":times[i],"kind":"H","price":h[i],"idx":i})
        if np.isfinite(l[i]) and l[i] == np.nanmin(wl) and (wl == l[i]).sum() == 1:
            rows.append({"confirm_time":ct,"pivot_time":times[i],"kind":"L","price":l[i],"idx":i})
    if not rows:
        return pd.DataFrame(columns=["confirm_time","pivot_time","kind","price","idx"])
    return pd.DataFrame(rows).sort_values(["confirm_time","pivot_time"]).reset_index(drop=True)


def line_value(a,b,t):
    dt=(b.pivot_time-a.pivot_time).total_seconds()
    if dt <= 0: return np.nan
    return float(a.price) + (float(b.price)-float(a.price))*((t-a.pivot_time).total_seconds()/dt)


def line_candidate(known, kind, atr):
    q=known[known.kind==kind].tail(2)
    if len(q)<2 or not np.isfinite(atr) or atr<=0: return None
    a,b=q.iloc[0],q.iloc[1]
    sep=int(b.idx-a.idx)
    if sep < MIN_PIVOT_SEP or sep > MAX_LINE_BARS: return None
    dp=float(b.price-a.price)
    # Resistance must descend; support must ascend.
    if kind=="H" and dp >= 0: return None
    if kind=="L" and dp <= 0: return None
    norm=abs(dp)/float(atr)/sep
    if not (MIN_NORM_SLOPE <= norm <= MAX_NORM_SLOPE): return None
    return a,b,norm


def build_candidates(m15):
    x=enrich(m15)
    piv=confirmed_pivots(x)
    out=[]; armed=[]; used=set()
    for i,(t,r) in enumerate(x.iterrows()):
        known=piv[piv.confirm_time <= t]
        # First process previously armed breaks. Retest cannot occur on break bar.
        next_armed=[]
        for a in armed:
            if i > a["expires_i"]: continue
            level=line_value(a["p0"],a["p1"],t)
            if not np.isfinite(level) or not np.isfinite(r.atr):
                next_armed.append(a); continue
            tol=RETEST_TOL_ATR*float(r.atr)
            if a["side"]==1:
                touched=r.low <= level+tol
                confirmed=touched and r.close > level
                stop=min(float(a["p0"].price),float(a["p1"].price))
            else:
                touched=r.high >= level-tol
                confirmed=touched and r.close < level
                stop=max(float(a["p0"].price),float(a["p1"].price))
            if confirmed:
                entry=float(r.close); risk=abs(entry-stop)
                if risk>0 and ((a["side"]==1 and stop<entry) or (a["side"]==-1 and stop>entry)):
                    out.append({"entry_time":t,"signal_available_time":t,"side":a["side"],"entry":entry,
                                "stop":stop,"risk":risk,"pattern":a["pattern"],"line_kind":a["kind"],
                                "line_pivot0":a["p0"].pivot_time,"line_pivot1":a["p1"].pivot_time,
                                "break_time":a["break_time"],"break_level":a["break_level"],
                                "retest_level":level,"normalized_slope":a["norm"],"vol_ratio":a["vol_ratio"]})
                    used.add(a["key"])
                continue
            next_armed.append(a)
        armed=next_armed

        if len(known)<2 or not np.isfinite(r.vol_ratio) or r.vol_ratio < BREAK_VOL:
            continue
        # Descending resistance break -> long. Ascending support break -> short.
        for kind,side,name in [("H",1,"DESC_RES_BREAK_RETEST"),("L",-1,"ASC_SUP_BREAK_RETEST")]:
            lc=line_candidate(known,kind,float(r.atr) if np.isfinite(r.atr) else np.nan)
            if lc is None: continue
            p0,p1,norm=lc
            key=(kind,p0.pivot_time,p1.pivot_time)
            if key in used or any(z["key"]==key for z in armed): continue
            level=line_value(p0,p1,t)
            if not np.isfinite(level): continue
            # Break is confirmed at this 15m close; entry waits for a later retest.
            broke=(r.close>level and r.open<=level) if side==1 else (r.close<level and r.open>=level)
            if broke:
                armed.append({"key":key,"kind":kind,"side":side,"pattern":name,"p0":p0,"p1":p1,
                              "norm":norm,"break_time":t,"break_level":level,"vol_ratio":float(r.vol_ratio),
                              "expires_i":i+RETEST_BARS})
    c=pd.DataFrame(out)
    if not c.empty:
        c=c.sort_values("entry_time").drop_duplicates(["entry_time","pattern"])
    return c


def main():
    name=base.os.environ["ASSET"]
    m15,m1,daily,prov=causal.load_asset_causal(name)
    c=build_candidates(m15)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    c.to_csv(out/"candidates.csv",index=False)
    models=[("FIXED_2R",lambda: exits.fixed_tp(c,m1,2.0)),
            ("FIXED_3R",lambda: exits.fixed_tp(c,m1,3.0)),
            ("FIXED_5R",lambda: exits.fixed_tp(c,m1,5.0)),
            ("PURE_SWING",lambda: exits.trailing(c,m1,"PURE_SWING")),
            ("ADAPTIVE_TRAIL",lambda: exits.trailing(c,m1,"ADAPTIVE_TRAIL")),
            ("HALF_2R_BE_RUNNER",lambda: exits.half_profit_be_runner(c,m1,2.0,0.5))]
    rows=[]
    for label,fn in models:
        tr=fn(); tr.to_csv(out/f"trades_{label}.csv",index=False)
        if tr.empty:
            rows.append({"asset":name,"exit_model":label,"split":"ALL","trades":0}); continue
        tr["split"]=tr.entry_time.map(base.split)
        for sp,g in [("ALL",tr),*list(tr.groupby("split"))]:
            m=exits.metrics_ext(g); m.update({"asset":name,"exit_model":label,"split":sp}); rows.append(m)
    pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
    (out/"meta.json").write_text(json.dumps({
        "asset":name,"provenance":prov,"candidate_count":len(c),
        "entry_family":"DIAGONAL_BREAK_RETEST","zigzag":False,
        "pivot_rule":f"causal confirmed 15m pivots L={PIVOT_L}, R={PIVOT_R}",
        "line_rules":"last two descending confirmed highs or last two ascending confirmed lows",
        "slope":"ATR-normalized price change per 15m bar; no screen-angle degrees",
        "break_volume_min":BREAK_VOL,"retest_bars":RETEST_BARS,"retest_tolerance_atr":RETEST_TOL_ATR,
        "selection_rule":"DEV/VAL for selection; OOS verification only"
    },indent=2,default=str))
    print(pd.DataFrame(rows).to_markdown(index=False))

if __name__=="__main__": main()
