"""Prop-seed holding-period comparison v4.0.

Frozen entry logic: v2.8 Session H/L -> H1 FVG -> M5 execution.
Assets: GOLD, SILVER.
Purpose: compare low-frequency day vs intraday-swing vs swing holding for prop-seed extraction.

Holding models (same entry + structural stop):
A) DAY_8H: max 8h
B) INTRADAY_SWING_16H: max 16h
C) SWING_72H: max 72h

Exit before horizon if H1 trend failure (2 consecutive completed H1 closes across SMA50).
No parameter is selected on OOS. Research only.
"""
from __future__ import annotations

import math, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_fvg_h1_m5_v27 as v27
import session_fvg_h1_m5_v28 as v28

OUT = Path("research/results/prop_seed_holding_v40")
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = ["GOLD", "SILVER"]
COST_R = 0.05
SEED = 42
MC_PATHS = 20000
MC_TRADES = 100
RISK_GRID = [0.0015,0.0020,0.0025,0.0030,0.0040,0.0050]
PROP_DD_LIMIT = 0.04  # normalized 4% failure budget for comparison only

MODELS = {"DAY_8H":8, "INTRADAY_SWING_16H":16, "SWING_72H":72}


def h1_fail_series(h1: pd.DataFrame, side: int) -> pd.Series:
    x=h1.copy()
    sma=x.close.rolling(50).mean()
    if side==1:
        f=(x.close<sma)&(x.close.shift(1)<sma.shift(1))
    else:
        f=(x.close>sma)&(x.close.shift(1)>sma.shift(1))
    return f.fillna(False)


def simulate_horizon(c,m5,h1,hours,model):
    rows=[]; busy_until=None
    for _,q in c.iterrows():
        et=pd.Timestamp(q.entry_time)
        if busy_until is not None and et<=busy_until: continue
        side=int(q.side); entry=float(q.entry); stop=float(q.stop); risk=float(q.risk)
        if risk<=0: continue
        end=et+pd.Timedelta(hours=hours)
        w=m5[(m5.index>=et)&(m5.index<=end)]
        if w.empty: continue
        fail=h1_fail_series(h1,side)
        xt=w.index[-1]; xp=float(w.close.iloc[-1]); reason="HORIZON"; mfe=0.0; mae=0.0
        for t,r in w.iterrows():
            hi=float(r.high); lo=float(r.low)
            fav=(hi-entry)/risk if side==1 else (entry-lo)/risk
            adv=(entry-lo)/risk if side==1 else (hi-entry)/risk
            mfe=max(mfe,fav); mae=max(mae,adv)
            hit=lo<=stop if side==1 else hi>=stop
            if hit:
                xt=t; xp=stop; reason="STOP"; break
            # only completed H1 information strictly before current M5 bar
            hist=fail[fail.index<t]
            if len(hist) and bool(hist.iloc[-1]):
                xt=t; xp=float(r.open); reason="H1_FAIL"; break
        gross=side*(xp-entry)/risk
        rows.append({**q.to_dict(),"exit_time":xt,"exit_model":model,"gross_R":gross,
                     "net_R":gross-COST_R,"reason":reason,"MFE_R":mfe,"MAE_R":mae,
                     "hold_hours":float((xt-et).total_seconds()/3600.0)})
        busy_until=xt
    return pd.DataFrame(rows)


def split(ts):
    y=pd.Timestamp(ts).year
    return "DEV" if y<=2022 else "VAL" if y<=2024 else "OOS"


def metrics(tr):
    if tr.empty:return {}
    r=tr.net_R.astype(float); wins=r[r>0]; losses=r[r<=0]
    eq=pd.concat([pd.Series([0.0]),r.cumsum().reset_index(drop=True)],ignore_index=True)
    dd=eq-eq.cummax(); sd=r.std(ddof=1)
    return {"trades":int(len(r)),"win_rate":float((r>0).mean()),"expectancy_R":float(r.mean()),
            "PF":float(wins.sum()/abs(losses.sum())) if losses.sum()<0 else np.nan,
            "total_R":float(r.sum()),"max_dd_R":float(abs(dd.min())),
            "trade_sharpe":float(r.mean()/sd*math.sqrt(len(r))) if sd>0 else np.nan,
            "avg_MFE_R":float(tr.MFE_R.mean()),"avg_MAE_R":float(tr.MAE_R.mean()),
            "avg_hold_hours":float(tr.hold_hours.mean()),"p95_R":float(r.quantile(.95)),
            "max_win_R":float(r.max())}


def mc_prop(rvals):
    arr=np.asarray(rvals,float)
    rng=np.random.default_rng(SEED)
    out=[]
    if len(arr)==0:return out
    for risk in RISK_GRID:
        breaches=0; finals=[]; mdds=[]
        for _ in range(MC_PATHS):
            p=rng.choice(arr,size=MC_TRADES,replace=True)*risk
            eq=np.cumsum(p)
            peak=np.maximum.accumulate(np.r_[0.0,eq])[:-1]
            dd=eq-peak; mdd=float(dd.min())
            if mdd<=-PROP_DD_LIMIT: breaches+=1
            finals.append(float(eq[-1])); mdds.append(abs(mdd))
        out.append({"risk_pct":risk*100,"mc_trades":MC_TRADES,
                    "breach_prob_4pct":breaches/MC_PATHS,
                    "ending_return_p05":float(np.quantile(finals,.05)),
                    "ending_return_p50":float(np.quantile(finals,.50)),
                    "ending_return_p95":float(np.quantile(finals,.95)),
                    "maxdd_p50":float(np.quantile(mdds,.50)),
                    "maxdd_p95":float(np.quantile(mdds,.95))})
    return out


def main():
    summary=[]; mcrows=[]; all_oos=[]
    for asset in ASSETS:
        m5,h1,d1,prov=v28.load_asset_v28(asset)
        c=v27.candidates(m5,h1,d1)
        for model,hours in MODELS.items():
            tr=simulate_horizon(c,m5,h1,hours,model)
            if tr.empty:continue
            tr["asset"]=asset; tr["split"]=tr.entry_time.apply(split)
            tr.to_csv(OUT/f"{asset}_{model}.csv",index=False)
            for sp in ["DEV","VAL","OOS"]:
                g=tr[tr.split==sp]
                if g.empty:continue
                summary.append({"asset":asset,"model":model,"split":sp,**metrics(g)})
            oos=tr[tr.split=="OOS"].copy()
            if not oos.empty:
                all_oos.append(oos)
                for row in mc_prop(oos.net_R): mcrows.append({"asset":asset,"model":model,**row})

    # Gold+Silver cluster: chronological, one active position at a time per model.
    if all_oos:
        combo=pd.concat(all_oos,ignore_index=True)
        for model,g in combo.groupby("exit_model"):
            g=g.sort_values("entry_time")
            acc=[]; active=None
            for _,q in g.iterrows():
                et=pd.Timestamp(q.entry_time)
                if active is not None and et<=active:continue
                acc.append(q); active=pd.Timestamp(q.exit_time)
            a=pd.DataFrame(acc)
            if not a.empty:
                summary.append({"asset":"METALS_CLUSTER","model":model,"split":"OOS",**metrics(a)})
                for row in mc_prop(a.net_R):mcrows.append({"asset":"METALS_CLUSTER","model":model,**row})
                a.to_csv(OUT/f"METALS_CLUSTER_{model}.csv",index=False)

    s=pd.DataFrame(summary); m=pd.DataFrame(mcrows)
    s.to_csv(OUT/"summary.csv",index=False); m.to_csv(OUT/"prop_mc.csv",index=False)
    print("=== PERFORMANCE ===")
    print(s.to_string(index=False))
    print("\n=== PROP RISK MONTE CARLO (normalized 4% DD budget, 100 trades) ===")
    print(m.to_string(index=False))
    print("\nLIMITATIONS: Gold/Silver were selected after prior OOS inspection; Dukascopy proxy; synthetic 0.05R cost; 4% DD is a normalized comparison budget, not any specific firm's current rule; overnight financing/gaps and exact prop intraday-equity rules are not modeled.")

if __name__=="__main__":main()
