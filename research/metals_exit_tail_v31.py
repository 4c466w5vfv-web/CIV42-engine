"""Metals exit/right-tail comparison v3.1.

Frozen entry logic: v2.8 Session H/L -> H1 FVG -> M5 execution.
Assets: GOLD, SILVER only.
Exit comparison:
A) FIXED_2R
B) TP1_2R_PLUS_H1_TRAIL: take 50% at +2R, trail remaining 50% by completed H1 structure.
C) TRAIL_AFTER_3R: no partial; once +3R is first reached, trail full position by completed H1 structure.

Research-only. Gold/Silver were selected after inspected OOS, so this is NOT pristine validation.
"""
from __future__ import annotations

import math, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import session_fvg_h1_m5_v27 as v27
import session_fvg_h1_m5_v28 as v28

OUT = Path("research/results/metals_exit_tail_v31")
OUT.mkdir(parents=True, exist_ok=True)
ASSETS = ["GOLD", "SILVER"]
COST_R = 0.05
MC_PATHS = 10_000
MC_DAYS = 252
BLOCK = 5
SEED = 42


def h1_structural_stop(h1, et, t, side, current_stop, current_close):
    hh = h1[(h1.index > et) & (h1.index <= t)]
    if len(hh) < 3:
        return current_stop
    if side == 1:
        ns = float(hh.low.iloc[-3:-1].min())
        if ns > current_stop and ns < current_close:
            return ns
    else:
        ns = float(hh.high.iloc[-3:-1].max())
        if ns < current_stop and ns > current_close:
            return ns
    return current_stop


def simulate_partial2_trail(c, m5, h1):
    rows=[]; busy_until=None
    for _,q in c.iterrows():
        et=pd.Timestamp(q.entry_time)
        if busy_until is not None and et <= busy_until: continue
        side=int(q.side); entry=float(q.entry); stop=float(q.stop); risk=float(q.risk)
        trigger=entry+side*2.0*risk
        w=m5[(m5.index>=et)&(m5.index<=et+pd.Timedelta(days=10))]
        if w.empty: continue
        hit=False; realized=0.0; xt=w.index[-1]; xp=float(w.close.iloc[-1]); reason="timeout"; mfe=mae=0.0
        for t,r in w.iterrows():
            fav=(float(r.high)-entry)/risk if side==1 else (entry-float(r.low))/risk
            adv=(entry-float(r.low))/risk if side==1 else (float(r.high)-entry)/risk
            mfe=max(mfe,fav); mae=max(mae,adv)
            sh=float(r.low)<=stop if side==1 else float(r.high)>=stop
            if sh:
                xp=stop; xt=t; reason="trail_stop" if hit else "stop"; break
            if not hit:
                th=float(r.high)>=trigger if side==1 else float(r.low)<=trigger
                if th:
                    hit=True; realized=1.0  # 50% * 2R
            if hit:
                stop=h1_structural_stop(h1,et,t,side,stop,float(r.close))
        gross=realized+0.5*(side*(xp-entry)/risk) if hit else side*(xp-entry)/risk
        rows.append({**q.to_dict(),"exit_time":xt,"exit_model":"TP1_2R_PLUS_H1_TRAIL","gross_R":gross,"net_R":gross-COST_R,"reason":reason,"MFE_R":mfe,"MAE_R":mae})
        busy_until=xt
    return pd.DataFrame(rows)


def simulate_full_trail_after3(c,m5,h1):
    rows=[]; busy_until=None
    for _,q in c.iterrows():
        et=pd.Timestamp(q.entry_time)
        if busy_until is not None and et <= busy_until: continue
        side=int(q.side); entry=float(q.entry); stop=float(q.stop); risk=float(q.risk)
        trigger=entry+side*3.0*risk
        w=m5[(m5.index>=et)&(m5.index<=et+pd.Timedelta(days=10))]
        if w.empty: continue
        active=False; xt=w.index[-1]; xp=float(w.close.iloc[-1]); reason="timeout"; mfe=mae=0.0
        for t,r in w.iterrows():
            fav=(float(r.high)-entry)/risk if side==1 else (entry-float(r.low))/risk
            adv=(entry-float(r.low))/risk if side==1 else (float(r.high)-entry)/risk
            mfe=max(mfe,fav); mae=max(mae,adv)
            sh=float(r.low)<=stop if side==1 else float(r.high)>=stop
            if sh:
                xp=stop; xt=t; reason="trail_stop" if active else "stop"; break
            if not active:
                th=float(r.high)>=trigger if side==1 else float(r.low)<=trigger
                if th: active=True
            if active:
                stop=h1_structural_stop(h1,et,t,side,stop,float(r.close))
        gross=side*(xp-entry)/risk
        rows.append({**q.to_dict(),"exit_time":xt,"exit_model":"TRAIL_AFTER_3R","gross_R":gross,"net_R":gross-COST_R,"reason":reason,"MFE_R":mfe,"MAE_R":mae})
        busy_until=xt
    return pd.DataFrame(rows)


def metrics(tr):
    if tr.empty: return {}
    r=tr.net_R.astype(float); wins=r[r>0]; losses=r[r<=0]
    eq=pd.concat([pd.Series([0.0]),r.cumsum().reset_index(drop=True)],ignore_index=True)
    dd=eq-eq.cummax(); sd=r.std(ddof=1)
    return {"trades":len(r),"win_rate":float((r>0).mean()),"expectancy_R":float(r.mean()),
            "PF":float(wins.sum()/abs(losses.sum())) if losses.sum()<0 else np.nan,
            "total_R":float(r.sum()),"max_dd_R":float(abs(dd.min())),
            "trade_sharpe":float(r.mean()/sd*math.sqrt(len(r))) if sd>0 else np.nan,
            "avg_MFE_R":float(tr.MFE_R.mean()),"avg_MAE_R":float(tr.MAE_R.mean()),
            "p90_R":float(r.quantile(.90)),"p95_R":float(r.quantile(.95)),"max_win_R":float(r.max())}


def split(ts):
    y=pd.Timestamp(ts).year
    return "DEV" if y<=2022 else "VAL" if y<=2024 else "OOS"


def block_bootstrap(arr,n,rng):
    out=[]; L=len(arr)
    while len(out)<n:
        s=int(rng.integers(0,max(1,L-BLOCK+1)))
        out.extend(arr[s:s+BLOCK].tolist())
    return np.asarray(out[:n],float)


def mc_trade_sequence(r):
    arr=np.asarray(r,float); rng=np.random.default_rng(SEED); vals=[]
    # convert expected OOS trade frequency to annual by preserving observed trade count per year approximately
    n=max(1,int(round(len(arr)/max(1,2))))  # OOS ~ 2025 through Jul-2026; conservative ~2 years
    for _ in range(MC_PATHS):
        p=block_bootstrap(arr,n,rng)
        eq=np.cumsum(p); peak=np.maximum.accumulate(np.r_[0.0,eq])[:-1]; dd=eq-peak
        vals.append((p.sum(),abs(dd.min())))
    x=np.asarray(vals)
    return {"mc_annual_R_p05":float(np.quantile(x[:,0],.05)),"mc_annual_R_p50":float(np.quantile(x[:,0],.50)),"mc_annual_R_p95":float(np.quantile(x[:,0],.95)),
            "mc_maxdd_R_p50":float(np.quantile(x[:,1],.50)),"mc_maxdd_R_p95":float(np.quantile(x[:,1],.95))}


def main():
    all_trades=[]; summary=[]
    for asset in ASSETS:
        m5,h1,d1,prov=v28.load_asset_v28(asset)
        c=v27.candidates(m5,h1,d1)
        models=[v27.simulate_fixed(c,m5,2.0),simulate_partial2_trail(c,m5,h1),simulate_full_trail_after3(c,m5,h1)]
        for tr in models:
            if tr.empty: continue
            tr["asset"]=asset; tr["split"]=tr.entry_time.apply(split)
            all_trades.append(tr)
            tr.to_csv(OUT/f"{asset}_{tr.exit_model.iloc[0]}.csv",index=False)
            for sp in ["DEV","VAL","OOS"]:
                g=tr[tr.split==sp]
                if g.empty: continue
                row={"asset":asset,"exit_model":tr.exit_model.iloc[0],"split":sp,**metrics(g)}
                if sp=="OOS": row.update(mc_trade_sequence(g.net_R.astype(float)))
                summary.append(row)

    # Metals cluster: chronological one-at-a-time, like v3.0.
    combo=pd.concat(all_trades,ignore_index=True)
    for model,g in combo.groupby("exit_model"):
        g=g[g.split=="OOS"].sort_values("entry_time")
        acc=[]; active_until=None
        for _,q in g.iterrows():
            et=pd.Timestamp(q.entry_time)
            if active_until is not None and et<=active_until: continue
            acc.append(q); active_until=pd.Timestamp(q.exit_time)
        a=pd.DataFrame(acc)
        if not a.empty:
            row={"asset":"METALS_CLUSTER","exit_model":model,"split":"OOS","gold_trades":int((a.asset=="GOLD").sum()),"silver_trades":int((a.asset=="SILVER").sum()),**metrics(a)}
            row.update(mc_trade_sequence(a.net_R.astype(float)))
            summary.append(row)
            a.to_csv(OUT/f"METALS_CLUSTER_{model}.csv",index=False)

    s=pd.DataFrame(summary)
    s.to_csv(OUT/"summary.csv",index=False)
    print(s.to_string(index=False))
    print("\nLIMITATIONS: Gold/Silver selected after inspected OOS; synthetic 0.05R cost; Dukascopy proxy; H1 structural trail uses completed bars only; MC is trade-sequence bootstrap, not exact prop intraday-equity simulation.")

if __name__=="__main__": main()
