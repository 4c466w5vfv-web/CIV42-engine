"""Edge-concentration stress + Monte Carlo v3.0.

Uses v2.8 Session/FVG/H1/M5 artifacts from run 33392019874.
Focuses only on strongest observed OOS candidates: GOLD and SILVER.
Compares GOLD only, SILVER only, and a metals-cluster portfolio.
Research only; not live-profit proof. Selection uses already-inspected OOS and is therefore not pristine validation.
"""
from __future__ import annotations

import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ARTIFACT_ROOT", "artifacts_v28"))
OUT = Path("research/results/edge_concentration_mc_v30")
OUT.mkdir(parents=True, exist_ok=True)
ACCOUNT = 100_000.0
MC_PATHS = 10_000
MC_DAYS = 252
BLOCK = 5
SEED = 42
DAILY_STOP_PCT = 0.02
STATIC_DD_PCT = 0.10
RISK_LEVELS = [0.005, 0.0075, 0.01]
STRESS = {"BASE":0.00,"EXEC_2X":0.05,"EXEC_3X":0.10,"EXEC_5X":0.20}
PORTFOLIOS = {
    "GOLD_ONLY":["GOLD"],
    "SILVER_ONLY":["SILVER"],
    "METALS_CLUSTER":["GOLD","SILVER"],
}


def load_asset(asset: str) -> pd.DataFrame:
    hits=list(ROOT.rglob(f"{asset}/trades_TP1_2R_PLUS_H1_RUNNER.csv"))
    if not hits:
        raise RuntimeError(f"missing {asset} artifact")
    d=pd.read_csv(hits[0])
    d["asset"]=asset
    d["entry_time"]=pd.to_datetime(d.entry_time,utc=True)
    d["exit_time"]=pd.to_datetime(d.exit_time,utc=True)
    # inspected OOS only
    d=d[d.entry_time.dt.year>=2025].copy()
    return d.sort_values("entry_time")


def select_portfolio(assets: list[str]) -> pd.DataFrame:
    d=pd.concat([load_asset(a) for a in assets],ignore_index=True).sort_values("entry_time")
    # Gold+Silver are one correlated metals thesis: at most one new 1R initial-risk trade active.
    accepted=[]; active_until=None; realized_by_day={}
    for _,q in d.iterrows():
        et=pd.Timestamp(q.entry_time); day=et.date()
        if realized_by_day.get(day,0.0) <= -2.0:
            continue
        if active_until is not None and active_until > et:
            continue
        accepted.append(q.to_dict())
        active_until=pd.Timestamp(q.exit_time)
        eday=active_until.date()
        realized_by_day[eday]=realized_by_day.get(eday,0.0)+float(q.net_R)
    a=pd.DataFrame(accepted)
    if a.empty: raise RuntimeError("no accepted trades")
    a["exit_day"]=pd.to_datetime(a.exit_time,utc=True).dt.floor("D")
    daily=a.groupby("exit_day").agg(net_R=("net_R","sum"),trades=("net_R","size")).reset_index()
    idx=pd.date_range(daily.exit_day.min(),daily.exit_day.max(),freq="B",tz="UTC")
    daily=daily.set_index("exit_day").reindex(idx,fill_value=0).rename_axis("day").reset_index()
    return a,daily


def block_bootstrap(arr,n,rng):
    out=[]; L=len(arr)
    while len(out)<n:
        s=int(rng.integers(0,max(1,L-BLOCK+1)))
        out.extend(arr[s:s+BLOCK].tolist())
    return np.asarray(out[:n],float)


def payout(path_pct, split, interval):
    bucket=[]; paid=0.0
    for i,r in enumerate(path_pct,1):
        bucket.append(float(r))
        if i%interval: continue
        total=sum(bucket); best=max([x for x in bucket if x>0],default=0.0)
        if total>0 and best <= 0.25*total + 1e-12:
            paid += total*ACCOUNT*split
            bucket=[]
    return paid


def hist_stats(x):
    x=np.asarray(x,float); eq=np.cumsum(x); peak=np.maximum.accumulate(np.r_[0.0,eq])[:-1]; dd=eq-peak
    ann=x.mean()*252; sd=x.std(ddof=1)*math.sqrt(252) if len(x)>1 else np.nan
    neg=x[x<0]; dsd=neg.std(ddof=1)*math.sqrt(252) if len(neg)>1 else np.nan
    mdd=abs(dd.min()) if len(dd) else np.nan
    return dict(annual_return_pct=100*ann,sharpe=ann/sd if sd>0 else np.nan,sortino=ann/dsd if dsd and dsd>0 else np.nan,max_dd_pct=100*mdd,calmar=ann/mdd if mdd>0 else np.nan)


def simulate(daily,risk_pct,extra_cost_R):
    hist=(daily.net_R.to_numpy(float)-extra_cost_R*daily.trades.to_numpy(float))*risk_pct
    hs=hist_stats(hist)
    rng=np.random.default_rng(SEED); rows=[]
    for _ in range(MC_PATHS):
        p=block_bootstrap(hist,MC_DAYS,rng)
        eq=np.cumsum(p); peak=np.maximum.accumulate(np.r_[0.0,eq])[:-1]; dd=eq-peak
        rows.append({
            "annual_return":float(p.sum()),
            "max_dd":float(abs(dd.min())),
            "daily_2pct_breach":bool((p < -DAILY_STOP_PCT-1e-12).any()),
            "static_10pct_breach":bool((abs(dd.min()) >= STATIC_DD_PCT)),
            "payout80_14d":payout(p,0.80,14),
            "payout100_30d":payout(p,1.00,30),
        })
    m=pd.DataFrame(rows)
    q=lambda c,z:float(m[c].quantile(z))
    mc={
        "annual_pct_p05":100*q("annual_return",.05),"annual_pct_p50":100*q("annual_return",.50),"annual_pct_p95":100*q("annual_return",.95),
        "maxdd_pct_p50":100*q("max_dd",.50),"maxdd_pct_p95":100*q("max_dd",.95),
        "daily2_breach_prob":float(m.daily_2pct_breach.mean()),"static10_breach_prob":float(m.static_10pct_breach.mean()),
        "payout80_p05":q("payout80_14d",.05),"payout80_p50":q("payout80_14d",.50),"payout80_p95":q("payout80_14d",.95),
        "payout100_p05":q("payout100_30d",.05),"payout100_p50":q("payout100_30d",.50),"payout100_p95":q("payout100_30d",.95),
    }
    return hs,mc


def main():
    rows=[]; diag=[]
    for pname,assets in PORTFOLIOS.items():
        a,daily=select_portfolio(assets)
        a.to_csv(OUT/f"accepted_{pname}.csv",index=False)
        for asset,g in a.groupby("asset"):
            r=g.net_R.astype(float); wins=r[r>0]; losses=r[r<=0]
            diag.append({"portfolio":pname,"asset":asset,"trades":len(r),"expectancy_R":r.mean(),"PF":wins.sum()/abs(losses.sum()) if losses.sum()<0 else np.nan})
        for risk in RISK_LEVELS:
            for sname,cost in STRESS.items():
                hs,mc=simulate(daily,risk,cost)
                rows.append({"portfolio":pname,"assets":"+".join(assets),"risk_pct":100*risk,"scenario":sname,"extra_cost_R":cost,"accepted_trades":len(a),**hs,**mc})
    pd.DataFrame(rows).to_csv(OUT/"summary.csv",index=False)
    pd.DataFrame(diag).to_csv(OUT/"diagnostics.csv",index=False)
    meta={"input_run":33392019874,"account":ACCOUNT,"portfolios":PORTFOLIOS,"risk_levels":RISK_LEVELS,"stress":STRESS,"mc_paths":MC_PATHS,"mc_days":MC_DAYS,"block":BLOCK,"daily_stop_pct":DAILY_STOP_PCT,"static_dd_pct":STATIC_DD_PCT,"limitations":["Gold/Silver were selected after inspecting aggregate OOS; this is concentration stress, not pristine validation","daily prop loss is equity-based; source artifacts do not contain full intraday portfolio equity","Gold and Silver are treated as one correlated metals risk cluster","payout model is simplified","costs are synthetic R stresses, not measured broker fills"]}
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2))
    print(pd.DataFrame(rows).to_string(index=False))

if __name__=="__main__": main()
