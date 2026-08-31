"""Prop portfolio stress + Monte Carlo v2.9.

Inputs: v2.8 Session/FVG/H1/M5 artifacts downloaded from run 33392019874.
Oil is excluded (WTI/BRENT). Strategy exit is frozen to TP1_2R_PLUS_H1_RUNNER.
Research only. Not a claim of live prop profitability.
"""
from __future__ import annotations

import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("ARTIFACT_ROOT", "artifacts_v28"))
OUT = Path("research/results/prop_portfolio_mc_v29")
OUT.mkdir(parents=True, exist_ok=True)

EXCLUDE = {"WTI", "BRENT"}
ACCOUNT = 100_000.0
RISK_PCT = 0.01
R_DOLLARS = ACCOUNT * RISK_PCT
DAILY_STOP_R = -2.0                # user's stricter daily stop: -2%
MAX_OPEN_R = 2.0                  # hard daily-loss-aware portfolio ceiling, stricter than old 3R
MAX_STATIC_DD_R = 10.0            # 10% static max loss proxy
MC_PATHS = 10_000
MC_DAYS = 252
BLOCK = 5
SEED = 42

# baseline v2.8 already subtracts 0.05R/trade synthetic cost.
STRESS = {
    "BASE": 0.00,
    "EXEC_2X": 0.05,
    "EXEC_3X": 0.10,
    "EXEC_5X": 0.20,
}


def find_trades():
    rows=[]
    for p in ROOT.rglob("trades_TP1_2R_PLUS_H1_RUNNER.csv"):
        asset = p.parent.name.upper()
        if asset in EXCLUDE:
            continue
        d = pd.read_csv(p)
        if d.empty:
            continue
        d["asset"] = asset
        d["entry_time"] = pd.to_datetime(d["entry_time"], utc=True)
        d["exit_time"] = pd.to_datetime(d["exit_time"], utc=True)
        # forward/OOS only
        d = d[d.entry_time.dt.year >= 2025].copy()
        rows.append(d)
    if not rows:
        raise RuntimeError(f"no trade files under {ROOT}")
    return pd.concat(rows, ignore_index=True).sort_values("entry_time")


def build_portfolio(tr: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Chronological acceptance with <=2R simultaneous initial risk and -2R daily realized stop.

    Each trade is 1R initial risk. Same-day stop uses already realized exits only; open-trade intraday
    equity path is unavailable in v2.8, so this is an approximation, not an exact prop daily-equity test.
    """
    active=[]
    accepted=[]
    realized_by_day={}
    for _, q in tr.iterrows():
        et=pd.Timestamp(q.entry_time)
        active=[x for x in active if x > et]
        day=et.date()
        if realized_by_day.get(day,0.0) <= DAILY_STOP_R:
            continue
        if len(active) >= int(MAX_OPEN_R):
            continue
        accepted.append(q.to_dict())
        xt=pd.Timestamp(q.exit_time)
        active.append(xt)
        eday=xt.date()
        realized_by_day[eday]=realized_by_day.get(eday,0.0)+float(q.net_R)
    a=pd.DataFrame(accepted)
    if a.empty: raise RuntimeError("portfolio filter left no trades")
    a["exit_day"]=pd.to_datetime(a.exit_time, utc=True).dt.floor("D")
    daily=a.groupby("exit_day").agg(net_R=("net_R","sum"),trades=("net_R","size")).reset_index()
    # complete business-day grid to preserve inactivity in annualization
    idx=pd.date_range(daily.exit_day.min(),daily.exit_day.max(),freq="B",tz="UTC")
    daily=daily.set_index("exit_day").reindex(idx,fill_value=0).rename_axis("day").reset_index()
    return a,daily


def stats(x: pd.Series):
    x=x.astype(float)
    eq=x.cumsum(); dd=eq-eq.cummax()
    neg=x[x<0]
    ann_mean=x.mean()*252
    ann_sd=x.std(ddof=1)*math.sqrt(252)
    sharpe=ann_mean/ann_sd if ann_sd>0 else np.nan
    downside=neg.std(ddof=1)*math.sqrt(252) if len(neg)>1 else np.nan
    sortino=ann_mean/downside if downside and downside>0 else np.nan
    mdd=abs(dd.min()) if len(dd) else np.nan
    calmar=ann_mean/mdd if mdd and mdd>0 else np.nan
    return {"annual_R":ann_mean,"sharpe":sharpe,"sortino":sortino,"max_dd_R":mdd,"calmar":calmar}


def block_bootstrap(arr: np.ndarray, n: int, rng: np.random.Generator):
    out=[]
    L=len(arr)
    while len(out)<n:
        s=int(rng.integers(0,max(1,L-BLOCK+1)))
        out.extend(arr[s:s+BLOCK].tolist())
    return np.array(out[:n],dtype=float)


def payout(path_R: np.ndarray, split: float, interval: int):
    """Simplified funded payout model with 25% best-day consistency.

    Profit accrues until payout checkpoint. Eligibility requires positive cumulative profit and
    best positive day <=25% of cumulative profit. On payout, eligible profit is paid at split and
    the profit bucket resets. This is an approximation of a prop firm's live-account mechanics.
    """
    bucket=[]; paid=0.0
    for i,r in enumerate(path_R,1):
        bucket.append(float(r))
        if i%interval:
            continue
        total=sum(bucket)
        best=max([z for z in bucket if z>0], default=0.0)
        if total>0 and best <= 0.25*total + 1e-12:
            paid += total*R_DOLLARS*split
            bucket=[]
    return paid


def simulate(daily: pd.DataFrame, extra_cost_R: float):
    # apply extra execution cost per accepted trade-day observation
    hist = daily.net_R.to_numpy(float) - extra_cost_R*daily.trades.to_numpy(float)
    hist_stats=stats(pd.Series(hist))
    rng=np.random.default_rng(SEED)
    rec=[]
    for _ in range(MC_PATHS):
        p=block_bootstrap(hist,MC_DAYS,rng)
        eq=np.cumsum(p); peak=np.maximum.accumulate(np.r_[0.0,eq])[:-1]
        dd=eq-peak
        maxdd=float(abs(dd.min()))
        daily_breach=bool((p < -2.0 - 1e-12).any())
        total_breach=bool(maxdd >= MAX_STATIC_DD_R)
        rec.append({
            "annual_R":float(p.sum()),
            "max_dd_R":maxdd,
            "daily_2pct_breach":daily_breach,
            "static_10pct_breach":total_breach,
            "payout_80_14d":payout(p,0.80,14),
            "payout_100_30d":payout(p,1.00,30),
        })
    m=pd.DataFrame(rec)
    q=lambda c,z: float(m[c].quantile(z))
    return hist_stats,{
        "annual_R_p05":q("annual_R",.05),"annual_R_p50":q("annual_R",.50),"annual_R_p95":q("annual_R",.95),
        "max_dd_R_p50":q("max_dd_R",.50),"max_dd_R_p95":q("max_dd_R",.95),
        "daily_2pct_breach_prob":float(m.daily_2pct_breach.mean()),
        "static_10pct_breach_prob":float(m.static_10pct_breach.mean()),
        "payout80_14d_p05":q("payout_80_14d",.05),"payout80_14d_p50":q("payout_80_14d",.50),"payout80_14d_p95":q("payout_80_14d",.95),
        "payout100_30d_p05":q("payout_100_30d",.05),"payout100_30d_p50":q("payout_100_30d",.50),"payout100_30d_p95":q("payout_100_30d",.95),
    },m


def main():
    tr=find_trades(); accepted,daily=build_portfolio(tr)
    assets=sorted(accepted.asset.unique().tolist())
    # per-asset OOS diagnostics before portfolio filters
    per=[]
    for asset,g in tr.groupby("asset"):
        r=g.net_R.astype(float); wins=r[r>0]; losses=r[r<=0]
        s=stats(r.reset_index(drop=True))
        per.append({"asset":asset,"trades":len(r),"expectancy_R":r.mean(),"PF":wins.sum()/abs(losses.sum()) if losses.sum()<0 else np.nan,
                    "trade_sharpe":r.mean()/r.std(ddof=1)*math.sqrt(len(r)) if r.std(ddof=1)>0 else np.nan})
    pd.DataFrame(per).sort_values("expectancy_R",ascending=False).to_csv(OUT/"per_asset_oos.csv",index=False)
    accepted.to_csv(OUT/"accepted_portfolio_trades.csv",index=False)
    daily.to_csv(OUT/"historical_daily_portfolio.csv",index=False)

    summary=[]
    for name,cost in STRESS.items():
        hs,mc,paths=simulate(daily,cost)
        row={"scenario":name,"extra_cost_R_per_trade":cost,**hs,**mc}
        summary.append(row)
        paths.to_csv(OUT/f"mc_{name}.csv",index=False)
    sdf=pd.DataFrame(summary)
    sdf.to_csv(OUT/"summary.csv",index=False)
    meta={
        "input_run":33392019874,"assets_used":assets,"excluded":["WTI","BRENT"],"exit_model":"TP1_2R_PLUS_H1_RUNNER",
        "account":ACCOUNT,"risk_per_trade_pct":RISK_PCT,"R_dollars":R_DOLLARS,"daily_stop_R":DAILY_STOP_R,
        "max_simultaneous_initial_open_R":MAX_OPEN_R,"static_dd_proxy_R":MAX_STATIC_DD_R,"mc_paths":MC_PATHS,"mc_days":MC_DAYS,
        "block_days":BLOCK,"limitations":[
            "v2.8 signal research is not live execution proof",
            "daily prop loss is equity-based but source artifacts lack intraday portfolio equity; daily stop is realized-PnL approximation",
            "payout consistency implementation is simplified",
            "synthetic costs are expressed in R, not measured FXIFY fills",
            "OOS has already been inspected; this is not a pristine untouched holdout"
        ]
    }
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2))
    print("ASSETS",assets)
    print("TRADES raw",len(tr),"accepted",len(accepted),"days",len(daily))
    print(sdf.to_string(index=False))

if __name__=="__main__": main()
