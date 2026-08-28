"""Capital Survival Layer v1.7.
Consumes v1.6 trade CSVs and evaluates a KRW 100,000,000 account.
No strategy re-optimization. Risk fractions: 0.25%, 0.50%, 1.00% of current equity/trade.
"""
from pathlib import Path
import numpy as np
import pandas as pd

INITIAL=100_000_000.0
RISKS=[0.0025,0.005,0.01]
MC_N=10_000
RNG=np.random.default_rng(42)


def max_dd(eq):
    a=np.r_[INITIAL,np.asarray(eq,float)]
    peak=np.maximum.accumulate(a)
    dd=(peak-a)/peak
    i=int(np.argmax(dd))
    return float(dd[i]), float((peak[i]-a[i]))


def simulate(rs,risk):
    e=INITIAL; curve=[]
    for r in rs:
        e*=max(0.0,1.0+risk*float(r))
        curve.append(e)
    dd,ddkrw=max_dd(curve)
    return e,dd,ddkrw,min([INITIAL]+curve)


def monte_carlo(rs,risk):
    rs=np.asarray(rs,float); n=len(rs)
    if n==0:return {}
    finals=[]; dds=[]; mins=[]
    for _ in range(MC_N):
        s=RNG.choice(rs,size=n,replace=True)
        f,d,_,m=simulate(s,risk); finals.append(f); dds.append(d); mins.append(m)
    finals=np.asarray(finals); dds=np.asarray(dds); mins=np.asarray(mins)
    return {
      'mc_final_p05':np.quantile(finals,.05),'mc_final_median':np.median(finals),'mc_final_p95':np.quantile(finals,.95),
      'mc_dd_p95':np.quantile(dds,.95),'mc_dd_p99':np.quantile(dds,.99),
      'p_below_80m':np.mean(mins<=80_000_000),'p_below_70m':np.mean(mins<=70_000_000),'p_below_50m':np.mean(mins<=50_000_000)}


def main():
    roots=[Path('research/results/mtf_v16'),Path('research/results/mtf_v16_parallel')]
    files=[]
    for root in roots:
        if root.exists(): files += list(root.rglob('trades*.csv'))
    rows=[]
    for f in files:
        try:d=pd.read_csv(f)
        except:continue
        rcol=next((c for c in ['R','r','net_R','net_r','result_R'] if c in d.columns),None)
        if not rcol:continue
        asset=d['asset'].iloc[0] if 'asset' in d and len(d) else f.parent.name
        tp=d['tp_R'].iloc[0] if 'tp_R' in d and len(d) else f.stem
        rs=pd.to_numeric(d[rcol],errors='coerce').dropna().values
        for risk in RISKS:
            final,dd,ddkrw,min_eq=simulate(rs,risk); mc=monte_carlo(rs,risk)
            rows.append({'asset':asset,'tp':tp,'risk_pct':risk*100,'trades':len(rs),'final_krw':final,'return_pct':(final/INITIAL-1)*100,'max_dd_pct':dd*100,'max_dd_krw':ddkrw,'min_equity_krw':min_eq,**mc})
    out=Path('research/results/capital_survival_v17');out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out/'capital_survival_100m.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
if __name__=='__main__':main()
