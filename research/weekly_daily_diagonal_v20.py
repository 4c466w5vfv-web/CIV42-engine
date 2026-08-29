"""v2.0 Weekly/Daily Diagonal + Volume-Level Break/Retest strategy.

No FVG/OB is used for primary setup generation. No ZigZag.
Primary evidence:
1) weekly structural regime,
2) daily causal diagonal structure,
3) high-volume confirmed daily swing highs/lows as horizontal S/R,
4) break -> later retest -> rejection/reclaim.

15m FVG is deliberately NOT a primary filter in this version. It is recorded later
as an optional confirmation feature so its incremental value can be ablation-tested
without contaminating the core price/volume edge.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal
import mtf_exit_engine_v17 as exits

OUT=Path('research/results/weekly_daily_diagonal_v20'); OUT.mkdir(parents=True,exist_ok=True)
PIVOT_L=PIVOT_R=2
VOL_N=20
VOL_SPIKE=1.5
ATR_N=14
RETEST_DAYS=10
RETEST_TOL_ATR=.20


def resample_ohlcv(x, rule):
    return x.resample(rule,label='right',closed='right').agg({'open':'first','high':'max','low':'min','close':'last','volume':'sum'}).dropna()


def enrich(x):
    z=x.copy(); prev=z.close.shift(1)
    tr=pd.concat([(z.high-z.low).abs(),(z.high-prev).abs(),(z.low-prev).abs()],axis=1).max(axis=1)
    z['atr']=tr.shift(1).rolling(ATR_N,min_periods=ATR_N).mean()
    z['vol_base']=z.volume.shift(1).rolling(VOL_N,min_periods=VOL_N).mean()
    z['vol_ratio']=z.volume/z.vol_base.replace(0,np.nan)
    return z


def pivots(x):
    rows=[]; h=x.high.to_numpy(float); l=x.low.to_numpy(float); ts=x.index
    for i in range(PIVOT_L,len(x)-PIVOT_R):
        wh=h[i-PIVOT_L:i+PIVOT_R+1]; wl=l[i-PIVOT_L:i+PIVOT_R+1]; ct=ts[i+PIVOT_R]
        if h[i]==np.nanmax(wh) and (wh==h[i]).sum()==1: rows.append({'confirm_time':ct,'pivot_time':ts[i],'kind':'H','price':h[i],'idx':i,'vol_ratio':x.vol_ratio.iloc[i]})
        if l[i]==np.nanmin(wl) and (wl==l[i]).sum()==1: rows.append({'confirm_time':ct,'pivot_time':ts[i],'kind':'L','price':l[i],'idx':i,'vol_ratio':x.vol_ratio.iloc[i]})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=['confirm_time','pivot_time','kind','price','idx','vol_ratio'])


def weekly_regime(d):
    w=enrich(resample_ohlcv(d,'7D')); p=pivots(w); out=[]
    for t in w.index:
        k=p[p.confirm_time<=t]; hs=k[k.kind=='H'].tail(2); ls=k[k.kind=='L'].tail(2); reg=0
        if len(hs)==2 and len(ls)==2:
            if hs.price.iloc[-1]>hs.price.iloc[-2] and ls.price.iloc[-1]>ls.price.iloc[-2]: reg=1
            elif hs.price.iloc[-1]<hs.price.iloc[-2] and ls.price.iloc[-1]<ls.price.iloc[-2]: reg=-1
        out.append((t,reg))
    return pd.Series(dict(out),name='weekly_regime')


def line(a,b,t):
    dt=(b.pivot_time-a.pivot_time).total_seconds()
    return np.nan if dt<=0 else float(a.price)+(float(b.price)-float(a.price))*((t-a.pivot_time).total_seconds()/dt)


def build_candidates(daily):
    d=enrich(daily); p=pivots(d); wr=weekly_regime(daily)
    reg=wr.reindex(d.index,method='ffill').fillna(0)
    armed=[]; used=set(); out=[]
    for i,(t,r) in enumerate(d.iterrows()):
        known=p[p.confirm_time<=t]
        # Retest only on a later daily bar; primary stop is the broken high-volume pivot level's structural anchor.
        nxt=[]
        for a in armed:
            if i>a['expiry']: continue
            tol=RETEST_TOL_ATR*float(r.atr) if np.isfinite(r.atr) else 0
            lv=a['level']; side=a['side']
            touched=(r.low<=lv+tol) if side==1 else (r.high>=lv-tol)
            confirmed=touched and ((r.close>lv) if side==1 else (r.close<lv))
            if confirmed:
                entry=float(r.close); stop=float(a['stop']); risk=abs(entry-stop)
                if risk>0 and ((side==1 and stop<entry) or (side==-1 and stop>entry)):
                    out.append({'entry_time':t,'signal_available_time':t,'side':side,'entry':entry,'stop':stop,'risk':risk,
                                'setup':a['setup'],'break_time':a['break_time'],'sr_level':lv,'sr_pivot_time':a['pivot_time'],
                                'sr_vol_ratio':a['vol_ratio'],'weekly_regime':a['regime'],'diagonal_level':a['diag']})
                    used.add(a['key'])
                continue
            nxt.append(a)
        armed=nxt
        if int(reg.loc[t])==0 or len(known)<4 or not np.isfinite(r.atr): continue
        side=int(reg.loc[t])
        # High-volume swing S/R: the pivot itself must have exceptional volume relative to prior daily bars.
        sr=known[(known.kind==('H' if side==1 else 'L')) & (known.vol_ratio>=VOL_SPIKE)].tail(1)
        if sr.empty: continue
        s=sr.iloc[-1]; level=float(s.price)
        # Daily diagonal context from last two lows in long regime / highs in short regime.
        q=known[known.kind==('L' if side==1 else 'H')].tail(2)
        if len(q)<2: continue
        a0,a1=q.iloc[0],q.iloc[1]
        if side==1 and not (a1.price>a0.price): continue
        if side==-1 and not (a1.price<a0.price): continue
        diag=line(a0,a1,t)
        if not np.isfinite(diag): continue
        key=(side,s.pivot_time,a0.pivot_time,a1.pivot_time)
        if key in used or any(z['key']==key for z in armed): continue
        broke=(r.close>level and r.open<=level) if side==1 else (r.close<level and r.open>=level)
        if broke:
            # Explicit structural invalidation: latest diagonal anchor swing.
            stop=float(a1.price)
            armed.append({'key':key,'side':side,'level':level,'stop':stop,'pivot_time':s.pivot_time,
                          'vol_ratio':float(s.vol_ratio),'regime':side,'diag':diag,'break_time':t,
                          'setup':'HV_SR_BREAK_RETEST_WITH_DIAGONAL','expiry':i+RETEST_DAYS})
    c=pd.DataFrame(out)
    if not c.empty: c=c.sort_values('entry_time').drop_duplicates(['entry_time','setup'])
    return c


def main():
    name=base.os.environ['ASSET']
    m15,m1,daily,prov=causal.load_asset_causal(name)
    c=build_candidates(daily)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True); c.to_csv(out/'candidates.csv',index=False)
    models=[('FIXED_2R',lambda:exits.fixed_tp(c,m1,2.0)),('FIXED_3R',lambda:exits.fixed_tp(c,m1,3.0)),
            ('FIXED_5R',lambda:exits.fixed_tp(c,m1,5.0)),('PURE_SWING',lambda:exits.trailing(c,m1,'PURE_SWING')),
            ('ADAPTIVE_TRAIL',lambda:exits.trailing(c,m1,'ADAPTIVE_TRAIL')),
            ('HALF_2R_BE_RUNNER',lambda:exits.half_profit_be_runner(c,m1,2.0,.5))]
    rows=[]
    for label,fn in models:
        tr=fn(); tr.to_csv(out/f'trades_{label}.csv',index=False)
        if tr.empty: rows.append({'asset':name,'exit_model':label,'split':'ALL','trades':0}); continue
        tr['split']=tr.entry_time.map(base.split)
        for sp,g in [('ALL',tr),*list(tr.groupby('split'))]:
            m=exits.metrics_ext(g); m.update({'asset':name,'exit_model':label,'split':sp}); rows.append(m)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'meta.json').write_text(json.dumps({'asset':name,'provenance':prov,'candidate_count':len(c),
      'primary_setup':'weekly structure + daily diagonal + high-volume daily swing S/R + break/retest',
      'zigzag':False,'fvg_ob_primary_filter':False,'fvg_15m_policy':'confirmation feature only; not used in v2.0 entry selection',
      'volume_spike':VOL_SPIKE,'retest_days':RETEST_DAYS,'stop':'latest confirmed daily diagonal anchor swing',
      'selection_rule':'DEV/VAL only; OOS verification only'},indent=2,default=str))
    print(pd.DataFrame(rows).to_markdown(index=False))

if __name__=='__main__': main()
