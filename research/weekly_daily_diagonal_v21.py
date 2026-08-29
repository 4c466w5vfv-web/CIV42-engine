"""v2.1 actual diagonal interaction + LTF retest-stop experiment.

Purpose: fix the two structural defects identified in v2.0 without tuning on OOS.
1) Diagonal must be an actual price interaction, not metadata only.
2) Daily break/retest defines setup; 15m causal reclaim defines entry and local risk.

Core v2.0 parameters remain frozen. FVG is not required and is not tuned here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal
import mtf_exit_engine_v17 as exits
import weekly_daily_diagonal_v20 as v20

OUT=Path('research/results/weekly_daily_diagonal_v21'); OUT.mkdir(parents=True,exist_ok=True)
LTF_LOOKBACK=8
LTF_CONFIRM_MINUTES=24*60
DIAG_TOL_ATR=.25


def actual_diagonal_candidates(daily):
    """Same weekly/HV-SR skeleton as v2.0, but require the break bar or later retest
    to interact with the causal diagonal. Candidate is a daily setup, not final entry.
    """
    d=v20.enrich(daily); p=v20.pivots(d); wr=v20.weekly_regime(daily)
    reg=wr.reindex(d.index,method='ffill').fillna(0)
    armed=[]; used=set(); out=[]
    for i,(t,r) in enumerate(d.iterrows()):
        known=p[p.confirm_time<=t]
        nxt=[]
        for a in armed:
            if i>a['expiry']: continue
            tol=v20.RETEST_TOL_ATR*float(r.atr) if np.isfinite(r.atr) else 0.0
            dtol=DIAG_TOL_ATR*float(r.atr) if np.isfinite(r.atr) else 0.0
            lv=a['level']; side=a['side']; diag_now=v20.line(a['a0'],a['a1'],t)
            touched=(r.low<=lv+tol) if side==1 else (r.high>=lv-tol)
            reclaimed=touched and ((r.close>lv) if side==1 else (r.close<lv))
            diag_touch=np.isfinite(diag_now) and (r.low-dtol<=diag_now<=r.high+dtol)
            if reclaimed and (a['break_diag_touch'] or diag_touch):
                out.append({'setup_time':t,'side':side,'sr_level':lv,'sr_pivot_time':a['pivot_time'],
                            'sr_vol_ratio':a['vol_ratio'],'weekly_regime':a['regime'],
                            'diagonal_level':float(diag_now),'diagonal_break_touch':a['break_diag_touch'],
                            'diagonal_retest_touch':bool(diag_touch),'daily_anchor_stop':float(a['a1'].price),
                            'setup':'ACTUAL_DIAGONAL_HVSR_BREAK_RETEST'})
                used.add(a['key']); continue
            nxt.append(a)
        armed=nxt
        if int(reg.loc[t])==0 or len(known)<4 or not np.isfinite(r.atr): continue
        side=int(reg.loc[t]); sr=known[(known.kind==('H' if side==1 else 'L')) & (known.vol_ratio>=v20.VOL_SPIKE)].tail(1)
        if sr.empty: continue
        s=sr.iloc[-1]; level=float(s.price)
        q=known[known.kind==('L' if side==1 else 'H')].tail(2)
        if len(q)<2: continue
        a0,a1=q.iloc[0],q.iloc[1]
        if side==1 and not (a1.price>a0.price): continue
        if side==-1 and not (a1.price<a0.price): continue
        diag=v20.line(a0,a1,t)
        if not np.isfinite(diag): continue
        key=(side,s.pivot_time,a0.pivot_time,a1.pivot_time)
        if key in used or any(z['key']==key for z in armed): continue
        broke=(r.close>level and r.open<=level) if side==1 else (r.close<level and r.open>=level)
        if broke:
            dtol=DIAG_TOL_ATR*float(r.atr)
            diag_touch=(r.low-dtol<=diag<=r.high+dtol)
            armed.append({'key':key,'side':side,'level':level,'pivot_time':s.pivot_time,'vol_ratio':float(s.vol_ratio),
                          'regime':side,'a0':a0,'a1':a1,'break_diag_touch':bool(diag_touch),
                          'break_time':t,'expiry':i+v20.RETEST_DAYS})
    return pd.DataFrame(out)


def ltf_entries(setups,m15):
    """After the daily setup is available, wait for a later 15m reclaim/break and use
    prior 15m structure as the stop. Current bar never defines its own stop.
    """
    if setups.empty: return pd.DataFrame()
    x=m15.copy(); x['prev_hi']=x.high.shift(1).rolling(LTF_LOOKBACK,min_periods=LTF_LOOKBACK).max(); x['prev_lo']=x.low.shift(1).rolling(LTF_LOOKBACK,min_periods=LTF_LOOKBACK).min()
    rows=[]
    for _,s in setups.iterrows():
        st=pd.Timestamp(s.setup_time); side=int(s.side)
        w=x[(x.index>st)&(x.index<=st+pd.Timedelta(minutes=LTF_CONFIRM_MINUTES))]
        for t,r in w.iterrows():
            if not np.isfinite(r.prev_hi) or not np.isfinite(r.prev_lo): continue
            confirm=(r.close>r.prev_hi and r.close>r.open) if side==1 else (r.close<r.prev_lo and r.close<r.open)
            if not confirm: continue
            entry=float(r.close); stop=float(r.prev_lo if side==1 else r.prev_hi)
            if (side==1 and stop>=entry) or (side==-1 and stop<=entry): continue
            risk=abs(entry-stop)
            if risk<=0: continue
            q=s.to_dict(); q.update({'entry_time':t,'signal_available_time':t,'entry':entry,'stop':stop,'risk':risk,
                                      'stop_model':'LTF_PRIOR_15M_STRUCTURE','ltf_lookback_bars':LTF_LOOKBACK})
            rows.append(q); break
    c=pd.DataFrame(rows)
    if not c.empty: c=c.sort_values('entry_time').drop_duplicates(['entry_time','setup'])
    return c


def main():
    name=base.os.environ['ASSET']; m15,m1,daily,prov=causal.load_asset_causal(name)
    setups=actual_diagonal_candidates(daily); c=ltf_entries(setups,m15)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    setups.to_csv(out/'daily_setups.csv',index=False); c.to_csv(out/'candidates.csv',index=False)
    models=[('FIXED_2R',lambda:exits.fixed_tp(c,m1,2.0)),('FIXED_3R',lambda:exits.fixed_tp(c,m1,3.0)),('FIXED_5R',lambda:exits.fixed_tp(c,m1,5.0)),
            ('PURE_SWING',lambda:exits.trailing(c,m1,'PURE_SWING')),('ADAPTIVE_TRAIL',lambda:exits.trailing(c,m1,'ADAPTIVE_TRAIL')),
            ('HALF_2R_BE_RUNNER',lambda:exits.half_profit_be_runner(c,m1,2.0,.5))]
    rows=[]
    for label,fn in models:
        tr=fn(); tr.to_csv(out/f'trades_{label}.csv',index=False)
        if tr.empty: rows.append({'asset':name,'exit_model':label,'split':'ALL','trades':0}); continue
        tr['split']=tr.entry_time.map(base.split)
        for sp,g in [('ALL',tr),*list(tr.groupby('split'))]:
            m=exits.metrics_ext(g); m.update({'asset':name,'exit_model':label,'split':sp}); rows.append(m)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'meta.json').write_text(json.dumps({'version':'2.1','asset':name,'provenance':prov,'daily_setups':len(setups),'candidate_count':len(c),
      'frozen_from_v20':{'pivot_lr':v20.PIVOT_L,'volume_spike':v20.VOL_SPIKE,'retest_days':v20.RETEST_DAYS,'retest_tol_atr':v20.RETEST_TOL_ATR},
      'new_test_only':{'actual_diagonal_interaction_tol_atr':DIAG_TOL_ATR,'ltf_confirm':'later 15m structure break','ltf_stop':'prior 8x15m structure'},
      'fvg_policy':'not required; no parameter tuning in v2.1','selection_rule':'DEV/VAL only; OOS verification only'},indent=2,default=str))
    print(pd.DataFrame(rows).to_markdown(index=False))

if __name__=='__main__': main()
