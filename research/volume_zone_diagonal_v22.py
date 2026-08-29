"""v2.2 Volume-spike inventory zone + long-term diagonal experiment.
Hypothesis test, not production strategy.
Frozen direction/setup skeleton from v2.1. Tests whether volume-spike candle inventory zones
and long-term diagonal can define entry/partial-exit/final-exit coherently.
No OOS parameter selection. FVG remains unrequired.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal
import mtf_exit_engine_v17 as exits
import weekly_daily_diagonal_v20 as v20
import weekly_daily_diagonal_v21 as v21

OUT=Path('research/results/volume_zone_diagonal_v22'); OUT.mkdir(parents=True,exist_ok=True)
VOL_SPIKE=v20.VOL_SPIKE
ZONE_MODES=('FULL','BODY','MID')


def volume_zones(daily):
    d=v20.enrich(daily); rows=[]
    for t,r in d.iterrows():
        if not np.isfinite(r.vol_ratio) or r.vol_ratio<VOL_SPIKE: continue
        lo=float(r.low); hi=float(r.high); blo=float(min(r.open,r.close)); bhi=float(max(r.open,r.close)); mid=(blo+bhi)/2
        rows.append({'zone_time':t,'vol_ratio':float(r.vol_ratio),'full_lo':lo,'full_hi':hi,'body_lo':blo,'body_hi':bhi,'mid':mid})
    return pd.DataFrame(rows)


def attach_zone(setups,zones):
    if setups.empty or zones.empty: return pd.DataFrame()
    rows=[]
    for _,s in setups.iterrows():
        z=zones[zones.zone_time<=s.setup_time]
        if z.empty: continue
        # nearest prior volume-spike candle whose range contains/overlaps the v2.1 S/R level
        lv=float(s.sr_level); q=z[(z.full_lo<=lv)&(z.full_hi>=lv)].tail(1)
        if q.empty: continue
        r=s.to_dict(); r.update(q.iloc[-1].to_dict()); rows.append(r)
    return pd.DataFrame(rows)


def zone_bounds(r,mode):
    if mode=='FULL': return float(r.full_lo),float(r.full_hi)
    if mode=='BODY': return float(r.body_lo),float(r.body_hi)
    return float(r.mid),float(r.mid)


def make_entries(setups,m15,mode):
    """After causal daily setup, wait for a later 15m rejection/reclaim of the selected inventory zone.
    Stop is prior 15m structure, so HTF decides location and LTF decides risk.
    """
    if setups.empty:return pd.DataFrame()
    x=m15.copy(); x['prev_hi']=x.high.shift(1).rolling(v21.LTF_LOOKBACK,min_periods=v21.LTF_LOOKBACK).max(); x['prev_lo']=x.low.shift(1).rolling(v21.LTF_LOOKBACK,min_periods=v21.LTF_LOOKBACK).min()
    rows=[]
    for _,s in setups.iterrows():
        side=int(s.side); zl,zh=zone_bounds(s,mode); st=pd.Timestamp(s.setup_time)
        w=x[(x.index>st)&(x.index<=st+pd.Timedelta(minutes=v21.LTF_CONFIRM_MINUTES))]
        for t,r in w.iterrows():
            if not np.isfinite(r.prev_hi) or not np.isfinite(r.prev_lo): continue
            if side==1:
                interacted=r.low<=zh and r.high>=zl; confirm=interacted and r.close>zh and r.close>r.open
                stop=float(min(r.prev_lo,zl)); entry=float(r.close)
            else:
                interacted=r.high>=zl and r.low<=zh; confirm=interacted and r.close<zl and r.close<r.open
                stop=float(max(r.prev_hi,zh)); entry=float(r.close)
            if not confirm or (side==1 and stop>=entry) or (side==-1 and stop<=entry): continue
            risk=abs(entry-stop)
            if risk<=0:continue
            q=s.to_dict(); q.update({'entry_time':t,'signal_available_time':t,'entry':entry,'stop':stop,'risk':risk,'zone_mode':mode,'zone_lo':zl,'zone_hi':zh,'stop_model':'LTF_STRUCTURE_PLUS_ZONE_INVALIDATION'})
            rows.append(q); break
    return pd.DataFrame(rows)


def next_opposing_zone(entry,side,zones,after):
    z=zones[zones.zone_time<after]
    if side==1: z=z[z.full_lo>entry].sort_values('full_lo')
    else: z=z[z.full_hi<entry].sort_values('full_hi',ascending=False)
    if z.empty:return np.nan
    r=z.iloc[0]; return float(r.full_lo if side==1 else r.full_hi)


def annotate_targets(c,zones):
    if c.empty:return c
    q=c.copy(); q['next_volume_zone']=q.apply(lambda r:next_opposing_zone(float(r.entry),int(r.side),zones,pd.Timestamp(r.entry_time)),axis=1)
    # diagonal is causal setup-time line; use as structural reference metadata, not future-fitted line.
    q['diagonal_exit_reference']=q.diagonal_level
    return q


def main():
    name=base.os.environ['ASSET']; m15,m1,daily,prov=causal.load_asset_causal(name)
    core=v21.actual_diagonal_candidates(daily); zones=volume_zones(daily); setup=attach_zone(core,zones)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True); zones.to_csv(out/'volume_zones.csv',index=False); setup.to_csv(out/'setups.csv',index=False)
    rows=[]
    for mode in ZONE_MODES:
        c=annotate_targets(make_entries(setup,m15,mode),zones); c.to_csv(out/f'candidates_{mode}.csv',index=False)
        models=[('FIXED_2R',lambda:exits.fixed_tp(c,m1,2.0)),('FIXED_3R',lambda:exits.fixed_tp(c,m1,3.0)),('FIXED_5R',lambda:exits.fixed_tp(c,m1,5.0)),('PURE_SWING',lambda:exits.trailing(c,m1,'PURE_SWING')),('ADAPTIVE_TRAIL',lambda:exits.trailing(c,m1,'ADAPTIVE_TRAIL'))]
        for label,fn in models:
            tr=fn(); tr.to_csv(out/f'trades_{mode}_{label}.csv',index=False)
            if tr.empty: rows.append({'asset':name,'zone_mode':mode,'exit_model':label,'split':'ALL','trades':0}); continue
            tr['split']=tr.entry_time.map(base.split)
            for sp,g in [('ALL',tr),*list(tr.groupby('split'))]:
                m=exits.metrics_ext(g); m.update({'asset':name,'zone_mode':mode,'exit_model':label,'split':sp}); rows.append(m)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False)
    (out/'meta.json').write_text(json.dumps({'version':'2.2','asset':name,'provenance':prov,'volume_spike_threshold':VOL_SPIKE,'zone_modes':ZONE_MODES,
      'hypothesis':'volume-spike candle inventory zone + actual long-term diagonal can improve location/risk and provide opposing-zone exit references',
      'entry':'daily v2.1 setup then later 15m zone reclaim/rejection','stop':'prior 15m structure plus zone invalidation','fvg':'not required','selection':'DEV/VAL only; OOS verification only'},indent=2,default=str))
    print(pd.DataFrame(rows).to_markdown(index=False))
if __name__=='__main__':main()
