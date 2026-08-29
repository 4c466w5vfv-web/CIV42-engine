"""v2.3 confidence-weighted position sizing validation.
Run only after v2.2 artifacts exist. No OOS tuning.
Tests whether ex-ante setup quality justifies 0.25/0.50/1.00% risk instead of fixed 0.25%.
Portfolio heat remains a separate 4% cap; no averaging down.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SRC=Path('research/results/volume_zone_diagonal_v22')
OUT=Path('research/results/confidence_weighted_sizing_v23'); OUT.mkdir(parents=True,exist_ok=True)
START=10000.0
RISK={'B':0.0025,'A':0.005,'A+':0.01}


def tier(r):
    # Ex-ante only: no realized-return inputs.
    score=0
    score += int(bool(r.get('diagonal_break_touch',False)))
    score += int(bool(r.get('diagonal_retest_touch',False)))
    vr=float(r.get('vol_ratio',r.get('sr_vol_ratio',0)) or 0)
    score += int(vr>=2.0)
    # zone reclaim plus actual diagonal interaction are already entry prerequisites.
    # Reward-space proxy: opposing volume zone distance in initial R, if causally known.
    nz=r.get('next_volume_zone',np.nan); entry=float(r.entry); risk=float(r.risk); side=int(r.side)
    if np.isfinite(nz) and risk>0:
        room=(float(nz)-entry)*side/risk
        score += int(room>=3.0)
    return 'A+' if score>=3 else ('A' if score==2 else 'B')


def path_stats(tr,mode):
    eq=START; peak=START; maxdd=0.; vals=[]
    for _,r in tr.sort_values('entry_time').iterrows():
        pct=0.0025 if mode=='FIXED_025' else RISK[r.tier]
        pnl=eq*pct*float(r.net_r); eq+=pnl; peak=max(peak,eq); maxdd=max(maxdd,(peak-eq)/peak); vals.append(eq)
    return {'terminal_equity':eq,'return_pct':(eq/START-1)*100,'mdd_pct':maxdd*100}


def main():
    rows=[]
    for asset_dir in SRC.iterdir() if SRC.exists() else []:
      if not asset_dir.is_dir(): continue
      for f in asset_dir.glob('trades_*_*.csv'):
        tr=pd.read_csv(f)
        if tr.empty or 'net_r' not in tr: continue
        cand_name=f.name.replace('trades_','candidates_').split('_FIXED')[0].split('_PURE')[0].split('_ADAPTIVE')[0]+'.csv'
        # Match candidate attributes by entry time when available.
        mode=f.name.split('_')[1]
        cf=asset_dir/f'candidates_{mode}.csv'
        if cf.exists():
          c=pd.read_csv(cf); keep=[x for x in ['entry_time','diagonal_break_touch','diagonal_retest_touch','vol_ratio','sr_vol_ratio','next_volume_zone','entry','risk','side'] if x in c]
          if 'entry_time' in keep: tr=tr.merge(c[keep].drop_duplicates('entry_time'),on='entry_time',how='left',suffixes=('','_c'))
        tr['tier']=tr.apply(tier,axis=1)
        tr['split']=tr.entry_time.map(lambda x:'DEV' if str(x)<'2023-01-01' else ('VAL' if str(x)<'2025-01-01' else 'OOS'))
        for sp,g in tr.groupby('split'):
          for sm in ['FIXED_025','TIERED']:
            st=path_stats(g,sm); st.update({'asset':asset_dir.name,'file':f.name,'split':sp,'sizing':sm,'trades':len(g),'Aplus':int((g.tier=='A+').sum()),'A':int((g.tier=='A').sum()),'B':int((g.tier=='B').sum())}); rows.append(st)
    pd.DataFrame(rows).to_csv(OUT/'sizing_metrics.csv',index=False)
    print(pd.DataFrame(rows).to_markdown(index=False))
if __name__=='__main__': main()
