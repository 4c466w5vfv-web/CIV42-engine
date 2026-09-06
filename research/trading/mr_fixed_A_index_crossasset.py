from pathlib import Path
import math
import pandas as pd

# Cross-asset holdout for the FROZEN MR candidate discovered on commodity data.
# Frozen filter: SMA200 down-slope deceleration <= 0.50 D1 ATR/20d AND
# post-shock median volume of next 3 completed H4 bars <= 0.60 x shock volume.
# Frozen exit: +1R 50% partial + 2ATR runner; cost=0.05R. No retuning by asset.

base_path = Path(__file__).with_name('mr_70_event_study.py')
src = base_path.read_text(encoding='utf-8')
cut = src.find('all_rows = []')
if cut < 0: raise RuntimeError('mr_70_event_study main boundary not found')
g={'__name__':'mr_index_holdout_lib','__file__':str(base_path)}
exec(compile(src[:cut], str(base_path), 'exec'), g)
core=g['ns']; prior_completed_h4_atr=g['prior_completed_h4_atr']; COST_R=g['COST_R']; MAX_HOLD_DAYS=g['MAX_HOLD_DAYS']
START=pd.Timestamp('2018-01-01'); END=pd.Timestamp('2026-12-31 23:59:59')
VOL_CONTRACTION_RATIO=0.60; SLOPE_DECEL=0.50

ROOT='https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Index-Cash'
SYMS=['NAS100','SPX500','UK100','US30']
SOURCES={s:{tf:f'{ROOT}/{s}/{s}_{tf}.csv' for tf in ['D1','H4','H1']} for s in SYMS}

def completed_d1(d1,dt):
    cutoff=pd.Timestamp(dt).normalize(); j=d1['dt'].searchsorted(cutoff,side='left')-1
    return None if j<0 else d1.iloc[int(j)]

def features(d1,h4,sh):
    i=int(sh['shock_i']); hb=h4.iloc[i]
    if pd.isna(hb['atr']) or float(hb['atr'])<=0: return None
    nxt=h4.iloc[i+1:min(len(h4),i+4)]
    if len(nxt)<3: return None
    sv=float(hb['vol']) if pd.notna(hb['vol']) else float('nan')
    mv=float(nxt['vol'].median()) if 'vol' in nxt.columns else float('nan')
    vr=mv/sv if pd.notna(sv) and sv>0 else float('nan')
    dr=completed_d1(d1,sh['shock_dt']); slope=float('nan')
    if dr is not None and pd.notna(dr['sma_slope']) and pd.notna(dr['atr']) and float(dr['atr'])>0:
        slope=-float(dr['sma_slope'])/float(dr['atr'])
    return vr,slope,pd.Timestamp(nxt.iloc[-1]['dt'])+pd.Timedelta(hours=4)

def eval_A(h1,h4,sch):
    i0=int(sch['accept_i'])
    if i0>=len(h1): return None
    edt=pd.Timestamp(h1.iloc[i0]['dt'])
    if edt<START or edt>END: return None
    entry=float(h1.iloc[i0]['open']); a0=prior_completed_h4_atr(h4,edt)
    if a0 is None or a0<=0: return None
    risk=2*a0; stop=entry-risk; high=entry; remain=1.0; real=0.0; partial=False; last=entry
    end=edt+pd.Timedelta(days=MAX_HOLD_DAYS)
    for i in range(i0,len(h1)):
        row=h1.iloc[i]; dt=pd.Timestamp(row['dt'])
        if dt>end or dt>END: break
        op=float(row['open']); lo=float(row['low']); hi=float(row['high']); cl=float(row['close']); last=cl
        if lo<=stop:
            xp=op if op<stop else stop; real+=remain*((xp-entry)/risk); remain=0; break
        if (not partial) and hi>=entry+risk:
            real+=0.5; remain=0.5; partial=True
        high=max(high,hi)
        a=prior_completed_h4_atr(h4,dt)
        if a is not None and a>0: stop=max(stop,high-2*a)
    if remain>0: real+=remain*((last-entry)/risk)
    return real-COST_R

def stats(vals):
    s=pd.Series(vals,dtype=float)
    if s.empty:return None
    w=s[s>0]; l=s[s<0]; gp=float(w.sum()); gl=float(-l.sum()); eq=s.cumsum(); dd=eq-eq.cummax()
    return {'n':len(s),'win':100*float((s>0).mean()),'mean':float(s.mean()),'median':float(s.median()),'net':float(s.sum()),'pf':gp/gl if gl>0 else math.inf,'mdd':float(dd.min())}

rows=[]
print('# ARK-42 FIXED-A INDEX CROSS-ASSET HOLDOUT')
print('Frozen commodity-discovery rule applied unchanged to NAS100/SPX500/UK100/US30.')
print('Filter: slope<=0.50 D1 ATR/20d; post-shock 3xH4 median volume<=0.60x shock volume. Exit:+1R half +2ATR runner. Cost=0.05R.')
for sym,srcs in SOURCES.items():
    d1=core['load'](srcs['D1']); h4=core['load'](srcs['H4']); h1=core['load'](srcs['H1'])
    print(sym,'raw_ranges',str(d1['dt'].min()),str(d1['dt'].max()),'|',str(h4['dt'].min()),str(h4['dt'].max()),'|',str(h1['dt'].min()),str(h1['dt'].max()))
    d1,h4,h1,supports=core['prep'](d1,h4,h1); shocks=core['build_shocks'](d1,h4,supports)
    raw_sched=0; fixed_n=0
    for sh in shocks:
        sch=core['find_mr_schedule'](h1,sh)
        if not sch: continue
        raw_sched+=1
        feat=features(d1,h4,sh)
        if not feat: continue
        vr,slope,avail=feat; edt=pd.Timestamp(h1.iloc[int(sch['accept_i'])]['dt'])
        if edt<START or edt>END or avail>edt: continue
        if not (pd.notna(slope) and slope<=SLOPE_DECEL and pd.notna(vr) and vr<=VOL_CONTRACTION_RATIO): continue
        r=eval_A(h1,h4,sch)
        if r is not None:
            fixed_n+=1; rows.append({'symbol':sym,'entry_dt':edt,'year':edt.year,'r':float(r)})
    print(sym,'raw_schedules=',raw_sched,'fixed_A=',fixed_n)

st=stats([x['r'] for x in rows])
if not st: print('ALL | n=0')
else:
    print(f"ALL | n={st['n']} win={st['win']:.1f}% meanR={st['mean']:+.3f} medianR={st['median']:+.3f} netR={st['net']:+.2f} PF={st['pf']:.2f} maxDD={st['mdd']:.2f}R")
    df=pd.DataFrame(rows)
    print('BY_ASSET')
    for sym,gp in df.groupby('symbol'):
        q=stats(gp['r'].tolist()); print(f"{sym} | n={q['n']} win={q['win']:.1f}% meanR={q['mean']:+.3f} netR={q['net']:+.2f} PF={q['pf']:.2f} maxDD={q['mdd']:.2f}R")
    print('BY_YEAR'); print(df.groupby('year')['r'].agg(['count','sum','mean']).round(3).to_string())
