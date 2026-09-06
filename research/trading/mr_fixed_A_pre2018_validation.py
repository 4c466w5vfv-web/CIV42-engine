from pathlib import Path
import math
import pandas as pd

# Temporal validation of the FIXED candidate discovered on 2018-2026:
# Important-zone shock machinery unchanged; filter is frozen as
# SMA200 down-slope deceleration <= 0.50 D1 ATR/20d AND
# post-shock median volume (next 3 completed H4) <= 0.60 x shock volume.
# Exit is frozen: +1R 50% partial + 2ATR runner; cost=0.05R.
# We intentionally evaluate only PRE-2018 observations to avoid reusing the
# 2018-2026 discovery window. This is historical temporal holdout, not live OOS.

mr_path = Path(__file__).with_name('mr_70_event_study.py')
src = mr_path.read_text(encoding='utf-8')
src = src.replace("START = pd.Timestamp('2018-01-01')", "START = pd.Timestamp('2000-01-01')")
src = src.replace("END = pd.Timestamp('2026-12-31 23:59:59')", "END = pd.Timestamp('2017-12-31 23:59:59')")
cut = src.find('all_rows = []')
if cut < 0:
    raise RuntimeError('mr_70_event_study main boundary not found')
g = {'__name__':'mr_pre2018_lib','__file__':str(mr_path)}
exec(compile(src[:cut], str(mr_path), 'exec'), g)

SOURCES=g['SOURCES']; START=g['START']; END=g['END']; COST_R=g['COST_R']; MAX_HOLD_DAYS=g['MAX_HOLD_DAYS']
core=g['ns']; prior_completed_h4_atr=g['prior_completed_h4_atr']
VOL_CONTRACTION_RATIO=0.60
SLOPE_DECEL=0.50


def completed_d1(d1, dt):
    cutoff=pd.Timestamp(dt).normalize()
    j=d1['dt'].searchsorted(cutoff, side='left')-1
    return None if j<0 else d1.iloc[int(j)]


def shock_features(d1,h4,sh):
    hi=int(sh['shock_i']); hb=h4.iloc[hi]
    if pd.isna(hb['atr']) or float(hb['atr'])<=0: return None
    nxt=h4.iloc[hi+1:min(len(h4),hi+4)]
    if len(nxt)<3: return None
    shock_vol=float(hb['vol']) if pd.notna(hb['vol']) else float('nan')
    med=float(nxt['vol'].median()) if 'vol' in nxt.columns else float('nan')
    vol_ratio=(med/shock_vol) if pd.notna(shock_vol) and shock_vol>0 else float('nan')
    dr=completed_d1(d1,sh['shock_dt'])
    slope_atr=float('nan')
    if dr is not None and pd.notna(dr['sma_slope']) and pd.notna(dr['atr']) and float(dr['atr'])>0:
        slope_atr=-float(dr['sma_slope'])/float(dr['atr'])
    return {
        'vol_ratio':vol_ratio,
        'slope_atr':slope_atr,
        'feature_end_dt':pd.Timestamp(nxt.iloc[-1]['dt'])+pd.Timedelta(hours=4),
    }


def eval_A(h1,h4,sch):
    i0=int(sch['accept_i'])
    if i0>=len(h1): return None
    entry_dt=pd.Timestamp(h1.iloc[i0]['dt'])
    if entry_dt<START or entry_dt>END: return None
    entry=float(h1.iloc[i0]['open'])
    a0=prior_completed_h4_atr(h4,entry_dt)
    if a0 is None or a0<=0: return None
    init_risk=2.0*a0; stop=entry-init_risk; high_since=entry
    remain=1.0; realized=0.0; partial=False; last_close=entry
    end_dt=entry_dt+pd.Timedelta(days=MAX_HOLD_DAYS)
    for i in range(i0,len(h1)):
        row=h1.iloc[i]; dt=pd.Timestamp(row['dt'])
        if dt>end_dt or dt>END: break
        op=float(row['open']); lo=float(row['low']); hi=float(row['high']); cl=float(row['close']); last_close=cl
        if lo<=stop:
            xp=op if op<stop else stop
            realized += remain*((xp-entry)/init_risk); remain=0.0; break
        if (not partial) and hi>=entry+init_risk:
            realized += 0.5; remain=0.5; partial=True
        high_since=max(high_since,hi)
        a=prior_completed_h4_atr(h4,dt)
        if a is not None and a>0:
            stop=max(stop,high_since-2.0*a)
    if remain>0:
        realized += remain*((last_close-entry)/init_risk)
    return realized-COST_R


def stats(vals):
    s=pd.Series(vals,dtype=float)
    if s.empty: return None
    wins=s[s>0]; losses=s[s<0]; gp=float(wins.sum()); gl=float(-losses.sum())
    eq=s.cumsum(); dd=eq-eq.cummax()
    return dict(n=len(s),win=100*float((s>0).mean()),mean=float(s.mean()),median=float(s.median()),net=float(s.sum()),pf=(gp/gl if gl>0 else math.inf),mdd=float(dd.min()))

rows=[]
print('# ARK-42 FIXED-A PRE-2018 TEMPORAL VALIDATION')
print('Frozen rule: SMA200 deceleration <=0.50 D1 ATR/20d + post-shock volume contraction <=0.60x; +1R half + 2ATR runner; cost=0.05R.')
print('Window:',START,'to',END)
for sym,srcs in SOURCES.items():
    d1=core['load'](srcs['D1']); h4=core['load'](srcs['H4']); h1=core['load'](srcs['H1'])
    print(sym,'raw_ranges',str(d1['dt'].min()),str(d1['dt'].max()),'|',str(h4['dt'].min()),str(h4['dt'].max()),'|',str(h1['dt'].min()),str(h1['dt'].max()))
    d1,h4,h1,supports=core['prep'](d1,h4,h1)
    shocks=core['build_shocks'](d1,h4,supports)
    for sh in shocks:
        sch=core['find_mr_schedule'](h1,sh)
        if not sch: continue
        feat=shock_features(d1,h4,sh)
        if not feat: continue
        entry_dt=pd.Timestamp(h1.iloc[int(sch['accept_i'])]['dt'])
        if entry_dt<START or entry_dt>END or feat['feature_end_dt']>entry_dt: continue
        fixed=(pd.notna(feat['slope_atr']) and feat['slope_atr']<=SLOPE_DECEL and pd.notna(feat['vol_ratio']) and feat['vol_ratio']<=VOL_CONTRACTION_RATIO)
        if not fixed: continue
        r=eval_A(h1,h4,sch)
        if r is not None: rows.append({'symbol':sym,'entry_dt':entry_dt,'year':entry_dt.year,'r':float(r)})

st=stats([r['r'] for r in rows])
if not st:
    print('ALL | n=0')
else:
    print(f"ALL | n={st['n']} win={st['win']:.1f}% meanR={st['mean']:+.3f} medianR={st['median']:+.3f} netR={st['net']:+.2f} PF={st['pf']:.2f} maxDD={st['mdd']:.2f}R")
    df=pd.DataFrame(rows)
    print('BY_ASSET')
    for sym,gp in df.groupby('symbol'):
        q=stats(gp['r'].tolist()); print(f"{sym} | n={q['n']} win={q['win']:.1f}% meanR={q['mean']:+.3f} netR={q['net']:+.2f} PF={q['pf']:.2f} maxDD={q['mdd']:.2f}R")
    print('BY_YEAR')
    print(df.groupby('year')['r'].agg(['count','sum','mean']).round(3).to_string())
