from pathlib import Path
import math
import pandas as pd

# VCP + volume diagnostic on the existing strict MR event universe.
# IMPORTANT: in-sample diagnostic only. We are testing structure definitions,
# not validating an edge. Entry/exit logic remains unchanged.

base_path = Path(__file__).with_name('mr_70_event_study.py')
base_src = base_path.read_text(encoding='utf-8')
cut = base_src.find('all_rows = []')
if cut < 0:
    raise RuntimeError('mr_70_event_study main boundary not found')
ns0 = {'__name__':'mr_vcp_lib','__file__':str(base_path)}
exec(compile(base_src[:cut], str(base_path), 'exec'), ns0)

SOURCES = ns0['SOURCES']
START = ns0['START']; END = ns0['END']; COST_R = ns0['COST_R']; MAX_HOLD_DAYS = ns0['MAX_HOLD_DAYS']
core = ns0['ns']; prior_completed_h4_atr = ns0['prior_completed_h4_atr']

# Frozen candidate from prior diagnostic
SLOPE_DECEL = 0.50
VOL_CONTRACTION_RATIO = 0.60

# Predeclared VCP definitions for this diagnostic:
# VCP-2: two successive completed H4 bars after shock have shrinking true range AND shrinking volume.
# VCP-3: three successive completed H4 bars after shock have monotonic shrinking true range AND volume.
# ATR contraction: H4 ATR at the 3rd post-shock bar <= 90% of shock-bar ATR.
# Pivot hold: min low of next 3 H4 bars is no more than 0.25 shock ATR below shock low.
ATR_CONTRACTION_RATIO = 0.90
FAILED_DOWNSIDE_ATR = 0.25


def completed_d1(d1, dt):
    cutoff = pd.Timestamp(dt).normalize()
    j = d1['dt'].searchsorted(cutoff, side='left') - 1
    return None if j < 0 else d1.iloc[int(j)]


def stats(rows, key='r'):
    if not rows: return None
    s=pd.Series([float(r[key]) for r in rows if pd.notna(r.get(key))], dtype=float)
    if s.empty: return None
    wins=s[s>0]; losses=s[s<0]; gp=float(wins.sum()); gl=float(-losses.sum())
    eq=s.cumsum(); dd=eq-eq.cummax()
    return dict(n=len(s), win=100*float((s>0).mean()), mean=float(s.mean()), median=float(s.median()), net=float(s.sum()), pf=(gp/gl if gl>0 else math.inf), mdd=float(dd.min()))


def fmt(label, rows):
    q=stats(rows)
    if not q:
        print(label,'| n=0'); return
    print(f"{label} | n={q['n']} win={q['win']:.1f}% meanR={q['mean']:+.3f} medianR={q['median']:+.3f} netR={q['net']:+.2f} PF={q['pf']:.2f} maxDD={q['mdd']:.2f}R")


def feature_pack(d1,h4,sh):
    hi=int(sh['shock_i']); hb=h4.iloc[hi]
    if pd.isna(hb['atr']) or float(hb['atr'])<=0: return None
    nxt=h4.iloc[hi+1:min(len(h4),hi+4)]
    if len(nxt)<3: return None
    atr=float(hb['atr']); shock_low=float(hb['low'])
    shock_vol=float(hb['vol']) if pd.notna(hb['vol']) else float('nan')
    vols=[float(x) for x in nxt['vol'].tolist()]
    ranges=[float(r['high']-r['low']) for _,r in nxt.iterrows()]
    med_vol=float(pd.Series(vols).median())
    vol_ratio=(med_vol/shock_vol) if pd.notna(shock_vol) and shock_vol>0 else float('nan')
    downside=max(0.0, shock_low-float(nxt['low'].min()))/atr
    dr=completed_d1(d1, sh['shock_dt'])
    slope_atr=float('nan')
    if dr is not None and pd.notna(dr['sma_slope']) and pd.notna(dr['atr']) and float(dr['atr'])>0:
        slope_atr=-float(dr['sma_slope'])/float(dr['atr'])
    atr3=float(nxt.iloc[2]['atr']) if pd.notna(nxt.iloc[2]['atr']) else float('nan')
    atr_contract=pd.notna(atr3) and atr3 <= ATR_CONTRACTION_RATIO*atr
    vcp2 = ranges[1] < ranges[0] and vols[1] < vols[0]
    vcp3 = ranges[2] < ranges[1] < ranges[0] and vols[2] < vols[1] < vols[0]
    return {
        'vol_ratio':vol_ratio,
        'decelerating':pd.notna(slope_atr) and slope_atr<=SLOPE_DECEL,
        'vol_contract':pd.notna(vol_ratio) and vol_ratio<=VOL_CONTRACTION_RATIO,
        'pivot_hold':downside<=FAILED_DOWNSIDE_ATR,
        'vcp2':vcp2,
        'vcp3':vcp3,
        'atr_contract':atr_contract,
        'feature_end_dt':pd.Timestamp(nxt.iloc[-1]['dt'])+pd.Timedelta(hours=4),
    }


def eval_A(h1,h4,sch):
    i0=int(sch['accept_i'])
    if i0>=len(h1): return None
    entry_dt=pd.Timestamp(h1.iloc[i0]['dt']); entry=float(h1.iloc[i0]['open'])
    a0=prior_completed_h4_atr(h4,entry_dt)
    if a0 is None or a0<=0: return None
    risk=2.0*a0; stop=entry-risk; high_since=entry; remain=1.0; real=0.0; partial=False; last_close=entry
    end_dt=entry_dt+pd.Timedelta(days=MAX_HOLD_DAYS)
    for i in range(i0,len(h1)):
        row=h1.iloc[i]; dt=pd.Timestamp(row['dt'])
        if dt>end_dt or dt>END: break
        op=float(row['open']); lo=float(row['low']); hi=float(row['high']); cl=float(row['close']); last_close=cl
        if lo<=stop:
            xp=op if op<stop else stop
            real += remain*((xp-entry)/risk); remain=0; break
        if (not partial) and hi>=entry+risk:
            real += 0.5; remain=0.5; partial=True
        high_since=max(high_since,hi)
        a=prior_completed_h4_atr(h4,dt)
        if a is not None and a>0: stop=max(stop,high_since-2.0*a)
    if remain>0: real += remain*((last_close-entry)/risk)
    return real-COST_R

rows=[]
for sym,src in SOURCES.items():
    d1=core['load'](src['D1']); h4=core['load'](src['H4']); h1=core['load'](src['H1'])
    d1,h4,h1,supports=core['prep'](d1,h4,h1)
    for sh in core['build_shocks'](d1,h4,supports):
        sch=core['find_mr_schedule'](h1,sh)
        if not sch: continue
        feat=feature_pack(d1,h4,sh)
        if not feat: continue
        entry_dt=pd.Timestamp(h1.iloc[int(sch['accept_i'])]['dt'])
        if feat['feature_end_dt']>entry_dt: continue
        r=eval_A(h1,h4,sch)
        if r is None: continue
        rows.append({**feat,'symbol':sym,'year':entry_dt.year,'entry_dt':entry_dt,'r':float(r)})

print('# ARK-42 MR VCP + Volume Diagnostic v4')
print('IN-SAMPLE STRUCTURE DIAGNOSTIC ONLY. Do not call winning subgroup validated edge.')
print('Exit fixed: +1R half + 2ATR runner; cost=0.05R. Entry logic unchanged.')
print('VCP-2 = next H4 bar2 range<bar1 and volume<bar1; VCP-3 = 3-bar monotonic range+volume contraction.')

fmt('ALL',rows)
base=[r for r in rows if r['decelerating'] and r['vol_contract']]
fmt('FIXED-A deceleration + volume contraction',base)
fmt('FIXED-A + VCP2',[r for r in base if r['vcp2']])
fmt('FIXED-A + VCP3',[r for r in base if r['vcp3']])
fmt('FIXED-A + ATR contraction',[r for r in base if r['atr_contract']])
fmt('FIXED-A + pivot hold',[r for r in base if r['pivot_hold']])
fmt('FIXED-A + VCP2 + pivot hold',[r for r in base if r['vcp2'] and r['pivot_hold']])
fmt('FIXED-A + ATR contraction + pivot hold',[r for r in base if r['atr_contract'] and r['pivot_hold']])

print('\n## BASE ASSET/YEAR')
if base:
    df=pd.DataFrame(base)
    print('by_asset=',df.groupby('symbol')['r'].agg(['count','sum','mean']).round(3).to_dict('index'))
    print('by_year=',df.groupby('year')['r'].agg(['count','sum','mean']).round(3).to_dict('index'))

for label,flt in [
    ('VCP2',[r for r in base if r['vcp2']]),
    ('VCP3',[r for r in base if r['vcp3']]),
    ('ATR_CONTRACT',[r for r in base if r['atr_contract']]),
    ('VCP2_PIVOT',[r for r in base if r['vcp2'] and r['pivot_hold']]),
]:
    print(f'\n## {label} CONCENTRATION')
    if flt:
        df=pd.DataFrame(flt)
        print('by_asset=',df.groupby('symbol')['r'].agg(['count','sum','mean']).round(3).to_dict('index'))
        print('by_year=',df.groupby('year')['r'].agg(['count','sum','mean']).round(3).to_dict('index'))
