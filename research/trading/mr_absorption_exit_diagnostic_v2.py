from pathlib import Path
import math
import pandas as pd

# Reuse strict MR event-study machinery without executing its report.
base_path = Path(__file__).with_name('mr_70_event_study.py')
base_src = base_path.read_text(encoding='utf-8')
cut = base_src.find('all_rows = []')
if cut < 0:
    raise RuntimeError('mr_70_event_study main boundary not found')
ns0 = {'__name__':'mr_absorption_lib','__file__':str(base_path)}
exec(compile(base_src[:cut], str(base_path), 'exec'), ns0)

SOURCES = ns0['SOURCES']
START = ns0['START']; END = ns0['END']
COST_R = ns0['COST_R']
MAX_HOLD_DAYS = ns0['MAX_HOLD_DAYS']
core = ns0['ns']
prior_completed_h4_atr = ns0['prior_completed_h4_atr']

# PREDECLARED BEFORE THIS RERUN:
# 1) post-shock volume contraction = median of next 3 completed H4 volumes <= 60% of shock volume
# 2) TIGHT failed downside = next 3 H4 lows do not extend >0.25 shock-ATR below shock low
#    (base shock builder already imposes a looser <=0.50 ATR condition, so 0.25 is intentionally discriminatory)
# 3) trend deceleration = D1 SMA200 20d down-slope <=0.5 D1 ATR
# 4) structural retracement target = 50% of the prior 12-H4-bar selloff leg (pre-shock high -> shock low)
VOL_CONTRACTION_RATIO = 0.60
FAILED_DOWNSIDE_ATR = 0.25
SLOPE_DECEL = 0.50
PRE_SHOCK_LOOKBACK_H4 = 12


def completed_d1(d1, dt):
    cutoff = pd.Timestamp(dt).normalize()
    j = d1['dt'].searchsorted(cutoff, side='left') - 1
    return None if j < 0 else d1.iloc[int(j)]


def stats(rows, key='r'):
    if not rows:
        return None
    s = pd.Series([float(r[key]) for r in rows if pd.notna(r.get(key))], dtype=float)
    if s.empty:
        return None
    wins=s[s>0]; losses=s[s<0]
    gp=float(wins.sum()); gl=float(-losses.sum())
    eq=s.cumsum(); dd=eq-eq.cummax()
    return {'n':len(s),'win%':100*float((s>0).mean()),'meanR':float(s.mean()),'medianR':float(s.median()),'netR':float(s.sum()),'PF':float(gp/gl) if gl>0 else math.inf,'maxDD_R':float(dd.min())}


def fmt(label, rows, key='r'):
    st=stats(rows,key)
    if not st:
        print(label,'| n=0')
        return
    print(f"{label} | n={st['n']} win={st['win%']:.1f}% meanR={st['meanR']:+.3f} medianR={st['medianR']:+.3f} netR={st['netR']:+.2f} PF={st['PF']:.2f} maxDD={st['maxDD_R']:.2f}R")


def shock_features(d1,h4,sh):
    hi=int(sh['shock_i'])
    hb=h4.iloc[hi]
    if pd.isna(hb['atr']) or float(hb['atr'])<=0:
        return None
    atr=float(hb['atr']); shock_low=float(hb['low']); shock_high=float(hb['high'])
    nxt=h4.iloc[hi+1:min(len(h4),hi+4)]
    if len(nxt)<3:
        return None
    # Loader standardizes tick_volume/volume into `vol`.
    shock_vol=float(hb['vol']) if pd.notna(hb['vol']) else float('nan')
    next_med_vol=float(nxt['vol'].median()) if 'vol' in nxt.columns else float('nan')
    vol_ratio=(next_med_vol/shock_vol) if pd.notna(shock_vol) and shock_vol>0 else float('nan')
    next_min_low=float(nxt['low'].min())
    downside_ext=max(0.0, shock_low-next_min_low)/atr
    dr=completed_d1(d1,sh['shock_dt'])
    slope_atr=float('nan')
    if dr is not None and pd.notna(dr['sma_slope']) and pd.notna(dr['atr']) and float(dr['atr'])>0:
        slope_atr=-float(dr['sma_slope'])/float(dr['atr'])
    pre=h4.iloc[max(0,hi-PRE_SHOCK_LOOKBACK_H4):hi]
    prior_high=float(pre['high'].max()) if len(pre) else shock_high
    retrace50=shock_low+0.5*(prior_high-shock_low)
    return {
        'vol_ratio':vol_ratio,
        'downside_ext_atr':downside_ext,
        'slope_atr':slope_atr,
        'retrace50':retrace50,
        'shock_low':shock_low,
        'shock_high':shock_high,
        'feature_end_dt':pd.Timestamp(nxt.iloc[-1]['dt'])+pd.Timedelta(hours=4),
    }


def eval_modes(h1,h4,sch,sh,target50):
    i0=int(sch['accept_i'])
    if i0>=len(h1): return None
    entry_dt=pd.Timestamp(h1.iloc[i0]['dt'])
    entry=float(h1.iloc[i0]['open'])
    a0=prior_completed_h4_atr(h4,entry_dt)
    if a0 is None or a0<=0: return None
    init_risk=2.0*a0
    init_stop=entry-init_risk
    end_dt=entry_dt+pd.Timedelta(days=MAX_HOLD_DAYS)
    target_valid = target50 > entry

    states={
      'A': {'stop':init_stop,'high':entry,'remain':1.0,'real':0.0,'p1':False,'done':False},
      'B': {'stop':init_stop,'high':entry,'remain':1.0,'real':0.0,'done':False},
      'C': {'stop':init_stop,'high':entry,'remain':1.0,'real':0.0,'pt':False,'done':False},
    }
    last_close=entry
    for i in range(i0,len(h1)):
        row=h1.iloc[i]; dt=pd.Timestamp(row['dt'])
        if dt>end_dt or dt>END: break
        op=float(row['open']); lo=float(row['low']); hi=float(row['high']); cl=float(row['close']); last_close=cl
        for k,s in states.items():
            if s['done']: continue
            if lo<=s['stop']:
                xp=op if op<s['stop'] else s['stop']
                s['real'] += s['remain']*((xp-entry)/init_risk); s['remain']=0; s['done']=True; continue
            if k=='A':
                if (not s['p1']) and hi>=entry+init_risk:
                    s['real'] += 0.5; s['remain']=0.5; s['p1']=True
            elif k=='B':
                if target_valid and hi>=target50:
                    s['real'] += s['remain']*((target50-entry)/init_risk); s['remain']=0; s['done']=True; continue
            elif k=='C':
                if target_valid and (not s['pt']) and hi>=target50:
                    s['real'] += 0.5*((target50-entry)/init_risk); s['remain']=0.5; s['pt']=True
            if k in ('A','C') and not s['done']:
                s['high']=max(s['high'],hi)
                a=prior_completed_h4_atr(h4,dt)
                if a is not None and a>0:
                    s['stop']=max(s['stop'],s['high']-2.0*a)
    for s in states.values():
        if not s['done'] and s['remain']>0:
            s['real'] += s['remain']*((last_close-entry)/init_risk)
            s['remain']=0; s['done']=True
        s['real'] -= COST_R
    return {'r_A':states['A']['real'],'r_B':states['B']['real'] if target_valid else float('nan'),'r_C':states['C']['real'] if target_valid else float('nan'),'target_valid':target_valid,'entry':entry,'target50':target50}

rows=[]
for sym,src in SOURCES.items():
    d1=core['load'](src['D1']); h4=core['load'](src['H4']); h1=core['load'](src['H1'])
    d1,h4,h1,supports=core['prep'](d1,h4,h1)
    shocks=core['build_shocks'](d1,h4,supports)
    for sh in shocks:
        sch=core['find_mr_schedule'](h1,sh)
        if not sch: continue
        feat=shock_features(d1,h4,sh)
        if not feat: continue
        entry_dt=pd.Timestamp(h1.iloc[int(sch['accept_i'])]['dt'])
        if feat['feature_end_dt']>entry_dt: continue
        out=eval_modes(h1,h4,sch,sh,feat['retrace50'])
        if not out: continue
        row={**out,**feat,'symbol':sym,'entry_dt':str(entry_dt),'year':entry_dt.year,'rvol':float(sh['rvol'])}
        row['vol_contract']=pd.notna(row['vol_ratio']) and row['vol_ratio']<=VOL_CONTRACTION_RATIO
        row['failed_downside']=row['downside_ext_atr']<=FAILED_DOWNSIDE_ATR
        row['decelerating']=pd.notna(row['slope_atr']) and row['slope_atr']<=SLOPE_DECEL
        rows.append(row)

print('# ARK-42 MR Absorption + Exit Diagnostic v2')
print('IN-SAMPLE DIAGNOSTIC ONLY. Thresholds were fixed before this rerun; do not call a winning subgroup validated edge.')
print('No-future rule: post-shock 3xH4 volume/downside features must be fully completed before H1 entry confirmation.')
print(f'Features: volume contraction <= {VOL_CONTRACTION_RATIO:.2f}x shock volume; TIGHT downside extension <= {FAILED_DOWNSIDE_ATR:.2f} shock ATR; SMA200 down-slope <= {SLOPE_DECEL:.2f} D1 ATR/20d.')
print('Exit A=+1R 50% partial + 2ATR runner; B=full exit at 50% prior-12H4 selloff retracement; C=50% at retracement + 50% 2ATR runner. Cost=0.05R.')

print('\n## BASE AFTER NO-FUTURE FEATURE AVAILABILITY')
fmt('A baseline',rows,'r_A')
fmt('B retrace50',rows,'r_B')
fmt('C retrace50+runner',rows,'r_C')

filters=[
 ('Volume contraction',lambda r:r['vol_contract']),
 ('Tight failed downside',lambda r:r['failed_downside']),
 ('Trend deceleration',lambda r:r['decelerating']),
 ('Vol contraction + tight failed downside',lambda r:r['vol_contract'] and r['failed_downside']),
 ('Deceleration + vol contraction',lambda r:r['decelerating'] and r['vol_contract']),
 ('Deceleration + tight failed downside',lambda r:r['decelerating'] and r['failed_downside']),
 ('FULL absorption candidate',lambda r:r['decelerating'] and r['vol_contract'] and r['failed_downside']),
]
print('\n## PREDECLARED FILTERS — EXIT A')
for label,fn in filters: fmt(label,[r for r in rows if fn(r)],'r_A')

print('\n## FULL ABSORPTION CANDIDATE — EXIT COMPARISON')
full=[r for r in rows if r['decelerating'] and r['vol_contract'] and r['failed_downside']]
fmt('A +1R partial + 2ATR',full,'r_A')
fmt('B full retrace50',full,'r_B')
fmt('C retrace50 partial + 2ATR runner',full,'r_C')

print('\n## FULL CANDIDATE ASSET/YEAR CONCENTRATION')
if full:
    df=pd.DataFrame(full)
    for key in ['r_A','r_B','r_C']:
        print(key,'by_asset=',df.groupby('symbol')[key].sum().round(3).to_dict())
        print(key,'by_year=',df.groupby('year')[key].sum().round(3).to_dict())
    print('target_valid=',int(df['target_valid'].sum()),'/',len(df))

print('\n## DECELERATION + TIGHT FAILED DOWNSIDE — EXIT COMPARISON')
dfilt=[r for r in rows if r['decelerating'] and r['failed_downside']]
fmt('A',dfilt,'r_A'); fmt('B',dfilt,'r_B'); fmt('C',dfilt,'r_C')
