from pathlib import Path
import math
import pandas as pd

# Reuse the already-audited MR 70-event machinery without executing its report.
base_path = Path(__file__).with_name('mr_70_event_study.py')
base_src = base_path.read_text(encoding='utf-8')
cut = base_src.find("all_rows = []")
if cut < 0:
    raise RuntimeError('mr_70_event_study main boundary not found')
ns = {'__name__':'mr_subgroup_lib','__file__':str(base_path)}
exec(compile(base_src[:cut], str(base_path), 'exec'), ns)

SOURCES = ns['SOURCES']
START = ns['START']; END = ns['END']


def completed_d1(d1, dt):
    cutoff = pd.Timestamp(dt).normalize()
    j = d1['dt'].searchsorted(cutoff, side='left') - 1
    return None if j < 0 else d1.iloc[int(j)]


def stats(rows):
    if not rows:
        return None
    s = pd.Series([r['r'] for r in rows], dtype=float)
    wins=s[s>0]; losses=s[s<0]
    gp=float(wins.sum()); gl=float(-losses.sum())
    eq=s.cumsum(); dd=eq-eq.cummax()
    return {
        'n':len(s),
        'win%':100*float((s>0).mean()),
        'meanR':float(s.mean()),
        'medianR':float(s.median()),
        'netR':float(s.sum()),
        'PF':float(gp/gl) if gl>0 else math.inf,
        'maxDD_R':float(dd.min()),
    }


def fmt(label, rows):
    st=stats(rows)
    if not st:
        return
    print(f"{label} | n={st['n']} win={st['win%']:.1f}% meanR={st['meanR']:+.3f} medianR={st['medianR']:+.3f} netR={st['netR']:+.2f} PF={st['PF']:.2f} maxDD={st['maxDD_R']:.2f}R")

rows=[]
for sym,src in SOURCES.items():
    d1=ns['ns']['load'](src['D1']); h4=ns['ns']['load'](src['H4']); h1=ns['ns']['load'](src['H1'])
    d1,h4,h1,supports=ns['ns']['prep'](d1,h4,h1)
    shocks=ns['ns']['build_shocks'](d1,h4,supports)
    for sh in shocks:
        sch=ns['ns']['find_mr_schedule'](h1,sh)
        if not sch:
            continue
        out=ns['eval_one'](h1,h4,sch,sh)
        if out is None:
            continue
        hi=int(sh['shock_i'])
        hb=h4.iloc[hi]
        dr=completed_d1(d1, sh['shock_dt'])
        if dr is None or pd.isna(hb['atr']) or float(hb['atr'])<=0:
            continue
        shock_atr=float((hb['high']-hb['low'])/hb['atr'])
        support_dist_atr=abs(float(hb['low'])-float(sh['support']))/float(dr['atr']) if pd.notna(dr['atr']) and float(dr['atr'])>0 else float('nan')
        sma_dist_atr=(float(dr['sma200'])-float(dr['close']))/float(dr['atr']) if pd.notna(dr['sma200']) and pd.notna(dr['atr']) and float(dr['atr'])>0 else float('nan')
        slope_atr=(-float(dr['sma_slope'])/float(dr['atr'])) if pd.notna(dr['sma_slope']) and pd.notna(dr['atr']) and float(dr['atr'])>0 else float('nan')
        accept_dt=h1.iloc[int(sch['accept_i'])]['dt']
        confirm_delay_h=(pd.Timestamp(accept_dt)-pd.Timestamp(sh['signal_dt'])).total_seconds()/3600.0
        out.update({
            'symbol':sym,
            'rvol':float(sh['rvol']),
            'shock_atr':shock_atr,
            'support_dist_atr':support_dist_atr,
            'sma_dist_atr':sma_dist_atr,
            'slope_atr':slope_atr,
            'confirm_delay_h':confirm_delay_h,
            'year':pd.Timestamp(out['entry_dt']).year,
        })
        rows.append(out)

print('# ARK-42 MR Subgroup Diagnostic v1')
print('IMPORTANT: in-sample diagnostic only. Thresholds are predeclared before reading these subgroup results; do not call a winning subgroup validated edge.')
print('Base event definition unchanged: shock -> reclaim -> HL -> close-break -> acceptance; 2ATR stop/trail; +1R 50% partial; cost 0.05R.')
fmt('ALL', rows)

# Predeclared mechanism-oriented splits. Minimum n is reported, not optimized.
checks = [
    ('RVOL 1.5-2.0', lambda r: 1.5 <= r['rvol'] < 2.0),
    ('RVOL >=2.0', lambda r: r['rvol'] >= 2.0),
    ('Shock range <1.5 ATR', lambda r: r['shock_atr'] < 1.5),
    ('Shock range >=1.5 ATR', lambda r: r['shock_atr'] >= 1.5),
    ('Support distance <=0.25 D1 ATR', lambda r: pd.notna(r['support_dist_atr']) and r['support_dist_atr'] <= 0.25),
    ('Support distance >0.25 D1 ATR', lambda r: pd.notna(r['support_dist_atr']) and r['support_dist_atr'] > 0.25),
    ('Below SMA200 <=2 D1 ATR', lambda r: pd.notna(r['sma_dist_atr']) and r['sma_dist_atr'] <= 2.0),
    ('Below SMA200 >2 D1 ATR', lambda r: pd.notna(r['sma_dist_atr']) and r['sma_dist_atr'] > 2.0),
    ('SMA down-slope <=0.5 ATR/20d', lambda r: pd.notna(r['slope_atr']) and r['slope_atr'] <= 0.5),
    ('SMA down-slope >0.5 ATR/20d', lambda r: pd.notna(r['slope_atr']) and r['slope_atr'] > 0.5),
    ('Confirmation <=24h', lambda r: r['confirm_delay_h'] <= 24),
    ('Confirmation >24h', lambda r: r['confirm_delay_h'] > 24),
]

print('\n## ONE-FACTOR SPLITS')
for label,fn in checks:
    sub=[r for r in rows if fn(r)]
    fmt(label,sub)

print('\n## PREDECLARED INTERSECTIONS')
intersections = [
    ('Strong shock + close support', lambda r: r['rvol']>=2.0 and r['shock_atr']>=1.5 and pd.notna(r['support_dist_atr']) and r['support_dist_atr']<=0.25),
    ('Strong shock + fast confirm', lambda r: r['rvol']>=2.0 and r['confirm_delay_h']<=24),
    ('Close support + fast confirm', lambda r: pd.notna(r['support_dist_atr']) and r['support_dist_atr']<=0.25 and r['confirm_delay_h']<=24),
    ('Strong shock + close support + fast confirm', lambda r: r['rvol']>=2.0 and r['shock_atr']>=1.5 and pd.notna(r['support_dist_atr']) and r['support_dist_atr']<=0.25 and r['confirm_delay_h']<=24),
]
for label,fn in intersections:
    fmt(label,[r for r in rows if fn(r)])

print('\n## ASSET BREAKDOWN')
for sym in ['XAUUSD','USOIL','UKOIL']:
    fmt(sym,[r for r in rows if r['symbol']==sym])

print('\n## STABILITY RULE')
print('A subgroup is only a research candidate if n>=20, meanR>0, PF>1.10, and is not positive solely because of one asset/year. It still requires a fresh holdout/OOS test before promotion.')

# Concentration audit for any subgroup meeting the loose research-candidate gate.
print('\n## CANDIDATE CONCENTRATION')
for label,fn in checks+intersections:
    sub=[r for r in rows if fn(r)]
    st=stats(sub)
    if not st or st['n']<20 or st['meanR']<=0 or st['PF']<=1.10:
        continue
    print('\nCANDIDATE',label)
    df=pd.DataFrame(sub)
    print('by_asset_netR=',df.groupby('symbol')['r'].sum().round(3).to_dict())
    print('by_year_netR=',df.groupby('year')['r'].sum().round(3).to_dict())
