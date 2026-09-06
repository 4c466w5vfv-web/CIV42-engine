from pathlib import Path

p = Path(__file__).with_name('full_cycle_pilot.py')
s = p.read_text(encoding='utf-8')

# --- v4 correctness patches -------------------------------------------------
# Robust mixed source timestamps.
s = s.replace(
    "df['dt'] = pd.to_datetime(df[tcol], errors='coerce')",
    "df['dt'] = pd.to_datetime(df[tcol].astype(str), errors='coerce', format='mixed', utc=True).dt.tz_convert(None)"
)

# Pandas Series .dt is an accessor, not the row's dt value.
for old, new in {
    "r=h4.iloc[i]; dt=r.dt": "r=h4.iloc[i]; dt=r['dt']",
    "'signal_dt':h4.iloc[i+3].dt": "'signal_dt':h4.iloc[i+3]['dt']",
    "asof_row(d1,h1.iloc[i].dt)": "asof_row(d1,h1.iloc[i]['dt'])",
    "a=h4atr(h1.iloc[i].dt)": "a=h4atr(h1.iloc[i]['dt'])",
    "row=h1.iloc[i]; dt=row.dt": "row=h1.iloc[i]; dt=row['dt']",
    "h1.iloc[schedules[next_sched][1]['probe_i']].dt<dt": "h1.iloc[schedules[next_sched][1]['probe_i']]['dt']<dt",
    "h1.iloc[sch['probe_i']].dt==dt": "h1.iloc[sch['probe_i']]['dt']==dt",
}.items():
    s = s.replace(old, new)

# Preserve an MR thesis after its positions are flat so it can transition to trend mode.
s = s.replace(
    "next_sched=0; current_shock=None; trend_queue=[]",
    "next_sched=0; current_shock=None; trend_queue=[]; cycle_start_dt=None"
)
s = s.replace(
    "cycle_active=True;cycle_id+=1;current_shock=s;trend_mode=False;trend_add_count=0;last_add_entry=None;trend_queue=[]",
    "cycle_active=True;cycle_id+=1;current_shock=s;trend_mode=False;trend_add_count=0;last_add_entry=None;trend_queue=[];cycle_start_dt=dt"
)
old_term = """        # cycle ends when all entries used, no positions, and either trend queue exhausted or trend regime has failed
        if cycle_active and not pending and all(t.closed for t in tranches):
            dr=asof_row(d1,dt)
            regime_ok=dr is not None and pd.notna(dr.sma200) and dr.close>dr.sma200
            if (not trend_mode) or (not regime_ok) or not trend_queue:
                cycle_active=False;tranches=[];pending={};trend_queue=[];last_add_entry=None
"""
new_term = """        # Preserve the reversal thesis long enough to transition to trend mode.
        if cycle_active and not pending and all(t.closed for t in tranches):
            dr=asof_d1(d1,dt)
            regime_ok=dr is not None and pd.notna(dr.sma200) and dr.close>dr.sma200
            age_days=(dt-cycle_start_dt).days if cycle_start_dt is not None else 999
            terminate=False
            if not trend_mode:
                terminate = age_days>180
            else:
                terminate = (not regime_ok) and (not trend_queue)
            if terminate:
                cycle_active=False;tranches=[];pending={};trend_queue=[];last_add_entry=None;cycle_start_dt=None
"""
if old_term not in s:
    raise RuntimeError('v5 cycle termination patch target not found')
s = s.replace(old_term, new_term)

# --- strict completed-bar audit ---------------------------------------------
# D1 state used by intraday logic must come from the previous completed D1 bar.
needle = """def asof_row(df, dt):
    j=df['dt'].searchsorted(dt, side='right')-1
    return None if j<0 else df.iloc[int(j)]
"""
replacement = needle + """

def asof_d1(df, dt):
    cutoff = pd.Timestamp(dt).normalize()
    j=df['dt'].searchsorted(cutoff, side='left')-1
    return None if j<0 else df.iloc[int(j)]
"""
if needle not in s:
    raise RuntimeError('asof_row target not found')
s = s.replace(needle, replacement)

# All D1 reads inside the H1 trend/cycle state machine use the prior completed day.
s = s.replace("asof_row(d1,h1.iloc[i]['dt'])", "asof_d1(d1,h1.iloc[i]['dt'])")
s = s.replace("asof_row(d1,dt)", "asof_d1(d1,dt)")

# Previous completed H4 only. At H1 timestamp t, an H4 bar stamped x is complete only if x+4h <= t.
old_h4atr = """    def h4atr(dt):
        rr=asof_row(h4,dt)
        return None if rr is None or pd.isna(rr.atr) else float(rr.atr)
"""
new_h4atr = """    def h4atr(dt):
        cutoff=pd.Timestamp(dt)-pd.Timedelta(hours=4)
        rr=asof_row(h4,cutoff)
        return None if rr is None or pd.isna(rr.atr) else float(rr.atr)
"""
if old_h4atr not in s:
    raise RuntimeError('h4atr target not found')
s = s.replace(old_h4atr, new_h4atr)

# Trend-mode transition also uses only prior completed D1 bars.
old_gate = """            j=d1['dt'].searchsorted(dt.normalize(),side='right')-1
            if j>=1:
                a=d1.iloc[j];b=d1.iloc[j-1]
"""
new_gate = """            j=d1['dt'].searchsorted(dt.normalize(),side='left')-1
            if j>=1:
                a=d1.iloc[j];b=d1.iloc[j-1]
"""
if old_gate not in s:
    raise RuntimeError('D1 gate target not found')
s = s.replace(old_gate, new_gate)

# --- A/B: same base signals, Pyramid OFF vs risk-recycled Pyramid ON --------
s = s.replace("def run_asset(sym,src):", "def run_asset(sym,src,pyramid_enabled=True):")

# A pyramid is legal only when the immediately previous TREND/PYRAMID tranche is
# still open, profitable, and its ratcheted stop is at/above its own entry.
addtr_end = """        tranches.append(t);all_tr.append(t);last_add_entry=px
        return True
"""
addtr_plus = addtr_end + """
    def pyramid_ready(px):
        active=[t for t in tranches if (not t.closed) and t.module in ('TREND','PYRAMID')]
        if not active:
            return False
        prev=active[-1]
        if prev.stop + 1e-12 < prev.entry:
            return False
        if px <= prev.entry:
            return False
        return True
"""
if addtr_end not in s:
    raise RuntimeError('add_tr insertion target not found')
s = s.replace(addtr_end, addtr_plus)

old_add = """        if cycle_active and trend_mode and trend_queue and i==trend_queue[0]:
            trend_queue.pop(0)
            rf=(0.30 if trend_add_count==0 else 0.20)*ONE_R
            mod='TREND' if trend_add_count==0 else 'PYRAMID'
            if add_tr(mod,rf,i): trend_add_count+=1
"""
new_add = """        if cycle_active and trend_mode and trend_queue and i==trend_queue[0]:
            trend_queue.pop(0)
            if trend_add_count==0:
                # First post-SMA200 trend tranche is identical in both A/B arms.
                if add_tr('TREND',0.30*ONE_R,i): trend_add_count=1
            elif pyramid_enabled and trend_add_count<=3:
                # At most three pyramids. Reuse only risk already released by a
                # protected prior trend tranche; add_tr still enforces total open risk <=1R.
                px=float(h1.iloc[i].open)
                if pyramid_ready(px) and add_tr('PYRAMID',0.20*ONE_R,i):
                    trend_add_count+=1
"""
if old_add not in s:
    raise RuntimeError('trend add target not found')
s = s.replace(old_add, new_add)

# Replace the original one-arm main with a paired A/B report.
cut = s.find("\ndef main():")
if cut < 0:
    raise RuntimeError('main target not found')
s = s[:cut] + r'''

def main():
    print('# ARK-42 MR + Trend / Pyramid A-B Pilot v5')
    print('Audit: intraday D1 state = strictly previous completed D1; H4 ATR = strictly previous completed H4.')
    print('Common rules: same MR shocks/schedules, staged MR 0.25R/0.35R/0.40R, +1R 50% partial, SMA200 trend gate, structure signals, 2x H4 ATR ratchet, 0.10 ATR RT cost, 1R=1%, no averaging down, open-risk cap 1R.')
    print('A=PYRAMID_OFF: MR + first post-SMA200 TREND tranche only.')
    print('B=RISK_RECYCLED_ON: same A plus max 3 continuation pyramids; prior trend tranche must remain open, be profitable, and have stop >= entry; each pyramid 0.20R; total open risk remains <=1R.')
    for sym,src in SOURCES.items():
        try:
            a=run_asset(sym,src,pyramid_enabled=False)
            b=run_asset(sym,src,pyramid_enabled=True)
            print('\n##',sym)
            for label,r in [('OFF',a),('ON',b)]:
                print(label,
                      'return_pct=',r.get('return_pct'),
                      'sharpe=',r.get('sharpe'),
                      'sortino=',r.get('sortino'),
                      'mdd_pct=',r.get('mdd_pct'),
                      'calmar=',r.get('calmar'),
                      'pf_approx=',r.get('pf_approx'),
                      'tranches=',r.get('tranches'),
                      'modules=',r.get('module_pnl_pct'))
            print('DELTA_ON_MINUS_OFF',
                  'return_pp=',b['return_pct']-a['return_pct'],
                  'sharpe=',b['sharpe']-a['sharpe'],
                  'sortino=',b['sortino']-a['sortino'],
                  'mdd_pp=',b['mdd_pct']-a['mdd_pct'],
                  'tranches=',b['tranches']-a['tranches'])
        except Exception as e:
            print('\n##',sym,'ERROR',repr(e))

if __name__=='__main__':main()
'''

exec(compile(s, str(p), 'exec'), {'__name__':'__main__','__file__':str(p)})
