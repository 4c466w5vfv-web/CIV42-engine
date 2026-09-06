from pathlib import Path

p = Path(__file__).with_name('full_cycle_pilot.py')
s = p.read_text(encoding='utf-8')

# Robust source timestamps.
s = s.replace("df['dt'] = pd.to_datetime(df[tcol], errors='coerce')", "df['dt'] = pd.to_datetime(df[tcol].astype(str), errors='coerce', format='mixed', utc=True).dt.tz_convert(None)")

# Pandas Series .dt is an accessor, not the row's 'dt' value.
for old,new in {
    "r=h4.iloc[i]; dt=r.dt": "r=h4.iloc[i]; dt=r['dt']",
    "'signal_dt':h4.iloc[i+3].dt": "'signal_dt':h4.iloc[i+3]['dt']",
    "asof_row(d1,h1.iloc[i].dt)": "asof_row(d1,h1.iloc[i]['dt'])",
    "a=h4atr(h1.iloc[i].dt)": "a=h4atr(h1.iloc[i]['dt'])",
    "row=h1.iloc[i]; dt=row.dt": "row=h1.iloc[i]; dt=row['dt']",
    "h1.iloc[schedules[next_sched][1]['probe_i']].dt<dt": "h1.iloc[schedules[next_sched][1]['probe_i']]['dt']<dt",
    "h1.iloc[sch['probe_i']].dt==dt": "h1.iloc[sch['probe_i']]['dt']==dt",
}.items():
    s=s.replace(old,new)

# The v3 pilot accidentally killed a cycle as soon as MR tranches were flat, so
# SMA200 transition and pyramids could never occur. Preserve the thesis after MR
# flat for up to 180 days while waiting for an accepted SMA200 transition.
s=s.replace(
    "next_sched=0; current_shock=None; trend_queue=[]",
    "next_sched=0; current_shock=None; trend_queue=[]; cycle_start_dt=None"
)
s=s.replace(
    "cycle_active=True;cycle_id+=1;current_shock=s;trend_mode=False;trend_add_count=0;last_add_entry=None;trend_queue=[]",
    "cycle_active=True;cycle_id+=1;current_shock=s;trend_mode=False;trend_add_count=0;last_add_entry=None;trend_queue=[];cycle_start_dt=dt"
)
old="""        # cycle ends when all entries used, no positions, and either trend queue exhausted or trend regime has failed
        if cycle_active and not pending and all(t.closed for t in tranches):
            dr=asof_row(d1,dt)
            regime_ok=dr is not None and pd.notna(dr.sma200) and dr.close>dr.sma200
            if (not trend_mode) or (not regime_ok) or not trend_queue:
                cycle_active=False;tranches=[];pending={};trend_queue=[];last_add_entry=None
"""
new="""        # A mean-reversion thesis is allowed to become a trend thesis even after the
        # MR tranche is flat. Do not terminate merely because the MR stop/trail fired.
        if cycle_active and not pending and all(t.closed for t in tranches):
            dr=asof_row(d1,dt)
            regime_ok=dr is not None and pd.notna(dr.sma200) and dr.close>dr.sma200
            age_days=(dt-cycle_start_dt).days if cycle_start_dt is not None else 999
            terminate=False
            if not trend_mode:
                # Give the reversal up to 180 days to reclaim/accept SMA200.
                terminate = age_days>180
            else:
                # Once Trend Mode has existed, terminate only after the D1 regime
                # has failed and there are no remaining structural add candidates.
                terminate = (not regime_ok) and (not trend_queue)
            if terminate:
                cycle_active=False;tranches=[];pending={};trend_queue=[];last_add_entry=None;cycle_start_dt=None
"""
if old not in s:
    raise RuntimeError('cycle termination patch target not found')
s=s.replace(old,new)

exec(compile(s, str(p), 'exec'), {'__name__':'__main__','__file__':str(p)})
