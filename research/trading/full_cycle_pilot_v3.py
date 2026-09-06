from pathlib import Path

p = Path(__file__).with_name('full_cycle_pilot.py')
s = p.read_text(encoding='utf-8')

# Normalize mixed-offset timestamps to naive UTC.
s = s.replace("df['dt'] = pd.to_datetime(df[tcol], errors='coerce')", "df['dt'] = pd.to_datetime(df[tcol].astype(str), errors='coerce', format='mixed', utc=True).dt.tz_convert(None)")

# Pandas Series has a .dt accessor, so row.dt is not the value of the 'dt' column.
# Replace only row/iloc datetime access; leave DataFrame column .dt.hour intact.
repls = {
    "r=h4.iloc[i]; dt=r.dt": "r=h4.iloc[i]; dt=r['dt']",
    "'signal_dt':h4.iloc[i+3].dt": "'signal_dt':h4.iloc[i+3]['dt']",
    "asof_row(d1,h1.iloc[i].dt)": "asof_row(d1,h1.iloc[i]['dt'])",
    "a=h4atr(h1.iloc[i].dt)": "a=h4atr(h1.iloc[i]['dt'])",
    "row=h1.iloc[i]; dt=row.dt": "row=h1.iloc[i]; dt=row['dt']",
    "h1.iloc[schedules[next_sched][1]['probe_i']].dt<dt": "h1.iloc[schedules[next_sched][1]['probe_i']]['dt']<dt",
    "h1.iloc[sch['probe_i']].dt==dt": "h1.iloc[sch['probe_i']]['dt']==dt",
}
for old, new in repls.items():
    if old not in s:
        print('WARN missing patch target:', old)
    s = s.replace(old, new)

exec(compile(s, str(p), 'exec'), {'__name__': '__main__', '__file__': str(p)})
