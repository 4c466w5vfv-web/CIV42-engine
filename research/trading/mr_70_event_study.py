from pathlib import Path
import math
import pandas as pd

# Load the strict v5 namespace without running its built-in report.
v5_path = Path(__file__).with_name('full_cycle_pilot_v5.py')
v5_src = v5_path.read_text(encoding='utf-8')
old_exec = "exec(compile(s, str(p), 'exec'), {'__name__':'__main__','__file__':str(p)})"
new_exec = "ns={'__name__':'ark42_v5lib','__file__':str(p)}\nexec(compile(s, str(p), 'exec'), ns)"
if old_exec not in v5_src:
    raise RuntimeError('v5 final exec target not found')
v5_src = v5_src.replace(old_exec, new_exec)
g = {'__name__':'__main__','__file__':str(v5_path)}
exec(compile(v5_src, str(v5_path), 'exec'), g)
ns = g['ns']

SOURCES = {
    'XAUUSD': {
        'D1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Gold-Cash/XAUUSD/XAUUSD_D1.csv',
        'H4': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Gold-Cash/XAUUSD/XAUUSD_H4.csv',
        'H1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Gold-Cash/XAUUSD/XAUUSD_H1.csv',
    },
    'USOIL': {
        'D1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Oil-Cash/USOIL/USOIL_D1.csv',
        'H4': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Oil-Cash/USOIL/USOIL_H4.csv',
        'H1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Oil-Cash/USOIL/USOIL_H1.csv',
    },
    'UKOIL': {
        'D1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Oil-Cash/UKOIL/UKOIL_D1.csv',
        'H4': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Oil-Cash/UKOIL/UKOIL_H4.csv',
        'H1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Oil-Cash/UKOIL/UKOIL_H1.csv',
    },
}

START = pd.Timestamp('2018-01-01')
END = pd.Timestamp('2026-12-31 23:59:59')
MAX_HOLD_DAYS = 30
COST_R = 0.05  # 0.10 ATR round trip / 2ATR initial risk = 0.05R approximation


def prior_completed_h4_atr(h4, dt):
    cutoff = pd.Timestamp(dt) - pd.Timedelta(hours=4)
    j = h4['dt'].searchsorted(cutoff, side='right') - 1
    if j < 0:
        return None
    a = h4.iloc[int(j)]['atr']
    return None if pd.isna(a) else float(a)


def eval_one(h1, h4, sch, shock):
    i0 = int(sch['accept_i'])
    if i0 >= len(h1):
        return None
    entry_dt = h1.iloc[i0]['dt']
    if entry_dt < START or entry_dt > END:
        return None
    entry = float(h1.iloc[i0]['open'])
    a0 = prior_completed_h4_atr(h4, entry_dt)
    if a0 is None or a0 <= 0:
        return None
    init_risk = 2.0 * a0
    stop = entry - init_risk
    high_since = entry
    remain = 1.0
    realized_r = 0.0
    partial = False
    end_dt = entry_dt + pd.Timedelta(days=MAX_HOLD_DAYS)

    for i in range(i0, len(h1)):
        row = h1.iloc[i]
        dt = row['dt']
        if dt > end_dt or dt > END:
            # time exit at close
            realized_r += remain * ((float(row['close']) - entry) / init_risk)
            remain = 0.0
            break
        op = float(row['open']); lo = float(row['low']); hi = float(row['high'])

        # Stop first when stop and target coexist in same H1 bar: conservative ordering.
        if lo <= stop:
            exit_px = op if op < stop else stop
            realized_r += remain * ((exit_px - entry) / init_risk)
            remain = 0.0
            break

        if (not partial) and hi >= entry + init_risk:
            realized_r += 0.5 * 1.0
            remain = 0.5
            partial = True

        high_since = max(high_since, hi)
        # Trail update uses ATR known from a completed H4 bar and applies next H1 bar.
        a = prior_completed_h4_atr(h4, dt)
        if a is not None and a > 0:
            candidate = high_since - 2.0 * a
            stop = max(stop, candidate)

    if remain > 0:
        row = h1.iloc[min(len(h1)-1, i0 + 24*MAX_HOLD_DAYS)]
        realized_r += remain * ((float(row['close']) - entry) / init_risk)

    realized_r -= COST_R
    return {
        'entry_dt': str(entry_dt),
        'shock_dt': str(shock['shock_dt']),
        'r': float(realized_r),
        'rvol': float(shock['rvol']),
    }


def stats(rs):
    if not rs:
        return {}
    s = pd.Series(rs, dtype=float)
    wins = s[s > 0]
    losses = s[s < 0]
    gp = float(wins.sum())
    gl = float(-losses.sum())
    eq = s.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return {
        'n': int(len(s)),
        'win_rate_pct': float((s > 0).mean() * 100),
        'mean_r': float(s.mean()),
        'median_r': float(s.median()),
        'net_r': float(s.sum()),
        'pf': float(gp / gl) if gl > 0 else math.inf,
        'avg_win_r': float(wins.mean()) if len(wins) else None,
        'avg_loss_r': float(losses.mean()) if len(losses) else None,
        'max_sequential_dd_r': float(dd.min()),
    }

all_rows = []
print('# ARK-42 MR 70+ Event Study — XAU / WTI / Brent')
print('Boundary: mechanical signal-event study; each valid shock->reclaim->HL->break->acceptance schedule is evaluated independently.')
print('Not untouched OOS. Build_shocks already deduplicates shock clusters within 3 days. 30-day max hold; 50% partial at +1R; 2ATR ratchet; 0.05R cost approximation.')

for sym, src in SOURCES.items():
    d1 = ns['load'](src['D1']); h4 = ns['load'](src['H4']); h1 = ns['load'](src['H1'])
    d1, h4, h1, supports = ns['prep'](d1, h4, h1)
    shocks = ns['build_shocks'](d1, h4, supports)
    rows = []
    for sh in shocks:
        sch = ns['find_mr_schedule'](h1, sh)
        if not sch:
            continue
        r = eval_one(h1, h4, sch, sh)
        if r is not None:
            r['symbol'] = sym
            rows.append(r)
            all_rows.append(r)
    st = stats([x['r'] for x in rows])
    print('\n##', sym)
    print('shocks=', len(shocks), 'evaluated_events=', len(rows))
    for k, v in st.items():
        print(f'{k}={v}')

print('\n## COMBINED')
cs = stats([x['r'] for x in all_rows])
print('evaluated_events=', len(all_rows))
for k, v in cs.items():
    print(f'{k}={v}')

# Year distribution for concentration audit.
df = pd.DataFrame(all_rows)
if not df.empty:
    df['year'] = pd.to_datetime(df['entry_dt']).dt.year
    print('\n## YEAR_COUNTS')
    print(df.groupby(['symbol','year']).size().to_string())
    print('\n## YEAR_NET_R')
    print(df.groupby(['symbol','year'])['r'].sum().round(4).to_string())
