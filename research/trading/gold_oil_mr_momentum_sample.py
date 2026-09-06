from pathlib import Path

# Reuse the strict-completed-bar v5 engine without running its built-in A/B main.
v5_path = Path(__file__).with_name('full_cycle_pilot_v5.py')
v5_src = v5_path.read_text(encoding='utf-8')

old_exec = "exec(compile(s, str(p), 'exec'), {'__name__':'__main__','__file__':str(p)})"
new_exec = r'''
ns={'__name__':'ark42_v5lib','__file__':str(p)}
exec(compile(s, str(p), 'exec'), ns)

# Expanded commodity sample. Pyramid is deliberately OFF: this test isolates
# Mean Reversion + first post-SMA200 Directional/Momentum tranche.
commodity_sources = {
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

print('# ARK-42 Gold + Oil MR/Momentum Sample')
print('Engine: strict v5 completed-bar audit; Pyramid OFF; 0.10 ATR RT cost; 1R=1% engine normalization.')
print('Interpretation boundary: pilot-period mechanical sample, not untouched OOS and not yet the manual 0.5% prop sizing layer.')
for sym, src in commodity_sources.items():
    try:
        r=ns['run_asset'](sym, src, pyramid_enabled=False)
        print('\n##', sym)
        for k in [
            'sample','shocks','mr_schedules','cycles','tranches','return_pct','sharpe','sortino',
            'mdd_pct','cagr_pct','calmar','pf_approx','module_pnl_pct'
        ]:
            if k in r:
                print(f'{k}={r[k]}')
        print('RAW_RESULT', r)
    except Exception as e:
        print('\n##', sym, 'ERROR', repr(e))
'''

if old_exec not in v5_src:
    raise RuntimeError('v5 final exec target not found')
v5_src = v5_src.replace(old_exec, new_exec)
exec(compile(v5_src, str(v5_path), 'exec'), {'__name__':'__main__','__file__':str(v5_path)})
