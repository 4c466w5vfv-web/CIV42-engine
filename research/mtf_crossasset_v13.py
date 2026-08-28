import json, subprocess, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mtf_backtest_v12 import standardize, run_asset, OUT as OLD_OUT
import mtf_backtest_v12 as core

OUT = Path('research/results/mtf_v13')
OUT.mkdir(parents=True, exist_ok=True)
core.OUT = OUT

ASSETS = {
    'NQ_PROXY': 'usatechidxusd',
    'SP500_PROXY': 'usa500idxusd',
    'GOLD': 'xauusd',
    'SILVER': 'xagusd',
    'WTI': 'lightcmdusd',
    'COPPER': 'coppercmdusd',
    'NATGAS': 'gascmdusd',
    'USTBOND': 'ustbondtrusd',
}


def run_cli(inst, tf, start, end, dest):
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ['npx','dukascopy-cli','-i',inst,'-from',start,'-to',end,'-t',tf,'-f','csv','-v','-dir',str(dest),'--silent']
    p = subprocess.run(cmd, text=True, capture_output=True, timeout=2400)
    if p.returncode != 0:
        raise RuntimeError(f'{inst} {tf} failed: {p.stderr[-2000:]}')
    files = list(dest.rglob('*.csv'))
    if not files:
        raise RuntimeError(f'no csv produced for {inst} {tf}; stdout={p.stdout[-1000:]}')
    return max(files, key=lambda x: x.stat().st_size)


def read_duka(path):
    d = pd.read_csv(path)
    cols = {str(c).lower().strip(): c for c in d.columns}
    # dukascopy-cli typically emits timestamp/open/high/low/close/volume.
    dtcol = next((cols[k] for k in ['timestamp','datetime','date','time'] if k in cols), d.columns[0])
    ren = {dtcol:'datetime'}
    for k in ['open','high','low','close','volume']:
        if k in cols: ren[cols[k]] = k
    d = d.rename(columns=ren)
    if 'volume' not in d.columns:
        d['volume'] = 1.0
    # Handle numeric epoch timestamps if present.
    if pd.api.types.is_numeric_dtype(d['datetime']):
        med = pd.to_numeric(d['datetime'], errors='coerce').dropna().median()
        unit = 'ms' if med > 1e11 else 's'
        d['datetime'] = pd.to_datetime(d['datetime'], unit=unit, utc=True, errors='coerce')
    return standardize(d)


def main():
    all_rows=[]; metas=[]; errors=[]
    base=Path('research/duka_v13')
    for name,inst in ASSETS.items():
        try:
            p15=run_cli(inst,'m15','2019-01-01','2026-08-01',base/name/'m15')
            pd1=run_cli(inst,'d1','2010-01-01','2026-08-01',base/name/'d1')
            intr=read_duka(p15); daily=read_duka(pd1)
            meta,rows=run_asset(name,intr,daily,{
                'source':'Dukascopy public feed via dukascopy-cli',
                'instrument':inst,'m15_file':str(p15),'d1_file':str(pd1),
                'note':'CFD/spot proxy where applicable; volume is Dukascopy tick volume'
            })
            metas.append(meta); all_rows += rows
            print(f'DONE {name}: {len(intr):,} 15m bars')
        except Exception as e:
            errors.append({'asset':name,'error':repr(e)})
            print(f'ERROR {name}: {e}')

    df=pd.DataFrame(all_rows)
    df.to_csv(OUT/'metrics.csv',index=False)
    (OUT/'metadata.json').write_text(json.dumps(metas,indent=2,default=str))
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2,default=str))
    lines=['# Cross-Asset MTF v1.3','',
           'Same causal mechanical proxy and parameter set as v1.2. No per-asset optimization.','',
           'Dukascopy non-crypto volume is tick volume, not centralized exchange volume.','',
           '## Results','']
    if not df.empty:
        show=df.copy()
        for c in ['win_rate','expectancy_R','profit_factor','total_R','max_dd_R','avg_win_R','avg_loss_R','avg_hold_hours']:
            if c in show: show[c]=show[c].map(lambda x: '' if pd.isna(x) else f'{x:.3f}')
        lines.append(show.to_markdown(index=False))
    lines += ['', '## Data windows','']
    for m in metas:
        lines.append(f"- {m['asset']}: {m['intraday_start']} -> {m['intraday_end']} ({m['intraday_bars']:,} 15m bars); test {m['test_start']} -> {m['test_end']}")
    if errors:
        lines += ['', '## Errors','', '```json', json.dumps(errors,indent=2), '```']
    report='\n'.join(lines)
    (OUT/'report.md').write_text(report)
    print(report)

if __name__=='__main__':
    main()
