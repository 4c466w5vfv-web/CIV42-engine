import json, os
from pathlib import Path
import pandas as pd
import mtf_crossasset_v13 as v
from mtf_backtest_v12 import run_asset
import mtf_backtest_v12 as core

name=os.environ['ASSET']
inst=v.ASSETS[name]
out=Path('research/results/mtf_v13_parallel')/name
out.mkdir(parents=True,exist_ok=True)
core.OUT=out
base=Path('research/duka_v13_parallel')/name
try:
    p15=v.run_cli(inst,'m15','2020-01-01','2026-08-01',base/'m15')
    pd1=v.run_cli(inst,'d1','2010-01-01','2026-08-01',base/'d1')
    intr=v.read_duka(p15); daily=v.read_duka(pd1)
    meta,rows=run_asset(name,intr,daily,{'source':'Dukascopy public feed via dukascopy-cli','instrument':inst,'note':'CFD/spot proxy where applicable; volume=tick volume'})
    df=pd.DataFrame(rows)
    df.to_csv(out/'metrics.csv',index=False)
    (out/'metadata.json').write_text(json.dumps(meta,indent=2,default=str))
    print(df.to_markdown(index=False))
    print('WINDOW',meta['test_start'],meta['test_end'],'BARS',meta['intraday_bars'])
except Exception as e:
    (out/'error.txt').write_text(repr(e))
    print('ERROR',name,repr(e))
    raise
