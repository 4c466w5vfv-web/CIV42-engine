import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_crossasset_v13 as duka
import mtf_backtest_v12 as core

OUT = Path('research/results/mtf_v15')
OUT.mkdir(parents=True, exist_ok=True)

TP_LEVELS = [2.0, 3.0, 5.0]
VOL_MIN = 1.2
ENTRY_MODES = ['FVG_OR_OB', 'FVG_ONLY', 'OB_ONLY']
COST_R = 0.05


def entry_ok(r, side, mode):
    if side == 1:
        if not (r.get('regime',0) == 1 and r.get('wstruct',0) >= 0): return False
        p = r.get('p', np.nan)
        if not np.isfinite(p) or p > 0.45: return False
        if not bool(r.get('bull_bos', False)): return False
        if not bool(r.get('micro_bull', False)): return False
        fvg = bool(r.get('bull_fvg', False))
        ob = bool(r.get('bull_ob_retest', False))
    else:
        if not (r.get('regime',0) == -1 and r.get('wstruct',0) <= 0): return False
        p = r.get('p', np.nan)
        if not np.isfinite(p) or p < 0.55: return False
        if not bool(r.get('bear_bos', False)): return False
        if not bool(r.get('micro_bear', False)): return False
        fvg = bool(r.get('bear_fvg', False))
        ob = bool(r.get('bear_ob_retest', False))
    if mode == 'FVG_ONLY' and not fvg: return False
    if mode == 'OB_ONLY' and not ob: return False
    if mode == 'FVG_OR_OB' and not (fvg or ob): return False
    if not (r.get('vol_ratio', 0) >= VOL_MIN): return False
    return True


def backtest_fixed_tp(feat, mode, tp_r):
    trades=[]
    pos=0; entry=stop=risk=np.nan; et=None; held=0
    max_hold=20*24*4
    for t,r in feat.iterrows():
        if pos == 0:
            for side in (1,-1):
                if not entry_ok(r, side, mode):
                    continue
                e=float(r['close'])
                if side == 1:
                    s=float(r.get('swing_lo', np.nan))
                    if not np.isfinite(s) or s >= e: continue
                else:
                    s=float(r.get('swing_hi', np.nan))
                    if not np.isfinite(s) or s <= e: continue
                rr=abs(e-s)
                a=float(r.get('atr1h', np.nan))
                if not np.isfinite(a) or a <= 0: continue
                if rr < 0.5*a or rr > 3.0*a: continue
                pos=side; entry=e; stop=s; risk=rr; et=t; held=0
                break
            continue

        held += 1
        if pos == 1:
            tp=entry + tp_r*risk
            stop_hit=float(r['low']) <= stop
            tp_hit=float(r['high']) >= tp
            # Conservative same-bar collision: stop first.
            if stop_hit or tp_hit or held >= max_hold:
                if stop_hit: xp=stop; reason='stop'
                elif tp_hit: xp=tp; reason='tp'
                else: xp=float(r['close']); reason='timeout'
                gross=(xp-entry)/risk
                trades.append({'entry_time':et,'exit_time':t,'side':'LONG','gross_R':gross,'net_R':gross-COST_R,'bars':held,'reason':reason})
                pos=0
        else:
            tp=entry - tp_r*risk
            stop_hit=float(r['high']) >= stop
            tp_hit=float(r['low']) <= tp
            if stop_hit or tp_hit or held >= max_hold:
                if stop_hit: xp=stop; reason='stop'
                elif tp_hit: xp=tp; reason='tp'
                else: xp=float(r['close']); reason='timeout'
                gross=(entry-xp)/risk
                trades.append({'entry_time':et,'exit_time':t,'side':'SHORT','gross_R':gross,'net_R':gross-COST_R,'bars':held,'reason':reason})
                pos=0
    return pd.DataFrame(trades)


def metrics(tr):
    if tr.empty:
        return {'trades':0,'win_rate':None,'expectancy_R':None,'profit_factor':None,'total_R':0,'max_dd_R':None,'sharpe_trade':None,'avg_hold_h':None}
    r=tr.net_R.astype(float)
    wins=r[r>0]; losses=r[r<=0]
    pf=float(wins.sum()/abs(losses.sum())) if losses.sum()<0 else None
    eq=pd.concat([pd.Series([0.0]),r.cumsum().reset_index(drop=True)],ignore_index=True)
    dd=eq-eq.cummax()
    sd=float(r.std(ddof=1)) if len(r)>1 else np.nan
    sh=float(r.mean()/sd*math.sqrt(len(r))) if np.isfinite(sd) and sd>0 else None
    return {'trades':int(len(r)),'win_rate':float((r>0).mean()),'expectancy_R':float(r.mean()),'profit_factor':pf,'total_R':float(r.sum()),'max_dd_R':float(abs(dd.min())),'sharpe_trade':sh,'avg_hold_h':float(tr.bars.mean()*0.25)}


def split_label(ts):
    y=ts.year
    if y <= 2022: return 'DEV'
    if y <= 2024: return 'VAL'
    return 'OOS'


def run_one(name, intr, daily, provenance):
    feat=core.build_features(intr,daily)
    feat=feat[feat['sma200'].notna()].copy()
    rows=[]
    for mode in ENTRY_MODES:
        for tp in TP_LEVELS:
            tr=backtest_fixed_tp(feat,mode,tp)
            tr.to_csv(OUT/f'{name}_{mode}_TP{int(tp)}_trades.csv',index=False)
            m=metrics(tr); m.update({'asset':name,'entry_mode':mode,'tp_R':tp,'split':'ALL','volume_rule':f'vol_ratio>={VOL_MIN}'})
            rows.append(m)
            if not tr.empty:
                tr2=tr.copy(); tr2['entry_time']=pd.to_datetime(tr2.entry_time,utc=True)
                tr2['split']=tr2.entry_time.map(split_label)
                for sp,g in tr2.groupby('split'):
                    sm=metrics(g); sm.update({'asset':name,'entry_mode':mode,'tp_R':tp,'split':sp,'volume_rule':f'vol_ratio>={VOL_MIN}'})
                    rows.append(sm)
    meta={'asset':name,'bars':len(intr),'start':str(intr.index.min()),'end':str(intr.index.max()),'provenance':provenance}
    return rows,meta


def main():
    name=os.environ['ASSET']
    if name in duka.ASSETS:
        inst=duka.ASSETS[name]
        base=Path('research/duka_v15')/name
        p15=duka.run_cli(inst,'m15','2018-01-01','2026-08-01',base/'m15')
        pd1=duka.run_cli(inst,'d1','2008-01-01','2026-08-01',base/'d1')
        intr=duka.read_duka(p15); daily=duka.read_duka(pd1)
        rows,meta=run_one(name,intr,daily,{'source':'Dukascopy','instrument':inst,'volume':'tick volume'})
    elif name in ('BTC','ETH'):
        pair={'BTC':'BTCUSDT','ETH':'ETHUSDT'}[name]
        intr,_=core.fetch_binance_15m(pair,'2018-01')
        daily=core.resample_ohlcv(intr,'1D')
        rows,meta=run_one(name,intr,daily,{'source':'Binance spot public data','symbol':pair,'volume':'exchange traded spot volume'})
    else:
        raise ValueError(name)
    df=pd.DataFrame(rows)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True)
    df.to_csv(out/'metrics.csv',index=False)
    (out/'meta.json').write_text(json.dumps(meta,indent=2))
    print(df.to_markdown(index=False))

if __name__=='__main__':
    main()
