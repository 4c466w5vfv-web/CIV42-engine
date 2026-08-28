import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_crossasset_v13 as duka
import mtf_backtest_v12 as core

OUT=Path('research/results/mtf_v16'); OUT.mkdir(parents=True,exist_ok=True)
TP_LEVELS=[2.0,3.0,5.0]
VOL_MIN=1.2
COST_R=0.05


def fvg_zones(df, min_atr=0.10):
    x=df.copy(); x['atr']=core.atr(x,14)
    bull=(x.high.shift(2)<x.low)&((x.low-x.high.shift(2))>=min_atr*x.atr)
    bear=(x.low.shift(2)>x.high)&((x.low.shift(2)-x.high)>=min_atr*x.atr)
    out=[]
    for i,t in enumerate(x.index):
        if bull.iloc[i]: out.append((t,1,float(x.high.shift(2).iloc[i]),float(x.low.iloc[i]),'FVG'))
        if bear.iloc[i]: out.append((t,-1,float(x.high.iloc[i]),float(x.low.shift(2).iloc[i]),'FVG'))
    return out


def ob_zones(df):
    x=df.copy(); x['atr']=core.atr(x,14)
    body=(x.close-x.open).abs()
    ph=x.high.shift(1).rolling(10,min_periods=10).max(); pl=x.low.shift(1).rolling(10,min_periods=10).min()
    bull=(x.close>ph)&(x.close>x.open)&(body>=0.8*x.atr)
    bear=(x.close<pl)&(x.close<x.open)&(body>=0.8*x.atr)
    out=[]; r=x.reset_index()
    for i in range(len(r)):
        if bool(bull.iloc[i]):
            for j in range(i-1,max(-1,i-6),-1):
                if r.loc[j,'close']<r.loc[j,'open']:
                    out.append((r.loc[i,'datetime'],1,float(r.loc[j,'low']),float(r.loc[j,'high']),'OB')); break
        if bool(bear.iloc[i]):
            for j in range(i-1,max(-1,i-6),-1):
                if r.loc[j,'close']>r.loc[j,'open']:
                    out.append((r.loc[i,'datetime'],-1,float(r.loc[j,'low']),float(r.loc[j,'high']),'OB')); break
    return out


def active_zone_map(intr15,daily):
    h4=core.resample_ohlcv(intr15,'4h')
    d1=daily.copy()
    zones=[]
    for tf,df,max_age in [('4H',h4,pd.Timedelta(days=20)),('1D',d1,pd.Timedelta(days=60))]:
        for z in fvg_zones(df)+ob_zones(df): zones.append((*z,tf,max_age))
    zones.sort(key=lambda z:z[0])
    return zones


def prep_ltf(m15,m1,daily):
    dr=core.daily_regime(daily)
    a=m15.copy(); a['atr']=core.atr(a,14)
    a['vol_base']=a.volume.shift(1).rolling(20,min_periods=20).mean(); a['vol_ratio']=a.volume/a.vol_base.replace(0,np.nan)
    a['prev_hi']=a.high.shift(1).rolling(4,min_periods=4).max(); a['prev_lo']=a.low.shift(1).rolling(4,min_periods=4).min()
    a['bull15']=(a.close>a.prev_hi)&(a.close>a.open); a['bear15']=(a.close<a.prev_lo)&(a.close<a.open)
    a=pd.merge_asof(a.reset_index().sort_values('datetime'),dr[['regime']].reset_index().sort_values('datetime'),on='datetime',direction='backward').set_index('datetime')
    b=m1.copy(); b['hi5']=b.high.shift(1).rolling(5,min_periods=5).max(); b['lo5']=b.low.shift(1).rolling(5,min_periods=5).min()
    b['bull1']=(b.close>b.hi5)&(b.close>b.open); b['bear1']=(b.close<b.lo5)&(b.close<b.open)
    b['swing_lo']=b.low.shift(1).rolling(10,min_periods=10).min(); b['swing_hi']=b.high.shift(1).rolling(10,min_periods=10).max()
    return a,b


def candidates(m15,m1,daily):
    zones=active_zone_map(m15,daily); a,b=prep_ltf(m15,m1,daily); out=[]
    for t,r in a.iterrows():
        side=1 if r.regime==1 else -1 if r.regime==-1 else 0
        if side==0 or not np.isfinite(r.vol_ratio) or r.vol_ratio<VOL_MIN: continue
        if side==1 and not r.bull15: continue
        if side==-1 and not r.bear15: continue
        # HTF zone must exist before the 15m confirmation bar and be touched by that bar.
        elig=[]
        for zt,zs,zlo,zhi,kind,tf,max_age in zones:
            if zs!=side or zt>=t or t-zt>max_age: continue
            if r.low<=zhi and r.high>=zlo: elig.append((zt,zs,zlo,zhi,kind,tf,max_age))
        if not elig: continue
        z=max(elig,key=lambda z:z[0])
        w=b[(b.index>t-pd.Timedelta(minutes=15))&(b.index<=t+pd.Timedelta(minutes=60))]
        if side==1: w=w[w.bull1]
        else: w=w[w.bear1]
        if w.empty: continue
        ct=w.index[0]; cr=w.loc[ct]; entry=float(cr.close)
        if side==1:
            stop=min(float(cr.swing_lo),z[2]) if np.isfinite(cr.swing_lo) else z[2]
            if stop>=entry: continue
        else:
            stop=max(float(cr.swing_hi),z[3]) if np.isfinite(cr.swing_hi) else z[3]
            if stop<=entry: continue
        risk=abs(entry-stop)
        if risk<=0: continue
        out.append({'entry_time':ct,'side':side,'entry':entry,'stop':stop,'risk':risk,'zone_kind':z[4],'zone_tf':z[5],'zone_time':z[0],'vol_ratio':float(r.vol_ratio)})
    c=pd.DataFrame(out)
    if not c.empty: c=c.sort_values('entry_time').drop_duplicates('entry_time')
    return c


def simulate(c,m1,tp):
    rows=[]; busy_until=None
    for _,q in c.iterrows():
        et=pd.Timestamp(q.entry_time)
        if busy_until is not None and et<=busy_until: continue
        side=int(q.side); entry=float(q.entry); stop=float(q.stop); risk=float(q.risk)
        target=entry+side*tp*risk
        w=m1[(m1.index>et)&(m1.index<=et+pd.Timedelta(days=20))]
        if w.empty: continue
        xp=float(w.close.iloc[-1]); xt=w.index[-1]; reason='timeout'
        for t,r in w.iterrows():
            sh=(r.low<=stop) if side==1 else (r.high>=stop)
            th=(r.high>=target) if side==1 else (r.low<=target)
            if sh or th:
                # conservative if both touched in same 1m bar
                if sh: xp=stop; reason='stop'
                else: xp=target; reason='tp'
                xt=t; break
        gross=side*(xp-entry)/risk
        rows.append({**q.to_dict(),'exit_time':xt,'gross_R':gross,'net_R':gross-COST_R,'reason':reason})
        busy_until=xt
    return pd.DataFrame(rows)


def metrics(tr):
    if tr.empty:return {'trades':0}
    r=tr.net_R.astype(float); wins=r[r>0]; losses=r[r<=0]
    eq=pd.concat([pd.Series([0.0]),r.cumsum().reset_index(drop=True)],ignore_index=True); dd=eq-eq.cummax(); sd=r.std(ddof=1)
    return {'trades':len(r),'win_rate':(r>0).mean(),'expectancy_R':r.mean(),'PF':wins.sum()/abs(losses.sum()) if losses.sum()<0 else np.nan,'total_R':r.sum(),'max_dd_R':abs(dd.min()),'trade_sharpe':(r.mean()/sd*math.sqrt(len(r))) if sd>0 else np.nan,'avg_win_R':wins.mean() if len(wins) else np.nan,'avg_loss_R':losses.mean() if len(losses) else np.nan}


def split(ts):
    y=pd.Timestamp(ts).year
    return 'DEV' if y<=2022 else 'VAL' if y<=2024 else 'OOS'


def load_asset(name):
    if name in duka.ASSETS:
        inst=duka.ASSETS[name]; base=Path('research/duka_v16')/name
        p1=duka.run_cli(inst,'m1','2020-01-01','2026-08-01',base/'m1'); p15=duka.run_cli(inst,'m15','2020-01-01','2026-08-01',base/'m15'); pd1=duka.run_cli(inst,'d1','2008-01-01','2026-08-01',base/'d1')
        return duka.read_duka(p15),duka.read_duka(p1),duka.read_duka(pd1),{'source':'Dukascopy','instrument':inst,'volume':'tick volume'}
    pair={'ETH':'ETHUSDT','BTC':'BTCUSDT'}[name]
    m15,_=core.fetch_binance_15m(pair,'2018-01'); d1=core.resample_ohlcv(m15,'1D')
    # Binance 1m archive would be very large; use public monthly 1m from 2020 onward via direct adaptation.
    frames=[]
    import io,zipfile
    for p in core.month_range('2020-01'):
        url=f'https://data.binance.vision/data/spot/monthly/klines/{pair}/1m/{pair}-1m-{p}.zip'
        try:
            z=zipfile.ZipFile(io.BytesIO(core.get_bytes(url,45))); raw=z.read(z.namelist()[0]); cols=['open_time','open','high','low','close','volume','close_time','quote_volume','trades','taker_base','taker_quote','ignore']; d=pd.read_csv(io.BytesIO(raw),header=None,names=cols); ot=pd.to_numeric(d.open_time,errors='coerce'); unit='us' if ot.dropna().median()>1e14 else 'ms'; d['datetime']=pd.to_datetime(ot,unit=unit,utc=True,errors='coerce'); frames.append(d[['datetime','open','high','low','close','volume']])
        except Exception: pass
    m1=core.standardize(pd.concat(frames,ignore_index=True))
    return m15[m15.index>=m1.index.min()],m1,d1,{'source':'Binance spot public archive','symbol':pair,'volume':'exchange spot volume'}


def main():
    name=os.environ['ASSET']; m15,m1,daily,prov=load_asset(name); c=candidates(m15,m1,daily)
    out=OUT/name; out.mkdir(parents=True,exist_ok=True); c.to_csv(out/'candidates.csv',index=False)
    rows=[]
    for tp in TP_LEVELS:
        tr=simulate(c,m1,tp); tr.to_csv(out/f'trades_TP{int(tp)}R.csv',index=False)
        if tr.empty:
            rows.append({'asset':name,'tp_R':tp,'split':'ALL','trades':0}); continue
        tr['split']=tr.entry_time.map(split)
        m=metrics(tr); m.update({'asset':name,'tp_R':tp,'split':'ALL'}); rows.append(m)
        for sp,g in tr.groupby('split'):
            x=metrics(g); x.update({'asset':name,'tp_R':tp,'split':sp}); rows.append(x)
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False); (out/'meta.json').write_text(json.dumps({'asset':name,'provenance':prov,'m15_bars':len(m15),'m1_bars':len(m1),'candidate_count':len(c),'rules':'Daily regime; 4H/Daily FVG or OB touch; 15m BOS + volume>=1.2x; 1m micro BOS confirmation; structural stop; fixed 2R/3R/5R TP; 0.05R cost'},indent=2))
    print(pd.DataFrame(rows).to_markdown(index=False))

if __name__=='__main__': main()
