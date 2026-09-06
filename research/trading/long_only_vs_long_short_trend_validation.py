import math
import pandas as pd
from pathlib import Path

# Reuse data/ATR loader from the established daily trend pilot.
base_path = Path(__file__).with_name('trend_atr_pilot.py')
ns = {'__name__':'trend_lib','__file__':str(base_path)}
exec(compile(base_path.read_text(encoding='utf-8'), str(base_path), 'exec'), ns)

SOURCES=ns['SOURCES']; load_data=ns['load_data']; wilder_atr=ns['wilder_atr']
START_DATE=pd.Timestamp('2023-01-01')
SMA_LEN=200; SLOPE_LOOKBACK=20; ATR_MULT=2.0; COST_ATR_RT=0.10
RISK=0.005  # operational 0.5% risk per trade


def run_side(symbol, df, side):
    df=df.copy()
    df['atr14']=wilder_atr(df)
    df['sma200']=df['close'].rolling(SMA_LEN).mean()
    df['slope']=df['sma200']-df['sma200'].shift(SLOPE_LOOKBACK)
    if side=='LONG':
        df['ok']=(df['close']>df['sma200']) & (df['slope']>0)
    else:
        df['ok']=(df['close']<df['sma200']) & (df['slope']<0)
    in_pos=False; entry=entry_atr=risk_px=stop=None; extreme=None; rows=[]
    for i in range(1,len(df)):
        row=df.iloc[i]; prev=df.iloc[i-1]; dt=row['datetime']
        if (not in_pos) and dt>=START_DATE and bool(prev['ok']) and pd.notna(prev['atr14']):
            entry=float(row['open']); entry_atr=float(prev['atr14']); risk_px=ATR_MULT*entry_atr
            stop=entry-risk_px if side=='LONG' else entry+risk_px
            extreme=entry; in_pos=True; entry_dt=dt
        if not in_pos: continue
        op=float(row['open']); hi=float(row['high']); lo=float(row['low'])
        hit = lo<=stop if side=='LONG' else hi>=stop
        if hit:
            if side=='LONG': xp=op if op<stop else stop; rg=(xp-entry)/risk_px
            else: xp=op if op>stop else stop; rg=(entry-xp)/risk_px
            cost_r=(COST_ATR_RT*entry_atr)/risk_px
            rows.append({'symbol':symbol,'side':side,'entry':entry_dt,'exit':dt,'r':rg-cost_r})
            in_pos=False; continue
        if side=='LONG':
            extreme=max(extreme,hi)
            if pd.notna(row['atr14']): stop=max(stop,extreme-ATR_MULT*float(row['atr14']))
        else:
            extreme=min(extreme,lo)
            if pd.notna(row['atr14']): stop=min(stop,extreme+ATR_MULT*float(row['atr14']))
    return pd.DataFrame(rows)


def summarize(df):
    if df.empty: return {'n':0}
    s=df['r'].astype(float); w=s[s>0]; l=s[s<0]; gp=float(w.sum()); gl=float(-l.sum())
    eq=(1+RISK*s).cumprod(); dd=eq/eq.cummax()-1
    return {'n':len(s),'win':100*float((s>0).mean()),'meanR':float(s.mean()),'medianR':float(s.median()),'netR':float(s.sum()),'PF':gp/gl if gl>0 else math.inf,'ret':100*float(eq.iloc[-1]-1),'mdd':100*float(dd.min())}

print('# LONG-ONLY vs LONG+SHORT TREND VALIDATION')
print('Frozen symmetric rule: prior close vs SMA200 + 20D slope sign; next open entry; 2ATR initial/trailing stop; cost=0.10 ATR RT; 0.5% risk/trade.')
print('Purpose: test whether adding a naive mirrored short sleeve improves the established long trend engine. This is pilot-period comparison, not untouched OOS.')
all_long=[]; all_short=[]
for symbol,url in SOURCES.items():
    df=load_data(url)
    L=run_side(symbol,df,'LONG'); S=run_side(symbol,df,'SHORT')
    all_long.append(L); all_short.append(S)
    print(symbol,'LONG',summarize(L))
    print(symbol,'SHORT',summarize(S))
L=pd.concat(all_long,ignore_index=True); S=pd.concat(all_short,ignore_index=True)
print('COMBINED_LONG',summarize(L))
print('COMBINED_SHORT',summarize(S))
print('NOTE: simple concatenation is trade-stream evidence, not a synchronized covariance-aware portfolio backtest.')
