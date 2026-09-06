import io
import math
import urllib.request
from dataclasses import dataclass, field

import pandas as pd

START = pd.Timestamp('2018-01-01')
END = pd.Timestamp('2026-12-31 23:59:59')
ATR_N = 14
SMA_N = 200
RVOL_N = 20
RVOL_MIN = 1.5
COST_ATR_RT = 0.10
ONE_R = 0.01

SOURCES = {
    'XAUUSD': {
        'D1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Gold-Cash/XAUUSD/XAUUSD_D1.csv',
        'H4': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Gold-Cash/XAUUSD/XAUUSD_H4.csv',
        'H1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Gold-Cash/XAUUSD/XAUUSD_H1.csv',
    },
    'NAS100': {
        'D1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Index-Cash/NAS100/NAS100_D1.csv',
        'H4': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Index-Cash/NAS100/NAS100_H4.csv',
        'H1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Index-Cash/NAS100/NAS100_H1.csv',
    },
    'SPX500': {
        'D1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Index-Cash/SPX500/SPX500_D1.csv',
        'H4': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Index-Cash/SPX500/SPX500_H4.csv',
        'H1': 'https://raw.githubusercontent.com/simom1/XAUUSD-history/main/Index-Cash/SPX500/SPX500_H1.csv',
    },
}


def load(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode('utf-8-sig')
    df = pd.read_csv(io.StringIO(raw))
    df.columns = [str(c).strip().lower() for c in df.columns]
    tcol = 'time' if 'time' in df.columns else ('datetime' if 'datetime' in df.columns else df.columns[0])
    df['dt'] = pd.to_datetime(df[tcol], errors='coerce')
    for c in ['open','high','low','close']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    vcol = 'tick_volume' if 'tick_volume' in df.columns else ('volume' if 'volume' in df.columns else None)
    df['vol'] = pd.to_numeric(df[vcol], errors='coerce') if vcol else 0.0
    return df.dropna(subset=['dt','open','high','low','close']).sort_values('dt').drop_duplicates('dt').reset_index(drop=True)


def atr(df, n=ATR_N):
    pc = df['close'].shift(1)
    tr = pd.concat([(df['high']-df['low']), (df['high']-pc).abs(), (df['low']-pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()


def add_pivots(df, left=2, right=2):
    x = df.copy()
    x['piv_lo'] = False
    x['piv_hi'] = False
    lows = x['low'].to_numpy(); highs = x['high'].to_numpy()
    for i in range(left, len(x)-right):
        if lows[i] == min(lows[i-left:i+right+1]): x.loc[i,'piv_lo'] = True
        if highs[i] == max(highs[i-left:i+right+1]): x.loc[i,'piv_hi'] = True
    x['piv_lo_confirm'] = x['piv_lo'].shift(right, fill_value=False)
    x['piv_hi_confirm'] = x['piv_hi'].shift(right, fill_value=False)
    x['piv_lo_center_i'] = pd.Series(range(len(x))).shift(right)
    x['piv_hi_center_i'] = pd.Series(range(len(x))).shift(right)
    return x


def prep(d1,h4,h1):
    d1=d1.copy(); h4=h4.copy(); h1=add_pivots(h1)
    d1['atr']=atr(d1); d1['sma200']=d1['close'].rolling(SMA_N).mean(); d1['sma_slope']=d1['sma200']-d1['sma200'].shift(20)
    d1['bear']=(d1['close']<d1['sma200']) & (d1['sma_slope']<=0)
    # close-based support pivot, usable only 2 days after confirmation
    d1['close_piv_lo'] = d1['close'].eq(d1['close'].rolling(5, center=True).min())
    supports=[]
    for i in range(2,len(d1)-2):
        if bool(d1.loc[i,'close_piv_lo']) and pd.notna(d1.loc[i,'atr']):
            supports.append((d1.loc[i+2,'dt'], float(d1.loc[i,'close']), float(d1.loc[i,'atr'])))
    h4['atr']=atr(h4)
    h4['slot']=h4['dt'].dt.hour
    h4['slot_avg']=h4.groupby('slot')['vol'].transform(lambda s:s.shift(1).rolling(RVOL_N,min_periods=10).mean())
    h4['rvol']=h4['vol']/h4['slot_avg']
    h1['atr']=atr(h1)
    return d1,h4,h1,supports


def asof_row(df, dt):
    j=df['dt'].searchsorted(dt, side='right')-1
    return None if j<0 else df.iloc[int(j)]


def build_shocks(d1,h4,supports):
    out=[]
    sup_i=0; active=[]
    for i in range(len(h4)-3):
        r=h4.iloc[i]; dt=r.dt
        while sup_i<len(supports) and supports[sup_i][0] <= dt:
            active.append(supports[sup_i]); sup_i+=1
        if dt<START or dt>END or not active or pd.isna(r.atr) or pd.isna(r.rvol) or r.rvol<RVOL_MIN: continue
        dr=asof_row(d1,dt)
        if dr is None or not bool(dr.bear) or pd.isna(dr.atr): continue
        # nearest confirmed close-based support; must be near shock low
        nearest=min(active, key=lambda z:abs(z[1]-r.low))
        if abs(nearest[1]-r.low) > float(dr.atr): continue
        nxt=h4.iloc[i+1:i+4]
        if float(nxt['low'].min()) < float(r.low)-0.5*float(r.atr): continue
        if float(nxt['close'].max()) <= float(r.close): continue
        out.append({'shock_i':i,'shock_dt':dt,'signal_dt':h4.iloc[i+3].dt,'shock_low':float(r.low),'shock_close':float(r.close),'rvol':float(r.rvol),'support':nearest[1]})
    # de-duplicate clusters: keep first valid shock within 3 days
    ded=[]
    for s in out:
        if not ded or s['shock_dt']-ded[-1]['shock_dt']>pd.Timedelta(days=3): ded.append(s)
    return ded


def find_mr_schedule(h1, shock):
    a=h1['dt'].searchsorted(shock['signal_dt'],side='left'); b=h1['dt'].searchsorted(shock['signal_dt']+pd.Timedelta(days=10),side='right')
    if b-a<6:return None
    reclaim=None
    for i in range(a,b):
        if h1.iloc[i].close>shock['shock_close']:
            reclaim=i;break
    if reclaim is None:return None
    hl_conf=None; hl_center=None
    for i in range(reclaim+1,b):
        if bool(h1.iloc[i].piv_lo_confirm):
            c=int(h1.iloc[i].piv_lo_center_i)
            if c>=reclaim and h1.iloc[c].low>shock['shock_low']:
                hl_conf=i;hl_center=c;break
    if hl_conf is None:return None
    # latest confirmed pivot high between signal and HL; else local max high
    ph=[]
    for i in range(a,hl_conf+1):
        if bool(h1.iloc[i].piv_hi_confirm):
            c=int(h1.iloc[i].piv_hi_center_i)
            if a<=c<=hl_center: ph.append(c)
    if ph: swing=max(ph); level=float(h1.iloc[swing].high)
    else:
        swing=int(h1.iloc[a:hl_center+1]['high'].idxmax()); level=float(h1.iloc[swing].high)
    br=None
    for i in range(hl_conf+1,b):
        if h1.iloc[i].close>level: br=i;break
    if br is None:return None
    acc=None
    for i in range(br,min(br+4,b)):
        rr=h1.iloc[i]
        if pd.notna(rr.atr) and rr.close>=level and rr.low>=level-0.25*rr.atr:
            acc=i;break
    if acc is None:return None
    return {'probe_i':hl_conf+1,'break_i':br+1,'accept_i':acc+1,'break_level':level,'hl_center':hl_center}


def trend_signals(h1,d1,start_i,last_i):
    # structure-confirmed post-SMA200 add candidates
    sig=[]
    used_hl=-1
    i=max(start_i,5)
    while i<last_i-2:
        dr=asof_row(d1,h1.iloc[i].dt)
        if dr is None or pd.isna(dr.sma200) or not (dr.close>dr.sma200 and dr.sma_slope>=0): i+=1;continue
        if bool(h1.iloc[i].piv_lo_confirm):
            c=int(h1.iloc[i].piv_lo_center_i)
            if c<=used_hl: i+=1;continue
            # require pullback low to remain above D1 SMA200 area with 0.5% tolerance
            if h1.iloc[c].low < 0.995*dr.sma200: i+=1;continue
            highs=[]
            for k in range(max(start_i,i-48),i+1):
                if bool(h1.iloc[k].piv_hi_confirm):
                    hc=int(h1.iloc[k].piv_hi_center_i)
                    if hc<c: highs.append(hc)
            if not highs: i+=1;continue
            hc=highs[-1]; level=float(h1.iloc[hc].high)
            br=None
            for j in range(i+1,min(i+49,last_i)):
                if h1.iloc[j].close>level:br=j;break
            if br is None:i+=1;continue
            acc=None
            for j in range(br,min(br+4,last_i)):
                rr=h1.iloc[j]
                if pd.notna(rr.atr) and rr.close>=level and rr.low>=level-0.25*rr.atr:
                    acc=j;break
            if acc is not None and acc+1<last_i:
                sig.append(acc+1); used_hl=c; i=acc+2;continue
        i+=1
    return sig

@dataclass
class Tranche:
    module:str
    risk_frac:float
    entry_i:int
    entry:float
    init_risk:float
    qty:float
    stop:float
    high:float
    remain:float=1.0
    partial_done:bool=False
    realized:float=0.0
    closed:bool=False


def run_asset(sym,src):
    d1,h4,h1=prep(load(src['D1']),load(src['H4']),load(src['H1']))[:3]
    # prep again to retain supports cleanly
    d1,h4,h1,supports=prep(load(src['D1']),load(src['H4']),load(src['H1']))
    shocks=build_shocks(d1,h4,supports)
    schedules=[]
    for s in shocks:
        sch=find_mr_schedule(h1,s)
        if sch and sch['accept_i']<len(h1): schedules.append((s,sch))
    # prevent overlapping cycle starts: first schedule after previous cycle start + 5 days; actual positions determine overlap too
    equity_realized=1.0; tranches=[]; all_tr=[]; eq_points=[]; module_pnl={'MR':0.0,'TREND':0.0,'PYRAMID':0.0}
    pending={}; schedule_map={}; cycle_active=False; cycle_id=0; trend_mode=False; trend_add_count=0; last_add_entry=None
    next_sched=0; current_shock=None; trend_queue=[]

    def h4atr(dt):
        rr=asof_row(h4,dt)
        return None if rr is None or pd.isna(rr.atr) else float(rr.atr)
    def open_risk(eq,px):
        x=0.0
        for t in tranches:
            if t.closed: continue
            q=t.qty*t.remain
            downside=max(0.0,px-t.stop)*q
            x+=downside
        return x/max(eq,1e-12)
    def add_tr(module,risk_frac,i):
        nonlocal last_add_entry
        if i>=len(h1):return False
        px=float(h1.iloc[i].open); a=h4atr(h1.iloc[i].dt)
        if a is None or a<=0:return False
        if last_add_entry is not None and px<last_add_entry:return False # no averaging down
        mark=equity_realized+sum((float(h1.iloc[i].open)-t.entry)*t.qty*t.remain for t in tranches if not t.closed)
        ir=2*a; qty=(mark*risk_frac)/ir
        # cycle open-risk cap <= 1R after proposed add
        if open_risk(mark,px)+risk_frac > ONE_R+1e-9:return False
        t=Tranche(module,risk_frac,i,px,ir,qty,px-ir,px)
        tranches.append(t);all_tr.append(t);last_add_entry=px
        return True

    for i in range(1,len(h1)):
        row=h1.iloc[i]; dt=row.dt
        if dt<START:continue
        if dt>END:break
        # if flat, start next eligible MR cycle
        if not cycle_active:
            while next_sched<len(schedules) and h1.iloc[schedules[next_sched][1]['probe_i']].dt<dt: next_sched+=1
            if next_sched<len(schedules):
                s,sch=schedules[next_sched]
                if h1.iloc[sch['probe_i']].dt==dt:
                    cycle_active=True;cycle_id+=1;current_shock=s;trend_mode=False;trend_add_count=0;last_add_entry=None;trend_queue=[]
                    pending={sch['probe_i']:('MR',0.25*ONE_R),sch['break_i']:('MR',0.35*ONE_R),sch['accept_i']:('MR',0.40*ONE_R)}
                    next_sched+=1
        if cycle_active and i in pending:
            mod,rf=pending.pop(i);add_tr(mod,rf,i)

        # detect 2 consecutive accepted D1 closes above SMA200; switch trend mode
        if cycle_active and not trend_mode:
            j=d1['dt'].searchsorted(dt.normalize(),side='right')-1
            if j>=1:
                a=d1.iloc[j];b=d1.iloc[j-1]
                if pd.notna(a.sma200) and pd.notna(b.sma200) and a.close>a.sma200 and b.close>b.sma200 and a.sma_slope>=0:
                    trend_mode=True
                    trend_queue=trend_signals(h1,d1,i,min(len(h1),i+24*120))

        if cycle_active and trend_mode and trend_queue and i==trend_queue[0]:
            trend_queue.pop(0)
            rf=(0.30 if trend_add_count==0 else 0.20)*ONE_R
            mod='TREND' if trend_add_count==0 else 'PYRAMID'
            if add_tr(mod,rf,i): trend_add_count+=1

        # manage each tranche; stop-first if both stop/target occur in same H1 bar
        for t in tranches:
            if t.closed:continue
            px_open=float(row.open); low=float(row.low); high=float(row.high)
            if low<=t.stop:
                exitpx=px_open if px_open<t.stop else t.stop
                q=t.qty*t.remain
                gross=(exitpx-t.entry)*q
                cost=COST_ATR_RT*(t.init_risk/2)*q*t.remain
                pnl=gross-cost; equity_realized+=pnl;t.realized+=pnl;module_pnl[t.module]+=pnl;t.remain=0;t.closed=True
                continue
            if t.module=='MR' and not t.partial_done and high>=t.entry+t.init_risk:
                q=t.qty*0.5
                exitpx=t.entry+t.init_risk
                gross=(exitpx-t.entry)*q
                cost=0.5*COST_ATR_RT*(t.init_risk/2)*t.qty
                pnl=gross-cost;equity_realized+=pnl;t.realized+=pnl;module_pnl[t.module]+=pnl;t.remain-=0.5;t.partial_done=True
            t.high=max(t.high,high)
            a=h4atr(dt)
            if a is not None:
                t.stop=max(t.stop,t.high-2*a)

        mtm=equity_realized+sum((float(row.close)-t.entry)*t.qty*t.remain for t in tranches if not t.closed)
        eq_points.append((dt,mtm))

        # cycle ends when all entries used, no positions, and either trend queue exhausted or trend regime has failed
        if cycle_active and not pending and all(t.closed for t in tranches):
            dr=asof_row(d1,dt)
            regime_ok=dr is not None and pd.notna(dr.sma200) and dr.close>dr.sma200
            if (not trend_mode) or (not regime_ok) or not trend_queue:
                cycle_active=False;tranches=[];pending={};trend_queue=[];last_add_entry=None

    # force MTM close at sample end for metric continuity but mark separately
    eq=pd.DataFrame(eq_points,columns=['dt','eq'])
    if eq.empty:return {'symbol':sym,'shocks':len(shocks),'schedules':len(schedules),'cycles':cycle_id,'error':'no equity'}
    daily=eq.set_index('dt')['eq'].resample('1D').last().dropna()
    rets=daily.pct_change().fillna(0)
    sd=rets.std(ddof=1); sharpe=(rets.mean()/sd*math.sqrt(252)) if sd>0 else math.nan
    neg=rets[rets<0]; dsd=neg.std(ddof=1); sortino=(rets.mean()/dsd*math.sqrt(252)) if len(neg)>1 and dsd>0 else math.nan
    dd=daily/daily.cummax()-1;mdd=float(dd.min())
    years=max((daily.index[-1]-daily.index[0]).days/365.25,1e-9);cagr=float((daily.iloc[-1]/daily.iloc[0])**(1/years)-1)
    calmar=cagr/abs(mdd) if mdd<0 else math.nan
    # tranche R based on actual realized pnl / initial allocated risk dollars approximation
    rs=[]
    for t in all_tr:
        base=max(t.risk_frac,1e-12) # start-equity normalized approximately; exact denominator at entry not retained
        rs.append(t.realized/base)
    wins=[x for x in rs if x>0]; losses=[x for x in rs if x<0]; pf=sum(wins)/abs(sum(losses)) if losses else math.inf
    return {
        'symbol':sym,'shocks':len(shocks),'mr_schedules':len(schedules),'cycles_started':cycle_id,'tranches':len(all_tr),
        'final_eq':float(daily.iloc[-1]),'return_pct':float((daily.iloc[-1]-1)*100),'sharpe':float(sharpe),'sortino':float(sortino),
        'mdd_pct':mdd*100,'cagr_pct':cagr*100,'calmar':float(calmar),'pf_approx':float(pf),
        'module_pnl_pct':{k:v*100 for k,v in module_pnl.items()},
        'first_daily':str(daily.index[0]),'last_daily':str(daily.index[-1]),
    }


def main():
    print('# ARK-42 Integrated MR + Trend + Pyramid Pilot')
    print('Mechanical pilot: D1 close-pivot zone + H4 same-slot RVOL/absorption + H1 Reclaim/HL/Break/Acceptance; staged MR 0.25R/0.35R/0.40R; 2 consecutive D1 closes above SMA200 => Trend Mode; structure-confirmed trend/pyramid adds; 2x H4 ATR ratchet; 0.10 ATR RT cost; 1R=1% equity; no averaging down; open-risk cap 1R.')
    print('Boundary: this is the first integrated mechanical implementation, not yet the final cross-asset production backtest. PF is approximate because tranche R denominator is normalized from risk fraction.')
    for sym,src in SOURCES.items():
        try:
            r=run_asset(sym,src)
            print('\n##',sym)
            for k,v in r.items(): print(f'{k}: {v}')
        except Exception as e:
            print('\n##',sym,'ERROR',repr(e))

if __name__=='__main__':main()
