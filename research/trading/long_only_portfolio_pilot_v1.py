import math
import pandas as pd
from pathlib import Path

# Reuse the executed NDX/SPX long-only trend engine.
base = Path(__file__).with_name('trend_atr_pilot.py')
ns = {'__name__':'trend_lib','__file__':str(base)}
exec(compile(base.read_text(encoding='utf-8'), str(base), 'exec'), ns)

SOURCES = ns['SOURCES']
load_data = ns['load_data']
run_backtest = ns['run_backtest']
START_DATE = ns['START_DATE']

# FROZEN PORTFOLIO-RISK PILOT PARAMETERS
BASE_RISK = 0.005            # 0.5% equity per trade
PORTFOLIO_OPEN_RISK_CAP = 0.04
EQUITY_CLUSTER_CAP = 0.01    # NDX + SPX together <= 1.0% initial stop-risk
COST_ATR_RT = 0.10
TRADING_DAYS = 252

# DD throttle from prior design discussion
# <2%: 1.0x; 2-3%: .75x; 3-4%: .50x; >=4%: no new risk

def dd_mult(dd_abs):
    if dd_abs >= 0.04:
        return 0.0
    if dd_abs >= 0.03:
        return 0.50
    if dd_abs >= 0.02:
        return 0.75
    return 1.0

# Fetch + run each standalone engine to obtain no-lookahead closed trades.
data = {}
trades = []
for symbol, url in SOURCES.items():
    df = load_data(url)
    data[symbol] = df.set_index('datetime')
    res = run_backtest(symbol, df)
    tdf = res['trades'].copy()
    if not tdf.empty:
        for _, t in tdf.iterrows():
            trades.append({
                'symbol': symbol,
                'entry_date': pd.Timestamp(t['entry_date']),
                'exit_date': pd.Timestamp(t['exit_date']),
                'entry': float(t['entry']),
                'exit': float(t['exit']),
                'entry_atr': float(t['entry_atr']),
                'initial_risk_px': float(t['initial_risk']),
                'r_gross': float(t['r_gross']),
            })

trades = sorted(trades, key=lambda x: (x['entry_date'], x['symbol']))

# Unified business-day calendar using observed bars (union, not synthetic calendar).
calendar = sorted(set().union(*[set(df.index[df.index >= START_DATE]) for df in data.values()]))
if not calendar:
    raise RuntimeError('empty test calendar')

# Event maps
entries = {}
exits = {}
for i,t in enumerate(trades):
    entries.setdefault(t['entry_date'], []).append((i,t))
    exits.setdefault(t['exit_date'], []).append((i,t))

cash = 1.0
peak = 1.0
positions = {}  # trade id -> dict
curve = []
ledger = []
rejected = []
max_open_risk = 0.0
max_cluster_risk = 0.0

for dt in calendar:
    # Mark existing positions at current close first; missing symbol bars carry prior mark.
    mtm = cash
    for tid,p in positions.items():
        sdf = data[p['symbol']]
        if dt in sdf.index:
            px = float(sdf.loc[dt,'close'])
            p['last_px'] = px
        else:
            px = p['last_px']
        mtm += p['units'] * px
    equity_pre = mtm
    peak = max(peak, equity_pre)
    dd_abs = max(0.0, 1.0 - equity_pre/peak)

    # Exit at engine-defined exit price and charge 0.10 ATR RT.
    for tid,t in exits.get(dt, []):
        if tid not in positions:
            continue
        p = positions.pop(tid)
        proceeds = p['units'] * t['exit']
        cost = p['units'] * COST_ATR_RT * t['entry_atr']
        cash += proceeds - cost
        pnl = (t['exit'] - t['entry']) * p['units'] - cost
        ledger.append({**t,'risk_frac':p['risk_frac'],'pnl_equity_frac':pnl/p['entry_equity']})

    # Recompute equity after exits at same day before new entries.
    mtm = cash
    for p in positions.values():
        mtm += p['units'] * p['last_px']
    equity = mtm
    peak = max(peak, equity)
    dd_abs = max(0.0, 1.0 - equity/peak)
    mult = dd_mult(dd_abs)

    # Existing initial-stop risk budget is the risk fraction committed at entry.
    open_risk = sum(p['risk_frac'] for p in positions.values())
    cluster_risk = open_risk  # all instruments in this pilot are equity-index cluster

    for tid,t in entries.get(dt, []):
        if tid in positions:
            continue
        risk_frac = BASE_RISK * mult
        if risk_frac <= 0:
            rejected.append((dt,t['symbol'],'DD_STOP'))
            continue
        if open_risk + risk_frac > PORTFOLIO_OPEN_RISK_CAP + 1e-12:
            rejected.append((dt,t['symbol'],'PORTFOLIO_CAP'))
            continue
        if cluster_risk + risk_frac > EQUITY_CLUSTER_CAP + 1e-12:
            rejected.append((dt,t['symbol'],'CLUSTER_CAP'))
            continue
        # Allocate units so entry-to-initial-stop loss equals risk_frac * current equity.
        risk_dollars = equity * risk_frac
        units = risk_dollars / t['initial_risk_px']
        notional = units * t['entry']
        if notional > cash:
            # This pilot is cash-account long-only; scale units down to available cash.
            units = cash / t['entry']
            actual_risk_dollars = units * t['initial_risk_px']
            risk_frac = actual_risk_dollars / equity
            notional = units * t['entry']
        if units <= 0:
            rejected.append((dt,t['symbol'],'NO_CASH'))
            continue
        cash -= notional
        positions[tid] = {
            'symbol':t['symbol'],'units':units,'last_px':t['entry'],
            'risk_frac':risk_frac,'entry_equity':equity,
        }
        open_risk += risk_frac
        cluster_risk += risk_frac

    max_open_risk = max(max_open_risk, open_risk)
    max_cluster_risk = max(max_cluster_risk, cluster_risk)

    # End-day mark.
    mtm = cash
    for p in positions.values():
        sdf = data[p['symbol']]
        if dt in sdf.index:
            p['last_px'] = float(sdf.loc[dt,'close'])
        mtm += p['units'] * p['last_px']
    curve.append((dt, mtm, open_risk, dd_abs, len(positions)))

cdf = pd.DataFrame(curve, columns=['date','equity','open_risk','dd_pre','positions']).set_index('date')
ret = cdf['equity'].pct_change().fillna(0.0)
std = float(ret.std(ddof=1))
sharpe = float(ret.mean()/std*math.sqrt(TRADING_DAYS)) if std>0 else float('nan')
down = ret.clip(upper=0.0)
dev = float(math.sqrt((down.pow(2)).mean()))
sortino = float(ret.mean()/dev*math.sqrt(TRADING_DAYS)) if dev>0 else float('nan')
dd = cdf['equity']/cdf['equity'].cummax()-1.0
mdd = float(dd.min())
years = max((cdf.index[-1]-cdf.index[0]).days/365.25, 1/365.25)
cagr = float((cdf['equity'].iloc[-1]/cdf['equity'].iloc[0])**(1/years)-1)
calmar = cagr/abs(mdd) if mdd<0 else float('nan')

ldf = pd.DataFrame(ledger)
if len(ldf):
    wins = ldf.loc[ldf['pnl_equity_frac']>0,'pnl_equity_frac'].sum()
    losses = -ldf.loc[ldf['pnl_equity_frac']<0,'pnl_equity_frac'].sum()
    pf = wins/losses if losses>0 else float('inf')
else:
    pf = float('nan')

print('# ARK-42 LONG-ONLY PORTFOLIO PILOT v1')
print('Scope: NDX + SPX long momentum only; cash otherwise. MR/VCP is NOT included in this pilot yet.')
print('Rules: standalone no-lookahead SMA200+slope+2ATR trend engine, 0.5% base risk, 0.10ATR RT cost.')
print('Risk controls: equity-cluster initial stop-risk <=1.0%, portfolio hard cap <=4%, DD throttle 2/3/4%.')
print(f'window={cdf.index[0].date()} -> {cdf.index[-1].date()}')
print(f'closed_trades_executed={len(ldf)} rejected_entries={len(rejected)}')
print(f'final_return_pct={(cdf.equity.iloc[-1]-1)*100:+.3f}')
print(f'CAGR_pct={cagr*100:+.3f} Sharpe={sharpe:.3f} Sortino={sortino:.3f} MDD_pct={mdd*100:.3f} Calmar={calmar:.3f} PF_equityPnl={pf:.3f}')
print(f'max_open_stop_risk_pct={max_open_risk*100:.3f} max_equity_cluster_risk_pct={max_cluster_risk*100:.3f}')
print(f'days_at_or_above_2pct_DD={(dd<=-0.02).sum()} days_at_or_above_3pct_DD={(dd<=-0.03).sum()} days_at_or_above_4pct_DD={(dd<=-0.04).sum()}')
if rejected:
    rdf=pd.DataFrame(rejected,columns=['date','symbol','reason'])
    print('rejections=',rdf['reason'].value_counts().to_dict())
print('executed_by_symbol=',ldf['symbol'].value_counts().to_dict() if len(ldf) else {})
