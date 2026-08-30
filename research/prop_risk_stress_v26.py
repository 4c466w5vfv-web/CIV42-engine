"""v2.6 Conservative Prop Risk Stress Engine.

Consumes the frozen v2.4 trade sequence. It does NOT claim to repair v2.3 signal
lookahead. Purpose: compare risk policies under explicit execution/tail shocks.
All shock parameters are scenario multipliers or R-addons, NOT claimed broker facts.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

IN = Path('research/results/risk_scaling_antimartingale_v24/frozen_trade_sequence.csv')
OUT = Path('research/results/prop_risk_stress_v26'); OUT.mkdir(parents=True, exist_ok=True)
INITIAL = 100000.0
MODELS = {
 'FIXED_025': {'base':.0025,'win':.0025},
 'FIXED_050': {'base':.005,'win':.005},
 'WIN_100_RESET_050': {'base':.005,'win':.01},
}
# Additive adverse execution shock in R. These are stress scenarios, not empirical FXIFY estimates.
SHOCKS = {'BASE':0.0,'EXEC_2X_PROXY':0.05,'EXEC_3X_PROXY':0.10,'EXEC_5X_PROXY':0.20,'TAIL_GAP_PROXY':0.50}
# Research-only conservative internal de-risk ladder.
DD_LADDER = [(0.04,0.0),(0.03,.0025),(0.02,.0035),(0.0,None)]


def risk_after_dd(base_risk, equity, peak):
    dd = 1-equity/peak
    if dd >= .04: return 0.0
    if dd >= .03: return min(base_risk,.0025)
    if dd >= .02: return min(base_risk,.0035)
    return base_risk


def simulate(tr, model, shock_r=0.0, derisk=False):
    cfg=MODELS[model]; eq=INITIAL; peak=INITIAL; rf=cfg['base']; rows=[]
    day_start=INITIAL; current_day=None; stopped_day=False
    for _,r in tr.sort_values(['entry_time','asset']).iterrows():
        day=pd.Timestamp(r.entry_time).date()
        if day != current_day:
            current_day=day; day_start=eq; stopped_day=False
        use=rf
        if derisk: use=risk_after_dd(use,eq,peak)
        # Internal daily stop: once realized daily equity loss reaches 1%, stop new trades that day.
        if eq <= day_start*.99: stopped_day=True
        if stopped_day or use<=0: continue
        stressed_r=float(r.net_R)-shock_r
        before=eq; risk_d=before*use; pnl=risk_d*stressed_r; eq=before+pnl; peak=max(peak,eq)
        success=stressed_r>0
        rows.append({'asset':r.asset,'entry_time':r.entry_time,'net_R_raw':r.net_R,'net_R_stressed':stressed_r,
                     'risk_fraction':use,'risk_dollars':risk_d,'pnl':pnl,'equity':eq,'peak':peak,
                     'daily_loss_pct':max(0.0,1-eq/day_start),'dd_pct':max(0.0,1-eq/peak)})
        rf=cfg['win'] if success else cfg['base']
    return pd.DataFrame(rows)


def metrics(x):
    if x.empty:return {'trades':0}
    pnl=x.pnl; wins=pnl[pnl>0]; losses=pnl[pnl<=0]
    return {'trades':len(x),'return_pct':(x.equity.iloc[-1]/INITIAL-1)*100,
            'pf':wins.sum()/abs(losses.sum()) if losses.sum()<0 else np.nan,
            'max_dd_pct':x.dd_pct.max()*100,'worst_realized_daily_loss_pct':x.daily_loss_pct.max()*100,
            'max_risk_dollars':x.risk_dollars.max(),'ending_equity':x.equity.iloc[-1]}


def monte_carlo(tr, model, shock_r, derisk, n=10000, seed=42):
    # Resample frozen trade R outcomes. This estimates sequence sensitivity only;
    # it does not model serial dependence, exact gaps, or broker liquidation mechanics.
    rng=np.random.default_rng(seed); rs=tr.net_R.astype(float).to_numpy(); breach4=0; breach10=0; finals=[]
    for _ in range(n):
        sample=tr.copy(); sample['net_R']=rng.choice(rs,size=len(rs),replace=True)
        z=simulate(sample,model,shock_r,derisk)
        if z.empty: continue
        breach4 += bool((z.daily_loss_pct>=.04).any())
        breach10 += bool((z.dd_pct>=.10).any())
        finals.append(z.equity.iloc[-1])
    return {'mc_paths':n,'p_realized_daily_4pct_breach':breach4/n,'p_static_10pct_dd_breach':breach10/n,
            'median_ending_equity':float(np.median(finals)) if finals else np.nan,
            'p05_ending_equity':float(np.quantile(finals,.05)) if finals else np.nan}


def main():
    if not IN.exists(): raise FileNotFoundError(f'{IN} missing; run v2.4 first')
    tr=pd.read_csv(IN,parse_dates=['entry_time','exit_time'])
    summary=[]
    for model in MODELS:
      for s,shock in SHOCKS.items():
       for derisk in [False,True]:
        z=simulate(tr,model,shock,derisk); m=metrics(z); m.update({'model':model,'shock':s,'shock_R':shock,'derisk':derisk})
        # MC only on BASE and TAIL to keep CI runtime bounded.
        if s in ('BASE','TAIL_GAP_PROXY'): m.update(monte_carlo(tr,model,shock,derisk))
        summary.append(m)
    df=pd.DataFrame(summary); df.to_csv(OUT/'summary.csv',index=False)
    (OUT/'meta.json').write_text(json.dumps({'version':'2.6','initial_equity':INITIAL,'models':MODELS,'shocks':SHOCKS,
      'internal_daily_stop':'1% realized daily equity loss; research rule, not broker rule',
      'dd_derisk':'2%=>cap .35%, 3%=>cap .25%, 4%=>stop',
      'limits':['shock R values are synthetic stress assumptions, not measured spreads/slippage/gaps',
                'MC is iid bootstrap of observed R and does not preserve serial/cross-asset correlation',
                'inherits frozen v2.4/v2.3 trades and therefore inherits known signal lookahead contamination',
                'prop breach must eventually be simulated from intraday equity/floating PnL and exact current program rules']},indent=2))
    print(df.to_markdown(index=False))

if __name__=='__main__': main()
