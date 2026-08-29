"""v2.4 Risk Scaling experiment on frozen v2.3 trade outcomes.

Question: does 0.5% base risk -> 1.0% after a successful trade -> reset to
0.5% after a losing trade improve account outcomes versus fixed 0.5% risk?

This file changes POSITION SIZING ONLY. It does not optimize entry signals.
OOS is verification-only and must not be used to select a sizing model.
"""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

import inflection_effort_result_v23 as v23
import mtf_htf_zone_ltf_confirm_v16 as base
import mtf_htf_zone_ltf_confirm_v161 as causal

OUT = Path('research/results/risk_scaling_antimartingale_v24')
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_EQUITY = 100_000.0
RISK_MODELS = {
    'FIXED_050': {'base': 0.005, 'win': 0.005},
    'WIN_075_RESET_050': {'base': 0.005, 'win': 0.0075},
    'WIN_100_RESET_050': {'base': 0.005, 'win': 0.0100},
}
STAGE = 'RETEST_LTF'
EXIT_MODEL = 'PURE_SWING'


def build_trades(name):
    m15, m1, daily, prov = causal.load_asset_causal(name)
    spikes = v23.spike_events(daily)
    failure, disp = v23.classify_failure_and_displacement(spikes, daily)
    c = v23.retest_candidates(disp, daily, m15)
    tr = v23.run_model(c, m1, EXIT_MODEL)
    if not tr.empty:
        tr = tr.sort_values('entry_time').copy()
        tr['asset'] = name
        tr['split'] = tr.entry_time.map(base.split)
    return tr, prov


def apply_model(trades, model_name, initial_equity=INITIAL_EQUITY):
    cfg = RISK_MODELS[model_name]
    equity = float(initial_equity)
    risk_frac = cfg['base']
    rows = []
    for _, r in trades.sort_values(['entry_time', 'asset']).iterrows():
        net_r = float(r.net_R)
        equity_before = equity
        risk_dollars = equity_before * risk_frac
        pnl = risk_dollars * net_r
        equity = equity_before + pnl
        success = net_r > 0.0
        rows.append({
            'asset': r.asset, 'entry_time': r.entry_time, 'exit_time': r.exit_time,
            'split': r.split, 'net_R': net_r, 'model': model_name,
            'risk_fraction': risk_frac, 'risk_dollars': risk_dollars,
            'equity_before': equity_before, 'pnl_dollars': pnl, 'equity_after': equity,
            'success': success,
        })
        risk_frac = cfg['win'] if success else cfg['base']
    return pd.DataFrame(rows)


def metrics(x):
    if x.empty:
        return {'trades': 0}
    pnl = x.pnl_dollars.astype(float)
    eq = pd.concat([pd.Series([INITIAL_EQUITY]), x.equity_after.reset_index(drop=True)], ignore_index=True)
    peak = eq.cummax()
    dd_d = eq - peak
    dd_pct = eq / peak - 1.0
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    return {
        'trades': len(x),
        'win_rate': float((pnl > 0).mean()),
        'total_pnl': float(pnl.sum()),
        'return_pct': float((x.equity_after.iloc[-1] / INITIAL_EQUITY - 1.0) * 100.0),
        'ending_equity': float(x.equity_after.iloc[-1]),
        'profit_factor_dollars': float(wins.sum() / abs(losses.sum())) if losses.sum() < 0 else np.nan,
        'max_dd_dollars': float(abs(dd_d.min())),
        'max_dd_pct': float(abs(dd_pct.min()) * 100.0),
        'avg_risk_dollars': float(x.risk_dollars.mean()),
        'max_risk_dollars': float(x.risk_dollars.max()),
    }


def transition_stats(trades):
    """Test whether wins cluster: P(next win|win) versus P(next win|loss)."""
    z = trades.sort_values(['entry_time', 'asset']).copy()
    if len(z) < 2:
        return {}
    z['win'] = z.net_R.astype(float) > 0
    z['prev_win'] = z.win.shift(1)
    q = z.dropna(subset=['prev_win'])
    after_win = q[q.prev_win == True]
    after_loss = q[q.prev_win == False]
    return {
        'p_win_next_given_win': float(after_win.win.mean()) if len(after_win) else np.nan,
        'n_after_win': int(len(after_win)),
        'p_win_next_given_loss': float(after_loss.win.mean()) if len(after_loss) else np.nan,
        'n_after_loss': int(len(after_loss)),
    }


def main():
    names = [x.strip() for x in os.environ.get('ASSETS', 'NQ_PROXY,SP500_PROXY,WTI,GOLD,NATGAS').split(',') if x.strip()]
    all_trades = []
    provenance = {}
    for name in names:
        tr, prov = build_trades(name)
        provenance[name] = prov
        if not tr.empty:
            all_trades.append(tr)
    if not all_trades:
        raise RuntimeError('No trades produced')
    trades = pd.concat(all_trades, ignore_index=True).sort_values(['entry_time', 'asset'])
    trades.to_csv(OUT / 'frozen_trade_sequence.csv', index=False)

    summary = []
    for model in RISK_MODELS:
        sim = apply_model(trades, model)
        sim.to_csv(OUT / f'trades_{model}.csv', index=False)
        for split in ['DEV', 'VAL', 'OOS', 'ALL']:
            # Re-run each split from the same 100k starting equity so split comparisons are interpretable.
            subset = trades if split == 'ALL' else trades[trades.split == split]
            sx = apply_model(subset, model)
            m = metrics(sx)
            m.update({'model': model, 'split': split})
            summary.append(m)

    trans = transition_stats(trades)
    pd.DataFrame(summary).to_csv(OUT / 'metrics.csv', index=False)
    (OUT / 'meta.json').write_text(json.dumps({
        'version': '2.4-risk-scaling',
        'stage': STAGE,
        'exit_model': EXIT_MODEL,
        'initial_equity': INITIAL_EQUITY,
        'risk_models': RISK_MODELS,
        'transition_stats_all': trans,
        'assets_requested': names,
        'provenance': provenance,
        'selection_rule': 'Compare on DEV/VAL only. OOS is verification-only; prior OOS inspection means final proof requires new forward holdout.',
        'important_limit': 'This inherits v2.3 signal-generation semantics. It does not cure any causal/lookahead defect in v2.3; sizing conclusions remain provisional until the signal engine is patched and rerun.',
    }, indent=2, default=str))
    print(pd.DataFrame(summary).to_markdown(index=False))
    print('\nTransition stats:', json.dumps(trans, indent=2))


if __name__ == '__main__':
    main()
