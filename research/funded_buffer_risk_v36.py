from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import prop_evaluation_engine_v35 as v35

OUT = Path('research/results/funded_buffer_risk_v36')
OUT.mkdir(parents=True, exist_ok=True)

MODEL = 'TREND_ONLY'
TARGET_R = 1.5
START_BUFFERS = [0.0, 2.0, 5.0, 8.0]
STATIC_RISKS = [0.50, 0.75, 1.00, 1.25]
HORIZONS = [30, 60, 90]
SIMS = 20000
SEED = 6363
ABS_FLOOR = -10.0
PAYOUT_TARGET = 5.0
MAX_TRADES_PER_DAY = 1

# Candidate funded policy discussed with the user:
# protect immediately after funding, use 0.75% after a small buffer,
# and only test 1.0% once the buffer is substantial.
def adaptive_risk(equity_pct: float) -> float:
    if equity_pct < 2.0:
        return 0.50
    if equity_pct < 5.0:
        return 0.75
    return 1.00


def simulate_path(rvals, start_buffer, horizon_days, rng, static_risk=None):
    eq = float(start_buffer)
    peak = eq
    maxdd = 0.0
    breached = False
    hit_payout = False
    payout_day = np.nan
    below_start = False
    risks = []

    for day in range(1, horizon_days + 1):
        for _ in range(MAX_TRADES_PER_DAY):
            risk = float(static_risk if static_risk is not None else adaptive_risk(eq))
            risks.append(risk)
            r = float(rng.choice(rvals))
            eq += r * risk
            peak = max(peak, eq)
            maxdd = min(maxdd, eq - peak)
            if eq < start_buffer:
                below_start = True
            if not hit_payout and eq >= start_buffer + PAYOUT_TARGET:
                hit_payout = True
                payout_day = day
            if eq <= ABS_FLOOR:
                breached = True
                return {
                    'ending_equity_pct': eq,
                    'max_dd_pct': maxdd,
                    'breached': breached,
                    'hit_payout': hit_payout,
                    'payout_day': payout_day,
                    'retained_start_buffer': not below_start,
                    'avg_risk_pct': float(np.mean(risks)),
                }

    return {
        'ending_equity_pct': eq,
        'max_dd_pct': maxdd,
        'breached': breached,
        'hit_payout': hit_payout,
        'payout_day': payout_day,
        'retained_start_buffer': not below_start,
        'avg_risk_pct': float(np.mean(risks)),
    }


def evaluate_policy(rvals, start_buffer, horizon_days, rng, static_risk=None):
    rows = [simulate_path(rvals, start_buffer, horizon_days, rng, static_risk) for _ in range(SIMS)]
    df = pd.DataFrame(rows)
    payout_days = df.loc[df.hit_payout, 'payout_day'].dropna()
    return {
        'start_buffer_pct': start_buffer,
        'horizon_days': horizon_days,
        'policy': f'static_{static_risk:.2f}' if static_risk is not None else 'adaptive_0.50_0.75_1.00',
        'risk_pct': static_risk if static_risk is not None else np.nan,
        'breach_rate': float(df.breached.mean()),
        'survival_rate': float(1.0 - df.breached.mean()),
        'payout_hit_rate': float(df.hit_payout.mean()),
        'median_days_to_payout': float(np.median(payout_days)) if len(payout_days) else np.nan,
        'buffer_retention_rate': float(df.retained_start_buffer.mean()),
        'median_end_equity_pct': float(df.ending_equity_pct.median()),
        'p05_end_equity_pct': float(df.ending_equity_pct.quantile(.05)),
        'p05_max_dd_pct': float(df.max_dd_pct.quantile(.05)),
        'avg_risk_used_pct': float(df.avg_risk_pct.mean()),
    }


def main():
    trades = v35.build_oos_trades()
    g = trades[(trades.model == MODEL) & (trades.target_r == TARGET_R)].copy()
    rvals = g.r.to_numpy(dtype=float)
    base = {
        'n_oos_trades': int(len(g)),
        'win_rate': float((g.r > 0).mean()),
        'avg_r': float(g.r.mean()),
        'pf': float(g.loc[g.r > 0, 'r'].sum() / abs(g.loc[g.r < 0, 'r'].sum())),
    }

    rng = np.random.default_rng(SEED)
    out = []
    for buf in START_BUFFERS:
        for h in HORIZONS:
            for risk in STATIC_RISKS:
                out.append(evaluate_policy(rvals, buf, h, rng, static_risk=risk))
            out.append(evaluate_policy(rvals, buf, h, rng, static_risk=None))

    df = pd.DataFrame(out)
    # Primary funded objective: maximize payout probability and survival, penalize deep tail DD.
    df['funded_score'] = (
        0.50 * df.payout_hit_rate
        + 0.35 * df.survival_rate
        + 0.15 * df.buffer_retention_rate
        + 0.01 * df.p05_max_dd_pct
    )
    df = df.sort_values(['horizon_days', 'start_buffer_pct', 'funded_score'], ascending=[True, True, False])
    df.to_csv(OUT / 'buffer_risk_comparison.csv', index=False)

    meta = {
        'purpose': 'funded-account buffer/risk sizing proxy after passing evaluation',
        'model': MODEL,
        'target_r': TARGET_R,
        'start_buffers_pct': START_BUFFERS,
        'static_risks_pct': STATIC_RISKS,
        'adaptive_policy': {'buffer_lt_2': 0.50, 'buffer_2_to_5': 0.75, 'buffer_ge_5': 1.00},
        'horizons_days': HORIZONS,
        'absolute_floor_pct': ABS_FLOOR,
        'payout_target_above_start_buffer_pct': PAYOUT_TARGET,
        'sims': SIMS,
        'base_oos': base,
        'limitations': [
            'daily-bar US-stock OOS proxy, not actual swing CFD execution',
            'iid bootstrap does not preserve serial correlation, simultaneous exposure, gaps, or floating equity',
            'one sampled trade per day is a normalization for policy comparison, not a forecast of trade frequency',
            'absolute floor is modeled as -10% from starting account equity; firm-specific funded rules may differ',
            'payout target is a research threshold, not a firm rule',
            'survivorship bias remains in the current-stock universe',
        ],
    }
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))

    print('=== FUNDED BUFFER RISK v3.6 ===')
    print(json.dumps(meta, indent=2))
    print('\nTOP POLICY BY BUFFER/HORIZON')
    top = df.sort_values('funded_score', ascending=False).groupby(['start_buffer_pct', 'horizon_days'], as_index=False).head(1)
    print(top.sort_values(['start_buffer_pct', 'horizon_days']).to_string(index=False))
    print('\nFULL COMPARISON')
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
