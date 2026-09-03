from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

import prop_pass_speed_v34 as v34
import topview_sector_stock_10y_v33 as v33

OUT = Path('research/results/prop_evaluation_v35')
OUT.mkdir(parents=True, exist_ok=True)

# Focus on the configurations that matter for the user's objective:
# pass speed + consistency, not maximum raw expectancy.
CONFIGS = [
    ('TREND_ONLY', 3.0, 0.50),
    ('TREND_ONLY', 2.0, 0.50),
    ('TREND_ONLY', 1.5, 0.50),
    ('TREND_ONLY', 1.5, 0.75),
    ('TREND_ONLY', 1.0, 0.75),
]

PHASES = [('P1', 10.0), ('P2', 5.0)]
ABS_MAX_LOSS = -10.0
INTERNAL_DAY_STOP = -1.0
MAX_TRADES_PER_DAY = 2
SIMS = 20000
MAX_DAYS_PER_PHASE = 180
SEED = 5353


def build_oos_trades():
    data = v33.download()
    feat = v33.add_features(data)
    smap = v33.stock_sector_map()
    stocks = [s for s in smap if s in feat and v33.SECTORS[smap[s]]['etf'] in feat]
    needed = sorted(set((m, t) for m, t, _ in CONFIGS))
    rows = []
    for model, target_r in needed:
        for s in stocks:
            rows.extend(v34.simulate_fixed(feat, s, model, target_r))
    df = pd.DataFrame(rows).dropna(subset=['period'])
    return df[df.period == 'OOS'].copy()


def one_phase(rvals, target_pct, risk_pct, rng):
    equity = 0.0
    peak = 0.0
    maxdd = 0.0
    days = 0
    trades = 0
    max_loss_streak = 0
    loss_streak = 0
    daily_stop_hits = 0

    while days < MAX_DAYS_PER_PHASE:
        days += 1
        day_pnl = 0.0
        for _ in range(MAX_TRADES_PER_DAY):
            r = float(rng.choice(rvals))
            pnl = r * risk_pct
            equity += pnl
            day_pnl += pnl
            trades += 1
            peak = max(peak, equity)
            maxdd = min(maxdd, equity - peak)

            if r < 0:
                loss_streak += 1
                max_loss_streak = max(max_loss_streak, loss_streak)
            else:
                loss_streak = 0

            if equity >= target_pct:
                return 'pass', days, trades, maxdd, max_loss_streak, daily_stop_hits
            if equity <= ABS_MAX_LOSS:
                return 'fail', days, trades, maxdd, max_loss_streak, daily_stop_hits

            # Internal survival rule: once the day's realized loss reaches -1%, stop for the day.
            if day_pnl <= INTERNAL_DAY_STOP:
                daily_stop_hits += 1
                break

    return 'timeout', days, trades, maxdd, max_loss_streak, daily_stop_hits


def evaluate(rvals, risk_pct, rng):
    completed = failed = timed_out = 0
    total_days = []
    total_trades = []
    maxdds = []
    streaks = []
    day_stops = []

    for _ in range(SIMS):
        sim_days = sim_trades = sim_day_stops = 0
        sim_maxdd = 0.0
        sim_streak = 0
        ok = True
        for _, target in PHASES:
            outcome, days, trades, mdd, streak, dstop = one_phase(rvals, target, risk_pct, rng)
            sim_days += days
            sim_trades += trades
            sim_day_stops += dstop
            sim_maxdd = min(sim_maxdd, mdd)
            sim_streak = max(sim_streak, streak)
            if outcome == 'fail':
                failed += 1
                ok = False
                break
            if outcome == 'timeout':
                timed_out += 1
                ok = False
                break
        if ok:
            completed += 1
            total_days.append(sim_days)
            total_trades.append(sim_trades)
        maxdds.append(sim_maxdd)
        streaks.append(sim_streak)
        day_stops.append(sim_day_stops)

    return {
        'two_phase_pass_rate': completed / SIMS,
        'fail_rate': failed / SIMS,
        'timeout_rate': timed_out / SIMS,
        'median_days_to_complete': float(np.median(total_days)) if total_days else np.nan,
        'p75_days_to_complete': float(np.quantile(total_days, .75)) if total_days else np.nan,
        'median_trades_to_complete': float(np.median(total_trades)) if total_trades else np.nan,
        'p05_max_dd_pct': float(np.quantile(maxdds, .05)),
        'median_max_loss_streak': float(np.median(streaks)),
        'p95_max_loss_streak': float(np.quantile(streaks, .95)),
        'median_internal_day_stop_hits': float(np.median(day_stops)),
    }


def main():
    df = build_oos_trades()
    rng = np.random.default_rng(SEED)
    rows = []
    for model, target_r, risk_pct in CONFIGS:
        g = df[(df.model == model) & (df.target_r == target_r)]
        base = v34.stats(g.sort_values('entry_date'))
        res = evaluate(g.r.values, risk_pct, rng)
        rows.append({'model': model, 'target_r': target_r, 'risk_pct': risk_pct, **base, **res})

    out = pd.DataFrame(rows)
    # Higher completion, lower failures, fewer days, shallower tail DD.
    out['evaluation_score'] = (
        out.two_phase_pass_rate - 2.0 * out.fail_rate
        - 0.0015 * out.median_days_to_complete.fillna(MAX_DAYS_PER_PHASE * 2)
        + 0.015 * out.p05_max_dd_pct
    )
    out = out.sort_values('evaluation_score', ascending=False)
    out.to_csv(OUT / 'comparison.csv', index=False)
    meta = {
        'purpose': 'two-phase prop evaluation proxy for pass speed + consistency',
        'phases': PHASES,
        'absolute_max_loss_pct': ABS_MAX_LOSS,
        'internal_day_stop_pct': INTERNAL_DAY_STOP,
        'max_trades_per_day': MAX_TRADES_PER_DAY,
        'sims': SIMS,
        'max_days_per_phase': MAX_DAYS_PER_PHASE,
        'limitations': [
            'daily-bar stock proxy; not FTMO intraday CFD fills',
            'bootstrap samples trades iid and does not preserve correlation/concurrency',
            'internal day-stop is modeled on realized sampled trades, not floating equity',
            'this is not an official FTMO pass-probability estimate',
            'survivorship bias remains in the current-stock universe',
        ],
    }
    (OUT / 'meta.json').write_text(json.dumps(meta, indent=2))
    print('=== PROP EVALUATION ENGINE v3.5 ===')
    print(json.dumps(meta, indent=2))
    print('\nCOMPARISON')
    print(out.to_string(index=False))


if __name__ == '__main__':
    main()
