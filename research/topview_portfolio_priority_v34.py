from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.topview_sector_stock_10y_v33 import (
    download, add_features, stock_sector_map, simulate, period, PERIODS,
)

OUT = Path("research/results/topview_portfolio_priority_v34")
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["TREND_ONLY", "SECTOR_LEADER", "SECTOR_STOCK_LEADER", "FULL_TOPVIEW"]
RISK_PER_TRADE = 0.005
MAX_OPEN_POSITIONS = 4
MAX_PER_SECTOR = 1
INITIAL_EQUITY = 100_000.0


def build_trades(feat):
    smap = stock_sector_map()
    stocks = [s for s in smap if s in feat]
    rows = []
    for model in MODELS:
        for s in stocks:
            try:
                ts = simulate(feat, s, model, "FIXED_3R")
            except Exception:
                continue
            for t in ts:
                p = period(t.entry_date)
                if p:
                    rows.append(t.__dict__ | {"period": p})
    return pd.DataFrame(rows)


def mtm_r(row, date, feat):
    d = feat[row.symbol]
    if date < row.entry_date:
        return 0.0
    if date >= row.exit_date:
        return float(row.r)
    if date not in d.index:
        ix = d.index.searchsorted(date, side="right") - 1
        if ix < 0:
            return 0.0
        px = float(d.iloc[ix].Close)
    else:
        px = float(d.loc[date, "Close"])
    risk = float(row.entry - row.stop)
    if risk <= 0:
        return 0.0
    return (px - float(row.entry)) / risk


def run_portfolio(trades, feat, model, p):
    a, b = PERIODS[p]
    pool = trades[(trades.model == model) & (trades.period == p)].copy()
    if pool.empty:
        return None, pd.DataFrame()
    pool = pool.sort_values(["entry_date", "symbol"]).reset_index(drop=True)
    dates = pd.date_range(a, b, freq="D")

    accepted = []
    open_rows = []
    cursor = 0
    realized_equity = INITIAL_EQUITY
    curve = []

    for dt in dates:
        # Close positions whose exit date is today or earlier.
        still = []
        for r in open_rows:
            if r.exit_date <= dt:
                realized_equity *= (1.0 + RISK_PER_TRADE * float(r.r))
            else:
                still.append(r)
        open_rows = still

        # Add new positions at their precomputed next-open entry, with portfolio heat controls.
        while cursor < len(pool) and pool.loc[cursor, "entry_date"] <= dt:
            r = pool.loc[cursor]
            cursor += 1
            if r.entry_date != dt:
                continue
            sector_count = sum(1 for x in open_rows if x.sector == r.sector)
            if len(open_rows) >= MAX_OPEN_POSITIONS or sector_count >= MAX_PER_SECTOR:
                continue
            open_rows.append(r)
            accepted.append(r)

        floating_r = sum(mtm_r(r, dt, feat) for r in open_rows)
        equity = realized_equity * (1.0 + RISK_PER_TRADE * floating_r)
        curve.append((dt, equity, realized_equity, len(open_rows), floating_r))

    c = pd.DataFrame(curve, columns=["date", "equity", "realized_equity", "open_n", "floating_r"])
    peak = c.equity.cummax()
    dd = c.equity / peak - 1.0
    ret = c.equity.pct_change().fillna(0)
    ann = (c.equity.iloc[-1] / c.equity.iloc[0]) ** (365.25 / max((c.date.iloc[-1]-c.date.iloc[0]).days,1)) - 1.0
    vol = ret.std(ddof=1) * np.sqrt(365.25)
    sharpe = ann / vol if vol > 0 else np.nan
    acc = pd.DataFrame(accepted)
    stats = {
        "model": model,
        "period": p,
        "accepted_trades": int(len(acc)),
        "candidate_trades": int(len(pool)),
        "accept_rate": float(len(acc)/len(pool)),
        "final_equity": float(c.equity.iloc[-1]),
        "total_return": float(c.equity.iloc[-1]/INITIAL_EQUITY - 1.0),
        "annualized_return": float(ann),
        "max_drawdown": float(dd.min()),
        "daily_vol": float(vol),
        "return_over_dd": float(ann/abs(dd.min())) if dd.min() < 0 else np.nan,
        "daily_sharpe_proxy": float(sharpe) if np.isfinite(sharpe) else np.nan,
        "max_concurrent": int(c.open_n.max()),
        "avg_concurrent": float(c.open_n.mean()),
    }
    return stats, c


def main():
    data = download()
    feat = add_features(data)
    trades = build_trades(feat)
    trades.to_csv(OUT / "candidate_trades.csv", index=False)

    rows = []
    curves = []
    for p in ["DEV", "VAL", "OOS"]:
        for m in MODELS:
            s, c = run_portfolio(trades, feat, m, p)
            if s:
                rows.append(s)
                c["model"] = m
                c["period"] = p
                curves.append(c)

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "portfolio_summary.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(OUT / "equity_curves.csv", index=False)

    oos = res[res.period == "OOS"].sort_values("return_over_dd", ascending=False)
    meta = {
        "risk_per_trade": RISK_PER_TRADE,
        "max_open_positions": MAX_OPEN_POSITIONS,
        "max_per_sector": MAX_PER_SECTOR,
        "initial_equity": INITIAL_EQUITY,
        "selection": "No OOS tuning; same fixed portfolio constraints for all models",
        "survivorship_bias": True,
        "proprietary_bigview": False,
        "oos_best_return_over_dd": None if oos.empty else oos.iloc[0].to_dict(),
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("=== TOPVIEW PORTFOLIO PRIORITY v3.4 ===")
    print(json.dumps(meta, indent=2, default=str))
    print("\nSUMMARY")
    print(res.sort_values(["period", "return_over_dd"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
