from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import dukascopy_python as duka
from dukascopy_python.instruments import INSTRUMENT_FX_METALS_XAU_USD

OUT = Path("research/results/gold_30m_longrun_entry_ablation_v42")
OUT.mkdir(parents=True, exist_ok=True)

YEARS = 4
CHUNK_DAYS = 180
LOOKBACK = 20
RETEST_WINDOW = 6
RETEST_TOL_ATR = 0.30
ATR_N = 20
MAX_HOLD_BARS = 48
COST_BPS_PER_SIDE = 3.0

MODELS = {
    "BREAKOUT_IMMEDIATE": "30m breakout immediate next-bar entry",
    "BREAKOUT_RETEST": "30m breakout + retest/reclaim within 6 bars",
}


def fetch_xauusd_15m(start: datetime, end: datetime) -> pd.DataFrame:
    parts = []
    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=CHUNK_DAYS), end)
        print(f"FETCH {cur.date()} -> {nxt.date()}", flush=True)
        df = duka.fetch(
            instrument=INSTRUMENT_FX_METALS_XAU_USD,
            interval=duka.INTERVAL_MIN_15,
            offer_side=duka.OFFER_SIDE_BID,
            start=cur,
            end=nxt,
            max_retries=4,
        )
        if df is not None and not df.empty:
            parts.append(df.copy())
        cur = nxt
        time.sleep(0.25)
    if not parts:
        raise RuntimeError("Dukascopy returned no XAUUSD data")
    d = pd.concat(parts).sort_index()
    d = d[~d.index.duplicated(keep="last")]
    if not isinstance(d.index, pd.DatetimeIndex):
        if "timestamp" in d.columns:
            d = d.set_index("timestamp")
        d.index = pd.to_datetime(d.index, utc=True)
    else:
        d.index = pd.to_datetime(d.index, utc=True)
    cols = {str(c).lower(): c for c in d.columns}
    required = ["open", "high", "low", "close", "volume"]
    if not all(k in cols for k in required):
        raise RuntimeError(f"Missing OHLCV columns: {list(d.columns)}")
    d = d[[cols[k] for k in required]].copy()
    d.columns = ["Open", "High", "Low", "Close", "Volume"]
    return d


def to_30m(d15: pd.DataFrame) -> pd.DataFrame:
    d = d15.resample("30min", label="right", closed="right").agg(
        Open=("Open", "first"),
        High=("High", "max"),
        Low=("Low", "min"),
        Close=("Close", "last"),
        Volume=("Volume", "sum"),
    ).dropna(subset=["Open", "High", "Low", "Close"])
    pc = d.Close.shift(1)
    tr = pd.concat([
        d.High - d.Low,
        (d.High - pc).abs(),
        (d.Low - pc).abs(),
    ], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(ATR_N).mean()
    d["sma50"] = d.Close.rolling(50).mean()
    d["sma200"] = d.Close.rolling(200).mean()
    d["mom10h"] = d.Close.pct_change(20)
    d["mom48h"] = d.Close.pct_change(96)
    return d.dropna(subset=["atr20", "sma50", "sma200", "mom10h", "mom48h"])


def period_labels(idx: pd.DatetimeIndex) -> pd.Series:
    start = idx.min()
    end = idx.max()
    span = end - start
    dev_end = start + span * 0.50
    val_end = start + span * 0.75
    out = pd.Series(index=idx, dtype="object")
    out.loc[idx < dev_end] = "DEV"
    out.loc[(idx >= dev_end) & (idx < val_end)] = "VAL"
    out.loc[idx >= val_end] = "OOS"
    return out


def find_retest(d: pd.DataFrame, i: int, side: int, level: float, av: float) -> int | None:
    end = min(len(d) - 2, i + RETEST_WINDOW)
    for j in range(i + 1, end + 1):
        hi = float(d.High.iloc[j])
        lo = float(d.Low.iloc[j])
        cl = float(d.Close.iloc[j])
        if side > 0:
            touched = lo <= level + RETEST_TOL_ATR * av
            reclaimed = cl >= level
            if touched and reclaimed:
                return j
        else:
            touched = hi >= level - RETEST_TOL_ATR * av
            reclaimed = cl <= level
            if touched and reclaimed:
                return j
    return None


def candidate_events(d: pd.DataFrame) -> pd.DataFrame:
    labels = period_labels(d.index)
    rows = []
    warm = max(LOOKBACK, 200, 96)
    for i in range(warm, len(d) - RETEST_WINDOW - 3):
        m10 = float(d.mom10h.iloc[i])
        m48 = float(d.mom48h.iloc[i])
        score = 0.5 * m10 + 0.5 * m48
        if not np.isfinite(score) or abs(score) < 1e-12:
            continue
        side = 1 if score > 0 else -1
        # Same regime gate for both entry models; isolates entry timing.
        if side > 0 and float(d.Close.iloc[i]) <= float(d.sma200.iloc[i]):
            continue
        if side < 0 and float(d.Close.iloc[i]) >= float(d.sma200.iloc[i]):
            continue
        prev = d.iloc[i - LOOKBACK:i]
        if side > 0:
            level = float(prev.High.max())
            breakout = float(d.Close.iloc[i]) > level
        else:
            level = float(prev.Low.min())
            breakout = float(d.Close.iloc[i]) < level
        if not breakout:
            continue
        av = float(d.atr20.iloc[i])
        if not np.isfinite(av) or av <= 0:
            continue
        rj = find_retest(d, i, side, level, av)
        rows.append({
            "signal_i": i,
            "signal_time": d.index[i],
            "period": labels.iloc[i],
            "side": side,
            "score": score,
            "breakout_level": level,
            "atr": av,
            "retest_i": rj,
        })
    return pd.DataFrame(rows)


def simulate_one(d: pd.DataFrame, e, model: str):
    i = int(e.signal_i)
    side = int(e.side)
    if model == "BREAKOUT_RETEST":
        if pd.isna(e.retest_i):
            return None
        trigger_i = int(e.retest_i)
    else:
        trigger_i = i
    ei = trigger_i + 1
    if ei >= len(d):
        return None
    entry = float(d.Open.iloc[ei])
    av = float(d.atr20.iloc[trigger_i])
    if not np.isfinite(entry) or not np.isfinite(av) or av <= 0:
        return None
    if side > 0:
        swing = float(d.Low.iloc[max(0, trigger_i - 19):trigger_i + 1].min())
        stop = min(swing, entry - av)
        risk = entry - stop
    else:
        swing = float(d.High.iloc[max(0, trigger_i - 19):trigger_i + 1].max())
        stop = max(swing, entry + av)
        risk = stop - entry
    if risk <= 0 or risk / entry > 0.08:
        return None

    mfe, mae = 0.0, 0.0
    xi = None
    px = None
    reason = None
    last = min(len(d) - 1, ei + MAX_HOLD_BARS)
    for j in range(ei, last + 1):
        hi, lo = float(d.High.iloc[j]), float(d.Low.iloc[j])
        if side > 0:
            mfe = max(mfe, (hi - entry) / risk)
            mae = min(mae, (lo - entry) / risk)
            if lo <= stop:
                xi, px, reason = j, stop, "STOP"
                break
            if j >= ei + 2 and float(d.Close.iloc[j - 1]) < float(d.sma50.iloc[j - 1]) and float(d.Close.iloc[j - 2]) < float(d.sma50.iloc[j - 2]):
                xi, px, reason = j, float(d.Open.iloc[j]), "30M_SMA50_FAIL"
                break
        else:
            mfe = max(mfe, (entry - lo) / risk)
            mae = min(mae, (entry - hi) / risk)
            if hi >= stop:
                xi, px, reason = j, stop, "STOP"
                break
            if j >= ei + 2 and float(d.Close.iloc[j - 1]) > float(d.sma50.iloc[j - 1]) and float(d.Close.iloc[j - 2]) > float(d.sma50.iloc[j - 2]):
                xi, px, reason = j, float(d.Open.iloc[j]), "30M_SMA50_FAIL"
                break
    if xi is None:
        xi, px, reason = last, float(d.Close.iloc[last]), "TIME"
    gross = side * (px - entry) / risk
    cost = (2 * COST_BPS_PER_SIDE / 10000.0) * entry / risk
    return {
        "model": model,
        "period": e.period,
        "signal_time": e.signal_time,
        "entry_time": d.index[ei],
        "exit_time": d.index[xi],
        "side": "LONG" if side > 0 else "SHORT",
        "entry": entry,
        "stop": stop,
        "r": gross - cost,
        "mfe_r": mfe,
        "mae_r": mae,
        "hold_bars": xi - ei,
        "exit_reason": reason,
    }


def simulate(d: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        busy_until = None
        for _, e in events.sort_values("signal_time").iterrows():
            if busy_until is not None and pd.Timestamp(e.signal_time) <= busy_until:
                continue
            q = simulate_one(d, e, model)
            if q is None:
                continue
            rows.append(q)
            busy_until = pd.Timestamp(q["exit_time"])
    return pd.DataFrame(rows)


def metrics(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"n": 0}
    r = g.r.astype(float)
    wins = r[r > 0]
    losses = r[r < 0]
    eq = r.cumsum()
    dd = eq - eq.cummax()
    sd = r.std(ddof=1)
    return {
        "n": int(len(r)),
        "win_rate": float((r > 0).mean()),
        "avg_r": float(r.mean()),
        "pf": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else np.nan,
        "total_r": float(r.sum()),
        "max_dd_r": float(dd.min()),
        "trade_sharpe": float(r.mean() / sd * math.sqrt(len(r))) if sd > 0 else np.nan,
        "avg_mfe_r": float(g.mfe_r.mean()),
        "avg_mae_r": float(g.mae_r.mean()),
        "avg_hold_bars": float(g.hold_bars.mean()),
    }


def main():
    end = datetime.now(timezone.utc) - timedelta(days=2)
    start = end - timedelta(days=365 * YEARS)
    d15 = fetch_xauusd_15m(start, end)
    d = to_30m(d15)
    events = candidate_events(d)
    trades = simulate(d, events)

    rows = []
    for (m, p), g in trades.groupby(["model", "period"]):
        rows.append({"model": m, "description": MODELS[m], "period": p, **metrics(g)})
    summary = pd.DataFrame(rows)

    d.to_csv(OUT / "xauusd_30m.csv")
    events.to_csv(OUT / "events.csv", index=False)
    trades.to_csv(OUT / "trades.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    meta = {
        "purpose": "Gold-only long-run 30m entry-timing ablation",
        "data_source": "Dukascopy XAUUSD BID 15m resampled to 30m",
        "years": YEARS,
        "split": "chronological 50% DEV / 25% VAL / 25% OOS",
        "models": MODELS,
        "shared_signal": "20-bar breakout in 10h/48h momentum direction, aligned with 30m SMA200",
        "retest": f"within {RETEST_WINDOW} bars, tolerance {RETEST_TOL_ATR} ATR",
        "cost_bps_per_side": COST_BPS_PER_SIDE,
        "limitations": [
            "Spot XAUUSD BID proxy, not a specific CFD prop-firm execution feed.",
            "Spread/slippage modeled as fixed bps rather than timestamp-level quotes.",
            "Entry-ablation only; no prop-firm daily loss or trailing drawdown rules here.",
            "Single-asset result must not be generalized to all markets without separate validation.",
        ],
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    print("BARS", len(d), d.index.min(), d.index.max())
    print("EVENTS", len(events), "TRADES", len(trades))
    print(summary.sort_values(["period", "avg_r"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
