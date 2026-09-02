from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START = "2016-01-01"
END = "2026-09-01"
BENCH = "SPY"
SYMBOLS = [
    "SPY", "QQQ", "IWM", "DIA", "XLK", "SMH", "XLE", "XLF", "XLI", "XLB",
    "XLV", "XLY", "XLP", "XLU", "XLRE", "GLD", "GDX", "SLV"
]
TRADE_SYMBOLS = [s for s in SYMBOLS if s != BENCH]
COST_BPS_PER_SIDE = 5.0
SEED = 42
OUT = Path("research/results/bigview_ma_v32")
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = {
    "DEV": (pd.Timestamp("2016-01-01"), pd.Timestamp("2021-12-31")),
    "VAL": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
    "OOS": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-08-31")),
}


def download() -> dict[str, pd.DataFrame]:
    raw = yf.download(
        SYMBOLS, start=START, end=END, auto_adjust=True, actions=False,
        group_by="ticker", threads=True, progress=False
    )
    out: dict[str, pd.DataFrame] = {}
    for s in SYMBOLS:
        if isinstance(raw.columns, pd.MultiIndex):
            if s not in raw.columns.get_level_values(0):
                continue
            df = raw[s].copy()
        else:
            df = raw.copy()
        df.columns = [str(c).title() for c in df.columns]
        needed = ["Open", "High", "Low", "Close", "Volume"]
        if not set(needed).issubset(df.columns):
            continue
        df = df[needed].dropna(subset=["Open", "High", "Low", "Close"]).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if len(df) >= 500:
            out[s] = df
    if BENCH not in out:
        raise RuntimeError("SPY benchmark data unavailable")
    return out


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    pc = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - pc).abs(),
        (df["Low"] - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_features(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    spy = data[BENCH]["Close"]
    spy20 = spy.pct_change(20)
    spy60 = spy.pct_change(60)
    frames = {}
    tsi_parts = []

    for s, df0 in data.items():
        df = df0.copy()
        c = df["Close"]
        df["ema10"] = c.ewm(span=10, adjust=False).mean()
        df["ema20"] = c.ewm(span=20, adjust=False).mean()
        df["sma50"] = c.rolling(50).mean()
        df["sma200"] = c.rolling(200).mean()
        df["atr14"] = atr(df, 14)
        df["mom20"] = c.pct_change(20)
        df["mom60"] = c.pct_change(60)
        df["mom126"] = c.pct_change(126)
        df["tsi_raw"] = 0.2 * df["mom20"] + 0.35 * df["mom60"] + 0.45 * df["mom126"]
        # TP-P proxy: medium-term price leadership vs SPY.
        df["tpp"] = (0.4 * df["mom20"] + 0.6 * df["mom60"]) - (0.4 * spy20 + 0.6 * spy60)
        # TP-V proxy: signed volume participation, normalized by 20d mean volume.
        signed = np.sign(c.diff()).fillna(0.0) * df["Volume"].fillna(0.0)
        df["tpv"] = signed.rolling(20).sum() / df["Volume"].rolling(20).sum().replace(0, np.nan)
        df["trend_stack"] = (c > df["sma50"]) & (df["sma50"] > df["sma200"]) & (df["ema10"] > df["ema20"])
        df["rm10_proxy"] = (df["ema10"] > df["ema20"]) & (c > df["ema20"])
        df["rm50_proxy"] = (df["sma50"] > df["sma200"]) & (c > df["sma50"])
        # Pullback: within last 3 bars price touched EMA20, and today reclaimed EMA10 while stack remains bullish.
        touch = (df["Low"] <= (df["ema20"] + 0.20 * df["atr14"])) & (df["High"] >= (df["ema20"] - 0.20 * df["atr14"]))
        df["pullback_reclaim"] = touch.rolling(3).max().fillna(0).astype(bool) & (c > df["ema10"]) & (c.shift(1) <= df["ema10"].shift(1))
        frames[s] = df
        tsi_parts.append(df["tsi_raw"].rename(s))

    tsi = pd.concat(tsi_parts, axis=1)
    tsi_rank = tsi.rank(axis=1, pct=True)
    for s in frames:
        frames[s]["tsi_pct"] = tsi_rank[s]
        frames[s]["scanner_a"] = (
            frames[s]["trend_stack"] &
            frames[s]["rm10_proxy"] & frames[s]["rm50_proxy"] &
            (frames[s]["tsi_pct"] >= 0.70) &
            (frames[s]["tpp"] > 0) &
            (frames[s]["tpv"] > 0)
        )
    return frames


@dataclass
class Trade:
    symbol: str
    model: str
    exit_model: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry: float
    stop: float
    exit: float
    r: float
    mfe_r: float
    mae_r: float
    holding_days: int


def simulate_symbol(df: pd.DataFrame, symbol: str, model: str, exit_model: str) -> list[Trade]:
    if model == "TREND_ONLY":
        signal = df["trend_stack"]
    elif model == "SCANNER_IMMEDIATE":
        signal = df["scanner_a"]
    elif model == "SCANNER_PULLBACK":
        signal = df["scanner_a"] & df["pullback_reclaim"]
    else:
        raise ValueError(model)

    trades: list[Trade] = []
    i = 210
    n = len(df)
    while i < n - 2:
        if not bool(signal.iloc[i]):
            i += 1
            continue
        entry_i = i + 1
        entry = float(df["Open"].iloc[entry_i])
        swing = float(df["Low"].iloc[max(0, i-9):i+1].min())
        atrv = float(df["atr14"].iloc[i])
        if not np.isfinite(atrv) or atrv <= 0:
            i += 1
            continue
        # Structural stop candidate; ensure minimum 1 ATR room to avoid tiny denominator artifacts.
        stop = min(swing, entry - atrv)
        risk = entry - stop
        if risk <= 0 or risk / entry > 0.12:
            i += 1
            continue

        exit_price = None
        exit_i = None
        max_high = entry
        min_low = entry
        j = entry_i
        while j < n:
            hi = float(df["High"].iloc[j]); lo = float(df["Low"].iloc[j]); cl = float(df["Close"].iloc[j])
            max_high = max(max_high, hi); min_low = min(min_low, lo)
            stop_hit = lo <= stop
            target2 = entry + 2.0 * risk
            target3 = entry + 3.0 * risk
            if stop_hit:
                exit_price = stop
                exit_i = j
                break
            if exit_model == "FIXED_2R" and hi >= target2:
                exit_price = target2; exit_i = j; break
            if exit_model == "FIXED_3R" and hi >= target3:
                exit_price = target3; exit_i = j; break
            if exit_model == "EMA20_TRAIL":
                # Exit next day open after daily close loses EMA20, no lookahead at same close.
                if cl < float(df["ema20"].iloc[j]) and j + 1 < n:
                    exit_price = float(df["Open"].iloc[j+1]); exit_i = j+1; break
            if j - entry_i >= 60:
                exit_price = cl; exit_i = j; break
            j += 1
        if exit_i is None or exit_price is None:
            break

        gross_r = (exit_price - entry) / risk
        # Conservative round-trip cost expressed in R.
        cost = (2 * COST_BPS_PER_SIDE / 10000.0) * entry / risk
        net_r = gross_r - cost
        trades.append(Trade(
            symbol, model, exit_model, df.index[i], df.index[entry_i], df.index[exit_i],
            entry, stop, float(exit_price), float(net_r),
            float((max_high-entry)/risk), float((min_low-entry)/risk), int(exit_i-entry_i)
        ))
        i = max(exit_i + 1, i + 1)
    return trades


def summarize(g: pd.DataFrame) -> dict:
    if len(g) == 0:
        return {"n": 0}
    wins = g.loc[g.r > 0, "r"]
    losses = g.loc[g.r < 0, "r"]
    eq = g.r.cumsum()
    dd = eq - eq.cummax()
    pf = wins.sum() / abs(losses.sum()) if len(losses) and losses.sum() != 0 else np.nan
    std = g.r.std(ddof=1)
    return {
        "n": int(len(g)),
        "win_rate": float((g.r > 0).mean()),
        "avg_r": float(g.r.mean()),
        "median_r": float(g.r.median()),
        "avg_win_r": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_r": float(losses.mean()) if len(losses) else np.nan,
        "profit_factor": float(pf) if np.isfinite(pf) else np.nan,
        "trade_sharpe": float(g.r.mean()/std*math.sqrt(len(g))) if std and np.isfinite(std) else np.nan,
        "max_dd_r": float(dd.min()),
        "avg_mfe_r": float(g.mfe_r.mean()),
        "avg_mae_r": float(g.mae_r.mean()),
        "avg_hold_days": float(g.holding_days.mean()),
        "total_r": float(g.r.sum()),
    }


def period_name(ts: pd.Timestamp) -> str | None:
    for p, (a,b) in PERIODS.items():
        if a <= ts <= b:
            return p
    return None


def mc_prop(oos: pd.DataFrame, risk_pct: float = 0.005, sims: int = 10000, ntrades: int = 100) -> dict:
    # Simplified trade-sequence prop stress: ignores correlation, intraday equity, gaps, and daily-loss clock.
    rvals = oos["r"].dropna().to_numpy()
    if len(rvals) < 20:
        return {"n_oos": int(len(rvals)), "status": "INSUFFICIENT"}
    rng = np.random.default_rng(SEED)
    pass_count = fail_count = 0
    maxdds = []
    finals = []
    for _ in range(sims):
        seq = rng.choice(rvals, size=ntrades, replace=True)
        equity = 0.0
        peak = 0.0
        maxdd = 0.0
        state = "OPEN"
        for rr in seq:
            equity += rr * risk_pct
            peak = max(peak, equity)
            maxdd = min(maxdd, equity-peak)
            if equity >= 0.10:
                state = "PASS"; pass_count += 1; break
            if equity <= -0.10:
                state = "FAIL"; fail_count += 1; break
        maxdds.append(maxdd); finals.append(equity)
    return {
        "n_oos": int(len(rvals)), "risk_pct": risk_pct, "sims": sims, "trades_horizon": ntrades,
        "pass_before_fail_rate": pass_count/sims,
        "fail_before_pass_rate": fail_count/sims,
        "unfinished_rate": 1-(pass_count+fail_count)/sims,
        "median_final_return": float(np.median(finals)),
        "p05_max_drawdown": float(np.quantile(maxdds, 0.05)),
    }


def main():
    data = add_features(download())
    models = ["TREND_ONLY", "SCANNER_IMMEDIATE", "SCANNER_PULLBACK"]
    exits = ["FIXED_2R", "FIXED_3R", "EMA20_TRAIL"]
    all_trades = []
    for s in TRADE_SYMBOLS:
        if s not in data:
            continue
        for m in models:
            for e in exits:
                all_trades.extend(simulate_symbol(data[s], s, m, e))
    t = pd.DataFrame([x.__dict__ for x in all_trades])
    if t.empty:
        raise RuntimeError("No trades generated")
    t["period"] = t["entry_date"].map(period_name)
    t.to_csv(OUT / "trades.csv", index=False)

    rows = []
    for (m,e,p), g in t.dropna(subset=["period"]).groupby(["model","exit_model","period"]):
        row = {"model":m,"exit_model":e,"period":p}
        row.update(summarize(g.sort_values("entry_date")))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "summary.csv", index=False)

    oos = t[t.period == "OOS"].copy()
    mc = {}
    for (m,e), g in oos.groupby(["model","exit_model"]):
        mc[f"{m}|{e}"] = {
            "risk_0.25pct": mc_prop(g, 0.0025),
            "risk_0.50pct": mc_prop(g, 0.0050),
            "risk_0.75pct": mc_prop(g, 0.0075),
        }
    (OUT / "mc_prop.json").write_text(json.dumps(mc, indent=2), encoding="utf-8")

    # Selection rule is predeclared: choose by VAL avg_r, require DEV and VAL avg_r > 0 and n>=30 each.
    piv = summary.pivot_table(index=["model","exit_model"], columns="period", values=["avg_r","n","profit_factor"])
    candidates = []
    for idx in piv.index:
        try:
            dev_avg = float(piv.loc[idx, ("avg_r","DEV")]); val_avg = float(piv.loc[idx, ("avg_r","VAL")])
            dev_n = float(piv.loc[idx, ("n","DEV")]); val_n = float(piv.loc[idx, ("n","VAL")])
        except Exception:
            continue
        if np.isfinite(dev_avg) and np.isfinite(val_avg) and dev_avg > 0 and val_avg > 0 and dev_n >= 30 and val_n >= 30:
            candidates.append((val_avg, idx))
    selected = max(candidates)[1] if candidates else None

    verdict = {"selected_on_validation_only": selected, "selection_candidates": len(candidates), "oos_touched_for_selection": False}
    if selected is not None:
        m,e = selected
        o = summary[(summary.model==m)&(summary.exit_model==e)&(summary.period=="OOS")]
        if len(o):
            rr = o.iloc[0].to_dict()
            verdict["oos_result"] = rr
            verdict["oos_pass_basic"] = bool(rr.get("avg_r", -999) > 0 and rr.get("profit_factor", 0) > 1 and rr.get("n",0) >= 30)
    (OUT / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str), encoding="utf-8")

    print("=== BIGVIEW MA 10Y PROXY BACKTEST v3.2 ===")
    print(f"Data: {START} to {END}; symbols loaded={len(data)}; trades={len(t)}")
    print("IMPORTANT: BigView metrics are PUBLIC-DATA PROXIES, not proprietary historical MarketGauge values.")
    print(summary.sort_values(["period","avg_r"], ascending=[True,False]).to_string(index=False))
    print("\nVALIDATION-ONLY SELECTED:", selected)
    print(json.dumps(verdict, indent=2, default=str))
    print("\nOOS MONTE CARLO (simplified prop sequence model):")
    print(json.dumps(mc, indent=2))


if __name__ == "__main__":
    main()
