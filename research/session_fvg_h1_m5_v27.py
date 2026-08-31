"""Session High/Low -> H1 FVG -> M5 execution v2.7.

Research-only causal backtest for EURUSD/GOLD/WTI/BRENT.
Signal logic is frozen in session_fvg_h1_m5_v27_spec.md.
This is not a claim of live edge and does not model exact FXIFY execution.
"""
from __future__ import annotations

import json, math, os
from pathlib import Path
import numpy as np
import pandas as pd

import mtf_crossasset_v13 as duka
import mtf_backtest_v12 as core

OUT = Path("research/results/session_fvg_h1_m5_v27")
OUT.mkdir(parents=True, exist_ok=True)

ASSETS = {
    "EURUSD": "eurusd",
    "GOLD": "xauusd",
    "WTI": "lightcmdusd",
    "BRENT": "brentcmdusd",
}

ATR_N = 14
FVG_ATR_MIN = 0.10
M5_BOS_LOOKBACK = 4
H1_CONFIRM_HOURS = 12
M5_ENTRY_HOURS = 6
REFERENCE_HOURS = 18
COST_R = 0.05  # synthetic research baseline, NOT measured broker cost
TP_LEVELS = (2.0, 3.0, 5.0)


def available_at_close(df: pd.DataFrame, delta: pd.Timedelta) -> pd.DataFrame:
    x = df.copy()
    x.index = pd.DatetimeIndex(x.index) + delta
    x.index.name = "datetime"
    return x.sort_index()


def load_asset(name: str):
    inst = ASSETS[name]
    base = Path("research/duka_v27") / name
    p5 = duka.run_cli(inst, "m5", "2020-01-01", "2026-08-01", base / "m5")
    p1h = duka.run_cli(inst, "h1", "2019-01-01", "2026-08-01", base / "h1")
    pd1 = duka.run_cli(inst, "d1", "2007-01-01", "2026-08-01", base / "d1")
    m5 = available_at_close(duka.read_duka(p5), pd.Timedelta(minutes=5))
    h1 = available_at_close(duka.read_duka(p1h), pd.Timedelta(hours=1))
    d1 = available_at_close(duka.read_duka(pd1), pd.Timedelta(days=1))
    prov = {"source": "Dukascopy", "instrument": inst, "timing": "bar-open labels shifted to close availability"}
    return m5, h1, d1, prov


def daily_regime(d1: pd.DataFrame) -> pd.Series:
    x = d1.copy()
    sma = x.close.rolling(200, min_periods=200).mean()
    slope = sma - sma.shift(20)
    reg = pd.Series(0, index=x.index, dtype=int)
    reg[(x.close > sma) & (slope > 0)] = 1
    reg[(x.close < sma) & (slope < 0)] = -1
    return reg


def session_mask(idx: pd.DatetimeIndex, session: str) -> pd.Series:
    utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    if session == "ASIA":
        h = utc.hour + utc.minute / 60.0
        return (h >= 0) & (h < 8)
    if session == "LONDON":
        z = utc.tz_convert("Europe/London")
        h = z.hour + z.minute / 60.0
        return (h >= 8) & (h < 17)
    if session == "NEW_YORK":
        z = utc.tz_convert("America/New_York")
        h = z.hour + z.minute / 60.0
        return (h >= 8) & (h < 17)
    raise ValueError(session)


def session_date(idx: pd.DatetimeIndex, session: str) -> pd.Index:
    utc = idx.tz_convert("UTC") if idx.tz is not None else idx.tz_localize("UTC")
    if session == "ASIA": z = utc
    elif session == "LONDON": z = utc.tz_convert("Europe/London")
    else: z = utc.tz_convert("America/New_York")
    return pd.Index(z.date)


def completed_sessions(m5: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sess in ("ASIA", "LONDON", "NEW_YORK"):
        mask = session_mask(m5.index, sess)
        x = m5.loc[mask].copy()
        if x.empty: continue
        x["session_date"] = session_date(x.index, sess)
        for d, g in x.groupby("session_date"):
            if len(g) < 12:  # reject obviously incomplete windows
                continue
            rows.append({
                "session": sess, "session_date": str(d),
                "session_start": g.index.min(), "session_end": g.index.max(),
                "session_high": float(g.high.max()), "session_low": float(g.low.min()),
            })
    return pd.DataFrame(rows).sort_values("session_end") if rows else pd.DataFrame()


def prep_h1(h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    x = h1.copy()
    x["atr"] = core.atr(x, ATR_N)
    x["bull_fvg_lo"] = x.high.shift(2)
    x["bull_fvg_hi"] = x.low
    x["bear_fvg_lo"] = x.high
    x["bear_fvg_hi"] = x.low.shift(2)
    x["bull_fvg"] = (x.low > x.high.shift(2)) & ((x.low - x.high.shift(2)) >= FVG_ATR_MIN * x.atr)
    x["bear_fvg"] = (x.high < x.low.shift(2)) & ((x.low.shift(2) - x.high) >= FVG_ATR_MIN * x.atr)
    reg = daily_regime(d1).rename("regime")
    return pd.merge_asof(x.reset_index().sort_values("datetime"), reg.reset_index().sort_values("datetime"), on="datetime", direction="backward").set_index("datetime")


def prep_m5(m5: pd.DataFrame) -> pd.DataFrame:
    x = m5.copy()
    x["prev_hi"] = x.high.shift(1).rolling(M5_BOS_LOOKBACK, min_periods=M5_BOS_LOOKBACK).max()
    x["prev_lo"] = x.low.shift(1).rolling(M5_BOS_LOOKBACK, min_periods=M5_BOS_LOOKBACK).min()
    x["bull_bos"] = (x.close > x.prev_hi) & (x.close > x.open)
    x["bear_bos"] = (x.close < x.prev_lo) & (x.close < x.open)
    return x


def candidates(m5: pd.DataFrame, h1: pd.DataFrame, d1: pd.DataFrame) -> pd.DataFrame:
    sessions = completed_sessions(m5)
    if sessions.empty: return pd.DataFrame()
    H = prep_h1(h1, d1)
    M = prep_m5(m5)
    rows = []
    for _, s in sessions.iterrows():
        end = pd.Timestamp(s.session_end)
        ref_end = end + pd.Timedelta(hours=REFERENCE_HOURS)
        hw = H[(H.index > end) & (H.index <= ref_end)]
        if hw.empty: continue
        low = float(s.session_low); high = float(s.session_high)

        for side in (1, -1):
            sweep_extreme = None; signal_t = None; signal_r = None
            for t, r in hw.iterrows():
                if int(r.regime) != side or not np.isfinite(r.atr):
                    continue
                if side == 1:
                    swept = float(r.low) < low
                    if swept: sweep_extreme = float(r.low) if sweep_extreme is None else min(sweep_extreme, float(r.low))
                    confirmed = sweep_extreme is not None and float(r.close) > low and bool(r.bull_fvg) and float(r.close) > float(r.open)
                else:
                    swept = float(r.high) > high
                    if swept: sweep_extreme = float(r.high) if sweep_extreme is None else max(sweep_extreme, float(r.high))
                    confirmed = sweep_extreme is not None and float(r.close) < high and bool(r.bear_fvg) and float(r.close) < float(r.open)
                if confirmed:
                    signal_t, signal_r = t, r
                    break
            if signal_t is None: continue

            mw = M[(M.index > signal_t) & (M.index <= signal_t + pd.Timedelta(hours=M5_ENTRY_HOURS))]
            if side == 1: mw = mw[mw.bull_bos]
            else: mw = mw[mw.bear_bos]
            if mw.empty: continue
            confirm_t = mw.index[0]
            future = M[M.index > confirm_t]
            if future.empty: continue
            entry_t = future.index[0]
            entry = float(future.open.iloc[0])
            if side == 1:
                fvg_lo = float(signal_r.bull_fvg_lo)
                stop = min(float(sweep_extreme), fvg_lo)
                if stop >= entry: continue
            else:
                fvg_hi = float(signal_r.bear_fvg_hi)
                stop = max(float(sweep_extreme), fvg_hi)
                if stop <= entry: continue
            risk = abs(entry - stop)
            if risk <= 0: continue
            rows.append({
                "asset_session": s.session, "reference_session_date": s.session_date,
                "reference_session_end": end, "session_high": high, "session_low": low,
                "h1_signal_time": signal_t, "m5_confirm_time": confirm_t, "entry_time": entry_t,
                "side": side, "entry": entry, "stop": stop, "risk": risk,
                "regime": int(signal_r.regime), "sweep_extreme": float(sweep_extreme),
            })
    c = pd.DataFrame(rows)
    if not c.empty:
        c = c.sort_values("entry_time").drop_duplicates(["entry_time", "side"])
    return c


def simulate_fixed(c: pd.DataFrame, m5: pd.DataFrame, tp: float) -> pd.DataFrame:
    rows = []; busy_until = None
    for _, q in c.iterrows():
        et = pd.Timestamp(q.entry_time)
        if busy_until is not None and et <= busy_until: continue
        side = int(q.side); entry = float(q.entry); stop = float(q.stop); risk = float(q.risk)
        target = entry + side * tp * risk
        w = m5[(m5.index >= et) & (m5.index <= et + pd.Timedelta(days=5))]
        if w.empty: continue
        xp = float(w.close.iloc[-1]); xt = w.index[-1]; reason = "timeout"
        mfe = 0.0; mae = 0.0
        for t, r in w.iterrows():
            fav = (float(r.high)-entry)/risk if side == 1 else (entry-float(r.low))/risk
            adv = (entry-float(r.low))/risk if side == 1 else (float(r.high)-entry)/risk
            mfe = max(mfe, fav); mae = max(mae, adv)
            sh = float(r.low) <= stop if side == 1 else float(r.high) >= stop
            th = float(r.high) >= target if side == 1 else float(r.low) <= target
            if sh or th:
                if sh: xp, reason = stop, "stop"
                else: xp, reason = target, "tp"
                xt = t; break
        gross = side * (xp-entry) / risk
        rows.append({**q.to_dict(), "exit_time": xt, "exit_model": f"FIXED_{int(tp)}R", "gross_R": gross,
                     "net_R": gross-COST_R, "reason": reason, "MFE_R": mfe, "MAE_R": mae})
        busy_until = xt
    return pd.DataFrame(rows)


def simulate_runner(c: pd.DataFrame, m5: pd.DataFrame, h1: pd.DataFrame) -> pd.DataFrame:
    rows = []; busy_until = None
    for _, q in c.iterrows():
        et = pd.Timestamp(q.entry_time)
        if busy_until is not None and et <= busy_until: continue
        side = int(q.side); entry = float(q.entry); initial_stop = float(q.stop); risk = float(q.risk)
        tp1 = entry + side * 2.0 * risk
        w = m5[(m5.index >= et) & (m5.index <= et + pd.Timedelta(days=10))]
        if w.empty: continue
        stop = initial_stop; tp1_hit = False; realized = 0.0; xt = w.index[-1]; reason = "timeout"; runner_exit = float(w.close.iloc[-1])
        mfe = 0.0; mae = 0.0
        for t, r in w.iterrows():
            fav = (float(r.high)-entry)/risk if side == 1 else (entry-float(r.low))/risk
            adv = (entry-float(r.low))/risk if side == 1 else (float(r.high)-entry)/risk
            mfe = max(mfe, fav); mae = max(mae, adv)
            sh = float(r.low) <= stop if side == 1 else float(r.high) >= stop
            if sh:
                runner_exit = stop; xt = t; reason = "runner_stop" if tp1_hit else "stop"; break
            if not tp1_hit:
                hit = float(r.high) >= tp1 if side == 1 else float(r.low) <= tp1
                if hit:
                    tp1_hit = True; realized = 0.5 * 2.0
            if tp1_hit:
                hh = h1[(h1.index > et) & (h1.index <= t)]
                if len(hh) >= 3:
                    if side == 1:
                        new_stop = float(hh.low.iloc[-3:-1].min())
                        if new_stop > stop and new_stop < float(r.close): stop = new_stop
                    else:
                        new_stop = float(hh.high.iloc[-3:-1].max())
                        if new_stop < stop and new_stop > float(r.close): stop = new_stop
        if tp1_hit:
            runner_r = side * (runner_exit-entry)/risk
            gross = realized + 0.5 * runner_r
        else:
            gross = side * (runner_exit-entry)/risk
        rows.append({**q.to_dict(), "exit_time": xt, "exit_model": "TP1_2R_PLUS_H1_RUNNER", "gross_R": gross,
                     "net_R": gross-COST_R, "reason": reason, "MFE_R": mfe, "MAE_R": mae, "tp1_hit": tp1_hit})
        busy_until = xt
    return pd.DataFrame(rows)


def split(ts):
    y = pd.Timestamp(ts).year
    return "DEV" if y <= 2022 else "VAL" if y <= 2024 else "OOS"


def metrics(tr: pd.DataFrame):
    if tr.empty: return {"trades": 0}
    r = tr.net_R.astype(float); wins = r[r > 0]; losses = r[r <= 0]
    eq = pd.concat([pd.Series([0.0]), r.cumsum().reset_index(drop=True)], ignore_index=True)
    dd = eq - eq.cummax(); sd = r.std(ddof=1)
    return {
        "trades": len(r), "win_rate": float((r > 0).mean()), "expectancy_R": float(r.mean()),
        "PF": float(wins.sum()/abs(losses.sum())) if losses.sum() < 0 else np.nan,
        "total_R": float(r.sum()), "max_dd_R": float(abs(dd.min())),
        "avg_win_R": float(wins.mean()) if len(wins) else np.nan,
        "avg_loss_R": float(losses.mean()) if len(losses) else np.nan,
        "avg_MFE_R": float(tr.MFE_R.mean()), "avg_MAE_R": float(tr.MAE_R.mean()),
        "trade_sharpe": float(r.mean()/sd*math.sqrt(len(r))) if sd > 0 else np.nan,
    }


def main():
    name = os.environ["ASSET"].upper()
    if name not in ASSETS: raise KeyError(f"ASSET must be one of {list(ASSETS)}")
    m5, h1, d1, prov = load_asset(name)
    c = candidates(m5, h1, d1)
    out = OUT / name; out.mkdir(parents=True, exist_ok=True)
    c.to_csv(out / "candidates.csv", index=False)
    models = [simulate_fixed(c, m5, tp) for tp in TP_LEVELS]
    models.append(simulate_runner(c, m5, h1))
    rows = []
    for tr in models:
        if tr.empty: continue
        tr["split"] = tr.entry_time.map(split)
        label = str(tr.exit_model.iloc[0])
        tr.to_csv(out / f"trades_{label}.csv", index=False)
        for sess, sg in [("ALL", tr), *list(tr.groupby("asset_session"))]:
            for sp, g in [("ALL", sg), *list(sg.groupby("split"))]:
                m = metrics(g); m.update({"asset": name, "session": sess, "split": sp, "exit_model": label}); rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    meta = {
        "version": "2.7", "asset": name, "provenance": prov,
        "candidate_count": len(c), "risk_layer": "evaluate in R; fixed 1% prop candidate applied downstream",
        "cost_R": COST_R,
        "rules": "completed session H/L -> D1 SMA200 regime -> sweep/reclaim -> same-direction H1 FVG -> later M5 BOS -> next M5 open",
        "limits": [
            "Dukascopy feed is a research proxy and not exact FXIFY execution",
            "session windows are frozen research conventions",
            "SMA200 and mechanical FVG are proxies for discretionary context",
            "cost_R is synthetic and must later be replaced by measured spread/slippage/swap",
            "no true CVD or news filter in v2.7",
        ],
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
