from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("research/results/bigview_30m_pullback_scanner_v43")
OUT.mkdir(parents=True, exist_ok=True)

PERIOD = "59d"
INTERVAL = "30m"
ATR_N = 20

UNIVERSE = {
    "GC=F": "Gold",
    "SI=F": "Silver",
    "CL=F": "Crude Oil",
    "HG=F": "Copper",
    "ZB=F": "US 30Y Bond",
    "6E=F": "Euro FX",
    "6B=F": "British Pound",
    "6J=F": "Japanese Yen",
    "6A=F": "Australian Dollar",
    "6C=F": "Canadian Dollar",
    "DX-Y.NYB": "US Dollar Index",
    "NQ=F": "Nasdaq 100",
    "ES=F": "S&P 500",
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    need = ["Open", "High", "Low", "Close", "Volume"]
    if not set(need).issubset(df.columns):
        return pd.DataFrame()
    d = df[need].dropna(subset=["Open", "High", "Low", "Close"]).copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    d.index = idx
    d = d[~d.index.duplicated(keep="last")].sort_index()
    pc = d.Close.shift(1)
    tr = pd.concat([(d.High-d.Low), (d.High-pc).abs(), (d.Low-pc).abs()], axis=1).max(axis=1)
    d["atr20"] = tr.rolling(ATR_N).mean()
    d["ema20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["sma50"] = d.Close.rolling(50).mean()
    d["sma200"] = d.Close.rolling(200).mean()
    d["ret10h"] = d.Close.pct_change(20)
    d["ret48h"] = d.Close.pct_change(96)
    d["ret5d"] = d.Close.pct_change(240)
    return d


def fetch_all():
    out = {}
    for s in UNIVERSE:
        try:
            raw = yf.download(s, period=PERIOD, interval=INTERVAL, auto_adjust=True,
                              actions=False, progress=False, threads=False)
            d = clean(raw)
            if len(d) >= 300:
                out[s] = d
        except Exception:
            pass
    return out


def row_for(s: str, d: pd.DataFrame) -> dict | None:
    x = d.dropna().iloc[-1]
    px = float(x.Close)
    atr = float(x.atr20)
    if not np.isfinite(atr) or atr <= 0:
        return None

    trend_up = px > float(x.sma50) > float(x.sma200)
    trend_dn = px < float(x.sma50) < float(x.sma200)
    direction = 1 if trend_up else (-1 if trend_dn else 0)

    mom = 0.45 * float(x.ret48h) + 0.35 * float(x.ret5d) + 0.20 * float(x.ret10h)
    strength = direction * mom if direction != 0 else 0.0
    ext_ema_atr = (px - float(x.ema20)) / atr
    dist_sma50_atr = (px - float(x.sma50)) / atr
    recent = d.iloc[-12:]
    if direction > 0:
        short_pull = (recent.Close.iloc[-1] - recent.High.max()) / atr
        reclaim = px >= float(x.ema20)
    elif direction < 0:
        short_pull = (recent.Low.min() - recent.Close.iloc[-1]) / atr
        reclaim = px <= float(x.ema20)
    else:
        short_pull = 0.0
        reclaim = False

    if direction > 0:
        location_ok = -0.75 <= ext_ema_atr <= 0.35
        chase_penalty = max(0.0, ext_ema_atr - 0.75)
    elif direction < 0:
        location_ok = -0.35 <= ext_ema_atr <= 0.75
        chase_penalty = max(0.0, -ext_ema_atr - 0.75)
    else:
        location_ok = False
        chase_penalty = 1.0

    strength_score = max(0.0, abs(strength)) * 100.0
    pullback_score = max(0.0, min(2.0, abs(short_pull)))
    location_bonus = 1.0 if location_ok else 0.0
    reclaim_bonus = 0.5 if reclaim else 0.0
    score = 2.2*strength_score + 1.2*pullback_score + location_bonus + reclaim_bonus - 2.0*chase_penalty

    state = "NO_TRADE"
    if direction != 0 and location_ok:
        state = "WATCH_RETEST"
    if direction != 0 and location_ok and reclaim:
        state = "READY_FOR_30M_CONFIRMATION"

    return {
        "symbol": s,
        "asset": UNIVERSE[s],
        "time": d.index[-1],
        "direction": "LONG" if direction > 0 else ("SHORT" if direction < 0 else "NEUTRAL"),
        "price": px,
        "ret10h_pct": float(x.ret10h)*100,
        "ret48h_pct": float(x.ret48h)*100,
        "ret5d_pct": float(x.ret5d)*100,
        "ema20_ext_atr": ext_ema_atr,
        "sma50_dist_atr": dist_sma50_atr,
        "six_hour_pullback_atr": short_pull,
        "location_ok": bool(location_ok),
        "reclaim": bool(reclaim),
        "state": state,
        "bigview_pullback_score": score,
    }


def main():
    data = fetch_all()
    rows = []
    for s, d in data.items():
        r = row_for(s, d)
        if r:
            rows.append(r)
    if not rows:
        raise RuntimeError("No symbols loaded")
    df = pd.DataFrame(rows).sort_values("bigview_pullback_score", ascending=False)
    df.to_csv(OUT / "scanner.csv", index=False)
    top = df[df.state != "NO_TRADE"].head(7)
    top.to_csv(OUT / "shortlist.csv", index=False)
    meta = {
        "purpose": "BigView: medium-term strength + short-term 30m pullback scanner",
        "interval": INTERVAL,
        "period": PERIOD,
        "rule": "Strong trend first; prefer pullback near EMA20; never convert strength directly into entry permission.",
        "states": {
            "WATCH_RETEST": "location acceptable, wait for 30m retest/reclaim",
            "READY_FOR_30M_CONFIRMATION": "location + reclaim present; still requires discretionary/system confirmation",
            "NO_TRADE": "trend/location not suitable or already extended"
        },
        "limitations": [
            "Yahoo data is delayed/limited and not a prop execution feed.",
            "This is a scanner, not a trade signal.",
            "No news/event-risk filter yet.",
            "No proprietary TopView flow data; uses OHLCV/trend proxies."
        ]
    }
    (OUT / "meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("LOADED", list(data.keys()))
    print("\nTOP SHORTLIST")
    print(top[["symbol","asset","direction","state","ret48h_pct","ret5d_pct","ema20_ext_atr","six_hour_pullback_atr","bigview_pullback_score"]].to_string(index=False))
    print("\nFULL RANK")
    print(df[["symbol","asset","direction","state","bigview_pullback_score"]].to_string(index=False))


if __name__ == "__main__":
    main()

# CI trigger v4.3
