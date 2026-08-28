import io
import json
import math
import os
import zipfile
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

OUT = Path("research/results/mtf_v12")
OUT.mkdir(parents=True, exist_ok=True)

FUTURES_REPO = "axb0306/cme-futures-ohlc"
FUTURES = ["NQ", "ES", "GC", "CL", "HG", "NG", "ZN"]
CRYPTO = {"BTC": "BTCUSDT", "ETH": "ETHUSDT"}

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "CIV42-MTF-backtest/1.2"})


def get_json(url, timeout=30):
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.json()


def get_bytes(url, timeout=60):
    r = SESSION.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content


def standardize(df):
    x = df.copy()
    x.columns = [str(c).lower().strip() for c in x.columns]
    aliases = {"date": "datetime", "time": "datetime", "timestamp": "datetime"}
    x = x.rename(columns=aliases)
    req = ["datetime", "open", "high", "low", "close", "volume"]
    missing = [c for c in req if c not in x.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; got {list(x.columns)}")
    x = x[req]
    x["datetime"] = pd.to_datetime(x["datetime"], utc=True, errors="coerce")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["datetime", "open", "high", "low", "close"]).drop_duplicates("datetime")
    return x.sort_values("datetime").set_index("datetime")


def fetch_futures_file(symbol, token):
    items = get_json(f"https://api.github.com/repos/{FUTURES_REPO}/contents/{symbol}")
    cand = [i for i in items if i.get("type") == "file" and f"_{token}_" in i["name"] and i["name"].endswith(".csv")]
    if not cand:
        raise FileNotFoundError(f"{symbol} {token} file not found")
    # Prefer the file with the latest ending date embedded in name.
    item = sorted(cand, key=lambda z: z["name"])[-1]
    raw = get_bytes(item["download_url"])
    df = pd.read_csv(io.BytesIO(raw))
    return standardize(df), item["download_url"], item["name"]


def month_range(start="2018-01", end=None):
    s = pd.Period(start, freq="M")
    if end is None:
        now = pd.Timestamp.now(tz="UTC").to_period("M") - 1
        e = now
    else:
        e = pd.Period(end, freq="M")
    return pd.period_range(s, e, freq="M")


def fetch_binance_15m(symbol, start="2018-01"):
    frames = []
    urls = []
    for p in month_range(start=start):
        ym = str(p)
        url = f"https://data.binance.vision/data/spot/monthly/klines/{symbol}/15m/{symbol}-15m-{ym}.zip"
        try:
            b = get_bytes(url, timeout=45)
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                name = z.namelist()[0]
                raw = z.read(name)
            cols = ["open_time","open","high","low","close","volume","close_time","quote_volume","trades","taker_base","taker_quote","ignore"]
            d = pd.read_csv(io.BytesIO(raw), header=None, names=cols)
            ot = pd.to_numeric(d["open_time"], errors="coerce")
            # Binance spot switched to microsecond timestamps in 2025.
            unit = "us" if ot.dropna().median() > 1e14 else "ms"
            d["datetime"] = pd.to_datetime(ot, unit=unit, utc=True, errors="coerce")
            frames.append(d[["datetime","open","high","low","close","volume"]])
            urls.append(url)
        except Exception:
            continue
    if not frames:
        raise RuntimeError(f"no Binance data for {symbol}")
    return standardize(pd.concat(frames, ignore_index=True)), urls


def fetch_ura():
    # Yahoo intraday is intentionally treated as a short-window proxy only.
    import yfinance as yf
    intr = yf.download("URA", period="60d", interval="15m", auto_adjust=False, progress=False, threads=False)
    daily = yf.download("URA", period="max", interval="1d", auto_adjust=False, progress=False, threads=False)
    if intr.empty or daily.empty:
        raise RuntimeError("Yahoo returned no URA data")
    def flat(df):
        d = df.copy()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] for c in d.columns]
        d = d.reset_index()
        d = d.rename(columns={d.columns[0]: "datetime", "Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
        return standardize(d)
    return flat(intr), flat(daily)


def resample_ohlcv(df, rule):
    agg = {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
    return df.resample(rule, label="right", closed="right").agg(agg).dropna(subset=["open","high","low","close"])


def atr(df, n=14):
    pc = df["close"].shift(1)
    tr = pd.concat([(df["high"]-df["low"]).abs(), (df["high"]-pc).abs(), (df["low"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def daily_regime(daily):
    d = daily.copy()
    d["sma200"] = d["close"].rolling(200, min_periods=200).mean()
    d["slope20"] = d["sma200"] - d["sma200"].shift(20)
    d["regime"] = 0
    d.loc[(d["close"] > d["sma200"]) & (d["slope20"] > 0), "regime"] = 1
    d.loc[(d["close"] < d["sma200"]) & (d["slope20"] < 0), "regime"] = -1
    return d[["close","sma200","slope20","regime"]]


def weekly_structure(daily):
    w = resample_ohlcv(daily, "W-FRI")
    # Causal structure proxy: no centered/fractal future bars are used.
    prev_hi = w["high"].shift(1).rolling(4, min_periods=4).max()
    prev_lo = w["low"].shift(1).rolling(4, min_periods=4).min()
    w["break_up"] = w["close"] > prev_hi
    w["break_dn"] = w["close"] < prev_lo
    state = []
    s = 0
    for up, dn in zip(w["break_up"].fillna(False), w["break_dn"].fillna(False)):
        if up:
            s = 1
        elif dn:
            s = -1
        state.append(s)
    w["wstruct"] = state
    return w[["wstruct"]]


def build_features(intr15, daily):
    m15 = intr15.copy()
    h1 = resample_ohlcv(m15, "1h")
    h4 = resample_ohlcv(m15, "4h")

    # Daily / weekly filters, only information available by that timestamp is forward-filled.
    dr = daily_regime(daily)
    ws = weekly_structure(daily)

    # 4H dealing range and premium/discount location.
    h4["range_hi"] = h4["high"].shift(1).rolling(20, min_periods=20).max()
    h4["range_lo"] = h4["low"].shift(1).rolling(20, min_periods=20).min()
    den = (h4["range_hi"] - h4["range_lo"]).replace(0, np.nan)
    h4["p"] = (h4["close"] - h4["range_lo"]) / den
    h4["atr"] = atr(h4, 14)
    h4["trail_lo"] = h4["low"].shift(1).rolling(5, min_periods=5).min()
    h4["trail_hi"] = h4["high"].shift(1).rolling(5, min_periods=5).max()

    # 1H FVG.
    h1["atr"] = atr(h1, 14)
    h1["bull_fvg"] = (h1["high"].shift(2) < h1["low"]) & ((h1["low"] - h1["high"].shift(2)) >= 0.10*h1["atr"])
    h1["bear_fvg"] = (h1["low"].shift(2) > h1["high"]) & ((h1["low"].shift(2) - h1["high"]) >= 0.10*h1["atr"])

    # 1H displacement + BOS.
    body = (h1["close"] - h1["open"]).abs()
    prev10_hi = h1["high"].shift(1).rolling(10, min_periods=10).max()
    prev10_lo = h1["low"].shift(1).rolling(10, min_periods=10).min()
    h1["bull_bos"] = (h1["close"] > prev10_hi) & (h1["close"] > h1["open"]) & (body >= 0.8*h1["atr"])
    h1["bear_bos"] = (h1["close"] < prev10_lo) & (h1["close"] < h1["open"]) & (body >= 0.8*h1["atr"])

    # Mechanical OB proxy: last opposite candle in preceding 5 bars becomes valid only AFTER a BOS/displacement.
    bull_ob = pd.Series(False, index=h1.index)
    bear_ob = pd.Series(False, index=h1.index)
    bull_zone = [None] * len(h1)
    bear_zone = [None] * len(h1)
    rows = h1.reset_index()
    for i in range(len(rows)):
        if bool(rows.loc[i, "bull_bos"]):
            for j in range(i-1, max(-1, i-6), -1):
                if rows.loc[j, "close"] < rows.loc[j, "open"]:
                    bull_zone[i] = (rows.loc[j,"low"], rows.loc[j,"high"])
                    break
        if bool(rows.loc[i, "bear_bos"]):
            for j in range(i-1, max(-1, i-6), -1):
                if rows.loc[j, "close"] > rows.loc[j, "open"]:
                    bear_zone[i] = (rows.loc[j,"low"], rows.loc[j,"high"])
                    break
    # Carry zones max 20 hours; mark retest on current bar.
    active_bull = None; age_bull = 999
    active_bear = None; age_bear = 999
    bo=[]; so=[]
    for i, r in rows.iterrows():
        if bull_zone[i] is not None:
            active_bull=bull_zone[i]; age_bull=0
        else: age_bull += 1
        if bear_zone[i] is not None:
            active_bear=bear_zone[i]; age_bear=0
        else: age_bear += 1
        b = active_bull is not None and age_bull <= 20 and r["low"] <= active_bull[1] and r["high"] >= active_bull[0]
        s = active_bear is not None and age_bear <= 20 and r["low"] <= active_bear[1] and r["high"] >= active_bear[0]
        bo.append(b); so.append(s)
    h1["bull_ob_retest"] = bo
    h1["bear_ob_retest"] = so

    h1["vol_sma20"] = h1["volume"].shift(1).rolling(20, min_periods=20).mean()
    h1["vol_ratio"] = h1["volume"] / h1["vol_sma20"].replace(0, np.nan)
    h1["swing_lo"] = h1["low"].shift(1).rolling(5, min_periods=5).min()
    h1["swing_hi"] = h1["high"].shift(1).rolling(5, min_periods=5).max()

    # 15m micro BOS trigger.
    m15["micro_hi"] = m15["high"].shift(1).rolling(4, min_periods=4).max()
    m15["micro_lo"] = m15["low"].shift(1).rolling(4, min_periods=4).min()
    m15["micro_bull"] = m15["close"] > m15["micro_hi"]
    m15["micro_bear"] = m15["close"] < m15["micro_lo"]

    # Merge as-of: closed higher-TF bars only.
    x = m15.copy()
    for frame, cols in [
        (h1, ["bull_fvg","bear_fvg","bull_bos","bear_bos","bull_ob_retest","bear_ob_retest","vol_ratio","atr","swing_lo","swing_hi"]),
        (h4, ["p","atr","trail_lo","trail_hi"]),
        (dr, ["regime","sma200"]),
        (ws, ["wstruct"]),
    ]:
        tmp = frame[cols].copy()
        # Prefix duplicate ATR columns.
        if frame is h1:
            tmp = tmp.rename(columns={"atr":"atr1h"})
        if frame is h4:
            tmp = tmp.rename(columns={"atr":"atr4h"})
        x = pd.merge_asof(x.sort_index().reset_index(), tmp.sort_index().reset_index(), on="datetime", direction="backward").set_index("datetime")
    return x


@dataclass
class Variant:
    name: str
    need_zone: bool = False
    vol_min: float = 0.0
    narrow_location: bool = False


VARIANTS = [
    Variant("A_core_no_zone_no_volume", False, 0.0, False),
    Variant("B_plus_FVG_or_OB", True, 0.0, False),
    Variant("C_zone_vol1.2", True, 1.2, False),
    Variant("D_zone_vol1.5", True, 1.5, False),
    Variant("E_zone_vol2.0", True, 2.0, False),
    Variant("F_zone_vol1.2_priority_location", True, 1.2, True),
]


def allowed_entry(r, v, side):
    if side == 1:
        # Weekly structure is a confirming filter only when it has resolved bearish; neutral is allowed due short test windows.
        if not (r.get("regime",0) == 1 and r.get("wstruct",0) >= 0): return False
        p = r.get("p", np.nan)
        if not np.isfinite(p): return False
        if v.narrow_location:
            if not (0.21 <= p <= 0.38): return False
        else:
            if not (p <= 0.45): return False
        if not bool(r.get("bull_bos",False)): return False
        if not bool(r.get("micro_bull",False)): return False
        if v.need_zone and not (bool(r.get("bull_fvg",False)) or bool(r.get("bull_ob_retest",False))): return False
        if v.vol_min and not (r.get("vol_ratio",0) >= v.vol_min): return False
        return True
    else:
        if not (r.get("regime",0) == -1 and r.get("wstruct",0) <= 0): return False
        p = r.get("p", np.nan)
        if not np.isfinite(p): return False
        if v.narrow_location:
            if not (0.62 <= p <= 0.79): return False
        else:
            if not (p >= 0.55): return False
        if not bool(r.get("bear_bos",False)): return False
        if not bool(r.get("micro_bear",False)): return False
        if v.need_zone and not (bool(r.get("bear_fvg",False)) or bool(r.get("bear_ob_retest",False))): return False
        if v.vol_min and not (r.get("vol_ratio",0) >= v.vol_min): return False
        return True


def backtest(feat, variant, cost_r=0.05):
    trades=[]
    pos=0; entry=np.nan; stop=np.nan; risk=np.nan; et=None; mfe=-np.inf; mae=np.inf
    max_hold = 20*24*4  # 20 calendar days in 15m bars; conservative timeout.
    held=0
    for t, r in feat.iterrows():
        if pos == 0:
            for side in (1,-1):
                if allowed_entry(r, variant, side):
                    e = float(r["close"])
                    if side==1:
                        s = float(r.get("swing_lo", np.nan))
                        if not np.isfinite(s) or s >= e: continue
                    else:
                        s = float(r.get("swing_hi", np.nan))
                        if not np.isfinite(s) or s <= e: continue
                    rr = abs(e-s)
                    a = float(r.get("atr1h",np.nan))
                    if not np.isfinite(a) or a<=0: continue
                    # Reject unrealistically tiny/huge structural stops.
                    if rr < 0.5*a or rr > 3.0*a: continue
                    pos=side; entry=e; stop=s; risk=rr; et=t; held=0; mfe=0.0; mae=0.0
                    break
            continue

        held += 1
        if pos==1:
            mfe=max(mfe,(float(r["high"])-entry)/risk)
            mae=min(mae,(float(r["low"])-entry)/risk)
            stop_hit=float(r["low"]) <= stop
            struct_exit=np.isfinite(r.get("trail_lo",np.nan)) and float(r["close"]) < float(r["trail_lo"])
            regime_exit=r.get("regime",0) != 1
            timeout=held>=max_hold
            if stop_hit or struct_exit or regime_exit or timeout:
                xp = stop if stop_hit else float(r["close"])
                gross=(xp-entry)/risk
                reason="stop" if stop_hit else ("4h_structure" if struct_exit else ("regime" if regime_exit else "timeout"))
                trades.append({"entry_time":et,"exit_time":t,"side":"LONG","entry":entry,"exit":xp,"stop":stop,"gross_R":gross,"net_R":gross-cost_r,"mfe_R":mfe,"mae_R":mae,"bars":held,"reason":reason})
                pos=0
        else:
            mfe=max(mfe,(entry-float(r["low"]))/risk)
            mae=min(mae,(entry-float(r["high"]))/risk)
            stop_hit=float(r["high"]) >= stop
            struct_exit=np.isfinite(r.get("trail_hi",np.nan)) and float(r["close"]) > float(r["trail_hi"])
            regime_exit=r.get("regime",0) != -1
            timeout=held>=max_hold
            if stop_hit or struct_exit or regime_exit or timeout:
                xp = stop if stop_hit else float(r["close"])
                gross=(entry-xp)/risk
                reason="stop" if stop_hit else ("4h_structure" if struct_exit else ("regime" if regime_exit else "timeout"))
                trades.append({"entry_time":et,"exit_time":t,"side":"SHORT","entry":entry,"exit":xp,"stop":stop,"gross_R":gross,"net_R":gross-cost_r,"mfe_R":mfe,"mae_R":mae,"bars":held,"reason":reason})
                pos=0
    return pd.DataFrame(trades)


def metrics(trades):
    if trades.empty:
        return {"trades":0,"win_rate":None,"expectancy_R":None,"profit_factor":None,"total_R":0.0,"max_dd_R":None,"avg_win_R":None,"avg_loss_R":None,"max_consecutive_losses":None,"avg_hold_hours":None}
    r=trades["net_R"].astype(float)
    wins=r[r>0]; losses=r[r<=0]
    pf=wins.sum()/abs(losses.sum()) if losses.sum()<0 else np.inf
    eq=r.cumsum(); dd=eq-eq.cummax(); maxdd=abs(dd.min()) if len(dd) else 0
    max_l=0; cur=0
    for z in r:
        if z<=0: cur+=1; max_l=max(max_l,cur)
        else: cur=0
    return {
        "trades":int(len(r)),
        "win_rate":float((r>0).mean()),
        "expectancy_R":float(r.mean()),
        "profit_factor":float(pf) if np.isfinite(pf) else None,
        "total_R":float(r.sum()),
        "max_dd_R":float(maxdd),
        "avg_win_R":float(wins.mean()) if len(wins) else None,
        "avg_loss_R":float(losses.mean()) if len(losses) else None,
        "max_consecutive_losses":int(max_l),
        "avg_hold_hours":float(trades["bars"].mean()*0.25),
    }


def run_asset(name, intr15, daily, provenance):
    feat=build_features(intr15,daily)
    # Restrict test to where daily SMA200 is actually known.
    feat=feat[feat["sma200"].notna()].copy()
    meta={
        "asset":name,
        "intraday_start":str(intr15.index.min()),
        "intraday_end":str(intr15.index.max()),
        "intraday_bars":int(len(intr15)),
        "daily_start":str(daily.index.min()),
        "daily_end":str(daily.index.max()),
        "test_start":str(feat.index.min()) if len(feat) else None,
        "test_end":str(feat.index.max()) if len(feat) else None,
        "provenance":provenance,
    }
    rows=[]
    for v in VARIANTS:
        tr=backtest(feat,v)
        tr.to_csv(OUT/f"{name}_{v.name}_trades.csv",index=False)
        m=metrics(tr); m.update({"asset":name,"variant":v.name})
        rows.append(m)
    return meta,rows


def main():
    all_metrics=[]; metadata=[]; errors=[]

    for sym in FUTURES:
        try:
            intr, u15, f15 = fetch_futures_file(sym,"15min")
            daily, ud, fd = fetch_futures_file(sym,"daily")
            meta,rows=run_asset(sym,intr,daily,{"15m":u15,"daily":ud,"files":[f15,fd],"source":"TopstepX via public GitHub repo"})
            metadata.append(meta); all_metrics += rows
        except Exception as e:
            errors.append({"asset":sym,"error":repr(e)})

    for name,pair in CRYPTO.items():
        try:
            intr,urls=fetch_binance_15m(pair,"2018-01")
            daily=resample_ohlcv(intr,"1D")
            meta,rows=run_asset(name,intr,daily,{"source":"Binance public data","symbol":pair,"monthly_files":len(urls),"first_url":urls[0] if urls else None,"last_url":urls[-1] if urls else None})
            metadata.append(meta); all_metrics += rows
        except Exception as e:
            errors.append({"asset":name,"error":repr(e)})

    try:
        intr,daily=fetch_ura()
        meta,rows=run_asset("URA",intr,daily,{"source":"Yahoo Finance via yfinance","note":"60d intraday proxy; not uranium futures"})
        metadata.append(meta); all_metrics += rows
    except Exception as e:
        errors.append({"asset":"URA","error":repr(e)})

    metrics_df=pd.DataFrame(all_metrics)
    metrics_df.to_csv(OUT/"metrics.csv",index=False)
    with open(OUT/"metadata.json","w") as f: json.dump(metadata,f,indent=2,default=str)
    with open(OUT/"errors.json","w") as f: json.dump(errors,f,indent=2,default=str)

    # A compact markdown report suitable for connector retrieval.
    lines=["# MTF Strategy v1.2 — Preliminary Public-Data Backtest", "",
           "## Important scope", "",
           "This is a causal, no-lookahead *mechanical proxy* for the discussed discretionary rules. It is NOT an exact MT4/MT5 ZigZag 20/8/7 implementation because platform-specific Deviation semantics are ambiguous across instruments. Futures 15m history in the public source is short, so those rows are preliminary rather than validation-grade.", "",
           "Cost sensitivity: 0.05R deducted per round trip. No CFD swap/funding/contract-roll model is included.", "",
           "## Results", ""]
    if not metrics_df.empty:
        show=metrics_df.copy()
        for c in ["win_rate","expectancy_R","profit_factor","total_R","max_dd_R","avg_win_R","avg_loss_R","avg_hold_hours"]:
            if c in show: show[c]=show[c].map(lambda x: "" if pd.isna(x) else f"{x:.3f}")
        lines.append(show.to_markdown(index=False))
    lines += ["", "## Data windows", ""]
    for m in metadata:
        lines.append(f"- {m['asset']}: intraday {m['intraday_start']} → {m['intraday_end']} ({m['intraday_bars']:,} bars); test {m['test_start']} → {m['test_end']}")
    if errors:
        lines += ["", "## Errors", ""] + [f"- {e['asset']}: `{e['error']}`" for e in errors]
    (OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
