from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("research/results/ma_timeframe_ablation_v37")
OUT.mkdir(parents=True, exist_ok=True)

START = "2024-09-01"
END = "2026-09-02"
COST_BPS_PER_SIDE = 5.0
TARGET_R = 1.5
MAX_HOLD_H1_BARS = 120
ATR_N = 20

# Liquid cross-sector universe. Keep fixed ex ante to avoid cherry-picking by result.
SYMBOLS = [
    "AAPL","MSFT","NVDA","AVGO","AMD","META","GOOGL","AMZN","TSLA","NFLX",
    "JPM","BAC","GS","XOM","CVX","CAT","GE","LLY","UNH","WMT","COST",
    "NEE","PLD","LIN","FCX","BA","DE","MU","QCOM","CRM","ORCL"
]

PERIODS = {
    "DEV": (pd.Timestamp("2024-09-01"), pd.Timestamp("2025-06-30")),
    "VAL": (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-12-31")),
    "OOS": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-09-02")),
}

MODELS = {
    "A_D1_2050200": "D1 20EMA>50SMA>200SMA",
    "B_H4_2050200": "H4 20EMA>50SMA>200SMA",
    "C_H1_2050200": "H1 20EMA>50SMA>200SMA",
    "D_D1_H4": "D1 20EMA>50SMA>200SMA + H4 20EMA>50SMA",
    "E_D1_H4_H1": "D1 20EMA>50SMA>200SMA + H4 20EMA>50SMA + H1 20EMA>50SMA",
    "F_NO_MA": "No MA regime filter",
}


def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low), (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_ma(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["sma50"] = d.Close.rolling(50).mean()
    d["sma200"] = d.Close.rolling(200).mean()
    d["atr20"] = atr(d)
    return d


def download_h1(symbol: str) -> pd.DataFrame | None:
    try:
        d = yf.download(symbol, start=START, end=END, interval="1h", auto_adjust=True,
                        actions=False, progress=False, threads=False)
    except Exception:
        return None
    if d is None or len(d) < 500:
        return None
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = d.columns.get_level_values(0)
    d.columns = [str(c).title() for c in d.columns]
    need = ["Open","High","Low","Close","Volume"]
    if not set(need).issubset(d.columns):
        return None
    d = d[need].dropna(subset=["Open","High","Low","Close"]).copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    d.index = idx
    return d[~d.index.duplicated(keep="first")].sort_index()


def resample_ohlcv(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    return x.dropna(subset=["Open","High","Low","Close"])


def features(h1: pd.DataFrame):
    h1f = add_ma(h1)
    h4f = add_ma(resample_ohlcv(h1, "4h"))
    d1f = add_ma(resample_ohlcv(h1, "1D"))
    return h1f, h4f, d1f


def aligned(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    # Only use completed higher-timeframe values; shift one HTF bar to avoid look-ahead.
    s = series.shift(1)
    return s.reindex(idx, method="ffill")


def model_filter(model: str, h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame) -> pd.Series:
    idx = h1.index
    d1_2050200 = aligned((d1.ema20>d1.sma50) & (d1.sma50>d1.sma200) & (d1.Close>d1.ema20), idx).fillna(False)
    h4_2050200 = aligned((h4.ema20>h4.sma50) & (h4.sma50>h4.sma200) & (h4.Close>h4.ema20), idx).fillna(False)
    h4_2050 = aligned((h4.ema20>h4.sma50) & (h4.Close>h4.ema20), idx).fillna(False)
    h1_2050200 = ((h1.ema20>h1.sma50) & (h1.sma50>h1.sma200) & (h1.Close>h1.ema20)).fillna(False)
    h1_2050 = ((h1.ema20>h1.sma50) & (h1.Close>h1.ema20)).fillna(False)

    if model == "A_D1_2050200": return d1_2050200
    if model == "B_H4_2050200": return h4_2050200
    if model == "C_H1_2050200": return h1_2050200
    if model == "D_D1_H4": return d1_2050200 & h4_2050
    if model == "E_D1_H4_H1": return d1_2050200 & h4_2050 & h1_2050
    if model == "F_NO_MA": return pd.Series(True, index=idx)
    raise ValueError(model)


def common_setup(h1: pd.DataFrame) -> pd.Series:
    # Same H1 location/trigger for every MA model: pullback into prior 20-bar value zone,
    # followed by bullish reclaim. This isolates the MA-timeframe filter as much as practical.
    prior_low = h1.Low.rolling(20).min().shift(1)
    prior_high = h1.High.rolling(20).max().shift(1)
    rng = (prior_high-prior_low).replace(0, np.nan)
    lower_zone = prior_low + 0.35*rng
    touched = h1.Low <= lower_zone
    reclaim = (h1.Close > h1.Open) & (h1.Close > h1.Close.shift(1))
    not_extended = (h1.Close-h1.ema20).abs() <= 1.5*h1.atr20
    return (touched & reclaim & not_extended).fillna(False)


def period(ts):
    t = pd.Timestamp(ts)
    for k,(a,b) in PERIODS.items():
        if a <= t <= b + pd.Timedelta(days=1): return k
    return None


def simulate_symbol(symbol: str, model: str, h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame):
    filt = model_filter(model, h1, h4, d1)
    setup = common_setup(h1)
    sig = filt & setup
    rows = []
    i = 220
    n = len(h1)
    while i < n-2:
        if not bool(sig.iloc[i]):
            i += 1; continue
        ei = i+1
        entry = float(h1.Open.iloc[ei])
        av = float(h1.atr20.iloc[i])
        if not np.isfinite(av) or av <= 0:
            i += 1; continue
        swing = float(h1.Low.iloc[max(0,i-19):i+1].min())
        stop = min(swing, entry-av)
        risk = entry-stop
        if risk <= 0 or risk/entry > 0.12:
            i += 1; continue
        target = entry + TARGET_R*risk
        xp = None; xi = None; mh = entry; ml = entry
        for j in range(ei, min(n, ei+MAX_HOLD_H1_BARS+1)):
            hi=float(h1.High.iloc[j]); lo=float(h1.Low.iloc[j]); cl=float(h1.Close.iloc[j])
            mh=max(mh,hi); ml=min(ml,lo)
            # Conservative same-bar ordering: stop first.
            if lo <= stop:
                xp=stop; xi=j; break
            if hi >= target:
                xp=target; xi=j; break
        if xi is None:
            xi=min(n-1, ei+MAX_HOLD_H1_BARS)
            xp=float(h1.Close.iloc[xi])
        gross=(xp-entry)/risk
        cost=(2*COST_BPS_PER_SIDE/10000.0)*entry/risk
        r=float(gross-cost)
        rows.append({
            "symbol":symbol,"model":model,"signal_date":h1.index[i],"entry_date":h1.index[ei],"exit_date":h1.index[xi],
            "entry":entry,"stop":stop,"exit":float(xp),"r":r,
            "mfe_r":float((mh-entry)/risk),"mae_r":float((ml-entry)/risk),"holding_h1_bars":int(xi-ei),
            "period":period(h1.index[ei])
        })
        i=max(xi+1, i+1)
    return rows


def summary(g: pd.DataFrame):
    if len(g)==0: return {"n":0}
    wins=g.loc[g.r>0,"r"]; losses=g.loc[g.r<0,"r"]
    eq=g.sort_values("entry_date").r.cumsum(); dd=eq-eq.cummax()
    return {
        "n":int(len(g)), "win_rate":float((g.r>0).mean()), "avg_r":float(g.r.mean()), "median_r":float(g.r.median()),
        "pf":float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else np.nan,
        "max_dd_r":float(dd.min()), "avg_mfe_r":float(g.mfe_r.mean()), "avg_mae_r":float(g.mae_r.mean()),
        "avg_hold_h1_bars":float(g.holding_h1_bars.mean()), "target_hit_rate":float((g.r >= TARGET_R-0.10).mean()),
        "total_r":float(g.r.sum())
    }


def main():
    all_rows=[]; loaded=[]
    for s in SYMBOLS:
        h1=download_h1(s)
        if h1 is None: continue
        h1,h4,d1=features(h1)
        loaded.append(s)
        for m in MODELS:
            all_rows.extend(simulate_symbol(s,m,h1,h4,d1))
    df=pd.DataFrame(all_rows)
    if df.empty: raise RuntimeError("no trades")
    df=df.dropna(subset=["period"])
    rows=[]
    for (m,p),g in df.groupby(["model","period"]):
        rows.append({"model":m,"description":MODELS[m],"period":p,**summary(g)})
    res=pd.DataFrame(rows)

    # Choose on VAL only among configurations positive in DEV and VAL with >=50 trades each.
    cand=[]
    for m in MODELS:
        a=res[(res.model==m)&(res.period=="DEV")]
        b=res[(res.model==m)&(res.period=="VAL")]
        if len(a) and len(b):
            a=a.iloc[0]; b=b.iloc[0]
            if a.n>=50 and b.n>=50 and a.avg_r>0 and b.avg_r>0:
                score=float(b.avg_r) - 0.01*abs(float(b.max_dd_r))
                cand.append((score,m))
    cand.sort(reverse=True)
    selected=cand[0][1] if cand else None
    oos_selected=None
    if selected:
        x=res[(res.model==selected)&(res.period=="OOS")]
        if len(x): oos_selected=x.iloc[0].to_dict()

    df.to_csv(OUT/"trades.csv",index=False)
    res.sort_values(["period","avg_r"],ascending=[True,False]).to_csv(OUT/"summary.csv",index=False)
    meta={
        "purpose":"MA timeframe ablation for swing trend/location system",
        "start":START,"end":END,"interval":"1h downloaded; H4/D1 resampled",
        "symbols_requested":len(SYMBOLS),"symbols_loaded":len(loaded),"loaded":loaded,
        "models":MODELS,"target_r":TARGET_R,"cost_bps_per_side":COST_BPS_PER_SIDE,
        "selection":"VAL-only; OOS untouched","selected":selected,"selected_oos":oos_selected,
        "limitations":[
            "Yahoo 1h history is availability-limited and may not cover the requested full window uniformly",
            "US stocks only; not FTMO CFD fills",
            "current-symbol universe introduces survivorship bias",
            "H4 bars are clock-time resamples of H1, not exchange-session-native 4h candles",
            "pooled trade drawdown ignores concurrent portfolio exposure",
            "common H1 zone/reclaim setup is a proxy for discretionary H4/H1 supply-demand zones",
            "fixed 1.5R exit; the proposed 20EMA volume partial / 50SMA final exit is intentionally not mixed into this ablation"
        ]
    }
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2,default=str))
    print("=== MA TIMEFRAME ABLATION v3.7 ===")
    print(json.dumps(meta,indent=2,default=str))
    print("\nSUMMARY")
    print(res.sort_values(["period","avg_r"],ascending=[True,False]).to_string(index=False))

if __name__ == "__main__":
    main()
