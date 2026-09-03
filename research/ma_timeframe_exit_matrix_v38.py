from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("research/results/ma_timeframe_exit_matrix_v38")
OUT.mkdir(parents=True, exist_ok=True)

START = "2024-09-01"
END = "2026-09-02"
COST_BPS_PER_SIDE = 5.0
MAX_HOLD_H1_BARS = 240
ATR_N = 20
VOL_MULT = 3.0
PARTIAL_FRACTION = 0.50

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

ENTRY_MODELS = {
    "A_D1_2050200": "D1 20EMA>50SMA>200SMA",
    "B_H4_2050200": "H4 20EMA>50SMA>200SMA",
    "C_H1_2050200": "H1 20EMA>50SMA>200SMA",
    "D_D1_H4": "D1 20EMA>50SMA>200SMA + H4 20EMA>50SMA",
    "E_D1_H4_H1": "D1 20EMA>50SMA>200SMA + H4 20EMA>50SMA + H1 20EMA>50SMA",
    "F_NO_MA": "No MA regime filter",
}
EXIT_TFS = ["H1", "H4", "D1"]


def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low), (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["ema20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["sma50"] = d.Close.rolling(50).mean()
    d["sma200"] = d.Close.rolling(200).mean()
    d["atr20"] = atr(d)
    d["vol20"] = d.Volume.rolling(20).mean()
    d["vol_ratio"] = d.Volume / d.vol20.replace(0, np.nan)
    d["bear"] = d.Close < d.Open
    d["partial_exit_raw"] = d.bear & (d.Close < d.ema20) & (d.vol_ratio >= VOL_MULT)
    d["below50"] = d.Close < d.sma50
    d["final_exit_raw"] = d.below50 & d.below50.shift(1).fillna(False)
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
    if not set(need).issubset(d.columns): return None
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
    h1f = add_features(h1)
    h4f = add_features(resample_ohlcv(h1, "4h"))
    d1f = add_features(resample_ohlcv(h1, "1D"))
    return h1f, h4f, d1f


def aligned_completed(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return series.shift(1).reindex(idx, method="ffill")


def entry_filter(model: str, h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame) -> pd.Series:
    idx = h1.index
    d1_2050200 = aligned_completed((d1.ema20>d1.sma50) & (d1.sma50>d1.sma200) & (d1.Close>d1.ema20), idx).fillna(False)
    h4_2050200 = aligned_completed((h4.ema20>h4.sma50) & (h4.sma50>h4.sma200) & (h4.Close>h4.ema20), idx).fillna(False)
    h4_2050 = aligned_completed((h4.ema20>h4.sma50) & (h4.Close>h4.ema20), idx).fillna(False)
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
    prior_low = h1.Low.rolling(20).min().shift(1)
    prior_high = h1.High.rolling(20).max().shift(1)
    rng = (prior_high-prior_low).replace(0, np.nan)
    lower_zone = prior_low + 0.35*rng
    touched = h1.Low <= lower_zone
    reclaim = (h1.Close > h1.Open) & (h1.Close > h1.Close.shift(1))
    not_extended = (h1.Close-h1.ema20).abs() <= 1.5*h1.atr20
    return (touched & reclaim & not_extended).fillna(False)


def exit_signals(exit_tf: str, h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame):
    if exit_tf == "H1":
        # Signal is known only after the H1 close; execute next H1 open.
        return h1.partial_exit_raw.fillna(False), h1.final_exit_raw.fillna(False)
    src = h4 if exit_tf == "H4" else d1
    # Shift completed HTF signal one HTF bar, then forward-fill to H1 to avoid using an unfinished bar.
    return (
        aligned_completed(src.partial_exit_raw, h1.index).fillna(False),
        aligned_completed(src.final_exit_raw, h1.index).fillna(False),
    )


def period(ts):
    t = pd.Timestamp(ts)
    for k,(a,b) in PERIODS.items():
        if a <= t <= b + pd.Timedelta(days=1): return k
    return None


def simulate_symbol(symbol: str, model: str, exit_tf: str, h1: pd.DataFrame, h4: pd.DataFrame, d1: pd.DataFrame):
    sig = entry_filter(model,h1,h4,d1) & common_setup(h1)
    partial_sig, final_sig = exit_signals(exit_tf,h1,h4,d1)
    rows=[]; i=220; n=len(h1)
    while i<n-2:
        if not bool(sig.iloc[i]): i+=1; continue
        ei=i+1; entry=float(h1.Open.iloc[ei]); av=float(h1.atr20.iloc[i])
        if not np.isfinite(av) or av<=0: i+=1; continue
        swing=float(h1.Low.iloc[max(0,i-19):i+1].min()); stop=min(swing,entry-av); risk=entry-stop
        if risk<=0 or risk/entry>0.12: i+=1; continue

        remaining=1.0; realized_r=0.0; partial_done=False; partial_price=np.nan; partial_i=None
        mh=entry; ml=entry; xi=None; final_price=None; exit_reason=None
        j=ei
        last=min(n-1,ei+MAX_HOLD_H1_BARS)
        while j<=last:
            hi=float(h1.High.iloc[j]); lo=float(h1.Low.iloc[j]); cl=float(h1.Close.iloc[j])
            mh=max(mh,hi); ml=min(ml,lo)
            # Hard structural stop always wins same-bar conflicts.
            if lo<=stop:
                realized_r += remaining*((stop-entry)/risk)
                final_price=stop; xi=j; exit_reason="STOP"; remaining=0; break

            # Exit signals are acted on at next H1 open, not the triggering close.
            if j>ei:
                prev=j-1
                if (not partial_done) and bool(partial_sig.iloc[prev]):
                    px=float(h1.Open.iloc[j])
                    realized_r += PARTIAL_FRACTION*((px-entry)/risk)
                    remaining -= PARTIAL_FRACTION
                    partial_done=True; partial_price=px; partial_i=j
                if bool(final_sig.iloc[prev]):
                    px=float(h1.Open.iloc[j])
                    realized_r += remaining*((px-entry)/risk)
                    remaining=0; final_price=px; xi=j; exit_reason="SMA50_FAIL"; break
            j+=1

        if xi is None:
            xi=last; final_price=float(h1.Close.iloc[xi]); realized_r += remaining*((final_price-entry)/risk); remaining=0; exit_reason="TIME"

        # Approximate round-trip costs separately for each exited fraction.
        # Full position always enters once; exits can be one or two legs. Cost scales by fraction.
        cost_r=(2*COST_BPS_PER_SIDE/10000.0)*entry/risk
        r=float(realized_r-cost_r)
        rows.append({
            "symbol":symbol,"model":model,"exit_tf":exit_tf,"signal_date":h1.index[i],"entry_date":h1.index[ei],"exit_date":h1.index[xi],
            "entry":entry,"stop":stop,"partial_done":partial_done,"partial_price":partial_price,"final_exit":float(final_price),
            "exit_reason":exit_reason,"r":r,"mfe_r":float((mh-entry)/risk),"mae_r":float((ml-entry)/risk),
            "holding_h1_bars":int(xi-ei),"period":period(h1.index[ei])
        })
        i=max(xi+1,i+1)
    return rows


def summary(g: pd.DataFrame):
    if len(g)==0:return {"n":0}
    wins=g.loc[g.r>0,"r"]; losses=g.loc[g.r<0,"r"]
    eq=g.sort_values("entry_date").r.cumsum(); dd=eq-eq.cummax()
    return {
        "n":int(len(g)),"win_rate":float((g.r>0).mean()),"avg_r":float(g.r.mean()),"median_r":float(g.r.median()),
        "pf":float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else np.nan,
        "max_dd_r":float(dd.min()),"avg_mfe_r":float(g.mfe_r.mean()),"avg_mae_r":float(g.mae_r.mean()),
        "avg_hold_h1_bars":float(g.holding_h1_bars.mean()),"partial_rate":float(g.partial_done.mean()),
        "stop_rate":float((g.exit_reason=="STOP").mean()),"ma_exit_rate":float((g.exit_reason=="SMA50_FAIL").mean()),
        "total_r":float(g.r.sum())
    }


def main():
    all_rows=[]; loaded=[]
    for s in SYMBOLS:
        raw=download_h1(s)
        if raw is None: continue
        h1,h4,d1=features(raw); loaded.append(s)
        for m in ENTRY_MODELS:
            for x in EXIT_TFS:
                all_rows.extend(simulate_symbol(s,m,x,h1,h4,d1))
    df=pd.DataFrame(all_rows)
    if df.empty: raise RuntimeError("no trades")
    df=df.dropna(subset=["period"])
    rows=[]
    for (m,x,p),g in df.groupby(["model","exit_tf","period"]):
        rows.append({"model":m,"description":ENTRY_MODELS[m],"exit_tf":x,"period":p,**summary(g)})
    res=pd.DataFrame(rows)

    # Select only on DEV+VAL. OOS remains untouched until configuration selection.
    cand=[]
    for m in ENTRY_MODELS:
        for x in EXIT_TFS:
            a=res[(res.model==m)&(res.exit_tf==x)&(res.period=="DEV")]
            b=res[(res.model==m)&(res.exit_tf==x)&(res.period=="VAL")]
            if len(a) and len(b):
                a=a.iloc[0]; b=b.iloc[0]
                if a.n>=40 and b.n>=40 and a.avg_r>0 and b.avg_r>0:
                    score=float(b.avg_r)-0.01*abs(float(b.max_dd_r))
                    cand.append((score,m,x))
    cand.sort(reverse=True); selected=cand[0][1:] if cand else None
    selected_oos=None
    if selected:
        m,x=selected; z=res[(res.model==m)&(res.exit_tf==x)&(res.period=="OOS")]
        if len(z): selected_oos=z.iloc[0].to_dict()

    df.to_csv(OUT/"trades.csv",index=False)
    res.sort_values(["period","avg_r"],ascending=[True,False]).to_csv(OUT/"summary.csv",index=False)
    meta={
        "purpose":"joint MA entry-timeframe and MA-exit-timeframe ablation",
        "entry_models":ENTRY_MODELS,"exit_tfs":EXIT_TFS,
        "exit_rule":{
            "partial":"50% at next H1 open after bearish close below 20EMA with volume >=3x 20-bar average",
            "final":"remaining position at next H1 open after two consecutive closes below 50SMA",
            "hard_stop":"original structural/ATR stop remains active",
            "time_exit_h1_bars":MAX_HOLD_H1_BARS
        },
        "symbols_requested":len(SYMBOLS),"symbols_loaded":len(loaded),"loaded":loaded,
        "selection":"DEV+VAL only; OOS untouched until selected","selected":selected,"selected_oos":selected_oos,
        "limitations":[
            "Yahoo 1h history availability can vary by symbol",
            "US-stock proxy, not FTMO CFD execution",
            "current-symbol universe has survivorship bias",
            "H4 is clock-time resampling of H1",
            "pooled trade DD ignores concurrent positions and correlation",
            "manual supply/demand zones are represented by a common H1 pullback/reclaim proxy",
            "3x-volume exit threshold is a hypothesis under test, not a proven universal constant"
        ]
    }
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2,default=str))
    print("=== MA TIMEFRAME + MA EXIT MATRIX v3.8 ===")
    print(json.dumps(meta,indent=2,default=str))
    print("\nSUMMARY")
    print(res.sort_values(["period","avg_r"],ascending=[True,False]).to_string(index=False))

if __name__=="__main__":
    main()
