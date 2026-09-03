from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("research/results/topview_30m_entry_ablation_v41")
OUT.mkdir(parents=True, exist_ok=True)

PERIOD = "59d"
INTERVAL = "30m"
COST_BPS_PER_SIDE = 3.0
ATR_N = 20
TOP_N = 3
LOOKBACK = 20
RETEST_WINDOW = 6
EXTENSION_ATR_MAX = 0.75
RANGE_ATR_MAX = 1.25
RETEST_TOL_ATR = 0.30
MAX_HOLD_BARS = 48  # 24h on 30m bars

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
}

MODELS = {
    "BREAKOUT_IMMEDIATE": "30m breakout immediate next-bar entry",
    "BREAKOUT_NO_CHASE": "breakout + extension/range gate",
    "BREAKOUT_RETEST": "breakout + retest/reclaim within 6 bars",
    "BREAKOUT_NO_CHASE_RETEST": "breakout + no-chase + retest/reclaim",
}


def atr(df, n=ATR_N):
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low), (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def clean(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns={c: str(c).title() for c in df.columns})
    need = ["Open","High","Low","Close","Volume"]
    if not set(need).issubset(df.columns):
        return pd.DataFrame()
    d = df[need].dropna(subset=["Open","High","Low","Close"]).copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    d.index = idx
    d = d[~d.index.duplicated(keep="last")].sort_index()
    d["atr20"] = atr(d)
    d["ema20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["sma50"] = d.Close.rolling(50).mean()
    d["vol20"] = d.Volume.rolling(20).mean()
    d["vol_ratio"] = d.Volume / d.vol20.replace(0, np.nan)
    return d


def download():
    out = {}
    for s in UNIVERSE:
        try:
            d = yf.download(s, period=PERIOD, interval=INTERVAL, auto_adjust=True,
                            actions=False, progress=False, threads=False)
            d = clean(d)
            if len(d) >= 500:
                out[s] = d
        except Exception:
            pass
    return out


def period_labels(idx):
    start = idx.min().normalize()
    dev_end = start + pd.Timedelta(days=30)
    val_end = dev_end + pd.Timedelta(days=15)
    out = pd.Series(index=idx, dtype="object")
    out.loc[idx < dev_end] = "DEV"
    out.loc[(idx >= dev_end) & (idx < val_end)] = "VAL"
    out.loc[idx >= val_end] = "OOS"
    return out


def build_cross_section(data):
    common = None
    for d in data.values():
        common = d.index if common is None else common.union(d.index)
    common = common.sort_values()
    score = pd.DataFrame(index=common)
    aligned = {}
    for s, d0 in data.items():
        d = d0.reindex(common).ffill(limit=2)
        aligned[s] = d
        r10h = d.Close.pct_change(20)
        r48h = d.Close.pct_change(96)
        vol = np.log(d.vol_ratio.clip(lower=0.2, upper=5.0))
        score[s] = 0.45*r10h + 0.45*r48h + 0.10*np.sign(r10h)*vol.fillna(0)*0.01
    rank = score.abs().rank(axis=1, ascending=False, method="min")
    top = rank <= TOP_N
    return common, aligned, score, top


def find_retest(d, i, side, level, av):
    end = min(len(d)-2, i + RETEST_WINDOW)
    for j in range(i+1, end+1):
        hi, lo, cl = map(float, (d.High.iloc[j], d.Low.iloc[j], d.Close.iloc[j]))
        if side > 0:
            if lo <= level + RETEST_TOL_ATR*av and cl >= level:
                return j
        else:
            if hi >= level - RETEST_TOL_ATR*av and cl <= level:
                return j
    return None


def candidate_events(common, aligned, score, top):
    rows = []
    labels = period_labels(common)
    for s, d in aligned.items():
        for i in range(max(LOOKBACK, 120), len(common)-RETEST_WINDOW-3):
            if not bool(top[s].iloc[i]) or not np.isfinite(score[s].iloc[i]):
                continue
            side = 1 if score[s].iloc[i] >= 0 else -1
            av = float(d.atr20.iloc[i]) if np.isfinite(d.atr20.iloc[i]) else np.nan
            if not np.isfinite(av) or av <= 0:
                continue
            prev = d.iloc[i-LOOKBACK:i]
            if side > 0:
                level = float(prev.High.max())
                breakout = float(d.Close.iloc[i]) > level
            else:
                level = float(prev.Low.min())
                breakout = float(d.Close.iloc[i]) < level
            if not breakout:
                continue
            close = float(d.Close.iloc[i]); ema = float(d.ema20.iloc[i])
            hi = float(d.High.iloc[i]); lo = float(d.Low.iloc[i])
            ext = side*(close-ema)/av
            rng = (hi-lo)/av
            no_chase = (ext <= EXTENSION_ATR_MAX) and (rng <= RANGE_ATR_MAX)
            rj = find_retest(d, i, side, level, av)
            rows.append({"symbol":s,"asset":UNIVERSE[s],"signal_i":i,"signal_time":common[i],
                         "period":labels.iloc[i],"side":side,"score":float(score[s].iloc[i]),
                         "breakout_level":level,"extension_atr":ext,"range_atr":rng,
                         "no_chase":bool(no_chase),"retest_i":rj})
    return pd.DataFrame(rows)


def simulate_one(d, event, model):
    i = int(event.signal_i); side = int(event.side)
    if model in ("BREAKOUT_NO_CHASE","BREAKOUT_NO_CHASE_RETEST") and not bool(event.no_chase):
        return None
    if model in ("BREAKOUT_RETEST","BREAKOUT_NO_CHASE_RETEST"):
        if pd.isna(event.retest_i): return None
        trigger_i = int(event.retest_i)
    else:
        trigger_i = i
    ei = trigger_i + 1
    if ei >= len(d): return None
    entry = float(d.Open.iloc[ei])
    av = float(d.atr20.iloc[trigger_i])
    if not np.isfinite(entry) or not np.isfinite(av) or av <= 0: return None

    if side > 0:
        swing = float(d.Low.iloc[max(0,trigger_i-19):trigger_i+1].min())
        stop = min(swing, entry-av)
        risk = entry-stop
    else:
        swing = float(d.High.iloc[max(0,trigger_i-19):trigger_i+1].max())
        stop = max(swing, entry+av)
        risk = stop-entry
    if risk <= 0 or risk/entry > 0.08: return None

    mfe = 0.0; mae = 0.0; xi = None; px = None; reason = None
    last = min(len(d)-1, ei + MAX_HOLD_BARS)
    for j in range(ei, last+1):
        hi, lo = float(d.High.iloc[j]), float(d.Low.iloc[j])
        if side > 0:
            mfe=max(mfe,(hi-entry)/risk); mae=min(mae,(lo-entry)/risk)
            if lo <= stop: xi=j; px=stop; reason="STOP"; break
            # 30m SMA50 trend failure, two closes
            if j >= ei+2 and float(d.Close.iloc[j-1]) < float(d.sma50.iloc[j-1]) and float(d.Close.iloc[j-2]) < float(d.sma50.iloc[j-2]):
                xi=j; px=float(d.Open.iloc[j]); reason="30M_SMA50_FAIL"; break
        else:
            mfe=max(mfe,(entry-lo)/risk); mae=min(mae,(entry-hi)/risk)
            if hi >= stop: xi=j; px=stop; reason="STOP"; break
            if j >= ei+2 and float(d.Close.iloc[j-1]) > float(d.sma50.iloc[j-1]) and float(d.Close.iloc[j-2]) > float(d.sma50.iloc[j-2]):
                xi=j; px=float(d.Open.iloc[j]); reason="30M_SMA50_FAIL"; break
    if xi is None:
        xi=last; px=float(d.Close.iloc[xi]); reason="TIME"
    gross = side*(px-entry)/risk
    cost = (2*COST_BPS_PER_SIDE/10000.0)*entry/risk
    return {"model":model,"symbol":event.symbol,"asset":event.asset,"period":event.period,
            "signal_time":event.signal_time,"entry_time":d.index[ei],"exit_time":d.index[xi],
            "side":"LONG" if side>0 else "SHORT","entry":entry,"stop":stop,
            "r":gross-cost,"mfe_r":mfe,"mae_r":mae,"hold_bars":xi-ei,
            "extension_atr":float(event.extension_atr),"range_atr":float(event.range_atr),
            "exit_reason":reason}


def simulate(data, common, aligned, events):
    rows=[]
    for model in MODELS:
        for s in data:
            d=aligned[s]
            busy_until=None
            ev=events[events.symbol==s].sort_values("signal_time")
            for _,e in ev.iterrows():
                if busy_until is not None and pd.Timestamp(e.signal_time) <= busy_until:
                    continue
                q=simulate_one(d,e,model)
                if q is None: continue
                rows.append(q)
                busy_until=pd.Timestamp(q["exit_time"])
    return pd.DataFrame(rows)


def metrics(g):
    if g.empty: return {"n":0}
    r=g.r.astype(float); wins=r[r>0]; losses=r[r<0]
    eq=r.cumsum(); dd=eq-eq.cummax(); sd=r.std(ddof=1)
    return {"n":int(len(r)),"win_rate":float((r>0).mean()),"avg_r":float(r.mean()),
            "pf":float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum())>0 else np.nan,
            "total_r":float(r.sum()),"max_dd_r":float(dd.min()),
            "trade_sharpe":float(r.mean()/sd*math.sqrt(len(r))) if sd>0 else np.nan,
            "avg_mfe_r":float(g.mfe_r.mean()),"avg_mae_r":float(g.mae_r.mean()),
            "avg_hold_bars":float(g.hold_bars.mean())}


def main():
    data=download()
    if len(data)<5: raise RuntimeError(f"insufficient symbols loaded: {list(data)}")
    common,aligned,score,top=build_cross_section(data)
    events=candidate_events(common,aligned,score,top)
    trades=simulate(data,common,aligned,events)
    rows=[]
    for (m,p),g in trades.groupby(["model","period"]):
        rows.append({"model":m,"description":MODELS[m],"period":p,**metrics(g)})
    summary=pd.DataFrame(rows)
    trades.to_csv(OUT/"trades.csv",index=False)
    summary.to_csv(OUT/"summary.csv",index=False)
    meta={"purpose":"30m-only breakout entry-quality ablation for prop-seed research",
          "period":PERIOD,"interval":INTERVAL,"parameters":{"top_n":TOP_N,"lookback":LOOKBACK,
          "retest_window":RETEST_WINDOW,"extension_atr_max":EXTENSION_ATR_MAX,
          "range_atr_max":RANGE_ATR_MAX,"retest_tol_atr":RETEST_TOL_ATR},
          "limitations":["Not proprietary TopView data; uses OHLCV momentum/participation proxy.",
          "Yahoo 30m sample about 59 days; pilot only.",
          "Continuous futures/FX proxies differ from prop CFD execution.",
          "No exact firm daily-loss/trailing-drawdown simulation in this script."]}
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2,default=str))
    print("LOADED",list(data.keys()))
    print("EVENTS",len(events),"TRADES",len(trades))
    print(summary.sort_values(["period","avg_r"],ascending=[True,False]).to_string(index=False))

if __name__=="__main__": main()
