from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("research/results/topview_30m_crossasset_v39")
OUT.mkdir(parents=True, exist_ok=True)

# Yahoo intraday retention is limited. Keep this deliberately short and label it as a pilot.
PERIOD = "59d"
INTERVAL = "30m"
ATR_N = 20
COST_BPS_PER_SIDE = 3.0
MAX_HOLD_BARS = 320  # ~10 trading days for most futures sessions
TOP_N = 3

# Liquid futures / macro proxies. FX is represented with listed currency futures where available.
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
    "FLOW_ONLY": "TopView-style cross-sectional flow rank only",
    "FLOW_TREND": "Flow rank + completed H4 trend",
    "FLOW_ZONE": "Flow rank + H1 pullback/reclaim zone",
    "FLOW_TREND_ZONE": "Flow rank + H4 trend + H1 zone",
}


def atr(df: pd.DataFrame, n: int = ATR_N) -> pd.Series:
    pc = df.Close.shift(1)
    tr = pd.concat([(df.High-df.Low), (df.High-pc).abs(), (df.Low-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


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
    d["atr20"] = atr(d)
    d["ema20"] = d.Close.ewm(span=20, adjust=False).mean()
    d["sma50"] = d.Close.rolling(50).mean()
    d["vol20"] = d.Volume.rolling(20).mean()
    d["vol_ratio"] = d.Volume / d.vol20.replace(0, np.nan)
    return d


def download() -> dict[str, pd.DataFrame]:
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


def resample(d: pd.DataFrame, rule: str) -> pd.DataFrame:
    x = d.resample(rule).agg({"Open":"first","High":"max","Low":"min","Close":"last","Volume":"sum"})
    x = x.dropna(subset=["Open","High","Low","Close"])
    x["ema20"] = x.Close.ewm(span=20, adjust=False).mean()
    x["sma50"] = x.Close.rolling(50).mean()
    x["atr20"] = atr(x)
    return x


def completed(series: pd.Series, idx: pd.DatetimeIndex) -> pd.Series:
    return series.shift(1).reindex(idx, method="ffill")


def build_panel(data: dict[str, pd.DataFrame]):
    # Common 30m grid is required for causal cross-sectional ranking.
    common = None
    for d in data.values():
        common = d.index if common is None else common.union(d.index)
    common = common.sort_values()

    score = pd.DataFrame(index=common)
    direction = pd.DataFrame(index=common)
    trend_ok = pd.DataFrame(index=common)
    zone_ok = pd.DataFrame(index=common)

    aligned = {}
    for s, d0 in data.items():
        d = d0.reindex(common).ffill(limit=2)
        aligned[s] = d
        r10h = d.Close.pct_change(20)
        r48h = d.Close.pct_change(96)
        vol = np.log(d.vol_ratio.clip(lower=0.2, upper=5.0))
        # Signed flow proxy: medium/short momentum plus participation in the same direction.
        raw = 0.45*r10h + 0.45*r48h + 0.10*np.sign(r10h)*vol.fillna(0)*0.01
        score[s] = raw
        direction[s] = np.sign(raw)

        h4 = resample(d0, "4h")
        h4_long = (h4.Close > h4.ema20) & (h4.ema20 > h4.sma50)
        h4_short = (h4.Close < h4.ema20) & (h4.ema20 < h4.sma50)
        long_a = completed(h4_long, common).fillna(False)
        short_a = completed(h4_short, common).fillna(False)
        trend_ok[s] = np.where(direction[s] >= 0, long_a, short_a)

        h1 = resample(d0, "1h")
        # Zone = pullback into H1 20EMA / recent structure, followed by a directional reclaim.
        prior_low = h1.Low.rolling(12).min().shift(1)
        prior_high = h1.High.rolling(12).max().shift(1)
        near_ema = (h1.Low <= h1.ema20 + 0.25*h1.atr20) & (h1.High >= h1.ema20 - 0.25*h1.atr20)
        long_reclaim = near_ema & (h1.Close > h1.ema20) & (h1.Close > h1.Open) & (h1.Low <= prior_low + 0.45*(prior_high-prior_low))
        short_reclaim = near_ema & (h1.Close < h1.ema20) & (h1.Close < h1.Open) & (h1.High >= prior_high - 0.45*(prior_high-prior_low))
        zl = completed(long_reclaim, common).fillna(False)
        zs = completed(short_reclaim, common).fillna(False)
        zone_ok[s] = np.where(direction[s] >= 0, zl, zs)

    # Cross-sectional absolute rank: top N strongest capital-flow candidates each bar.
    abs_rank = score.abs().rank(axis=1, ascending=False, method="min")
    top = abs_rank <= TOP_N
    return common, aligned, score, direction, top, trend_ok.astype(bool), zone_ok.astype(bool)


def period_labels(idx: pd.DatetimeIndex) -> pd.Series:
    # 30/15/remaining day walk-forward split based on actual downloaded span.
    start = idx.min().normalize(); end = idx.max().normalize()
    dev_end = start + pd.Timedelta(days=30)
    val_end = dev_end + pd.Timedelta(days=15)
    out = pd.Series(index=idx, dtype="object")
    out.loc[idx < dev_end] = "DEV"
    out.loc[(idx >= dev_end) & (idx < val_end)] = "VAL"
    out.loc[idx >= val_end] = "OOS"
    return out


def model_signal(model, s, top, trend_ok, zone_ok):
    sig = top[s].fillna(False)
    if model in ("FLOW_TREND", "FLOW_TREND_ZONE"):
        sig &= trend_ok[s].fillna(False)
    if model in ("FLOW_ZONE", "FLOW_TREND_ZONE"):
        sig &= zone_ok[s].fillna(False)
    return sig


def simulate(data, common, aligned, score, direction, top, trend_ok, zone_ok):
    labels = period_labels(common)
    rows = []
    for model in MODELS:
        for s in data:
            d = aligned[s]
            sig = model_signal(model, s, top, trend_ok, zone_ok)
            i = 120
            while i < len(common)-3:
                if not bool(sig.iloc[i]) or not np.isfinite(score[s].iloc[i]):
                    i += 1; continue
                side = 1 if direction[s].iloc[i] >= 0 else -1
                ei = i+1
                entry = float(d.Open.iloc[ei]) if np.isfinite(d.Open.iloc[ei]) else np.nan
                av = float(d.atr20.iloc[i]) if np.isfinite(d.atr20.iloc[i]) else np.nan
                if not np.isfinite(entry) or not np.isfinite(av) or av <= 0:
                    i += 1; continue

                if side > 0:
                    swing = float(d.Low.iloc[max(0,i-19):i+1].min())
                    stop = min(swing, entry-av)
                    risk = entry-stop
                else:
                    swing = float(d.High.iloc[max(0,i-19):i+1].max())
                    stop = max(swing, entry+av)
                    risk = stop-entry
                if risk <= 0 or risk/entry > 0.08:
                    i += 1; continue

                # Exit: original structural stop, otherwise H1 50SMA trend failure; no fixed-R take profit.
                h1 = resample(data[s], "1h")
                if side > 0:
                    fail = (h1.Close < h1.sma50) & (h1.Close.shift(1) < h1.sma50.shift(1))
                else:
                    fail = (h1.Close > h1.sma50) & (h1.Close.shift(1) > h1.sma50.shift(1))
                fail30 = completed(fail, common).fillna(False)

                mfe = 0.0; mae = 0.0; xi = None; px = None; reason = None
                last = min(len(common)-1, ei+MAX_HOLD_BARS)
                j = ei
                while j <= last:
                    hi = d.High.iloc[j]; lo = d.Low.iloc[j]
                    if not np.isfinite(hi) or not np.isfinite(lo):
                        j += 1; continue
                    if side > 0:
                        mfe = max(mfe, (float(hi)-entry)/risk)
                        mae = min(mae, (float(lo)-entry)/risk)
                        if float(lo) <= stop:
                            xi=j; px=stop; reason="STOP"; break
                    else:
                        mfe = max(mfe, (entry-float(lo))/risk)
                        mae = min(mae, (entry-float(hi))/risk)
                        if float(hi) >= stop:
                            xi=j; px=stop; reason="STOP"; break
                    if j > ei and bool(fail30.iloc[j-1]):
                        op = d.Open.iloc[j]
                        if np.isfinite(op):
                            xi=j; px=float(op); reason="H1_50SMA_FAIL"; break
                    j += 1
                if xi is None:
                    xi=last; px=float(d.Close.iloc[xi]); reason="TIME"

                gross_r = side*(px-entry)/risk
                cost_r = (2*COST_BPS_PER_SIDE/10000.0)*entry/risk
                r = gross_r-cost_r
                rows.append({
                    "model":model,"symbol":s,"asset":UNIVERSE[s],"period":labels.iloc[ei],
                    "signal_time":common[i],"entry_time":common[ei],"exit_time":common[xi],
                    "side":"LONG" if side>0 else "SHORT","score":float(score[s].iloc[i]),
                    "entry":entry,"stop":stop,"r":float(r),"mfe_r":float(mfe),"mae_r":float(mae),
                    "hold_bars":int(xi-ei),"exit_reason":reason,
                })
                i = max(i+1, xi+1)
    return pd.DataFrame(rows)


def summarize(g):
    if g.empty: return {"n":0}
    wins = g.loc[g.r>0,"r"]; losses = g.loc[g.r<0,"r"]
    eq = g.sort_values("entry_time").r.cumsum(); dd = eq-eq.cummax()
    return {
        "n":int(len(g)),"win_rate":float((g.r>0).mean()),"avg_r":float(g.r.mean()),
        "median_r":float(g.r.median()),
        "pf":float(wins.sum()/abs(losses.sum())) if len(losses) and abs(losses.sum())>0 else np.nan,
        "total_r":float(g.r.sum()),"max_dd_r":float(dd.min()),
        "avg_mfe_r":float(g.mfe_r.mean()),"avg_mae_r":float(g.mae_r.mean()),
        "avg_hold_bars":float(g.hold_bars.mean()),
    }


def main():
    data = download()
    if len(data) < 5:
        raise RuntimeError(f"insufficient symbols loaded: {list(data)}")
    common, aligned, score, direction, top, trend_ok, zone_ok = build_panel(data)
    trades = simulate(data, common, aligned, score, direction, top, trend_ok, zone_ok)
    if trades.empty:
        raise RuntimeError("no trades")

    rows=[]
    for (m,p),g in trades.groupby(["model","period"]):
        rows.append({"model":m,"description":MODELS[m],"period":p,**summarize(g)})
    summary = pd.DataFrame(rows)

    # Selection strictly on DEV+VAL; OOS only reported after selection.
    candidates=[]
    for m in MODELS:
        a=summary[(summary.model==m)&(summary.period=="DEV")]
        b=summary[(summary.model==m)&(summary.period=="VAL")]
        if len(a) and len(b):
            a=a.iloc[0]; b=b.iloc[0]
            if a.n>=8 and b.n>=5 and a.avg_r>0 and b.avg_r>0:
                score_sel=float(b.avg_r)-0.02*abs(float(b.max_dd_r))
                candidates.append((score_sel,m))
    candidates.sort(reverse=True)
    selected=candidates[0][1] if candidates else None
    selected_oos=None
    if selected:
        z=summary[(summary.model==selected)&(summary.period=="OOS")]
        if len(z): selected_oos=z.iloc[0].to_dict()

    # 0.75% risk simple growth proxy, no concurrency netting.
    summary["growth_proxy_pct_075"] = summary.total_r * 0.75

    trades.to_csv(OUT/"trades.csv",index=False)
    summary.to_csv(OUT/"summary.csv",index=False)
    meta={
        "purpose":"pilot test of TopView-style capital-flow ranking with 30m entries in macro futures",
        "period":PERIOD,"interval":INTERVAL,"top_n":TOP_N,
        "loaded_symbols":list(data.keys()),"requested_symbols":list(UNIVERSE.keys()),
        "models":MODELS,"selected_on_dev_val":selected,"selected_oos":selected_oos,
        "risk_growth_proxy":"total_R * 0.75%; ignores overlapping positions, margin, compounding and firm rules",
        "important_limitations":[
            "This is NOT proprietary MarketGauge TopView/BigView data; capital flow is a causal OHLCV futures proxy using cross-sectional momentum and participation.",
            "FX uses listed currency futures proxies rather than retail spot/CFD feeds.",
            "Yahoo 30m retention limits the sample to about 59 days; this is a pilot, not sufficient evidence for a live edge.",
            "Futures continuous-contract behavior and roll effects differ from prop CFD execution.",
            "Costs are a simple 3 bps/side assumption and do not model symbol-specific spread/slippage.",
            "Pooled R drawdown does not model simultaneous portfolio correlation or prop daily-loss rules.",
        ],
    }
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2,default=str),encoding="utf-8")
    print("LOADED", list(data))
    print(summary.sort_values(["period","avg_r"],ascending=[True,False]).to_string(index=False))
    print("SELECTED", selected)
    print("SELECTED_OOS", selected_oos)

if __name__ == "__main__":
    main()
