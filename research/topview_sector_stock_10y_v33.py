from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

START = "2016-01-01"
END = "2026-09-01"
BENCH = "SPY"
COST_BPS_PER_SIDE = 5.0
SEED = 42
OUT = Path("research/results/topview_sector_stock_v33")
OUT.mkdir(parents=True, exist_ok=True)

# Current large/liquid US names grouped by sector. This introduces survivorship bias;
# the study is an edge-ablation experiment, not a constituent-history-perfect index backtest.
SECTORS = {
    "TECH": {"etf": "XLK", "stocks": ["AAPL","MSFT","NVDA","AVGO","ORCL","CRM","ADBE","CSCO","IBM","NOW"]},
    "SEMIS": {"etf": "SMH", "stocks": ["NVDA","AVGO","AMD","QCOM","TXN","MU","AMAT","LRCX","KLAC","INTC"]},
    "FIN": {"etf": "XLF", "stocks": ["JPM","BAC","WFC","GS","MS","C","BLK","SCHW","AXP","SPGI"]},
    "ENERGY": {"etf": "XLE", "stocks": ["XOM","CVX","COP","EOG","SLB","MPC","PSX","VLO","OXY","HAL"]},
    "INDUSTRIAL": {"etf": "XLI", "stocks": ["GE","CAT","HON","UNP","RTX","BA","UPS","DE","ETN","LMT"]},
    "MATERIALS": {"etf": "XLB", "stocks": ["LIN","APD","SHW","ECL","NEM","FCX","NUE","DOW","DD","PPG"]},
    "HEALTH": {"etf": "XLV", "stocks": ["LLY","UNH","JNJ","ABBV","MRK","TMO","ABT","AMGN","GILD","ISRG"]},
    "DISCRETIONARY": {"etf": "XLY", "stocks": ["AMZN","TSLA","HD","MCD","LOW","BKNG","TJX","SBUX","NKE","GM"]},
    "STAPLES": {"etf": "XLP", "stocks": ["WMT","COST","PG","KO","PEP","PM","MO","CL","MDLZ","KMB"]},
    "UTILITIES": {"etf": "XLU", "stocks": ["NEE","SO","DUK","CEG","AEP","SRE","D","EXC","XEL","PEG"]},
    "REAL_ESTATE": {"etf": "XLRE", "stocks": ["PLD","AMT","EQIX","WELL","SPG","O","PSA","DLR","CCI","CBRE"]},
    "COMM": {"etf": "XLC", "stocks": ["META","GOOGL","NFLX","TMUS","VZ","T","DIS","CMCSA","CHTR","EA"]},
}

PERIODS = {
    "DEV": (pd.Timestamp("2016-01-01"), pd.Timestamp("2021-12-31")),
    "VAL": (pd.Timestamp("2022-01-01"), pd.Timestamp("2023-12-31")),
    "OOS": (pd.Timestamp("2024-01-01"), pd.Timestamp("2026-08-31")),
}

MODELS = ["TREND_ONLY", "SECTOR_LEADER", "SECTOR_STOCK_LEADER", "FULL_TOPVIEW", "FULL_TOPVIEW_PULLBACK"]
EXITS = ["FIXED_3R", "EMA20_TRAIL"]


def unique_symbols():
    syms = {BENCH}
    for v in SECTORS.values():
        syms.add(v["etf"])
        syms.update(v["stocks"])
    return sorted(syms)


def atr(df: pd.DataFrame, n=14):
    pc = df["Close"].shift(1)
    tr = pd.concat([(df["High"]-df["Low"]), (df["High"]-pc).abs(), (df["Low"]-pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def download():
    syms = unique_symbols()
    raw = yf.download(syms, start=START, end=END, auto_adjust=True, actions=False, group_by="ticker", threads=True, progress=False)
    out = {}
    for s in syms:
        try:
            df = raw[s].copy() if isinstance(raw.columns, pd.MultiIndex) else raw.copy()
        except Exception:
            continue
        df.columns = [str(c).title() for c in df.columns]
        need = ["Open","High","Low","Close","Volume"]
        if not set(need).issubset(df.columns):
            continue
        df = df[need].dropna(subset=["Open","High","Low","Close"]).copy()
        df.index = pd.to_datetime(df.index).tz_localize(None)
        if len(df) >= 500:
            out[s] = df
    if BENCH not in out:
        raise RuntimeError("SPY missing")
    return out


def add_features(data):
    spy = data[BENCH]["Close"]
    spy20, spy60 = spy.pct_change(20), spy.pct_change(60)
    feat = {}
    for s, d0 in data.items():
        d = d0.copy(); c=d["Close"]
        d["ema10"] = c.ewm(span=10, adjust=False).mean(); d["ema20"] = c.ewm(span=20, adjust=False).mean()
        d["sma50"] = c.rolling(50).mean(); d["sma200"] = c.rolling(200).mean(); d["atr14"] = atr(d)
        d["mom20"] = c.pct_change(20); d["mom60"] = c.pct_change(60); d["mom126"] = c.pct_change(126)
        d["trend"] = (d["ema10"] > d["ema20"]) & (c > d["sma50"]) & (d["sma50"] > d["sma200"])
        d["rs"] = (0.4*d["mom20"] + 0.6*d["mom60"]) - (0.4*spy20 + 0.6*spy60)
        signed = np.sign(c.diff()).fillna(0) * d["Volume"].fillna(0)
        d["vp"] = signed.rolling(20).sum() / d["Volume"].rolling(20).sum().replace(0,np.nan)
        d["score_raw"] = 0.20*d["mom20"] + 0.35*d["mom60"] + 0.45*d["mom126"]
        touch = (d["Low"] <= d["ema20"] + 0.20*d["atr14"]) & (d["High"] >= d["ema20"] - 0.20*d["atr14"])
        d["pullback"] = touch.rolling(3).max().fillna(0).astype(bool) & (c > d["ema10"]) & (c.shift(1) <= d["ema10"].shift(1))
        feat[s]=d

    # Cross-sector rank on sector ETFs only.
    etfs = [v["etf"] for v in SECTORS.values() if v["etf"] in feat]
    sec_scores = pd.concat({s: feat[s]["score_raw"] for s in etfs}, axis=1)
    sec_rank = sec_scores.rank(axis=1, pct=True)
    for s in etfs:
        feat[s]["sector_rank"] = sec_rank[s]

    # Within-sector stock rank on medium/short relative strength.
    for sec, v in SECTORS.items():
        stocks = [s for s in v["stocks"] if s in feat]
        if not stocks: continue
        mat = pd.concat({s: feat[s]["score_raw"] for s in stocks}, axis=1)
        ranks = mat.rank(axis=1, pct=True)
        for s in stocks:
            feat[s]["stock_rank"] = ranks[s]
    return feat


def stock_sector_map():
    m={}
    for sec,v in SECTORS.items():
        for s in v["stocks"]: m[s]=sec
    return m


@dataclass
class Trade:
    symbol:str; sector:str; model:str; exit_model:str; signal_date:pd.Timestamp; entry_date:pd.Timestamp; exit_date:pd.Timestamp
    entry:float; stop:float; exit:float; r:float; mfe_r:float; mae_r:float; holding_days:int


def signal_for(feat, symbol, model):
    sec = stock_sector_map()[symbol]; etf = SECTORS[sec]["etf"]
    d=feat[symbol]; e=feat.get(etf)
    if e is None: return pd.Series(False,index=d.index)
    e = e.reindex(d.index)
    base = d["trend"].fillna(False)
    sec_leader = (e["trend"].fillna(False) & (e["rs"]>0) & (e.get("sector_rank", pd.Series(index=e.index,dtype=float))>=0.70)).fillna(False)
    stock_leader = (d.get("stock_rank", pd.Series(index=d.index,dtype=float))>=0.70) & (d["rs"]>0)
    if model=="TREND_ONLY": return base
    if model=="SECTOR_LEADER": return base & sec_leader
    if model=="SECTOR_STOCK_LEADER": return base & sec_leader & stock_leader
    full = base & sec_leader & stock_leader & (e["vp"]>0) & (d["vp"]>0) & (d["ema10"]>d["ema20"])
    if model=="FULL_TOPVIEW": return full
    if model=="FULL_TOPVIEW_PULLBACK": return full & d["pullback"]
    raise ValueError(model)


def simulate(feat, symbol, model, exit_model):
    d=feat[symbol]; sec=stock_sector_map()[symbol]; sig=signal_for(feat,symbol,model)
    trades=[]; i=210; n=len(d)
    while i<n-2:
        if not bool(sig.iloc[i]): i+=1; continue
        ei=i+1; entry=float(d["Open"].iloc[ei]); av=float(d["atr14"].iloc[i])
        if not np.isfinite(av) or av<=0: i+=1; continue
        swing=float(d["Low"].iloc[max(0,i-9):i+1].min()); stop=min(swing, entry-av); risk=entry-stop
        if risk<=0 or risk/entry>0.12: i+=1; continue
        xp=None; xi=None; mh=entry; ml=entry; j=ei
        while j<n:
            hi=float(d["High"].iloc[j]); lo=float(d["Low"].iloc[j]); cl=float(d["Close"].iloc[j]); mh=max(mh,hi); ml=min(ml,lo)
            if lo<=stop: xp=stop; xi=j; break
            if exit_model=="FIXED_3R" and hi>=entry+3*risk: xp=entry+3*risk; xi=j; break
            if exit_model=="EMA20_TRAIL" and cl<float(d["ema20"].iloc[j]) and j+1<n: xp=float(d["Open"].iloc[j+1]); xi=j+1; break
            if j-ei>=60: xp=cl; xi=j; break
            j+=1
        if xi is None: break
        gross=(xp-entry)/risk; cost=(2*COST_BPS_PER_SIDE/10000.0)*entry/risk; rr=gross-cost
        trades.append(Trade(symbol,sec,model,exit_model,d.index[i],d.index[ei],d.index[xi],entry,stop,float(xp),float(rr),float((mh-entry)/risk),float((ml-entry)/risk),int(xi-ei)))
        i=max(xi+1,i+1)
    return trades


def period(ts):
    for k,(a,b) in PERIODS.items():
        if a<=ts<=b: return k
    return None


def summary(g):
    if len(g)==0:return {"n":0}
    wins=g.loc[g.r>0,"r"]; losses=g.loc[g.r<0,"r"]; eq=g.r.cumsum(); dd=eq-eq.cummax(); std=g.r.std(ddof=1)
    return {"n":int(len(g)),"win_rate":float((g.r>0).mean()),"avg_r":float(g.r.mean()),"median_r":float(g.r.median()),
            "avg_win_r":float(wins.mean()) if len(wins) else np.nan,"avg_loss_r":float(losses.mean()) if len(losses) else np.nan,
            "profit_factor":float(wins.sum()/abs(losses.sum())) if len(losses) and losses.sum()!=0 else np.nan,
            "trade_sharpe":float(g.r.mean()/std*math.sqrt(len(g))) if std and np.isfinite(std) else np.nan,
            "max_dd_r":float(dd.min()),"avg_mfe_r":float(g.mfe_r.mean()),"avg_mae_r":float(g.mae_r.mean()),"avg_hold_days":float(g.holding_days.mean()),"total_r":float(g.r.sum())}


def main():
    data=download(); feat=add_features(data); smap=stock_sector_map(); stocks=[s for s in smap if s in feat and SECTORS[smap[s]]["etf"] in feat]
    alltr=[]
    for model in MODELS:
        for ex in EXITS:
            for s in stocks: alltr.extend(simulate(feat,s,model,ex))
    df=pd.DataFrame([t.__dict__ for t in alltr]); df["period"]=df.entry_date.map(period); df=df.dropna(subset=["period"])
    rows=[]
    for (m,e,p),g in df.groupby(["model","exit_model","period"]): rows.append({"model":m,"exit_model":e,"period":p,**summary(g.sort_values("entry_date"))})
    res=pd.DataFrame(rows).sort_values(["period","avg_r"],ascending=[True,False])

    # Predeclare selection: among models with DEV/VAL avg_r>0 and n>=100 in each, choose highest VAL avg_r.
    piv={}
    for _,r in res.iterrows(): piv[(r.model,r.exit_model,r.period)]=r
    cand=[]
    for m in MODELS:
        for e in EXITS:
            a=piv.get((m,e,"DEV")); b=piv.get((m,e,"VAL"))
            if a is not None and b is not None and a.n>=100 and b.n>=100 and a.avg_r>0 and b.avg_r>0: cand.append((float(b.avg_r),m,e))
    cand.sort(reverse=True); selected=cand[0][1:] if cand else None
    sel_oos=None
    if selected:
        m,e=selected; rr=piv.get((m,e,"OOS")); sel_oos=None if rr is None else rr.to_dict()

    # OOS sector diagnostics for full model only.
    secrows=[]
    oos=df[df.period=="OOS"]
    for (m,e,sec),g in oos.groupby(["model","exit_model","sector"]):
        if m in ["FULL_TOPVIEW","SECTOR_STOCK_LEADER","TREND_ONLY"]:
            secrows.append({"model":m,"exit_model":e,"sector":sec,**summary(g.sort_values("entry_date"))})
    secres=pd.DataFrame(secrows)

    df.to_csv(OUT/"trades.csv",index=False); res.to_csv(OUT/"summary.csv",index=False); secres.to_csv(OUT/"sector_oos.csv",index=False)
    meta={"start":START,"end":END,"symbols_requested":len(unique_symbols()),"symbols_loaded":len(data),"stocks_traded":len(stocks),"trades":len(df),
          "survivorship_bias":True,"bigview_proprietary":False,"cost_bps_per_side":COST_BPS_PER_SIDE,"selection":"VAL only; OOS untouched","selected":selected,"selected_oos":sel_oos}
    (OUT/"meta.json").write_text(json.dumps(meta,indent=2,default=str))
    print("=== TOPVIEW SECTOR->STOCK 10Y ABLATION v3.3 ===")
    print(json.dumps(meta,indent=2,default=str)); print("\nSUMMARY\n",res.to_string(index=False))
    if len(secres): print("\nOOS SECTOR DIAGNOSTICS\n",secres.sort_values(["model","exit_model","avg_r"],ascending=[True,True,False]).to_string(index=False))

if __name__=="__main__": main()
