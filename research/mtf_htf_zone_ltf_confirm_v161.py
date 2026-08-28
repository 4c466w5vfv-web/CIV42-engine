"""Causal-timing correction for MTF HTF Zone LTF Confirm v1.6.

Strategy logic and parameters are intentionally unchanged. This module only changes
when completed bars/zones become available to the strategy and when LTF confirmation
may begin.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import mtf_htf_zone_ltf_confirm_v16 as base

OUT = Path("research/results/mtf_v161")
OUT.mkdir(parents=True, exist_ok=True)
base.OUT = OUT


def available_at_close(df: pd.DataFrame, delta: pd.Timedelta) -> pd.DataFrame:
    """Treat source timestamps as bar-open labels and expose OHLCV only at bar close."""
    x = df.copy()
    x.index = pd.DatetimeIndex(x.index) + delta
    x.index.name = "datetime"
    return x.sort_index()


def load_asset_causal(name):
    m15, m1, daily, prov = base.load_asset(name)
    # Conservative causal convention shared across Binance and Dukascopy inputs:
    # raw timestamp identifies the interval start; full OHLCV is usable only after close.
    m15 = available_at_close(m15, pd.Timedelta(minutes=15))
    m1 = available_at_close(m1, pd.Timedelta(minutes=1))
    daily = available_at_close(daily, pd.Timedelta(days=1))
    prov = dict(prov)
    prov.update({
        "timing_convention": "raw timestamps treated as bar-open; OHLCV available at bar close",
        "m15_availability_shift": "+15m",
        "m1_availability_shift": "+1m",
        "daily_availability_shift": "+1d",
    })
    return m15, m1, daily, prov


def candidates_causal(m15, m1, daily):
    zones = base.active_zone_map(m15, daily)
    a, b = base.prep_ltf(m15, m1, daily)
    out = []

    for t, r in a.iterrows():
        side = 1 if r.regime == 1 else -1 if r.regime == -1 else 0
        if side == 0 or not np.isfinite(r.vol_ratio) or r.vol_ratio < base.VOL_MIN:
            continue
        if side == 1 and not r.bull15:
            continue
        if side == -1 and not r.bear15:
            continue

        # t is now the 15m bar CLOSE/availability timestamp. A zone may be used only
        # if it was already known when this 15m bar opened, preventing retroactive touch.
        bar_open = t - pd.Timedelta(minutes=15)
        elig = []
        for zt, zs, zlo, zhi, kind, tf, max_age in zones:
            if zs != side or zt > bar_open or t - zt > max_age:
                continue
            if r.low <= zhi and r.high >= zlo:
                elig.append((zt, zs, zlo, zhi, kind, tf, max_age))
        if not elig:
            continue
        z = max(elig, key=lambda q: q[0])

        # 1m confirmation begins strictly AFTER the completed 15m signal is available.
        w = b[(b.index > t) & (b.index <= t + pd.Timedelta(minutes=60))]
        if side == 1:
            w = w[w.bull1]
        else:
            w = w[w.bear1]
        if w.empty:
            continue

        ct = w.index[0]
        cr = w.loc[ct]
        entry = float(cr.close)
        if side == 1:
            stop = min(float(cr.swing_lo), z[2]) if np.isfinite(cr.swing_lo) else z[2]
            if stop >= entry:
                continue
        else:
            stop = max(float(cr.swing_hi), z[3]) if np.isfinite(cr.swing_hi) else z[3]
            if stop <= entry:
                continue
        risk = abs(entry - stop)
        if risk <= 0:
            continue

        out.append({
            "entry_time": ct,
            "signal_available_time": t,
            "signal_bar_open_time": bar_open,
            "side": side,
            "entry": entry,
            "stop": stop,
            "risk": risk,
            "zone_kind": z[4],
            "zone_tf": z[5],
            "zone_time": z[0],
            "zone_available_before_signal_bar": bool(z[0] <= bar_open),
            "vol_ratio": float(r.vol_ratio),
        })

    c = pd.DataFrame(out)
    if not c.empty:
        c = c.sort_values("entry_time").drop_duplicates("entry_time")
    return c


def main():
    name = base.os.environ["ASSET"]
    m15, m1, daily, prov = load_asset_causal(name)
    c = candidates_causal(m15, m1, daily)
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    c.to_csv(out / "candidates.csv", index=False)

    rows = []
    for tp in base.TP_LEVELS:
        tr = base.simulate(c, m1, tp)
        tr.to_csv(out / f"trades_TP{int(tp)}R.csv", index=False)
        if tr.empty:
            rows.append({"asset": name, "tp_R": tp, "split": "ALL", "trades": 0})
            continue
        tr["split"] = tr.entry_time.map(base.split)
        m = base.metrics(tr)
        m.update({"asset": name, "tp_R": tp, "split": "ALL"})
        rows.append(m)
        for sp, g in tr.groupby("split"):
            x = base.metrics(g)
            x.update({"asset": name, "tp_R": tp, "split": sp})
            rows.append(x)

    pd.DataFrame(rows).to_csv(out / "metrics.csv", index=False)
    (out / "meta.json").write_text(json.dumps({
        "asset": name,
        "provenance": prov,
        "m15_bars": len(m15),
        "m1_bars": len(m1),
        "candidate_count": len(c),
        "rules": "UNCHANGED from v1.6: Daily regime; 4H/Daily FVG or OB touch; 15m BOS + volume>=1.2x; 1m micro BOS confirmation; structural stop; fixed 2R/3R/5R TP; 0.05R cost",
        "causal_fix": "OHLCV shifted to close availability; HTF zone must predate 15m bar open; 1m confirmation starts after 15m close",
    }, indent=2, default=str))
    print(pd.DataFrame(rows).to_markdown(index=False))


if __name__ == "__main__":
    main()
