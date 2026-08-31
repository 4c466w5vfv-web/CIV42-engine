"""v2.8 cross-market discovery wrapper.

Runs the frozen v2.7 Session High/Low -> H1 FVG -> M5 execution logic
unchanged across a broader asset universe. No per-asset tuning.

Brent note: Dukascopy's daily download for brentcmdusd is unreliable in this
workflow. For Brent only, D1 is causally resampled from already-downloaded H1
bars. This changes data plumbing, not signal parameters.
"""
from pathlib import Path
import pandas as pd
import session_fvg_h1_m5_v27 as v27

v27.OUT = Path("research/results/session_fvg_h1_m5_v28")
v27.OUT.mkdir(parents=True, exist_ok=True)

v27.ASSETS = {
    # FX majors / crosses
    "EURUSD": "eurusd",
    "GBPUSD": "gbpusd",
    "USDJPY": "usdjpy",
    "EURJPY": "eurjpy",
    "GBPJPY": "gbpjpy",
    "AUDUSD": "audusd",
    "EURGBP": "eurgbp",

    # Metals
    "GOLD": "xauusd",
    "SILVER": "xagusd",

    # Equity indices / proxies already used elsewhere in this repo
    "NQ_PROXY": "usatechidxusd",
    "SP500_PROXY": "usa500idxusd",

    # Commodities / rates already used elsewhere in this repo
    "WTI": "lightcmdusd",
    "BRENT": "brentcmdusd",
    "NATGAS": "gascmdusd",
    "COPPER": "coppercmdusd",
    "USTBOND": "ustbondtrusd",
}

_original_load_asset = v27.load_asset

def _resample_h1_to_d1(h1: pd.DataFrame) -> pd.DataFrame:
    # h1 index is already shifted to close-availability by v27.available_at_close.
    # Aggregate only completed H1 bars into UTC calendar days, then shift the
    # resulting daily bar to next-day availability exactly like v27 D1 handling.
    x = h1.copy()
    if x.index.tz is None:
        x.index = x.index.tz_localize("UTC")
    else:
        x.index = x.index.tz_convert("UTC")
    d = x.resample("1D", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"
    }).dropna(subset=["open", "high", "low", "close"])
    # A UTC day is not knowable until it has closed.
    d.index = d.index + pd.Timedelta(days=1)
    d.index.name = "datetime"
    return d.sort_index()

def load_asset_v28(name: str):
    if name != "BRENT":
        return _original_load_asset(name)

    inst = v27.ASSETS[name]
    base = Path("research/duka_v27") / name
    p5 = v27.duka.run_cli(inst, "m5", "2020-01-01", "2026-08-01", base / "m5")
    p1h = v27.duka.run_cli(inst, "h1", "2019-01-01", "2026-08-01", base / "h1")
    m5 = v27.available_at_close(v27.duka.read_duka(p5), pd.Timedelta(minutes=5))
    h1 = v27.available_at_close(v27.duka.read_duka(p1h), pd.Timedelta(hours=1))
    d1 = _resample_h1_to_d1(h1)
    prov = {
        "source": "Dukascopy",
        "instrument": inst,
        "timing": "M5/H1 bar-open labels shifted to close availability; D1 causally resampled from H1",
        "d1_fallback": "H1->D1 because brentcmdusd direct D1 download failed repeatedly",
    }
    return m5, h1, d1, prov

v27.load_asset = load_asset_v28

if __name__ == "__main__":
    v27.main()
