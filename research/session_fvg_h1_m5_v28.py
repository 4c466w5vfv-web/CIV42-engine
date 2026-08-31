"""v2.8 cross-market discovery wrapper.

Runs the frozen v2.7 Session High/Low -> H1 FVG -> M5 execution logic
unchanged across a broader asset universe. No per-asset tuning.
"""
from pathlib import Path
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

if __name__ == "__main__":
    v27.main()
