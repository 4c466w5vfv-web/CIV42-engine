from pathlib import Path
import pandas as pd
import research.ma_timeframe_exit_matrix_v38 as base

# Yahoo 1h data must stay inside the provider's rolling 730-day limit.
base.START = "2024-09-06"
base.END = "2026-09-02"
base.PERIODS["DEV"] = (pd.Timestamp("2024-09-06"), pd.Timestamp("2025-06-30"))
base.OUT = Path("research/results/ma_timeframe_exit_matrix_v38b")
base.OUT.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    base.main()
