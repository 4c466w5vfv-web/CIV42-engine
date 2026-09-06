import io
import math
import urllib.request
from dataclasses import dataclass, asdict

import pandas as pd

SOURCES = {
    "NDX": "https://raw.githubusercontent.com/simom1/XAUUSD-history/main/TradingView_Deep_Datasets/TVC_NDX/TVC_NDX_D.csv",
    "SPX": "https://raw.githubusercontent.com/simom1/XAUUSD-history/main/TradingView_Deep_Datasets/TVC_SPX/TVC_SPX_D.csv",
}

START_DATE = pd.Timestamp("2023-01-01")
RISK_PER_TRADE = 0.01
SMA_LEN = 200
SLOPE_LOOKBACK = 20
ATR_LEN = 14
ATR_MULT = 2.0


def load_data(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=30) as r:
        raw = r.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    expected = ["timestamp", "datetime", "open", "high", "low", "close", "volume"]
    if list(df.columns) != expected:
        raise RuntimeError(f"unexpected columns: {df.columns.tolist()}")
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values("datetime").drop_duplicates("datetime").reset_index(drop=True)
    return df


def wilder_atr(df: pd.DataFrame, n: int = ATR_LEN) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry: float
    exit: float
    entry_atr: float
    initial_stop: float
    initial_risk: float
    r_gross: float
    exit_reason: str


def run_backtest(symbol: str, df: pd.DataFrame):
    df = df.copy()
    df["atr14"] = wilder_atr(df)
    df["sma200"] = df["close"].rolling(SMA_LEN).mean()
    df["sma_slope"] = df["sma200"] - df["sma200"].shift(SLOPE_LOOKBACK)
    df["trend_ok"] = (df["close"] > df["sma200"]) & (df["sma_slope"] > 0)

    # Warm up indicators using pre-2023 data but do not enter before START_DATE.
    in_pos = False
    entry = entry_atr = initial_stop = initial_risk = None
    entry_i = None
    active_stop = None
    highest_high = None
    trades = []

    equity = 1.0
    equity_curve = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev = df.iloc[i - 1]
        date = row["datetime"]

        # Entry uses only completed previous-day regime/ATR and fills at today's open.
        if (not in_pos) and date >= START_DATE and bool(prev["trend_ok"]) and pd.notna(prev["atr14"]):
            entry = float(row["open"])
            entry_atr = float(prev["atr14"])
            initial_risk = ATR_MULT * entry_atr
            initial_stop = entry - initial_risk
            active_stop = initial_stop
            highest_high = entry
            entry_i = i
            in_pos = True

        if in_pos:
            # Stop for this bar was fixed from information available before this bar.
            low = float(row["low"])
            open_ = float(row["open"])
            if low <= active_stop:
                exit_px = open_ if open_ < active_stop else active_stop
                r = (exit_px - entry) / initial_risk
                trades.append(
                    Trade(
                        symbol=symbol,
                        entry_date=str(df.iloc[entry_i]["datetime"]),
                        exit_date=str(date),
                        entry=entry,
                        exit=exit_px,
                        entry_atr=entry_atr,
                        initial_stop=initial_stop,
                        initial_risk=initial_risk,
                        r_gross=r,
                        exit_reason="2ATR_TRAIL",
                    )
                )
                equity *= max(0.0, 1.0 + RISK_PER_TRADE * r)
                equity_curve.append((date, equity))
                in_pos = False
                entry = entry_atr = initial_stop = initial_risk = None
                entry_i = None
                active_stop = None
                highest_high = None
                continue

            # End-of-bar update; applies from the next bar only.
            highest_high = max(highest_high, float(row["high"]))
            if pd.notna(row["atr14"]):
                candidate = highest_high - ATR_MULT * float(row["atr14"])
                active_stop = max(active_stop, candidate)

    # Mark-to-market final open trade at last close; keep separate from closed-trade stats.
    open_trade = None
    if in_pos:
        last = df.iloc[-1]
        open_trade = {
            "entry_date": str(df.iloc[entry_i]["datetime"]),
            "entry": entry,
            "last_date": str(last["datetime"]),
            "last_close": float(last["close"]),
            "active_stop": float(active_stop),
            "mtm_r": (float(last["close"]) - entry) / initial_risk,
        }

    tdf = pd.DataFrame([asdict(t) for t in trades])

    def stats_for_cost(cost_atr_rt: float):
        if tdf.empty:
            return {}
        # Round-trip cost expressed as a fraction of entry ATR in price units.
        cost_r = (cost_atr_rt * tdf["entry_atr"]) / tdf["initial_risk"]
        net_r = tdf["r_gross"] - cost_r
        wins = net_r[net_r > 0]
        losses = net_r[net_r < 0]
        pf = float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else math.inf
        eq = (1.0 + RISK_PER_TRADE * net_r).cumprod()
        dd = eq / eq.cummax() - 1.0
        return {
            "trades": int(len(net_r)),
            "win_rate": float((net_r > 0).mean()),
            "expectancy_R": float(net_r.mean()),
            "median_R": float(net_r.median()),
            "profit_factor": pf,
            "sum_R": float(net_r.sum()),
            "equity_return_pct_at_1pct_risk": float((eq.iloc[-1] - 1.0) * 100),
            "max_drawdown_pct_at_1pct_risk": float(dd.min() * 100),
            "best_R": float(net_r.max()),
            "worst_R": float(net_r.min()),
        }

    valid = df[(df["datetime"] >= START_DATE) & df["sma200"].notna()].copy()
    first_valid = valid.iloc[0]["datetime"] if len(valid) else None
    bh = None
    if len(valid) >= 2:
        bh = (float(valid.iloc[-1]["close"]) / float(valid.iloc[0]["close"]) - 1.0) * 100

    return {
        "symbol": symbol,
        "source_first": str(df.iloc[0]["datetime"]),
        "source_last": str(df.iloc[-1]["datetime"]),
        "first_testable_after_sma200": str(first_valid) if first_valid is not None else None,
        "buy_hold_pct_from_first_testable": bh,
        "cost_0": stats_for_cost(0.0),
        "cost_0_05ATR": stats_for_cost(0.05),
        "cost_0_10ATR": stats_for_cost(0.10),
        "open_trade": open_trade,
        "trades": tdf,
    }


def fmt(x, nd=2):
    if x is None:
        return "NA"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.{nd}f}"


def main():
    results = []
    for symbol, url in SOURCES.items():
        df = load_data(url)
        res = run_backtest(symbol, df)
        results.append(res)

    print("# SMA200 + 2ATR Trend Pilot")
    print(f"Rules: prev close > SMA200, 20D SMA200 slope > 0, next-day open entry, Wilder ATR14, 2ATR initial/trailing stop, 1% equity risk per trade, no lookahead stop update.")
    print("Data source: simom1/XAUUSD-history TradingView daily datasets. Volume is ignored.")
    print()
    for res in results:
        print(f"## {res['symbol']}")
        print(f"source: {res['source_first']} -> {res['source_last']}")
        print(f"first testable after SMA200 warmup: {res['first_testable_after_sma200']}")
        print(f"buy&hold from first testable close: {fmt(res['buy_hold_pct_from_first_testable'])}%")
        for label, key in [("0 cost", "cost_0"), ("0.05 ATR RT", "cost_0_05ATR"), ("0.10 ATR RT", "cost_0_10ATR")]:
            s = res[key]
            print(
                f"{label}: trades={s.get('trades','NA')}, win={fmt(s.get('win_rate',0)*100)}%, "
                f"exp={fmt(s.get('expectancy_R'))}R, median={fmt(s.get('median_R'))}R, "
                f"PF={fmt(s.get('profit_factor'))}, sum={fmt(s.get('sum_R'))}R, "
                f"eq@1%R={fmt(s.get('equity_return_pct_at_1pct_risk'))}%, "
                f"MDD@1%R={fmt(s.get('max_drawdown_pct_at_1pct_risk'))}%"
            )
        if res["open_trade"]:
            ot = res["open_trade"]
            print(f"open trade: entry {ot['entry_date']} @ {ot['entry']:.2f}; last {ot['last_date']} close {ot['last_close']:.2f}; stop {ot['active_stop']:.2f}; MTM {ot['mtm_r']:.2f}R")
        else:
            print("open trade: none")
        print("closed trades:")
        if res["trades"].empty:
            print("  none")
        else:
            for _, t in res["trades"].iterrows():
                print(f"  {t['entry_date']} -> {t['exit_date']} | {t['entry']:.2f} -> {t['exit']:.2f} | {t['r_gross']:.2f}R")
        print()


if __name__ == "__main__":
    main()
