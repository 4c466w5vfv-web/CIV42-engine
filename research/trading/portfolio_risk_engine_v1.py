from dataclasses import dataclass
from typing import List, Dict, Tuple

# ARK-42 Long-Only Portfolio Risk Engine v1
# Hard constraints are intentionally simple and auditable.
# 4% is a ceiling, never a target.
# Validation trigger: deterministic risk-control checks only; no performance claim.

TRADE_CAP = 0.005      # 0.50%
CLUSTER_CAP = 0.015    # 1.50%
SLEEVE_CAP = 0.020     # 2.00%
TOTAL_CAP = 0.040      # 4.00%

@dataclass(frozen=True)
class PositionRisk:
    symbol: str
    strategy: str      # MOMENTUM or MR
    cluster: str       # EQUITY, COMMODITY, RATES, FX, etc.
    risk_pct: float    # stop-defined loss as fraction of equity


def dd_multiplier(drawdown: float) -> float:
    """drawdown is a positive fraction: 0.025 == 2.5%."""
    if drawdown >= 0.04:
        return 0.0
    if drawdown >= 0.03:
        return 0.50
    if drawdown >= 0.02:
        return 0.75
    return 1.0


def aggregate(positions: List[PositionRisk]) -> Dict[str, object]:
    clusters: Dict[str, float] = {}
    sleeves: Dict[str, float] = {}
    total = 0.0
    for p in positions:
        total += p.risk_pct
        clusters[p.cluster] = clusters.get(p.cluster, 0.0) + p.risk_pct
        sleeves[p.strategy] = sleeves.get(p.strategy, 0.0) + p.risk_pct
    return {'total': total, 'clusters': clusters, 'sleeves': sleeves}


def approve_order(existing: List[PositionRisk], candidate: PositionRisk, drawdown: float) -> Tuple[str, float, List[str]]:
    """Return (decision, approved_risk_pct, reasons).

    The candidate is first throttled by portfolio drawdown. Then its risk is
    reduced as necessary to fit the per-trade, cluster, sleeve and total caps.
    No negative capacity is permitted. A zero-sized order is REJECT.
    """
    mult = dd_multiplier(drawdown)
    if mult == 0.0:
        return 'REJECT', 0.0, ['DD_STOP_4PCT']

    requested = min(max(candidate.risk_pct, 0.0), TRADE_CAP) * mult
    a = aggregate(existing)
    cluster_used = float(a['clusters'].get(candidate.cluster, 0.0))
    sleeve_used = float(a['sleeves'].get(candidate.strategy, 0.0))
    total_used = float(a['total'])

    capacity = min(
        requested,
        max(0.0, CLUSTER_CAP - cluster_used),
        max(0.0, SLEEVE_CAP - sleeve_used),
        max(0.0, TOTAL_CAP - total_used),
    )

    reasons = []
    if candidate.risk_pct > TRADE_CAP:
        reasons.append('TRADE_CAP_REDUCED')
    if mult < 1.0:
        reasons.append(f'DD_THROTTLE_{mult:.2f}X')
    if cluster_used + requested > CLUSTER_CAP + 1e-12:
        reasons.append('CLUSTER_CAP_REDUCED')
    if sleeve_used + requested > SLEEVE_CAP + 1e-12:
        reasons.append('SLEEVE_CAP_REDUCED')
    if total_used + requested > TOTAL_CAP + 1e-12:
        reasons.append('TOTAL_CAP_REDUCED')

    if capacity <= 1e-12:
        return 'REJECT', 0.0, reasons or ['NO_RISK_CAPACITY']
    decision = 'APPROVE' if abs(capacity - candidate.risk_pct) <= 1e-12 else 'REDUCE'
    return decision, capacity, reasons


def self_test() -> None:
    # 1) normal 0.5% trade passes
    d,r,_ = approve_order([], PositionRisk('NAS100','MOMENTUM','EQUITY',0.005), 0.0)
    assert d == 'APPROVE' and abs(r-0.005) < 1e-12

    # 2) 2.5% DD throttles 0.5% to 0.375%
    d,r,reasons = approve_order([], PositionRisk('NAS100','MOMENTUM','EQUITY',0.005), 0.025)
    assert d == 'REDUCE' and abs(r-0.00375) < 1e-12 and 'DD_THROTTLE_0.75X' in reasons

    # 3) equity cluster already at 1.25%, next 0.5% is reduced to 0.25%
    ex = [PositionRisk('SPX500','MOMENTUM','EQUITY',0.0075), PositionRisk('US30','MR','EQUITY',0.005)]
    d,r,reasons = approve_order(ex, PositionRisk('NAS100','MOMENTUM','EQUITY',0.005), 0.0)
    assert d == 'REDUCE' and abs(r-0.0025) < 1e-12 and 'CLUSTER_CAP_REDUCED' in reasons

    # 4) total risk at 4% rejects any new order
    ex = [
        PositionRisk('A','MOMENTUM','EQUITY',0.010),
        PositionRisk('B','MR','COMMODITY',0.010),
        PositionRisk('C','MOMENTUM','RATES',0.010),
        PositionRisk('D','MR','FX',0.010),
    ]
    d,r,_ = approve_order(ex, PositionRisk('E','MOMENTUM','CRYPTO',0.005), 0.0)
    assert d == 'REJECT' and r == 0.0

    # 5) portfolio DD >=4% is a hard stop independent of unused capacity
    d,r,reasons = approve_order([], PositionRisk('NAS100','MOMENTUM','EQUITY',0.005), 0.04)
    assert d == 'REJECT' and r == 0.0 and reasons == ['DD_STOP_4PCT']

    print('PORTFOLIO_RISK_ENGINE_V1: 5/5 deterministic tests PASS')
    print('CAPS trade=0.50% cluster=1.50% sleeve=2.00% total=4.00%')
    print('DD throttle <2%=1.00x, 2-3%=0.75x, 3-4%=0.50x, >=4%=STOP')


if __name__ == '__main__':
    self_test()
