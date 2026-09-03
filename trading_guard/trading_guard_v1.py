from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    sector: str
    direction: str  # LONG | SHORT
    account_equity: float
    entry: float
    stop: float
    risk_pct: float
    target_r: float = 1.5
    bigview_approved: bool = False
    htf_zone_reached: bool = False
    htf_trend_aligned: bool = False
    relative_strength_ok: bool = False
    trigger_confirmed: bool = False
    same_thesis_open_risk_pct: float = 0.0
    total_open_risk_pct: float = 0.0
    trades_today: int = 0
    losses_today: int = 0
    rule_violation_today: bool = False
    reverse_after_stop: bool = False
    h1_flip_confirmed: bool = False


@dataclass
class GuardConfig:
    evaluation_risk_pct: float = 0.75
    max_risk_pct: float = 0.75
    max_total_open_risk_pct: float = 1.50
    max_same_thesis_risk_pct: float = 0.75
    max_trades_per_day: int = 2
    block_after_losses: int = 1
    require_bigview: bool = True
    require_htf_zone: bool = True
    require_htf_trend: bool = True
    require_relative_strength: bool = True
    require_trigger: bool = True
    target_r: float = 1.5


@dataclass
class GuardDecision:
    status: str  # ENTER | BLOCK | WAIT
    reasons: List[str] = field(default_factory=list)
    risk_amount: Optional[float] = None
    stop_distance: Optional[float] = None
    position_units: Optional[float] = None
    take_profit: Optional[float] = None


def evaluate(intent: TradeIntent, cfg: GuardConfig | None = None) -> GuardDecision:
    cfg = cfg or GuardConfig()
    reasons: List[str] = []

    if intent.rule_violation_today:
        return GuardDecision("BLOCK", ["RULE_VIOLATION_DAY_BLOCK"])
    if intent.losses_today >= cfg.block_after_losses:
        return GuardDecision("BLOCK", ["LOSS_DAY_BLOCK"])
    if intent.trades_today >= cfg.max_trades_per_day:
        return GuardDecision("BLOCK", ["MAX_TRADES_REACHED"])
    if intent.reverse_after_stop and not intent.h1_flip_confirmed:
        return GuardDecision("BLOCK", ["REVERSAL_REQUIRES_H1_CONFIRMATION"])

    if cfg.require_bigview and not intent.bigview_approved:
        reasons.append("BIGVIEW_NOT_APPROVED")
    if cfg.require_htf_zone and not intent.htf_zone_reached:
        reasons.append("HTF_ZONE_NOT_REACHED")
    if cfg.require_htf_trend and not intent.htf_trend_aligned:
        reasons.append("HTF_TREND_NOT_ALIGNED")
    if cfg.require_relative_strength and not intent.relative_strength_ok:
        reasons.append("RELATIVE_STRENGTH_NOT_CONFIRMED")
    if cfg.require_trigger and not intent.trigger_confirmed:
        reasons.append("ENTRY_TRIGGER_NOT_CONFIRMED")

    if reasons:
        return GuardDecision("WAIT", reasons)

    if intent.risk_pct <= 0:
        return GuardDecision("BLOCK", ["INVALID_RISK"])
    if intent.risk_pct > cfg.max_risk_pct + 1e-12:
        return GuardDecision("BLOCK", ["RISK_CAP_EXCEEDED"])
    if intent.total_open_risk_pct + intent.risk_pct > cfg.max_total_open_risk_pct + 1e-12:
        return GuardDecision("BLOCK", ["TOTAL_OPEN_RISK_CAP_EXCEEDED"])
    if intent.same_thesis_open_risk_pct + intent.risk_pct > cfg.max_same_thesis_risk_pct + 1e-12:
        return GuardDecision("BLOCK", ["SAME_THESIS_RISK_CAP_EXCEEDED"])

    stop_distance = abs(intent.entry - intent.stop)
    if stop_distance <= 0:
        return GuardDecision("BLOCK", ["INVALID_STOP_DISTANCE"])

    risk_amount = intent.account_equity * (intent.risk_pct / 100.0)
    position_units = risk_amount / stop_distance

    if intent.direction.upper() == "LONG":
        if intent.stop >= intent.entry:
            return GuardDecision("BLOCK", ["LONG_STOP_MUST_BE_BELOW_ENTRY"])
        take_profit = intent.entry + cfg.target_r * stop_distance
    elif intent.direction.upper() == "SHORT":
        if intent.stop <= intent.entry:
            return GuardDecision("BLOCK", ["SHORT_STOP_MUST_BE_ABOVE_ENTRY"])
        take_profit = intent.entry - cfg.target_r * stop_distance
    else:
        return GuardDecision("BLOCK", ["INVALID_DIRECTION"])

    return GuardDecision(
        status="ENTER",
        reasons=["ALL_GATES_PASSED"],
        risk_amount=risk_amount,
        stop_distance=stop_distance,
        position_units=position_units,
        take_profit=take_profit,
    )
