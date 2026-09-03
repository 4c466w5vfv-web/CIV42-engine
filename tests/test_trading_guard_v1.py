from trading_guard.trading_guard_v1 import GuardConfig, TradeIntent, evaluate


def base_intent(**overrides):
    d = dict(
        symbol="TEST",
        sector="TECH",
        direction="LONG",
        account_equity=200_000,
        entry=100.0,
        stop=98.0,
        risk_pct=0.75,
        bigview_approved=True,
        htf_zone_reached=True,
        htf_trend_aligned=True,
        relative_strength_ok=True,
        trigger_confirmed=True,
    )
    d.update(overrides)
    return TradeIntent(**d)


def test_valid_trade_enters_and_sizes():
    d = evaluate(base_intent())
    assert d.status == "ENTER"
    assert round(d.risk_amount, 2) == 1500.00
    assert round(d.position_units, 2) == 750.00
    assert round(d.take_profit, 2) == 103.00


def test_waits_when_zone_not_reached():
    d = evaluate(base_intent(htf_zone_reached=False))
    assert d.status == "WAIT"
    assert "HTF_ZONE_NOT_REACHED" in d.reasons


def test_blocks_after_first_loss():
    d = evaluate(base_intent(losses_today=1))
    assert d.status == "BLOCK"
    assert d.reasons == ["LOSS_DAY_BLOCK"]


def test_blocks_risk_above_cap():
    d = evaluate(base_intent(risk_pct=1.0))
    assert d.status == "BLOCK"
    assert d.reasons == ["RISK_CAP_EXCEEDED"]


def test_blocks_same_thesis_cluster_over_cap():
    d = evaluate(base_intent(same_thesis_open_risk_pct=0.40))
    assert d.status == "BLOCK"
    assert d.reasons == ["SAME_THESIS_RISK_CAP_EXCEEDED"]


def test_blocks_reverse_without_h1_flip():
    d = evaluate(base_intent(reverse_after_stop=True, h1_flip_confirmed=False))
    assert d.status == "BLOCK"
    assert d.reasons == ["REVERSAL_REQUIRES_H1_CONFIRMATION"]


def test_short_tp_direction():
    d = evaluate(base_intent(direction="SHORT", entry=100.0, stop=102.0))
    assert d.status == "ENTER"
    assert round(d.take_profit, 2) == 97.00
