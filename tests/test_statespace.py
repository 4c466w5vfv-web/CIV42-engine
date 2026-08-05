from __future__ import annotations

"""42차원 상태공간 — 좌표계 무결성과 갱신 규약."""

import pytest

from ark42 import statespace as ss


def test_registry_is_a_valid_42_axis_coordinate_system():
    ss.validate_registry()
    assert len(ss.REGISTRY) == 42
    # 도메인 균형: 7개 도메인 × 정확히 6축
    domains = {}
    for v in ss.REGISTRY:
        domains.setdefault(v["domain"], []).append(v["id"])
    assert len(domains) == 7
    assert all(len(axes) == 6 for axes in domains.values())


def test_every_axis_declares_how_it_is_measured():
    for v in ss.REGISTRY:
        assert v["measurement"], v["id"]
        assert v["unit"], v["id"]


def test_observation_outside_coordinate_system_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK42_DATA_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="unknown axes"):
        ss.record_observation("s1", {"made.up_axis": 1}, source="test")
    with pytest.raises(ValueError, match="empty"):
        ss.record_observation("s1", {}, source="test")


def test_state_updates_carry_delta_and_history_is_append_only(tmp_path, monkeypatch):
    monkeypatch.setenv("ARK42_DATA_DIR", str(tmp_path))
    s1 = ss.record_observation("s2", {"resource.liquidity": 0.8},
                               source="ledger")
    assert s1["delta"]["resource.liquidity"] == {"from": None, "to": 0.8}
    s2 = ss.record_observation("s2", {"resource.liquidity": 2.5,
                                      "meaning.goal_clarity": "yes"},
                               source="ledger")
    # 최신 상태 = 이전 상태 + 이번 관측, 델타는 바뀐 것만
    assert s2["values"]["resource.liquidity"] == 2.5
    assert s2["delta"]["resource.liquidity"]["from"] == 0.8
    assert ss.load_history("s2")[0]["values"]["resource.liquidity"] == 0.8


def test_coverage_reports_honest_progress_toward_data_requirement(tmp_path,
                                                                  monkeypatch):
    monkeypatch.setenv("ARK42_DATA_DIR", str(tmp_path))
    ss.record_observation("s3", {"resource.liquidity": 1.0,
                                 "meaning.horizon": 50}, source="test")
    cov = ss.coverage("s3")
    assert cov["observed_axes"] == 2 and cov["total_axes"] == 42
    assert len(cov["missing"]) == 40      # 미관측은 미관측이라고 말한다
