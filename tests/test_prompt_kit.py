from __future__ import annotations

"""프롬프트 키트가 엔진의 실제 프롬프트와 어긋나지 않는지 고정한다.

키트를 보고 다른 AI로 만든 출력이 파이프라인 검증에서 떨어지는 사고는
사용자에게는 원인 불명의 실패로 보인다. 문서가 코드에서 생성된다는 사실만으로는
부족하고, 재생성을 잊었을 때를 잡아야 한다.
"""

from pathlib import Path

import pytest

from ark42 import prompts
from ark42.disciplines import DISCIPLINES
from ark42.ontology import CRITERIA

KIT = Path(__file__).resolve().parent.parent / "docs" / "PROMPT_KIT.md"


@pytest.fixture(scope="module")
def kit() -> str:
    assert KIT.exists(), "PROMPT_KIT.md 없음 — tools/export_prompt_kit.py 실행 필요"
    return KIT.read_text(encoding="utf-8")


def test_all_four_system_prompts_are_verbatim(kit):
    """4단계 시스템 프롬프트가 코드와 한 글자도 다르지 않아야 한다."""
    for name, text in (
            ("FRAMER", prompts.FRAMER_SYSTEM),
            ("SELECTOR", prompts.SELECTOR_SYSTEM),
            ("ANALYST", prompts.ANALYST_SYSTEM),
            ("FORECASTER", prompts.FORECASTER_SYSTEM)):
        assert text.strip() in kit, f"{name}_SYSTEM 이 키트와 불일치"


def test_every_allowed_key_is_documented(kit):
    """엔진이 허용하는 키만 쓰라고 했으면, 그 목록이 전부 실려 있어야 한다."""
    for key in CRITERIA:
        assert f"`{key}`" in kit, f"기준 {key} 누락"
    for key in DISCIPLINES:
        assert f"`{key}`" in kit, f"학문 {key} 누락"


def test_output_filenames_match_recorded_backend_contract(kit):
    """RecordedBackend 는 <key>.json 을 읽는다. 파이프라인이 쓰는 키와
    키트가 안내하는 파일명이 같아야 한다."""
    for fname in ("framing.json", "selection.json", "forecasts.json",
                  "analysis_<학문키>.json"):
        assert fname in kit, f"{fname} 안내 누락"


def test_kit_states_honest_limits(kit):
    """모델이 다르면 판단도 다르다는 사실을 감춘 키트는 신뢰를 판다."""
    assert "같은 **판단**을 보장하지 않습니다" in kit
    assert "상관계수" in kit


def test_validation_bounds_are_stated(kit):
    """검증 범위를 모르면 사용자는 원인 모를 실패를 겪는다."""
    for bound in ("score_mean", "score_std", "confidence", "3~5"):
        assert bound in kit
