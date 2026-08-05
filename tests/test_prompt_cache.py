from __future__ import annotations

"""프롬프트 캐싱 — 요청 형태와 원가 계상의 정직성.

캐싱은 돈을 아끼지만, 아낀 만큼 장부가 정확해야 한다. 캐시 쓰기(1.25배)와
읽기(0.1배)를 계상에서 빼면 원가가 실제보다 싸게 보이고, 그 위에서 계산한
마진은 거짓이 된다.
"""

import json

import pytest

from ark42 import llm


class _Resp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()

    def read(self):
        return self._b

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_api(captured, usage=None):
    def _open(req, timeout=0):
        captured.append(json.loads(req.data.decode()))
        return _Resp({
            "content": [{"text": '{"ok": true}'}],
            "stop_reason": "end_turn",
            "usage": usage or {"input_tokens": 10, "output_tokens": 5},
        })
    return _open


@pytest.fixture()
def backend(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k" * 20)
    monkeypatch.delenv("ARK42_PROMPT_CACHE", raising=False)
    return llm.AnthropicBackend()


def test_long_system_prompt_is_marked_cacheable(backend, monkeypatch):
    sent = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_api(sent))
    backend.complete("framing", "시스템 " * 500, "사용자 프롬프트")
    system = sent[0]["system"]
    assert isinstance(system, list), "긴 시스템 프롬프트는 캐시 블록이어야 한다"
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_short_system_prompt_skips_cache(backend, monkeypatch):
    """최소 토큰 요건에 미달하면 캐시가 생성되지 않는다 — 시도하면
    캐시 쓰기 비용(1.25배)만 내고 읽기 이득은 없다."""
    sent = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_api(sent))
    backend.complete("framing", "짧다", "사용자 프롬프트")
    assert isinstance(sent[0]["system"], str)


def test_cache_can_be_disabled_by_env(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k" * 20)
    monkeypatch.setenv("ARK42_PROMPT_CACHE", "0")
    b = llm.AnthropicBackend()
    sent = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_api(sent))
    b.complete("framing", "시스템 " * 500, "프롬프트")
    assert isinstance(sent[0]["system"], str)


def test_cache_tokens_are_recorded_for_the_ledger(backend, monkeypatch):
    sent = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", _fake_api(sent, usage={
        "input_tokens": 100, "output_tokens": 50,
        "cache_creation_input_tokens": 2000,
        "cache_read_input_tokens": 8000,
    }))
    backend.complete("framing", "시스템 " * 500, "프롬프트")
    u = backend.usage[0]
    assert u["cache_creation_input_tokens"] == 2000
    assert u["cache_read_input_tokens"] == 8000


def test_cost_counts_cache_tokens_at_their_real_rates(tmp_path):
    """캐시 토큰을 원가에서 누락하면 마진이 부풀려진다."""
    from ark42.pipeline import Run

    run = Run.__new__(Run)
    run.dir = tmp_path
    run.backend = type("B", (), {"usage": [
        {"key": "a", "input_tokens": 0, "output_tokens": 0,
         "cache_creation_input_tokens": 1_000_000,
         "cache_read_input_tokens": 0},
    ]})()
    doc = run.record_cost(in_price_per_mtok=3.0, out_price_per_mtok=15.0,
                          krw_per_usd=1000.0, credit_value_krw=100_000.0)
    # 캐시 쓰기 100만 토큰 = 3.0 USD × 1.25 = 3.75 USD
    assert doc["usd"] == pytest.approx(3.75, abs=1e-6)
    assert doc["cache_write_tokens"] == 1_000_000
