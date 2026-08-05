"""LLM backends. The engine never depends on a specific vendor:
`LLMBackend.complete(system, prompt) -> str(JSON)` is the whole contract.

- AnthropicBackend: production. Needs ANTHROPIC_API_KEY.
- RecordedBackend: replays stored JSON responses keyed by stage name.
  Used for tests, demos without a key, and full-run reproducibility.
Every call is logged to the run directory for the audit trail.
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol


class LLMBackend(Protocol):
    def complete(self, key: str, system: str, prompt: str) -> str: ...


class AnthropicBackend:
    """Direct Messages API call; no SDK dependency to keep the engine small."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 8000,
                 log_dir: Path | None = None, max_retries: int = 3):
        self.api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        self.model = model
        self.max_tokens = max_tokens
        self.log_dir = log_dir
        self.max_retries = max_retries
        self.usage: list[dict] = []   # per-call token usage for cost tracking
        #: Stable identity of who produced an answer. Independence is only
        #: real ACROSS distinct provider_ids — two disciplines on the same
        #: provider_id are the same voice under different labels.
        self.provider_id = f"anthropic:{model}"

    #: 캐시 쓰기는 표준 입력가의 1.25배, 캐시 읽기는 0.1배다. 한 실행에서
    #: 같은 system 프롬프트가 여러 번 재사용되므로(학문 수만큼) 두 번째
    #: 호출부터 그 부분의 입력비가 90% 줄어든다. 짧은 프롬프트는 최소
    #: 토큰 요건에 미달해 캐시가 생성되지 않으므로 시도 자체를 하지 않는다.
    _CACHE_MIN_CHARS = 1200

    def complete(self, key: str, system: str, prompt: str) -> str:
        cacheable = (os.environ.get("ARK42_PROMPT_CACHE", "1") != "0"
                     and len(system) >= self._CACHE_MIN_CHARS)
        system_field = ([{"type": "text", "text": system,
                          "cache_control": {"type": "ephemeral"}}]
                        if cacheable else system)
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": system_field,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=body,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
        )
        data = self._send_with_retry(req, key)
        # Truncation must fail loudly: a max_tokens cutoff yields invalid JSON
        # that would otherwise corrupt the run several calls later, after the
        # money for those calls has already been spent.
        if data.get("stop_reason") == "max_tokens":
            raise RuntimeError(
                f"{key}: response hit max_tokens={self.max_tokens} and was "
                "truncated. Raise max_tokens or tighten the prompt; the run is "
                "aborted here rather than saving corrupt output.")
        usage = data.get("usage", {}) or {}
        self.usage.append({
            "key": key,
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            # 캐시 효과를 원가 계산에 반영하려면 기록되어야 한다 — 기록되지
            # 않는 절감은 장부에서 존재하지 않는 절감이다.
            "cache_creation_input_tokens":
                usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens":
                usage.get("cache_read_input_tokens", 0),
            "stop_reason": data.get("stop_reason"),
        })
        text = "".join(b.get("text", "") for b in data.get("content", []))
        text = _strip_fences(text)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{key}.json").write_text(text, encoding="utf-8")
        return text

    def _send_with_retry(self, req, key: str) -> dict:
        """Retry transient API failures (429 rate limit, 5xx overload) with
        exponential backoff. Without this a single 429 destroys a run that has
        already spent money on earlier calls."""
        delay = 2.0
        last = None
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                last = e
                if e.code not in (429, 500, 502, 503, 529) or attempt == self.max_retries:
                    raise
                wait = float(e.headers.get("retry-after") or delay)
                time.sleep(min(wait, 60.0))
                delay *= 2
            except urllib.error.URLError as e:
                last = e
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 2
        raise last  # unreachable, kept for clarity


class RecordedBackend:
    """Replays responses from <dir>/<key>.json. Raises if a key is missing,
    so a demo can never silently invent an answer."""

    def __init__(self, directory: Path, provider_id: str | None = None):
        self.directory = Path(directory)
        self.provider_id = provider_id or f"recorded:{self.directory.name}"

    def complete(self, key: str, system: str, prompt: str) -> str:
        path = self.directory / f"{key}.json"
        if not path.exists():
            raise FileNotFoundError(
                f"RecordedBackend: no recorded response for {key!r} in {self.directory}")
        return _strip_fences(path.read_text(encoding="utf-8"))


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.rstrip().endswith("```"):
            t = t.rstrip()[:-3]
    return t.strip()
