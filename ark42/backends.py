"""Multi-provider backends — making "독립 다중 주체" real.

The independence measurement proved the engine's central weakness: eight
"disciplines" run on one model family produce ~one effective perspective
(measured r=0.93, n_eff≈1). Labels differ; the voice does not.

The only structural fix is genuinely different voices. This module lets each
discipline be answered by a DIFFERENT provider (a different model, a
different vendor), and — crucially — records which provider answered each
call so `independence.measure` can split correlation into within-provider
(expected high, the same voice) and cross-provider (the real independence
signal). Diversity you cannot measure is diversity you cannot claim.

Honest limits, stated in code:
- This session has no live API keys, so real cross-model correlation cannot
  be verified here. The architecture and the measurement are built; the
  verification waits for keys. The constitution keeps clause 7 at [부분].
- A single-provider deployment is still allowed and is the default. In that
  case the panel is one voice and `independence` reports single_provider.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from .llm import AnthropicBackend, RecordedBackend, _strip_fences


class FailoverExhausted(RuntimeError):
    """Raised when every provider in a FailoverBackend failed a single call.

    Carries each provider's error so the operator can see WHY every vendor
    failed — this is deliberately loud. Silently returning an empty or
    garbage string would let a decision run continue on a corrupt answer,
    which is strictly worse than an honest, fully-attributed failure.
    """

    def __init__(self, key: str, errors: list[tuple[str, BaseException]]):
        self.key = key
        #: [(provider_id, exception)] in the order they were tried.
        self.errors = errors
        detail = "; ".join(f"{pid}: {e!r}" for pid, e in errors)
        super().__init__(
            f"all {len(errors)} provider(s) failed for {key!r}: {detail}")


#: OpenAI-compatible gateways, by short name. A gateway speaks the same
#: /chat/completions grammar, so it is a CONFIGURATION of this backend, not a
#: new one — adding a second class per vendor is how request-shape bugs end up
#: fixed in one copy and not the other.
OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY",
               "gpt-4o"),
    # 카페24 LLM Router — 국내 게이트웨이. 원화 청구·세금계산서, 크레딧 종량제.
    # 100여 개 모델(Claude, Gemini, Qwen, DeepSeek, Llama …)을 한 엔드포인트로
    # 라우팅한다. 키 형식은 sk-cafe24-*.
    "cafe24": ("https://llm-router.cafe24.com/api/v1/chat/completions",
               "CAFE24_LLM_API_KEY", "Qwen/Qwen3-32B"),
    # Groq — 미국, 무료 티어는 카드 등록 없이 사용 가능. 단, 무료 TPM이
    # 분당 수천 토큰 수준이라 이 파이프라인(회당 수만 토큰)에는 부족하다 —
    # max_tokens 만으로 한도를 넘겨 413이 난다(2026-08-04 프로덕션 확인).
    "groq": ("https://api.groq.com/openai/v1/chat/completions",
             "GROQ_API_KEY", "llama-3.3-70b-versatile"),
    # Gemini — 미국, 무료 티어 카드 불필요 + Flash급 분당 1M 토큰으로 이
    # 파이프라인이 여유 있게 돈다. OpenAI 호환 게이트웨이 경유.
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai"
               "/chat/completions",
               "GEMINI_API_KEY", "gemini-2.5-flash"),
}


class OpenAIBackend:
    """Any OpenAI-compatible Chat Completions endpoint, urllib-only.

    `vendor` selects an entry in :data:`OPENAI_COMPATIBLE` (endpoint + key env
    var + default model), or pass `base_url`/`api_key_env` directly for a
    gateway not listed there. Untested live in this session — no key — but the
    request shape matches the documented API, so a key is all it needs.
    """

    def __init__(self, model: str | None = None, max_tokens: int = 8000,
                 log_dir: Path | None = None, max_retries: int = 3,
                 vendor: str = "openai", base_url: str | None = None,
                 api_key_env: str | None = None):
        default_url, default_env, default_model = OPENAI_COMPATIBLE.get(
            vendor, OPENAI_COMPATIBLE["openai"])
        self.vendor = vendor
        self.base_url = base_url or default_url
        self.api_key_env = api_key_env or default_env
        self.api_key = os.environ.get(self.api_key_env, "")
        if not self.api_key:
            raise RuntimeError(f"{self.api_key_env} is not set")
        self.model = model or default_model
        # Env override: providers meter max_tokens against per-request caps
        # (Groq's 413), so the operator can shrink it without a code change.
        self.max_tokens = int(os.environ.get("ARK42_LLM_MAX_TOKENS",
                                             max_tokens))
        self.log_dir = log_dir
        self.max_retries = max_retries
        self.usage: list[dict] = []
        self.provider_id = f"{vendor}:{self.model}"

    def complete(self, key: str, system: str, prompt: str) -> str:
        body = json.dumps({
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }).encode()
        req = urllib.request.Request(
            self.base_url, data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "content-type": "application/json",
                     # Groq's Cloudflare edge bans the default Python-urllib
                     # user agent outright (error 1010, verified 2026-08-03:
                     # same request 401s with a normal UA but 403s with
                     # urllib's). Identify honestly as this application.
                     "User-Agent": "ark42-engine/1.0 (+https://civ42.com)"})
        data = self._send_with_retry(req)
        finish = (data.get("choices") or [{}])[0].get("finish_reason")
        if finish == "length":
            raise RuntimeError(
                f"{key}: {self.vendor} response hit the token limit and was "
                "truncated. "
                "Raise max_tokens or tighten the prompt; aborting rather than "
                "saving corrupt output.")
        usage = data.get("usage", {}) or {}
        self.usage.append({"key": key,
                           "input_tokens": usage.get("prompt_tokens", 0),
                           "output_tokens": usage.get("completion_tokens", 0),
                           "stop_reason": finish})
        text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "")
        text = _strip_fences(text)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            (self.log_dir / f"{key}.json").write_text(text, encoding="utf-8")
        return text

    def _send_with_retry(self, req) -> dict:
        delay = 2.0
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    return json.loads(resp.read())
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503) or attempt == self.max_retries:
                    # Surface the provider's own words: a bare "HTTP 413" hides
                    # the limit numbers the body spells out, and that opacity
                    # cost a night of guessing (2026-08-04). Keep it short.
                    try:
                        detail = e.read().decode("utf-8", "replace")[:300]
                    except Exception:  # noqa: BLE001
                        detail = ""
                    raise RuntimeError(
                        f"{self.vendor} HTTP {e.code}: {detail or e.reason}"
                    ) from e
                time.sleep(min(float(e.headers.get("retry-after") or delay), 60.0))
                delay *= 2
            except urllib.error.URLError:
                if attempt == self.max_retries:
                    raise
                time.sleep(delay)
                delay *= 2


class PanelBackend:
    """Routes each discipline's analysis to a different provider from a pool,
    deterministically, and records the assignment.

    Orchestration calls (framing / selection / forecast) go to panel 0 — they
    are single-voice tasks (generate options, pick disciplines) where diversity
    would only add noise. The per-discipline ANALYSIS calls are what must be
    independent, so those are spread across the pool by a stable hash of the
    discipline name. Same discipline → same provider on every run (stable
    provenance); different disciplines → spread across the pool.
    """

    def __init__(self, panels: list):
        if not panels:
            raise ValueError("PanelBackend needs at least one panel")
        self.panels = panels
        self.usage: list[dict] = []
        self.provider_of: dict[str, str] = {}     # key -> provider_id
        # A panel of one is honestly single-voice; callers can check this.
        self.provider_ids = [getattr(p, "provider_id", "unknown") for p in panels]
        self.multi = len(set(self.provider_ids)) > 1

    def _panel_for(self, key: str):
        if key.startswith("analysis_") and len(self.panels) > 1:
            h = int(hashlib.sha256(key.encode()).hexdigest(), 16)
            return self.panels[h % len(self.panels)]
        return self.panels[0]

    def complete(self, key: str, system: str, prompt: str) -> str:
        b = self._panel_for(key)
        self.provider_of[key] = getattr(b, "provider_id", "unknown")
        out = b.complete(key, system, prompt)
        # Roll up sub-backend usage so cost accounting still works.
        for u in getattr(b, "usage", [])[len(self.usage):]:
            self.usage.append(u)
        return out

    def provider_map(self) -> dict:
        """discipline -> provider_id, from the analysis calls made so far."""
        return {k[len("analysis_"):]: v for k, v in self.provider_of.items()
                if k.startswith("analysis_")}


class FailoverBackend:
    """Redundancy across EQUIVALENT providers: try the primary, and if it is
    down or erroring, fall back to the next provider rather than failing the
    run. Vendor resilience — no single-vendor point of failure.

    This is a DIFFERENT concern from PanelBackend. PanelBackend spreads
    disciplines across providers ON PURPOSE (diversity: different voices for
    different disciplines). FailoverBackend treats its providers as
    interchangeable substitutes for the SAME call and only ever uses one of
    them per call — the first that answers.

    Which provider actually answered is recorded per call (`provider_of`) and
    every skipped/failed provider is recorded in `failures`, so downstream
    provenance code sees the truth: a fell-over call was answered by a
    different voice than the primary. We surface that, never hide it.
    """

    def __init__(self, backends: list, retry_terminal: bool = True):
        if not backends:
            raise ValueError("FailoverBackend needs at least one backend")
        self.backends = backends
        #: Fail over on a terminal (retry-exhausted / non-transient) HTTP error
        #: too. Sub-backends already retry transient 429/5xx internally, so an
        #: HTTPError that reaches us is terminal. True (default): treat that as
        #: grounds to try the next vendor — a 500/401/overload on one vendor
        #: says nothing about the next. False: re-raise it immediately without
        #: burning the other providers (e.g. when a 4xx means the request
        #: itself is malformed and would fail identically everywhere).
        self.retry_terminal = retry_terminal
        self.usage: list[dict] = []
        self.provider_of: dict[str, str] = {}          # key -> answering provider_id
        self.failures: list[tuple[str, str, str]] = []  # (provider_id, key, error-repr)
        self.provider_ids = [getattr(b, "provider_id", "unknown") for b in backends]
        # Mirror PanelBackend: more than one distinct provider = genuinely multi.
        self.multi = len(set(self.provider_ids)) > 1
        self._usage_seen: dict[int, int] = {}

    @property
    def provider_id(self) -> str:
        return "failover(" + ",".join(self.provider_ids) + ")"

    def _rollup(self, b) -> None:
        """Append only the sub-backend's NEW usage entries (per-backend
        watermark), so cost accounting stays correct across many calls and
        many distinct sub-backends."""
        seen = self._usage_seen.get(id(b), 0)
        new = getattr(b, "usage", [])[seen:]
        self.usage.extend(new)
        self._usage_seen[id(b)] = seen + len(new)

    def complete(self, key: str, system: str, prompt: str) -> str:
        errors: list[tuple[str, BaseException]] = []
        for b in self.backends:
            pid = getattr(b, "provider_id", "unknown")
            try:
                out = b.complete(key, system, prompt)
            except urllib.error.HTTPError as e:
                self.failures.append((pid, key, repr(e)))
                errors.append((pid, e))
                if not self.retry_terminal:
                    # Operator asked NOT to fail over on terminal HTTP errors.
                    raise
                continue
            except Exception as e:                              # noqa: BLE001
                # Any other error (URLError, timeout, truncation RuntimeError,
                # missing recorded key, ...) → this provider is out, try next.
                self.failures.append((pid, key, repr(e)))
                errors.append((pid, e))
                continue
            # First success wins. Record who answered and roll up its usage.
            self.provider_of[key] = pid
            self._rollup(b)
            return out
        raise FailoverExhausted(key, errors)

    def provider_map(self) -> dict:
        """discipline -> provider_id that ACTUALLY answered its analysis_*
        call. Mirrors PanelBackend.provider_map so provenance/independence
        code treats a failover panel uniformly."""
        return {k[len("analysis_"):]: v for k, v in self.provider_of.items()
                if k.startswith("analysis_")}


def build_backend(log_dir: Path | None = None):
    """Construct the backend from env. Precedence (first match wins):

      ARK42_FAILOVER="anthropic:claude-sonnet-4-5,openai:gpt-4o"
          → a FailoverBackend across those providers (redundancy: try the
            first, fall back to the next on failure). Checked FIRST because it
            is the most explicit resilience request; an operator who sets it
            is asking for no-single-vendor survival. Reuses _provider_from_spec
            (same spec grammar as ARK42_PANELS, incl. recorded: for keyless
            demos/tests).
      ARK42_PANELS="anthropic:claude-sonnet-4-5,openai:gpt-4o"
          → a PanelBackend across those providers (real multi-model diversity).
      ANTHROPIC_API_KEY set (no panels)
          → single AnthropicBackend (today's default; single voice).
      CAFE24_LLM_API_KEY set (no panels, no Anthropic key)
          → single OpenAIBackend against the Cafe24 LLM Router gateway.
            Model from ARK42_CAFE24_MODEL. Billed in KRW with a tax invoice,
            which is what makes it usable by a Korean sole proprietor without a
            USD card.
      GROQ_API_KEY set (none of the above)
          → single OpenAIBackend against Groq. Model from ARK42_GROQ_MODEL.
            Free tier needs no card — the zero-cost pilot configuration.
      ARK42_RECORDED_DIR set
          → RecordedBackend (demos/tests without keys).
      ARK42_RECORDED_PANELS="dirA,dirB"
          → PanelBackend across recorded dirs (used to DEMONSTRATE the
            mechanism here where no live keys exist).

    Every existing path is unchanged: FailoverBackend only activates when
    ARK42_FAILOVER is set, so setups that don't set it behave exactly as before.
    """
    failover_env = os.environ.get("ARK42_FAILOVER")
    if failover_env:
        backends = [_provider_from_spec(s.strip(), log_dir)
                    for s in failover_env.split(",") if s.strip()]
        return FailoverBackend(backends)

    panels_env = os.environ.get("ARK42_PANELS")
    if panels_env:
        panels = [_provider_from_spec(s.strip(), log_dir)
                  for s in panels_env.split(",") if s.strip()]
        return PanelBackend(panels)

    rec_panels = os.environ.get("ARK42_RECORDED_PANELS")
    if rec_panels:
        panels = [RecordedBackend(Path(d.strip()))
                  for d in rec_panels.split(",") if d.strip()]
        return PanelBackend(panels)

    if os.environ.get("ANTHROPIC_API_KEY"):
        return AnthropicBackend(log_dir=log_dir)
    if os.environ.get("CAFE24_LLM_API_KEY"):
        # 국내 게이트웨이 단독 사용. 모델은 ARK42_CAFE24_MODEL 로 지정.
        return OpenAIBackend(model=os.environ.get("ARK42_CAFE24_MODEL"),
                             log_dir=log_dir, vendor="cafe24")
    if os.environ.get("GEMINI_API_KEY"):
        # Gemini 단독 사용. GROQ 키보다 먼저 확인한다 — 무료 TPM이 커서
        # 키만 추가하면 Groq 설정을 지우지 않고도 이쪽으로 넘어온다.
        return OpenAIBackend(model=os.environ.get("ARK42_GEMINI_MODEL"),
                             log_dir=log_dir, vendor="gemini")
    if os.environ.get("GROQ_API_KEY"):
        # Groq 단독 사용. 모델은 ARK42_GROQ_MODEL 로 지정 (기본 llama-3.3-70b).
        return OpenAIBackend(model=os.environ.get("ARK42_GROQ_MODEL"),
                             log_dir=log_dir, vendor="groq")
    if os.environ.get("ARK42_RECORDED_DIR"):
        return RecordedBackend(Path(os.environ["ARK42_RECORDED_DIR"]))
    # No key, no recorded dir: the constructor raises, run ends failed —
    # the existing honest failure path.
    return AnthropicBackend(log_dir=log_dir)


def _provider_from_spec(spec: str, log_dir):
    vendor, _, model = spec.partition(":")
    if vendor == "anthropic":
        return AnthropicBackend(model=model or "claude-sonnet-4-5", log_dir=log_dir)
    if vendor in OPENAI_COMPATIBLE:
        return OpenAIBackend(model=model or None, log_dir=log_dir, vendor=vendor)
    if vendor == "recorded":
        return RecordedBackend(Path(model))
    raise ValueError(
        f"unknown provider vendor in ARK42_PANELS: {spec!r} "
        f"(알려진 벤더: anthropic, recorded, {', '.join(sorted(OPENAI_COMPATIBLE))})")
