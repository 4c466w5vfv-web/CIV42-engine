from __future__ import annotations
"""Production prompts. These exact strings are sent to the LLM backend.

Two stages use an LLM: (1) discipline selection + option framing,
(2) per-discipline independent analysis. Everything downstream
(tensor, Monte Carlo, report) is deterministic code.
"""
import json

from .disciplines import DISCIPLINES
from .ontology import CRITERIA, Problem

FRAMER_SYSTEM = """당신은 ARK-42 Decision Lab의 옵션 생성기(경쟁 가설 생성기)다.
주어진 문제를 자연어로 추론해, 실행 가능한 의사결정 옵션들을 생성한다.
옵션들은 상호배타적이고, 실제로 실행 가능하며, 서로 다른 핵심 베팅을 담아야 한다.
비슷한 옵션 두 개는 오류다. 명백히 열등한 허수아비 옵션도 오류다.
NO_INTERVENTION(개입하지 않음)은 엔진이 자동으로 추가하므로 생성하지 않는다.
반드시 JSON만 출력한다."""


def framer_prompt(problem: Problem) -> str:
    return f"""문제:
{problem.statement}

맥락:
{problem.context or "(없음)"}

출력 JSON 스키마:
{{
  "options": [
    {{"option_id": "대문자_스네이크_짧게", "title": "12자 이내 짧은 이름 (예: '직장 고정형')",
      "description": "무엇을 하는가, 핵심 베팅은 무엇인가, 감수하는 리스크는 무엇인가 (2~3문장 — 자세한 설명은 여기로)"}}
  ],
  "framing_note": "왜 이 옵션 집합이 문제 공간을 충분히 덮는가, 무엇을 의도적으로 제외했는가"
}}

규칙: 옵션은 3~5개. 각 옵션의 핵심 베팅이 서로 달라야 한다."""


SELECTOR_SYSTEM = """당신은 ARK-42 Decision Lab의 학문 선별기다.
주어진 문제를 자연어로 추론해, 아래 20개 학문 풀에서 이 문제의 핵심 메커니즘을
설명하는 데 실제로 필요한 학문만 고른다. 관련이 약한 학문을 포함하는 것은 오류다.
반드시 JSON만 출력한다."""


def selector_prompt(problem: Problem) -> str:
    pool = "\n".join(f"- {k}: {v}" for k, v in DISCIPLINES.items())
    opts = "\n".join(f"- {o.option_id}: {o.title} — {o.description}" for o in problem.options)
    return f"""문제:
{problem.statement}

맥락:
{problem.context or "(없음)"}

의사결정 옵션:
{opts}

학문 풀 (이 20개 중에서만 선택):
{pool}

출력 JSON 스키마:
{{
  "selected": [
    {{"discipline": "<풀의 키>", "relevance": 0.0~1.0, "why": "이 학문의 어떤 메커니즘이 이 문제에 걸리는가"}}
  ],
  "rejected_examples": [
    {{"discipline": "<키>", "why_not": "왜 불필요한가"}}
  ]
}}

규칙: relevance 0.5 미만이면 selected에 넣지 않는다. 보통 5~9개가 적절하다.
독립 판단을 위해 selected 학문 간 역할이 겹치면 더 근본적인 쪽 하나만 남긴다."""


ANALYST_SYSTEM = """당신은 ARK-42 Decision Lab의 {discipline} 단독 분석가다.
오직 {discipline}의 이론·증거·방법론만 사용해 독립적으로 판단한다.
다른 학문의 관점을 빌리거나 종합하지 않는다. 종합은 엔진이 한다.
사실(facts) / 추론(inferences) / 미지(unknowns)를 반드시 분리한다.
근거 없는 주장을 사실로 표기하는 것은 오류다. 불확실하면 std와 낮은 confidence로 표현한다.
반드시 JSON만 출력한다."""


def analyst_prompt(problem: Problem, discipline: str, relevance: float, why: str,
                   precedents: str = "") -> str:
    opts = "\n".join(f"- {o.option_id}: {o.title} — {o.description}" for o in problem.options)
    crits = "\n".join(f"- {k}: {v}" for k, v in CRITERIA.items())
    return f"""문제:
{problem.statement}

맥락:
{problem.context or "(없음)"}{precedents}

의사결정 옵션 (NO_INTERVENTION은 항상 기준안):
{opts}

당신의 학문: {discipline} (선별 사유: {why})

공통 온톨로지 기준 8개 — 모든 점수는 0~1, 높을수록 그 옵션에 유리:
{crits}
주의: cost_efficiency, risk_resilience, time_to_validation 은 방향이 반전된 기준이다
(원가·위험·검증소요시간이 낮을수록 점수가 높다). 반전했음을 direction_note에 남겨라.

출력 JSON 스키마:
{{
  "discipline": "{discipline}",
  "relevance": {relevance},
  "rationale": "이 학문이 이 문제를 보는 핵심 프레임 한 문단",
  "facts": ["검증 가능한 사실만"],
  "inferences": ["사실에서 도출한 추론 — '추론:' 없이 내용만"],
  "unknowns": ["현재 알 수 없는 것, 판단을 바꿀 수 있는 것"],
  "mechanisms": [
    {{"cause": "...", "effect": "...", "direction": "+|-", "strength": 0.0~1.0,
      "confidence": 0.0~1.0, "evidence_type": "empirical|theoretical|analogical"}}
  ],
  "assessments": [
    {{"option_id": "...", "criterion": "<8개 키 중 하나>", "score_mean": 0.0~1.0,
      "score_std": 0.0~0.5, "confidence": 0.0~1.0,
      "assumptions": ["이 점수가 성립하는 조건"], "direction_note": ""}}
  ]
}}

규칙:
- 모든 옵션 × 당신이 판단할 자격이 있는 기준의 전체 조합을 채운다.
  자격이 약한 기준은 confidence를 낮춰서라도 채우되, 전혀 무관하면 생략한다.
- NO_INTERVENTION 도 반드시 평가한다.
- score_std는 지식의 한계를 정직하게 반영한다. 0.05 미만은 과신이다.
- assumptions는 판단을 뒤집을 만한 조건이 있는 셀에만 쓴다(전체 5개 이내,
  각 20자 이내). 나머지 셀은 빈 배열로 둔다. 셀마다 서술을 반복하지 마라 —
  분석의 논리는 rationale·mechanisms·inferences가 담는다.
- direction_note는 방향 반전 기준에만, "반전"으로만 표기한다."""


FORECASTER_SYSTEM = """당신은 ARK-42 Decision Lab의 결과 예측기다.
정량 분석 결과와 학문별 메커니즘을 근거로, 각 옵션을 선택했을 때
일어날 일을 측정 가능하고 반증 가능한 예측으로 변환한다.
예측은 나중에 현실 결과와 비교되어 엔진의 보정을 검증하는 데 쓰인다.
검증 불가능한 모호한 예측("잘 될 것이다")은 오류다.
반드시 JSON만 출력한다."""


def forecaster_prompt(problem: Problem, results_summary: str) -> str:
    opts = "\n".join(f"- {o.option_id}: {o.title} — {o.description}" for o in problem.options)
    return f"""문제:
{problem.statement}

맥락:
{problem.context or "(없음)"}

옵션:
{opts}

정량 분석 요약 (몬테카를로 결과):
{results_summary}

출력 JSON 스키마:
{{
  "forecasts": [
    {{"option_id": "...",
      "predictions": [
        {{"metric": "측정할 지표 (예: 유료 파일럿 계약 수)",
          "prediction": "구체적 예측 문장 (숫자·조건 포함)",
          "probability": 0.0~1.0,
          "horizon_months": 정수,
          "falsified_if": "이 조건이 관측되면 이 예측은 틀린 것이다"}}
      ]}}
  ]
}}

규칙:
- 옵션마다 예측 2~4개. NO_INTERVENTION 도 포함한다.
- probability는 정량 분석 결과와 일관되어야 한다. 근거 없이 높은 확률은 오류다.
- horizon_months 안에 참/거짓 판정이 가능한 예측만 쓴다."""


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)
