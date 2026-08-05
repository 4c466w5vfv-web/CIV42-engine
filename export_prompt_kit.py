#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""이식 가능한 프롬프트 키트 생성기 → docs/PROMPT_KIT.md

왜 생성기인가
-------------
엔진에 실제로 전송되는 문자열은 ark42/prompts.py 에만 있다. 키트를 손으로
옮겨 적으면 코드가 바뀔 때마다 조용히 어긋나고, 어긋난 프롬프트로 만든
출력은 파이프라인 검증에서 떨어진다. 그래서 키트는 **코드에서 뽑는다**.

이 키트가 하는 일: ChatGPT·DeepSeek·Gemini 등 어떤 모델에서도 4단계 출력을
만들어 `--recorded` 폴더에 넣으면 CIV42 엔진(검증·텐서·몬테카를로·리포트)이
그대로 돈다. 즉 **엔진은 모델에 종속되지 않는다.**

정직한 한계: 같은 *형식*을 보장하지, 같은 *판단*을 보장하지 않는다. 모델이
다르면 점수가 다르고 결론도 달라질 수 있다 — 그건 결함이 아니라 설계
의도(독립 다중 주체)이며, 엔진은 그 차이를 상관계수로 측정해 보고한다.

사용: python3 tools/export_prompt_kit.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ark42 import prompts                                    # noqa: E402
from ark42.disciplines import DISCIPLINES                    # noqa: E402
from ark42.ontology import CRITERIA                          # noqa: E402

OUT = ROOT / "docs" / "PROMPT_KIT.md"

P_STATEMENT = "«여기에 결정 내용을 한 문단으로»"
P_CONTEXT = "«여기에 배경·제약·가용 자원. 없으면 (없음)»"


class _Opt:
    def __init__(self, i):
        self.option_id = f"«OPTION_{i}_ID»"
        self.title = f"«옵션 {i} 이름»"
        self.description = f"«옵션 {i} 설명 — framing 단계 출력에서 그대로 복사»"


class _Problem:
    """프롬프트 렌더용 자리표시자. 실제 Problem 과 같은 속성만 노출한다."""

    def __init__(self, with_options: bool):
        self.statement = P_STATEMENT
        self.context = P_CONTEXT
        self.options = ([_Opt(1), _Opt(2), _Opt(3)] +
                        [_NoIntervention()]) if with_options else []


class _NoIntervention:
    option_id = "NO_INTERVENTION"
    title = "개입하지 않음 (기준안)"
    description = "아무 조치도 하지 않고 현 상태를 유지한다."


def _fence(text: str) -> str:
    return "```text\n" + text.rstrip() + "\n```"


def main() -> int:
    bare = _Problem(with_options=False)
    framed = _Problem(with_options=True)

    disciplines_list = "\n".join(f"- `{k}` — {v}" for k, v in DISCIPLINES.items())
    criteria_list = "\n".join(f"- `{k}` — {v}" for k, v in CRITERIA.items())

    doc = f"""# CIV42 프롬프트 키트 — 어떤 AI로도 엔진을 돌리는 법

> 이 문서는 `tools/export_prompt_kit.py` 가 `ark42/prompts.py` 에서 **자동
> 생성**합니다. 손으로 고치지 마세요 — 코드가 바뀌면 다시 생성하면 됩니다.

## 무엇을 할 수 있나

ChatGPT·DeepSeek·Gemini·Claude 등 **아무 모델**에서 아래 4단계를 실행해
출력 JSON을 파일로 저장하면, CIV42 엔진이 그 파일들을 읽어 나머지를
전부 수행합니다 — 온톨로지 검증, 점수 텐서, 몬테카를로 20,000회, 예측
등록, 리포트 생성. **API 키도, 엔진의 모델 접근도 필요 없습니다.**

```bash
python3 -m ark42 problem.json --recorded <저장한_폴더> --draws 20000 --client
```

### 저장할 파일 이름 (정확히 지킬 것)

| 단계 | 파일명 | 개수 |
|---|---|---|
| 1. 옵션 생성 | `framing.json` | 1 |
| 2. 학문 선별 | `selection.json` | 1 |
| 3. 독립 분석 | `analysis_<학문키>.json` | 선별된 학문 수만큼 |
| 4. 결과 예측 | `forecasts.json` | 1 |

`problem.json` 은 직접 만듭니다:

```json
{{
  "problem_id": "case-2026-08-05-01",
  "statement": "{P_STATEMENT}",
  "context": "{P_CONTEXT}",
  "options": []
}}
```

### 지켜야 하는 규칙 (엔진이 실제로 검사합니다)

- 출력은 **JSON만**. 설명 문장이나 코드펜스를 함께 쓰면 파싱에 실패합니다.
- 옵션은 **3~5개**, `option_id` 중복 금지, `NO_INTERVENTION` 은 직접 만들지
  않습니다(엔진이 기준안으로 자동 추가).
- 분석 단계의 점수: `score_mean` 0~1, `score_std` 0~0.5, `confidence` 0~1
  범위를 벗어나면 검증에서 실행이 중단됩니다.
- 기준 8개와 학문 20개는 **아래 목록의 키만** 사용합니다.

---

## 1단계 — 옵션 생성 → `framing.json`

**시스템 프롬프트**

{_fence(prompts.FRAMER_SYSTEM)}

**사용자 프롬프트**

{_fence(prompts.framer_prompt(bare))}

---

## 2단계 — 학문 선별 → `selection.json`

1단계에서 받은 옵션들을 채워 넣습니다.

**시스템 프롬프트**

{_fence(prompts.SELECTOR_SYSTEM)}

**사용자 프롬프트**

{_fence(prompts.selector_prompt(framed))}

---

## 3단계 — 독립 분석 → `analysis_<학문키>.json`

**선별된 학문 하나당 대화를 새로 시작해서** 한 번씩 실행합니다. 한 대화에서
여러 학문을 연달아 시키면 앞선 답에 이끌려 독립성이 깨지고, 엔진이 측정하는
상관계수(ρ)가 의미를 잃습니다.

`{{discipline}}` 자리에는 학문 키(예: `economics`)를 넣습니다.

**시스템 프롬프트**

{_fence(prompts.ANALYST_SYSTEM)}

**사용자 프롬프트**

{_fence(prompts.analyst_prompt(framed, "«학문키»", 0.85, "«2단계에서 받은 선별 사유»"))}

---

## 4단계 — 결과 예측 → `forecasts.json`

`«정량 분석 요약»` 자리에는 3단계까지 끝낸 뒤 엔진이 출력한 몬테카를로 요약을
넣습니다. 엔진 없이 먼저 만들려면 각 옵션의 강·약점을 2~3줄로 요약해 넣어도
동작합니다(그 경우 확률의 근거가 약해지므로 리포트에 그 사실이 남습니다).

**시스템 프롬프트**

{_fence(prompts.FORECASTER_SYSTEM)}

**사용자 프롬프트**

{_fence(prompts.forecaster_prompt(framed, "«정량 분석 요약»"))}

---

## 참고 — 허용된 키 목록

### 기준 8개 (criterion)

{criteria_list}

방향 반전 기준: `cost_efficiency`, `risk_resilience`, `time_to_validation`
— 원가·위험·소요시간이 **낮을수록 점수가 높습니다.**

### 학문 20개 (discipline)

{disciplines_list}

---

## 정직한 한계

이 키트는 같은 **형식**을 보장하지 같은 **판단**을 보장하지 않습니다. 모델이
다르면 점수와 결론이 달라질 수 있고, 그건 결함이 아니라 설계 의도입니다 —
CIV42 는 서로 다른 관점의 독립성을 전제로 하고, 엔진은 그 관점들이 실제로
얼마나 독립적이었는지를 상관계수로 **측정해서 리포트에 적습니다**. 관점이
사실상 한 목소리였다면 리포트가 그렇게 말합니다.

리포트를 판매·배포할 때는 어떤 모델로 어느 단계를 만들었는지 기록해 두십시오.
나중에 예측이 채점될 때 그 기록이 있어야 "무엇이 잘 맞았는지"를 학습할 수
있습니다.
"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"prompt kit → {OUT} ({len(doc)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
