# CIV42 — a decision engine that keeps score

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Docs](https://img.shields.io/badge/methodology-CC--BY--4.0-lightgrey)](docs/PROMPT_KIT.md)

Most decision frameworks are never scored. Advice is given in words, and when
the outcome is bad the execution gets blamed, not the advice.

CIV42 takes one hard decision and does four things with it: separates what is
**known** from what is **assumed**, finds the assumptions that would break the
conclusion, adds the options nobody considered, and designs the smallest
reversible test. Then it does the part almost nobody does — it writes down
**falsifiable predictions with resolution criteria**, so the advice can be
proven wrong later.

We publish our own accuracy at **[civ42.com/ledger](https://civ42.com/ledger)**.
Wrong predictions stay on the page.

---

## Run it in two minutes — no API key

A complete worked example is included. It replays recorded model outputs, so
the whole pipeline runs offline:

```bash
pip install -r requirements.txt
python3 -m ark42 example/problem.json --recorded example/recorded --client
```

You get:

```
[0/5] 옵션: 4개 — ['PAID_ONLY', 'PAID_PLUS_INVITE', 'AUDIENCE_FIRST', 'NO_INTERVENTION']
[1/5] 학문 선별: 3개 — ['economics', 'behavioral_psychology', 'statistics_data']
[2/5] 독립 분석 완료: 96개 온톨로지 셀, 검증 통과
[3/5] 텐서 4×8×3 · 몬테카를로 20,000 draws 완료
[4/5] 결과 예상치: 9개 예측 생성·검증
[5/5] 리포트: runs/…/report.html
[+] 고객용 리포트: runs/…/client_report.html
```

The example is a real decision — CIV42's own launch strategy — and its
predictions are registered in the public ledger.

## Run it with *your* AI

The engine does not require our models, or any vendor's. **[docs/PROMPT_KIT.md](docs/PROMPT_KIT.md)**
contains the exact four prompts the engine sends, generated from the source so
they can never drift. Paste them into ChatGPT, DeepSeek, Gemini, a local
model — anything. Save the four JSON outputs to a folder and:

```bash
python3 -m ark42 problem.json --recorded my_outputs/ --client
```

Everything downstream is deterministic code: ontology validation, the
option × criterion × discipline score tensor, a 20,000-draw Monte Carlo with a
Gaussian copula, forecast generation, and report rendering.

## Run it against an API

```bash
export ANTHROPIC_API_KEY=...      # or GEMINI_API_KEY / GROQ_API_KEY / CAFE24_LLM_API_KEY
python3 -m ark42 problem.json --draws 20000 --client
```

Prompt caching is on by default (`ARK42_PROMPT_CACHE=0` to disable). Cache
tokens are billed into the cost record at their real rates — the ledger never
flatters itself.

---

## What is unusual here

**Independence is measured, not claimed.** Running eight "disciplines" on one
model gives you one voice wearing eight labels. The engine measures the
correlation between them and reports it. In the included example it found
ρ = 0.95 and widened its own confidence accordingly — the engine argues
against its own certainty.

**No option wins on expected value alone.** `NO_INTERVENTION` is always
evaluated as a real alternative with its own risks. Reports carry reversal
conditions, not just rankings.

**Nothing is invented.** Facts, inferences and unknowns are kept separate; an
LLM response that hits the token limit aborts the run rather than saving
truncated JSON; a missing recorded response raises instead of guessing.

There is also an optional statevector qubit simulator
(`ark42/qubit.py`, `ARK42_QUANTUM=1`) that amplitude-encodes the Monte Carlo
rank distribution and re-measures it as a self-consistency check. It is a
classical simulation of a quantum circuit and claims no speedup — the label
says so in the output, because a diagnostic that oversells itself is worse
than none.

## What is in this repository — and what is not

**Included** — the whole engine: prompts, ontology and validation, score
tensor, Monte Carlo, forecasting, learning weights, report renderers, the CLI,
the prompt kit and its generator, tests.

**Not included** — the hosted service's plumbing (accounts, credits, payments,
mail) and, by definition, the things that cannot be copied: client decision
data, the reliability weights learned from *resolved* predictions, and the
CIV42 trademark. See [NOTICE](NOTICE) for the exact boundary.

You may run, modify, fork and sell services built on this code. You may not
call your fork CIV42 — Apache-2.0 grants no trademark rights (§6).

## Contributing

Sign your commits (`git commit -s`) — we use the
[DCO](https://developercertificate.org/), not a CLA. Changes that alter the
criteria, disciplines, prompts, or distributional assumptions need a reason
why the new value is *more accurate*, not merely different. See
[CONTRIBUTING.md](CONTRIBUTING.md), including the four honesty rules that get
PRs reverted most often.

## License

Code: **Apache-2.0**. Methodology and documentation: **CC BY 4.0**.
The Methods Paper is registered on OSF with a DOI.

---

<details>
<summary>한국어</summary>

CIV42는 **자기 정확도를 공개 채점하는 의사결정 엔진**입니다.

되돌리기 어려운 결정 하나를 네 단계로 해부합니다 — 사실과 가정의 분리
(Reality Map), 결론을 뒤집을 가정 찾기(Blind Spot), 검토된 적 없는 선택지
추가(Choice Set), 되돌릴 수 있는 최소 검증 설계(Smallest Test). 그리고
거의 아무도 하지 않는 일을 합니다: **판정 기준이 붙은 반증 가능한 예측을
등록**하고, 3개월 뒤 실제 결과와 대조해 [공개 대장](https://civ42.com/ledger)에
그대로 적습니다. 틀린 예측도 지우지 않습니다.

**API 키 없이 2분 만에 실행**할 수 있습니다(위 `--recorded` 예제). 엔진은
특정 모델에 종속되지 않으며, `docs/PROMPT_KIT.md`의 4단계 프롬프트를 아무
AI에나 붙여넣어 출력을 저장하면 그대로 돌아갑니다.

특이한 점: **독립성을 주장하지 않고 측정합니다.** 한 모델로 여러 학문을
돌리면 라벨만 다른 한 목소리가 되는데, 엔진은 그 상관계수를 재서 보고하고
스스로의 확신을 낮춥니다. 포함된 예제에서 ρ=0.95가 측정되었습니다.

코드는 Apache-2.0, 방법론 문서는 CC BY 4.0입니다. 포크·수정·상업적 사용
모두 자유지만 "CIV42"라는 이름은 상표이므로 포크에 쓰실 수 없습니다
([NOTICE](NOTICE) 참조).

</details>
