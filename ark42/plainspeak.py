"""쉬운 말 번역 — 전문가용 원문은 그대로 두고, 표층만 옮긴다.

왜 별도 단계인가. 분석 프롬프트는 "학문 단독 분석가"에게 쓰라고 지시하고,
그래서 결과물은 분석가가 분석가에게 쓴 글이 된다. 그 정밀도는 유료 자문의
근거이므로 낮추면 안 된다. 동시에 그 글을 못 읽는 고객에게는 팔리지 않는다.
둘 다 만족시키는 방법은 하나뿐이다 — **원문을 보존한 채 번역본을 따로 만든다.**

이 모듈이 지키는 규칙:

1. **숫자를 바꾸지 않는다.** 번역본에 나오는 모든 수는 원문에 있던 수여야 한다.
   :func:`verify_numbers` 가 이를 기계로 검사하고, 새 숫자가 나타나면 번역을
   폐기한다. 쉽게 쓰다가 "약 60%"처럼 반올림하는 것이 가장 흔한 사고이고,
   확률을 파는 제품에서 그건 제품 결함이다.
2. **원문을 덮어쓰지 않는다.** 번역본은 `plain.json` 에 따로 저장되며,
   동결된 예측 스냅샷과 results.json 은 손대지 않는다.
3. **없으면 없다고 한다.** 키가 없거나 검증에 실패하면 번역본을 만들지 않고
   이유를 남긴다. 원문은 언제나 볼 수 있으므로 기능 부재가 장애가 아니다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

PLAIN_SYSTEM = """당신은 전문 분석 보고서를 일반 독자용으로 옮기는 번역가다.
독자는 이 분야를 전혀 모르는 사람이고, 이 문서를 읽고 자기 돈이 걸린 결정을 한다.

지켜야 할 것:
- 숫자·날짜·비율은 원문 그대로 옮긴다. 반올림·요약·생략을 절대 하지 않는다.
  "63.7%"를 "약 60%"로 바꾸는 것은 심각한 오류다.
- 전문용어를 일상어로 바꾼다. 파일 이름, 영어 식별자, 변수명은 문장에서 없앤다.
- 조건과 단서를 지운 채 단정하지 않는다. 원문이 "~일 수 있다"면 번역도 그렇다.
- 문장을 짧게 끊는다. 한 문장에 한 가지만 담는다.
- 원문에 없는 조언·해석·격려를 덧붙이지 않는다. 옮기는 것이지 쓰는 것이 아니다.

반드시 JSON만 출력한다."""


def plain_prompt(items: list[dict]) -> str:
    body = "\n\n".join(f'[{i["key"]}]\n{i["text"]}' for i in items)
    return f"""아래 각 조각을 일반 독자용으로 옮겨라. key는 그대로 두고 text만 바꾼다.

{body}

출력 JSON 스키마:
{{"items": [{{"key": "<위의 key 그대로>", "text": "옮긴 문장"}}]}}

규칙: 숫자·날짜·비율은 원문과 글자 그대로 일치해야 한다. 조각 개수는 입력과 같아야 한다."""


#: 숫자로 인정하는 패턴 — 정수/소수/퍼센트/날짜 조각.
_NUM = re.compile(r"\d+(?:[.,]\d+)*")

#: 원문의 수를 하나도 잃으면 안 되는 조각. 반증 조건에서 마감일이나 임계값이
#: 빠지면 그건 더 이상 반증 조건이 아니다 — 이 제품이 파는 것이 바로 반증
#: 가능성이므로, 숫자를 잃으면 안 되는 유일한 자리가 무방비였다(적대 감사 C1/F1).
STRICT_KEYS = ("falsified_if", "prediction")

#: LLM 번역에서 아예 제외하는 키. 독립성 판정문은 이 제품의 서명이고,
#: 뒤집히면(“사실상 하나의 관점” → “서로 다른 관점에서 독립적으로 판단”)
#: 숫자를 그대로 둔 채 제품 주장이 정반대가 된다. 실제로 통과했다.
#: 그래서 번역하지 않고, 측정값에서 결정론적으로 만든다.
NEVER_LLM = ("independence.verdict",)


def numbers_in(text: str) -> list[str]:
    """텍스트에 나타난 수를 정규화해 반환 (천단위 콤마 제거)."""
    return [m.group(0).replace(",", "") for m in _NUM.finditer(text or "")]


def verify_numbers(original: str, plain: str, strict: bool = False) -> dict:
    """번역본이 원문의 수를 날조하거나(항상) 잃지 않았는지(strict) 검사한다.

    기본 규칙: 번역본의 모든 수가 원문에도 있어야 한다. 날조는 금지.

    `strict=True`: 여기에 더해 **원문의 모든 수가 번역본에도 남아 있어야** 한다.
    반증 조건과 예측 문장에 적용한다. 감사에서 "2026-09-29까지 … 1건 이상"을
    "이 조건이 확실히 성립한다"로 바꾼 번역이 통과했는데, 마감일과 임계값을
    잃은 반증 조건은 반증 조건이 아니다.

    **이 검사가 막지 못하는 것을 분명히 해 둔다.** 숫자를 건드리지 않는 왜곡은
    통과한다 — 부정어 뒤집기, "절반/두 배/대부분" 같은 비수치 크기 표현,
    단위 바꿔치기("63.7%"→"63.7배"). 적대 감사에서 시도한 12가지 왜곡이 전부
    통과했다. 그래서 가장 위험한 조각은 아예 LLM에 보내지 않는다(NEVER_LLM)
    — 이 함수는 방어선의 전부가 아니라 일부다.
    """
    have = set(numbers_in(original))
    got = set(numbers_in(plain))
    invented = sorted(got - have)
    lost = sorted(have - got) if strict else []
    return {"ok": not invented and not lost, "invented": invented, "lost": lost,
            "n_original": len(have), "n_plain": len(got)}


def plain_independence(run_dir: Path) -> str | None:
    """독립성 판정문을 측정값에서 **결정론적으로** 만든다. LLM을 쓰지 않는다.

    LLM 번역에 맡겼더니 "사실상 하나의 관점처럼 판단했습니다"가 "서로 다른
    관점에서 독립적으로 판단했습니다"로 뒤집힌 채 숫자 검사를 통과했다. 제품의
    핵심 주장이 정반대가 되는데 검사기는 통과시킨다. 여기서는 문장을 코드가
    쓰므로 뒤집힐 수 없다.
    """
    p = Path(run_dir) / "results.json"
    if not p.exists():
        return None
    try:
        ind = (json.loads(p.read_text(encoding="utf-8")).get("independence") or {})
    except (OSError, json.JSONDecodeError):
        return None
    if ind.get("reliable") is not True:
        return ("이번에는 각 분야의 판단이 얼마나 비슷했는지 재지 못했습니다. "
                "재지 못한 값은 비워 둡니다.")
    D, mr, ne = ind.get("n_disciplines"), ind.get("mean_r"), ind.get("n_effective")
    if not isinstance(D, int) or mr is None or ne is None:
        return None
    same = "한 명" if ne < 1.5 else f"{ne:.1f}명"
    tail = ("서로 다른 회사의 인공지능을 섞어야 진짜로 다른 생각이 됩니다."
            if ind.get("single_provider") else "")
    return (f"{D}명에게 물었지만, 생각이 많이 비슷해서 사실은 {same} 조금 넘는 "
            f"사람에게 물은 것과 비슷했습니다(겹친 정도 {mr:.2f}, 실제로 다른 "
            f"생각 {ne:.2f}명). 그만큼 확률을 깎아서 적었습니다. {tail}".strip())


def collect(run_dir: Path) -> list[dict]:
    """번역 대상 조각을 모은다. 원문 파일은 읽기만 한다."""
    run_dir = Path(run_dir)
    items: list[dict] = []

    def add(key, text):
        if text and str(text).strip():
            items.append({"key": key, "text": str(text).strip()})

    res_p = run_dir / "results.json"
    if res_p.exists():
        res = json.loads(res_p.read_text(encoding="utf-8"))
        ind = res.get("independence") or {}
        add("independence.verdict", ind.get("verdict"))
        add("independence.method", ind.get("method"))
    prob_p = run_dir / "problem.json"
    if prob_p.exists():
        prob = json.loads(prob_p.read_text(encoding="utf-8"))
        for o in prob.get("options", []):
            add(f"option.{o['option_id']}.title", o.get("title"))
            add(f"option.{o['option_id']}.description", o.get("description"))
    fc_p = run_dir / "forecasts.json"
    if fc_p.exists():
        fc = json.loads(fc_p.read_text(encoding="utf-8"))
        for f in fc.get("forecasts", []):
            for i, pr in enumerate(f.get("predictions", [])):
                add(f"forecast.{f['option_id']}.{i}.prediction", pr.get("prediction"))
                add(f"forecast.{f['option_id']}.{i}.falsified_if", pr.get("falsified_if"))
    for a in sorted((run_dir / "analyses").glob("*.json")) if (run_dir / "analyses").exists() else []:
        doc = json.loads(a.read_text(encoding="utf-8"))
        add(f"analysis.{a.stem}.rationale", doc.get("rationale"))
    return items


def translate(run_dir: Path, backend, max_items: int = 60) -> dict:
    """원문 → 쉬운 말. 결과를 plain.json 에 쓰고 요약을 반환한다.

    검증에 실패한 조각은 **버린다** — 원문이 그 자리에 남고, 어떤 조각이 왜
    버려졌는지 plain.json 에 기록된다. 조용히 잘못된 숫자를 보여주느니
    번역이 비어 있는 편이 낫다.
    """
    run_dir = Path(run_dir)
    # 뒤집히면 치명적인 조각은 LLM에 보내지 않는다
    items = [i for i in collect(run_dir) if i["key"] not in NEVER_LLM][:max_items]
    if not items:
        return {"ok": False, "reason": "번역할 원문 조각이 없습니다"}

    raw = backend.complete("plainspeak", PLAIN_SYSTEM, plain_prompt(items))
    try:
        got = json.loads(raw)["items"]
    except Exception as e:
        return {"ok": False, "reason": f"번역 결과를 읽지 못했습니다: {e!r}"}

    src = {i["key"]: i["text"] for i in items}
    out, rejected = {}, []
    for g in got:
        k, txt = g.get("key"), (g.get("text") or "").strip()
        if k not in src or not txt:
            continue
        strict = any(k.endswith(s) for s in STRICT_KEYS)
        chk = verify_numbers(src[k], txt, strict=strict)
        # 길이가 극단적으로 어긋나면 다른 조각이 잘못 배정됐을 가능성이 높다
        # (감사 D9: 조각 교차 배정은 숫자 검사를 통과한다). 값싼 정합성 검사다.
        ratio = len(txt) / max(1, len(src[k]))
        if not chk["ok"]:
            rejected.append({"key": k, "invented_numbers": chk["invented"],
                             "lost_numbers": chk.get("lost") or []})
        elif ratio > 3.0 or ratio < 0.3:
            rejected.append({"key": k, "reason": f"길이 비 {ratio:.2f} — 조각이 "
                                                 "뒤바뀌었을 수 있음"})
        else:
            out[k] = txt

    # 결정론적으로 만든 조각을 덮어쓴다 (LLM이 손대지 않은 자리)
    det = plain_independence(run_dir)
    if det:
        out["independence.verdict"] = det

    doc = {
        "items": out,
        "deterministic_keys": ["independence.verdict"] if det else [],
        "n_source": len(items),
        "n_translated": len(out),
        "rejected": rejected,
        "note": ("쉬운 말 번역본입니다. 원문은 그대로 보존되어 있고 언제든 볼 수 "
                 "있습니다. 숫자를 날조한 조각과, 반증 조건에서 마감일·임계값을 "
                 "잃은 조각은 자동으로 버려집니다 — 그 자리에는 원문이 그대로 "
                 "보입니다. 다만 숫자를 건드리지 않는 왜곡(부정어 뒤집기, "
                 "'절반·두 배' 같은 표현)까지 막지는 못합니다. 그래서 독립성 "
                 "판정문은 번역하지 않고 측정값에서 직접 만듭니다."),
    }
    from .lineage import atomic_write_text
    atomic_write_text(run_dir / "plain.json",
                      json.dumps(doc, ensure_ascii=False, indent=2))
    return {"ok": True, "n_translated": len(out), "n_source": len(items),
            "n_rejected": len(rejected)}


def load(run_dir: Path) -> dict | None:
    p = Path(run_dir) / "plain.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
