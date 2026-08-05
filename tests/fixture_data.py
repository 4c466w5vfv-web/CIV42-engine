from __future__ import annotations
"""Deterministic fixture: a tiny recorded-LLM response set.

Two disciplines with OPPOSING views (economics favors FIX_A, marketing
favors FIX_B) so that a resolved outcome produces genuinely different
alignment per discipline. This is test design, not result manipulation:
no score or weight is set to force Run B's numbers — they follow from
the update rule alone.

Everything here is synthetic and must only ever touch sandbox stores.
"""
import json
from pathlib import Path

FRAMING = {
    "options": [
        {"option_id": "FIX_A", "title": "고정 A안",
         "description": "테스트 픽스처 옵션 A. 경제학이 선호하는 구조."},
        {"option_id": "FIX_B", "title": "고정 B안",
         "description": "테스트 픽스처 옵션 B. 마케팅이 선호하는 구조."},
        {"option_id": "FIX_C", "title": "고정 C안",
         "description": "테스트 픽스처 옵션 C. 두 학문 모두 중립인 구조."},
    ],
    "framing_note": "test fixture: 두 학문이 상반된 판단을 하도록 설계됨",
}

SELECTION = {
    "selected": [
        {"discipline": "economics", "relevance": 0.9, "why": "fixture"},
        {"discipline": "marketing", "relevance": 0.8, "why": "fixture"},
    ],
    "rejected_examples": [],
}


def _analysis(discipline, relevance, favored, disfavored):
    def cells(option, base):
        return [{"option_id": option, "criterion": c, "score_mean": base,
                 "score_std": 0.08, "confidence": 0.8, "assumptions": [],
                 "direction_note": ""}
                for c in ("demand", "feasibility", "revenue_potential")]
    return {
        "discipline": discipline, "relevance": relevance,
        "rationale": "fixture rationale",
        "facts": ["fixture fact"], "inferences": ["fixture inference"],
        "unknowns": ["fixture unknown"],
        "mechanisms": [],
        "assessments": (cells(favored, 0.75) + cells(disfavored, 0.45)
                        + cells("FIX_C", 0.55) + cells("NO_INTERVENTION", 0.30)),
    }


FORECASTS = {
    "forecasts": [
        {"option_id": "FIX_A", "predictions": [
            {"metric": "지표1", "prediction": "6개월 내 X 달성", "probability": 0.7,
             "horizon_months": 6, "falsified_if": "X 미달성"},
            {"metric": "지표2", "prediction": "12개월 내 Y 달성", "probability": 0.6,
             "horizon_months": 12, "falsified_if": "Y 미달성"},
        ]},
        {"option_id": "FIX_B", "predictions": [
            {"metric": "지표3", "prediction": "6개월 내 Z 달성", "probability": 0.5,
             "horizon_months": 6, "falsified_if": "Z 미달성"},
        ]},
        {"option_id": "NO_INTERVENTION", "predictions": [
            {"metric": "지표4", "prediction": "현상 유지 시 W 발생", "probability": 0.5,
             "horizon_months": 6, "falsified_if": "W 미발생"},
        ]},
    ]
}

PROBLEM = {
    "problem_id": None,   # set per run
    "statement": "픽스처 문제: 고정 A안과 B안 중 무엇을 택할 것인가 (테스트 전용)",
    "context": "synthetic test fixture — 현실 증거 아님",
}


def write_recorded(dst: Path) -> Path:
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "framing.json").write_text(json.dumps(FRAMING, ensure_ascii=False))
    (dst / "selection.json").write_text(json.dumps(SELECTION, ensure_ascii=False))
    (dst / "analysis_economics.json").write_text(json.dumps(
        _analysis("economics", 0.9, favored="FIX_A", disfavored="FIX_B"),
        ensure_ascii=False))
    (dst / "analysis_marketing.json").write_text(json.dumps(
        _analysis("marketing", 0.8, favored="FIX_B", disfavored="FIX_A"),
        ensure_ascii=False))
    (dst / "forecasts.json").write_text(json.dumps(FORECASTS, ensure_ascii=False))
    return dst


class OptionAwareBackend:
    """A deterministic stand-in that answers about THIS problem's options.

    :class:`~ark42.llm.RecordedBackend` replays one frozen response set, so it
    can only be used on the fixture problem — point it at a real case and every
    forecast comes back about FIX_A/FIX_B and nothing resolves. This backend
    instead parses the option ids out of the prompt it is given and emits
    well-formed responses for them, which is what makes it possible to run real
    cases end-to-end with no API key and no network.

    IT PRODUCES NO INFORMATION. Scores are a fixed function of the option's
    position, not of its content. Use it to prove the PLUMBING carries real
    cases — schema, verdict mapping, scoring, leakage split. Any Brier it
    yields is an artefact of this file and must never be reported as a
    calibration result.
    """

    provider_id = "synthetic:option-aware"

    def __init__(self, disciplines=("economics", "marketing", "law_regulation")):
        self.disciplines = list(disciplines)

    @staticmethod
    def _option_ids(prompt: str) -> list[str]:
        """Option ids as the prompt lists them, in order, deduplicated.

        The prompt renders options as ``- <ID>: <title> — <desc>`` lines under
        a '의사결정 옵션' header. Two things nearby look identical to a naive
        matcher and both broke this: a JSON schema example containing the
        literal ``"option_id": "..."``, and the shared criteria list, which is
        also ``- <name>: <desc>`` lines. So scope the match to the option block
        — from its header to the first blank line — instead of the whole text.
        """
        import re
        # The analyst prompt heads the block '의사결정 옵션 (...)'; the
        # forecaster prompt heads it just '옵션:'. Missing the second one made
        # every forecast come back for a default A/B pair, which silently
        # unresolved any case whose real winner was option C.
        head = -1
        for marker in ("의사결정 옵션", "\n옵션:\n"):
            head = prompt.find(marker)
            if head != -1:
                break
        if head == -1:
            return []
        block = prompt[head:].split("\n\n", 1)[0]
        seen, out = set(), []
        for m in re.finditer(r'^-\s+([A-Za-z][A-Za-z0-9_\-]{0,31}):\s',
                             block, re.MULTILINE):
            oid = m.group(1)
            if oid not in seen:
                seen.add(oid)
                out.append(oid)
        return out

    def complete(self, key: str, system: str, prompt: str) -> str:
        ids = self._option_ids(prompt) or ["A", "B"]
        if key == "framing":
            return json.dumps({"options": [
                {"option_id": i, "title": f"옵션 {i}",
                 "description": "synthetic backend — 내용 없음"} for i in ids],
                "framing_note": "synthetic"}, ensure_ascii=False)
        if key == "selection":
            return json.dumps({"selected": [
                {"discipline": d, "relevance": 0.9 - 0.1 * n, "why": "synthetic"}
                for n, d in enumerate(self.disciplines)],
                "rejected_examples": []}, ensure_ascii=False)
        if key.startswith("analysis_"):
            disc = key[len("analysis_"):]
            # Deterministic, content-free: the k-th option always scores the
            # same. Varying it by discipline index keeps disciplines from being
            # perfectly identical, which would make the independence measure
            # degenerate rather than merely uninformative.
            shift = 0.05 * (self.disciplines.index(disc)
                            if disc in self.disciplines else 0)
            cells = []
            for n, oid in enumerate(ids):
                base = max(0.05, min(0.95, 0.70 - 0.12 * n + shift))
                cells += [{"option_id": oid, "criterion": c, "score_mean": base,
                           "score_std": 0.10, "confidence": 0.7,
                           "assumptions": [], "direction_note": ""}
                          for c in ("demand", "feasibility", "revenue_potential")]
            return json.dumps({
                "discipline": disc, "relevance": 0.8,
                "rationale": "synthetic backend — 실제 분석 아님",
                "facts": [], "inferences": [], "unknowns": ["everything"],
                "mechanisms": [], "assessments": cells}, ensure_ascii=False)
        if key == "forecasts":
            return json.dumps({"forecasts": [
                {"option_id": oid, "predictions": [
                    {"metric": f"지표{n + 1}",
                     "prediction": "synthetic prediction",
                     "probability": max(0.05, min(0.95, 0.60 - 0.15 * n)),
                     "horizon_months": 12,
                     "falsified_if": "synthetic"}]}
                for n, oid in enumerate(ids)]}, ensure_ascii=False)
        raise FileNotFoundError(f"OptionAwareBackend: unhandled key {key!r}")


def execute_run(run_id: str, runs_dir: Path, recorded_dir: Path,
                library_dir: Path | None, channel: str = "sandbox"):
    """Run the full pipeline on the fixture. Returns (Run, results)."""
    from ark42.llm import RecordedBackend
    from ark42.pipeline import Run, make_problem
    spec = dict(PROBLEM, problem_id=run_id)
    p = make_problem(spec["problem_id"], spec["statement"], spec["context"], [])
    run = Run(p, RecordedBackend(recorded_dir), runs_dir,
              library_dir=library_dir, library_channel=channel)
    run.frame_options()
    chosen = run.select_disciplines()
    analyses = run.analyze(chosen)
    results = run.quantify(analyses, n_draws=4000)
    run.forecast(results)
    return run, results
