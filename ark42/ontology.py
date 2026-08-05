"""ARK-42 common ontology.

Every discipline's independent analysis is translated into this single
schema. Facts, inferences, and unknowns are kept separate by construction;
uncertainty is carried as (mean, std, confidence), never collapsed.
Scores are oriented so that HIGHER = MORE FAVORABLE for the option,
regardless of the criterion's natural direction (cost, risk are inverted
by the analyst at translation time and flagged via `direction_note`).
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

# Shared evaluation criteria (the "common ontology" axes).
CRITERIA: dict[str, str] = {
    "demand": "실수요 존재 — 대상 고객이 이 옵션의 결과물에 돈/시간을 쓸 근거",
    "feasibility": "실행 가능성 — 현재 자원·역량으로 실제 수행 가능한 정도",
    "cost_efficiency": "원가 효율 — 낮은 원가/자원 소모일수록 높은 점수 (방향 반전)",
    "revenue_potential": "수익 잠재력 — 매출·현금흐름 창출 가능성",
    "defensibility": "방어 가능성 — 경쟁 진입에 대한 구조적 방어",
    "risk_resilience": "위험 내성 — 규제·의존성·실패 위험이 낮을수록 높은 점수 (방향 반전)",
    "time_to_validation": "검증 속도 — 가설이 참/거짓으로 판명되기까지 짧을수록 높은 점수",
    "sustainability": "지속 가능성 — 유지보수·운영이 계속 굴러갈 수 있는 구조",
}


@dataclass
class Mechanism:
    """A causal claim: cause → effect, with signed strength and confidence."""
    cause: str
    effect: str
    direction: str          # "+" amplifies, "-" suppresses
    strength: float         # 0..1
    confidence: float       # 0..1
    evidence_type: str      # "empirical" | "theoretical" | "analogical"


@dataclass
class OptionAssessment:
    """One discipline's score for one option on one criterion."""
    option_id: str
    criterion: str          # key of CRITERIA
    score_mean: float       # 0..1, higher = more favorable
    score_std: float        # 0..0.5, epistemic spread
    confidence: float       # 0..1, how qualified this discipline is to judge this cell
    assumptions: list[str] = field(default_factory=list)
    direction_note: str = ""  # set when a naturally-negative criterion was inverted


@dataclass
class DisciplineAnalysis:
    """A single discipline's full analysis, translated into the ontology."""
    discipline: str
    relevance: float                     # 0..1 from the selector
    rationale: str
    facts: list[str] = field(default_factory=list)
    inferences: list[str] = field(default_factory=list)
    unknowns: list[str] = field(default_factory=list)
    mechanisms: list[Mechanism] = field(default_factory=list)
    assessments: list[OptionAssessment] = field(default_factory=list)

    def validate(self, option_ids: set[str]) -> list[str]:
        errors: list[str] = []
        for a in self.assessments:
            if a.criterion not in CRITERIA:
                errors.append(f"{self.discipline}: unknown criterion {a.criterion!r}")
            if a.option_id not in option_ids:
                errors.append(f"{self.discipline}: unknown option {a.option_id!r}")
            if not (0.0 <= a.score_mean <= 1.0):
                errors.append(f"{self.discipline}/{a.option_id}/{a.criterion}: score_mean out of [0,1]")
            if not (0.0 <= a.score_std <= 0.5):
                errors.append(f"{self.discipline}/{a.option_id}/{a.criterion}: score_std out of [0,0.5]")
            if not (0.0 <= a.confidence <= 1.0):
                errors.append(f"{self.discipline}/{a.option_id}/{a.criterion}: confidence out of [0,1]")
        return errors


@dataclass
class DecisionOption:
    option_id: str
    title: str
    description: str
    is_baseline: bool = False   # NO_INTERVENTION baseline


@dataclass
class Problem:
    problem_id: str
    statement: str               # natural-language problem as given
    context: str = ""
    options: list[DecisionOption] = field(default_factory=list)

    def ensure_baseline(self) -> None:
        """ARK-42 invariant: a NO_INTERVENTION baseline is always present."""
        if not any(o.is_baseline for o in self.options):
            self.options.append(DecisionOption(
                option_id="NO_INTERVENTION",
                title="개입하지 않음 (기준안)",
                description="아무 조치도 하지 않고 현 상태를 유지한다. 모든 옵션은 이 기준안 대비로 평가된다.",
                is_baseline=True,
            ))


def to_dict(obj: Any) -> Any:
    return asdict(obj)


def analysis_from_dict(d: dict) -> DisciplineAnalysis:
    return DisciplineAnalysis(
        discipline=d["discipline"],
        relevance=float(d["relevance"]),
        rationale=d.get("rationale", ""),
        facts=list(d.get("facts", [])),
        inferences=list(d.get("inferences", [])),
        unknowns=list(d.get("unknowns", [])),
        mechanisms=[Mechanism(**m) for m in d.get("mechanisms", [])],
        assessments=[OptionAssessment(**a) for a in d.get("assessments", [])],
    )
