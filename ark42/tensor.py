"""Score tensor and deterministic multi-dimensional analysis.

The tensor is T[option, criterion, discipline] holding (mean, std,
confidence). This is the honest name for what the product spec called
"tensor network analysis": a 3-axis score tensor with weighted
aggregation, disagreement measurement, and sensitivity analysis.
Missing cells (a discipline that declined a criterion) carry weight 0.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ontology import CRITERIA, DisciplineAnalysis, Problem


@dataclass
class ScoreTensor:
    options: list[str]
    criteria: list[str]
    disciplines: list[str]
    mean: np.ndarray        # (O, C, D)
    std: np.ndarray         # (O, C, D)
    conf: np.ndarray        # (O, C, D), 0 where cell missing
    relevance: np.ndarray   # (D,) selector relevance per discipline

    @property
    def cell_weight(self) -> np.ndarray:
        """Weight of each cell = discipline relevance × cell confidence."""
        return self.conf * self.relevance[None, None, :]


def build_tensor(problem: Problem, analyses: list[DisciplineAnalysis],
                 reliability: dict[str, float] | None = None) -> ScoreTensor:
    """reliability: optional per-discipline multiplier from the learning
    layer (library/reliability.json), bounded [0.5, 1.5]. Unknown
    disciplines default to 1.0 — no evidence, no adjustment."""
    options = [o.option_id for o in problem.options]
    criteria = list(CRITERIA.keys())
    disciplines = [a.discipline for a in analyses]
    O, C, D = len(options), len(criteria), len(disciplines)
    mean = np.full((O, C, D), np.nan)
    std = np.zeros((O, C, D))
    conf = np.zeros((O, C, D))
    rel = reliability or {}
    relevance = np.array([a.relevance * rel.get(a.discipline, 1.0)
                          for a in analyses])
    oi = {o: i for i, o in enumerate(options)}
    ci = {c: i for i, c in enumerate(criteria)}
    for d, a in enumerate(analyses):
        for s in a.assessments:
            i, j = oi[s.option_id], ci[s.criterion]
            mean[i, j, d] = s.score_mean
            std[i, j, d] = s.score_std
            conf[i, j, d] = s.confidence
    return ScoreTensor(options, criteria, disciplines, mean, std, conf, relevance)


def aggregate(t: ScoreTensor, criterion_weights: np.ndarray | None = None) -> dict:
    """Collapse the tensor deterministically (point estimate, no sampling)."""
    w = t.cell_weight                                    # (O,C,D)
    m = np.nan_to_num(t.mean)
    wsum = w.sum(axis=2)                                 # (O,C)
    crit_score = np.divide(np.sum(m * w, axis=2), wsum,
                           out=np.full(wsum.shape, np.nan), where=wsum > 0)  # (O,C)
    cw = _norm_weights(criterion_weights, len(t.criteria))
    have = ~np.isnan(crit_score)
    eff = np.where(have, cw[None, :], 0.0)
    eff = eff / eff.sum(axis=1, keepdims=True)
    utility = np.nansum(crit_score * eff, axis=1)        # (O,)

    # Disagreement: weighted std of discipline means per (option, criterion),
    # then weight-averaged over criteria. High = the disciplines conflict.
    dev = np.where(w > 0, (m - np.nan_to_num(crit_score)[:, :, None]) ** 2, 0.0)
    disp = np.divide(np.sum(dev * w, axis=2), wsum,
                     out=np.zeros(wsum.shape), where=wsum > 0) ** 0.5      # (O,C)
    disagreement = np.sum(disp * eff, axis=1)            # (O,)

    coverage = (w.sum(axis=(1, 2)) > 0).astype(float)    # crude sanity flag
    return {
        "criterion_scores": crit_score,                  # (O,C)
        "utility": utility,                              # (O,)
        "disagreement": disagreement,                    # (O,)
        "criterion_weights": cw,
        "coverage_ok": bool(coverage.all()),
    }


def sensitivity(t: ScoreTensor, n: int = 2000, concentration: float = 30.0,
                seed: int = 7) -> dict:
    """Rank stability under criterion-weight perturbation (Dirichlet).

    Answers: does the winning option depend on how the criteria are weighted?
    Returns P(rank 1) per option under weight uncertainty alone (scores fixed
    at their means — score uncertainty is Monte Carlo's job, kept separate
    so the two uncertainty sources are attributable)."""
    rng = np.random.default_rng(seed)
    C = len(t.criteria)
    base = np.full(C, 1.0 / C)
    wins = np.zeros(len(t.options))
    for _ in range(n):
        cw = rng.dirichlet(base * concentration)
        util = aggregate(t, cw)["utility"]
        wins[int(np.nanargmax(util))] += 1
    return {"p_rank1_weights_only": wins / n}


def _norm_weights(w: np.ndarray | None, c: int) -> np.ndarray:
    if w is None:
        return np.full(c, 1.0 / c)
    w = np.asarray(w, dtype=float)
    return w / w.sum()
