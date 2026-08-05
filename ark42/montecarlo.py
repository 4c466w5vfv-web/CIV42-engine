"""Monte Carlo simulation over the score tensor.

Uncertainty sources sampled jointly per draw:
  1. Cell scores  — Beta distributions fit to each cell's (mean, std).
  2. Criterion weights — Dirichlet around the baseline.
  3. Discipline weights — relevance × confidence, jittered by Dirichlet.

Outputs per option: expected utility, percentiles, P(rank 1),
P(beats NO_INTERVENTION baseline), expected regret.
"""
from __future__ import annotations

import os

import numpy as np

from .tensor import ScoreTensor


def _beta_params(mean: np.ndarray, std: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Method-of-moments Beta fit, clipped away from degenerate edges."""
    m = np.clip(mean, 0.02, 0.98)
    v = np.clip(std, 0.02, 0.5) ** 2
    v = np.minimum(v, m * (1 - m) * 0.95)   # keep variance feasible for Beta
    k = m * (1 - m) / v - 1
    return m * k, (1 - m) * k


def simulate(t: ScoreTensor, baseline_id: str = "NO_INTERVENTION",
             n: int = 20000, weight_concentration: float = 30.0,
             seed: int = 42, rho: float = 0.0) -> dict:
    """rho: within-option correlation of cell scores across criteria and
    disciplines. rho=0 samples every cell independently — mathematically
    convenient but it averages away most input uncertainty (the scores all
    come from one LLM family and are in reality correlated), so P(rank1)
    comes out overconfident. rho>0 imposes a per-(draw, option) common
    factor via a Gaussian copula, keeping each cell's Beta marginal intact
    while widening each option's utility spread — a more honest, wider
    ranking distribution. Means (expected_utility) are unchanged by rho."""
    rng = np.random.default_rng(seed)
    O, C, D = t.mean.shape
    present = t.conf > 0
    a, b = _beta_params(np.nan_to_num(t.mean, nan=0.5), t.std)

    cell_w = t.cell_weight                     # (O,C,D)
    crit_base = np.full(C, 1.0 / C)
    util = np.empty((n, O))

    # Vectorized over draws in chunks to bound memory.
    chunk = 2000
    done = 0
    while done < n:
        k = min(chunk, n - done)
        if rho <= 0.0:
            scores = rng.beta(a[None], b[None], size=(k, O, C, D))
        else:
            scores = _correlated_scores(rng, a, b, k, O, C, D, rho)
        scores = np.where(present[None], scores, 0.0)
        dw = rng.dirichlet(np.full(D, weight_concentration), size=k)      # (k,D)
        w = cell_w[None] * dw[:, None, None, :]                            # (k,O,C,D)
        wsum = w.sum(axis=3)                                               # (k,O,C)
        crit = np.divide((scores * w).sum(axis=3), wsum,
                         out=np.full(wsum.shape, np.nan), where=wsum > 0)  # (k,O,C)
        cw = rng.dirichlet(crit_base * weight_concentration, size=k)       # (k,C)
        have = ~np.isnan(crit)
        eff = np.where(have, cw[:, None, :], 0.0)
        eff = eff / eff.sum(axis=2, keepdims=True)
        util[done:done + k] = np.nansum(crit * eff, axis=2)
        done += k

    order = np.argsort(-util, axis=1)
    p_rank1 = np.bincount(order[:, 0], minlength=O) / n
    best = util.max(axis=1, keepdims=True)
    regret = (best - util).mean(axis=0)

    bi = t.options.index(baseline_id) if baseline_id in t.options else None
    p_beats_baseline = ((util > util[:, [bi]]).mean(axis=0) if bi is not None
                        else np.full(O, np.nan))

    pct = np.percentile(util, [5, 25, 50, 75, 95], axis=0)
    # Opt-in quantum-circuit self-consistency check (ARK42_QUANTUM=1): the
    # rank-1 distribution is amplitude-encoded and re-measured on a simulated
    # qubit register (ark42.qubit). Diagnostic only — it never changes the
    # classical results above, and its output is labeled as simulation.
    quantum = None
    if os.environ.get("ARK42_QUANTUM") == "1" and len(t.options) >= 2:
        try:
            from .qubit import rank_consistency_check
            quantum = rank_consistency_check(p_rank1, seed=seed)
        except Exception:   # a diagnostic must never fail the run
            quantum = None
    return {
        **({"quantum_rank_check": quantum} if quantum else {}),
        "n_draws": n,
        "rho": rho,
        "options": t.options,
        "expected_utility": util.mean(axis=0),
        "utility_std": util.std(axis=0),
        "percentiles": {"p5": pct[0], "p25": pct[1], "p50": pct[2],
                        "p75": pct[3], "p95": pct[4]},
        "p_rank1": p_rank1,
        "p_beats_baseline": p_beats_baseline,
        "expected_regret": regret,
        "utility_samples_head": util[:2000],  # for the report's distribution plot
    }


def _correlated_scores(rng, a, b, k, O, C, D, rho):
    """Gaussian copula: per-(draw, option) common factor shared across the
    option's criteria and disciplines, correlation rho, plus per-cell noise.
    Each cell keeps its Beta(a,b) marginal via the inverse-CDF transform."""
    from scipy.stats import beta as _beta, norm as _norm
    zc = rng.standard_normal((k, O, 1, 1))                 # per-option factor
    ze = rng.standard_normal((k, O, C, D))                 # per-cell noise
    z = np.sqrt(rho) * zc + np.sqrt(1.0 - rho) * ze        # corr rho within option
    u = _norm.cdf(z)
    u = np.clip(u, 1e-6, 1 - 1e-6)
    return _beta.ppf(u, a[None], b[None])
