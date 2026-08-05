"""Cross-discipline interaction via a genuine tensor contraction.

This module makes "tensor networking" a real multilinear operation rather
than a metaphor. The rest of the engine (tensor.py, montecarlo.py) collapses
the score tensor with axis-wise weighted means — every discipline is treated
as additively separable. That is deliberately conservative but it cannot
express *interactions*: the idea that scoring well on, say, economics AND
law together is worth more (or less) than the sum of the two parts.

We model those interactions with a symmetric coupling tensor J over the
discipline axis and contract it against each option's discipline-score
vector V[o]:

    U_int[o] = V[o] · J · V[o]        (a quadratic form; einsum 'od,de,oe->o')

This is a true tensor-network contraction over the discipline dimension, not
a mean. J is symmetric with a zero diagonal (no discipline couples with
itself — that signal already lives in the additive term).

HONESTY / PROVENANCE
--------------------
There are NO invented coupling numbers in this file. J defaults to
`zero_coupling(D)` — all zeros — which makes the whole contraction an exact
no-op (U_int == 0, adjusted == additive). Any non-zero J must be supplied by
the caller from coefficients LEARNED from recorded run outcomes elsewhere in
the system (the learning layer), never hardcoded here. Shipping this module
with the default J is therefore reversible and safe: it changes no decision
until real, learned evidence turns a coupling on.
"""
from __future__ import annotations

import numpy as np

from .tensor import ScoreTensor


def zero_coupling(D: int) -> np.ndarray:
    """The safe default: no coupling at all. Shape (D, D), all zeros.

    With this J the contraction is an exact no-op — see the module docstring
    and `interaction_utility`. This is what any non-learned run must use."""
    return np.zeros((D, D), dtype=np.float64)


def normalize_coupling(J: np.ndarray) -> np.ndarray:
    """Project an arbitrary matrix onto a valid coupling tensor.

    Enforces the two structural invariants a coupling must satisfy:
      1. symmetric — J[d1,d2] == J[d2,d1] (interaction is mutual), via (J+J.T)/2
      2. zero diagonal — no self-coupling (that signal is the additive term)

    Returns a new float64 array; the input is not mutated."""
    J = np.asarray(J, dtype=np.float64)
    if J.ndim != 2 or J.shape[0] != J.shape[1]:
        raise ValueError(f"coupling must be square (D,D), got {J.shape}")
    sym = (J + J.T) / 2.0
    np.fill_diagonal(sym, 0.0)
    return sym


def discipline_vectors(tensor: ScoreTensor) -> np.ndarray:
    """Per-option discipline-score vectors V, shape (O, D).

    V[o,d] is the criterion-weighted mean of discipline d's scores for option
    o, using the tensor's own cell weights (conf × relevance):

        V[o,d] = sum_c cell_weight[o,c,d] * mean[o,c,d]
                 / sum_c cell_weight[o,c,d]

    A cell whose weights sum to zero (discipline declined every criterion for
    that option) yields V[o,d] = 0 rather than NaN — consistent with how
    tensor.aggregate treats empty cells."""
    w = tensor.cell_weight                       # (O, C, D)
    m = np.nan_to_num(tensor.mean)               # (O, C, D), NaN cells -> 0
    wsum = w.sum(axis=1)                          # (O, D)
    num = np.sum(m * w, axis=1)                   # (O, D)
    V = np.divide(num, wsum,
                  out=np.zeros_like(num, dtype=np.float64),
                  where=wsum > 0)
    return V.astype(np.float64)


def interaction_utility(V: np.ndarray, J: np.ndarray) -> np.ndarray:
    """Contract V against the coupling J: U_int[o] = V[o] · J · V[o].

    Implemented as the genuine tensor-network contraction
    np.einsum('od,de,oe->o', V, J, V). Shape (O,).

    INVARIANT: if J is all zeros the result is exactly 0 for every option
    (einsum of a zero operand is exactly 0.0, no floating-point residue), so
    a zero coupling is an exact no-op on the adjusted utility."""
    V = np.asarray(V, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    return np.einsum('od,de,oe->o', V, J, V)


def combined_utility(U_add, V: np.ndarray, J: np.ndarray,
                     lam: float = 1.0) -> dict:
    """Combine the existing additive utility with the interaction term.

    lam scales how much the interaction is allowed to move the decision
    (lam=0 disables it entirely, independent of J). Returns numpy arrays:

        additive    — U_add, unchanged
        interaction — U_int = V·J·V per option
        adjusted    — U_add + lam * U_int
        delta       — lam * U_int (the adjustment, for provenance)

    When J == zero_coupling(D), interaction and delta are exactly 0 and
    adjusted equals additive elementwise (np.array_equal holds)."""
    U_add = np.asarray(U_add, dtype=np.float64)
    U_int = interaction_utility(V, J)
    delta = lam * U_int
    return {
        "additive": U_add,
        "interaction": U_int,
        "adjusted": U_add + delta,
        "delta": delta,
    }


def contributing_pairs(V: np.ndarray, J: np.ndarray,
                       options: list, disciplines: list,
                       top_k: int = 5) -> list:
    """Rank the discipline pairs that most drive the interaction adjustment.

    Contribution of an unordered pair (d1, d2) is scored by the magnitude of
    its coupling times a typical magnitude of the product of the two
    disciplines' scores across options:

        |J[d1,d2]| * mean_o |V[o,d1] * V[o,d2]|

    Only the upper triangle (d1 < d2) is considered — J is symmetric, so each
    unordered pair is counted once. Pairs with zero coupling are omitted
    (with the default zero J this returns an empty list — honest: nothing
    contributed). Each entry:

        {d1, d2, coupling: J[d1,d2], sign: 'amplify' | 'dampen',
         contribution: <float>}

    sign is 'amplify' when the coupling is positive (scoring well on both
    disciplines raises adjusted utility) and 'dampen' when negative."""
    V = np.asarray(V, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    D = J.shape[0]
    typical = np.mean(np.abs(V[:, :, None] * V[:, None, :]), axis=0)  # (D, D)
    rows = []
    for i in range(D):
        for j in range(i + 1, D):
            c = J[i, j]
            if c == 0.0:
                continue
            rows.append({
                "d1": disciplines[i],
                "d2": disciplines[j],
                "coupling": float(c),
                "sign": "amplify" if c > 0 else "dampen",
                "contribution": float(abs(c) * typical[i, j]),
            })
    rows.sort(key=lambda r: r["contribution"], reverse=True)
    return rows[:top_k]
