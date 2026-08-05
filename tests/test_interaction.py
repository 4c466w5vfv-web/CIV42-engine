from __future__ import annotations
"""Tests for the cross-discipline interaction contraction (ark42.interaction).

Written as plain `def test_*()` functions with bare asserts — no pytest
fixtures, no pytest.approx — so they run both under pytest AND under a plain
python harness that imports this module and calls each test_* in turn.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ark42.interaction import (                                    # noqa: E402
    combined_utility,
    contributing_pairs,
    discipline_vectors,
    interaction_utility,
    normalize_coupling,
    zero_coupling,
)
from ark42.tensor import ScoreTensor                               # noqa: E402


def _tensor(mean, conf=None, relevance=None, std=None):
    mean = np.asarray(mean, dtype=float)
    O, C, D = mean.shape
    if conf is None:
        conf = np.full((O, C, D), 0.8)
    if std is None:
        std = np.full((O, C, D), 0.1)
    if relevance is None:
        relevance = np.ones(D)
    return ScoreTensor([f"O{i}" for i in range(O)],
                       [f"C{i}" for i in range(C)],
                       [f"d{i}" for i in range(D)],
                       mean, std, np.asarray(conf, dtype=float),
                       np.asarray(relevance, dtype=float))


def test_zero_coupling_is_exact_no_op():
    """J=0 → adjusted is bit-for-bit equal to additive, and interaction/delta
    are exactly zero. This is the reversibility guarantee for shipping."""
    rng = np.random.default_rng(0)
    V = rng.random((4, 6))
    U_add = rng.random(4)
    J = zero_coupling(6)
    out = combined_utility(U_add, V, J)
    assert np.array_equal(out["adjusted"], U_add)
    assert np.array_equal(out["interaction"], np.zeros(4))
    assert np.array_equal(out["delta"], np.zeros(4))


def test_contraction_matches_hand_computation():
    """einsum('od,de,oe->o') must equal V[o] @ J @ V[o] computed by hand for a
    tiny 2-option / 2-discipline case."""
    V = np.array([[1.0, 2.0],
                  [3.0, 4.0]])
    J = np.array([[0.0, 0.5],
                  [0.5, 0.0]])
    # by hand: U[o] = 2 * J[0,1] * V[o,0] * V[o,1]  (symmetric, zero diagonal)
    hand = np.array([2 * 0.5 * 1.0 * 2.0,      # = 2.0
                     2 * 0.5 * 3.0 * 4.0])     # = 12.0
    got = interaction_utility(V, J)
    assert np.allclose(got, hand)
    # also equals the explicit matrix product per option
    explicit = np.array([V[o] @ J @ V[o] for o in range(2)])
    assert np.allclose(got, explicit)


def test_normalize_coupling_symmetrizes_and_zeroes_diagonal():
    J = np.array([[9.0, 1.0, 0.0],
                  [3.0, 7.0, 2.0],
                  [0.0, 4.0, 5.0]])
    N = normalize_coupling(J)
    assert np.allclose(N, N.T)                 # symmetric
    assert np.allclose(np.diag(N), 0.0)        # zero diagonal
    # off-diagonal is the average of the two original entries
    assert abs(N[0, 1] - (1.0 + 3.0) / 2) < 1e-12
    assert abs(N[1, 2] - (2.0 + 4.0) / 2) < 1e-12
    # input not mutated
    assert J[0, 0] == 9.0


def test_symmetric_J_equals_its_transpose():
    """A symmetric coupling and its transpose give identical utilities (and a
    normalized coupling is invariant under transpose)."""
    rng = np.random.default_rng(1)
    V = rng.random((5, 4))
    J = normalize_coupling(rng.random((4, 4)))
    a = interaction_utility(V, J)
    b = interaction_utility(V, J.T)
    assert np.allclose(a, b)
    assert np.allclose(J, J.T)


def test_discipline_vectors_handles_zero_weight_cell():
    """A cell where the discipline declined every criterion for an option must
    give V=0, no NaN, no crash."""
    mean = np.full((2, 3, 2), 0.5)
    conf = np.full((2, 3, 2), 0.8)
    conf[0, :, 1] = 0.0                          # discipline 1 skipped option 0
    t = _tensor(mean, conf=conf)
    V = discipline_vectors(t)
    assert V.shape == (2, 2)
    assert not np.isnan(V).any()
    assert V[0, 1] == 0.0                        # zero-weight cell -> 0
    assert abs(V[0, 0] - 0.5) < 1e-12            # normal cell -> weighted mean


def test_positive_coupling_amplifies_high_high_option():
    """A positive coupling between two disciplines on which one option scores
    high on BOTH must raise that option's adjusted utility above its additive
    utility (amplification is real and directional)."""
    # option 0 scores high on both disciplines; option 1 high on only one
    mean = np.zeros((2, 1, 2))
    mean[0, 0, 0] = 0.9
    mean[0, 0, 1] = 0.9
    mean[1, 0, 0] = 0.9
    mean[1, 0, 1] = 0.1
    t = _tensor(mean)
    V = discipline_vectors(t)
    J = normalize_coupling(np.array([[0.0, 1.0], [1.0, 0.0]]))
    U_add = np.zeros(2)
    out = combined_utility(U_add, V, J)
    # option 0 (high-high) is amplified above additive; option 1 much less
    assert out["adjusted"][0] > out["additive"][0]
    assert out["adjusted"][0] > out["adjusted"][1]
    # provenance names the pair and calls it amplification
    pairs = contributing_pairs(V, J, t.options, t.disciplines)
    assert len(pairs) == 1
    assert pairs[0]["sign"] == "amplify"
    assert {pairs[0]["d1"], pairs[0]["d2"]} == {"d0", "d1"}


def test_contributing_pairs_empty_for_zero_coupling():
    """With the default zero coupling, nothing contributes — honest empty
    provenance rather than invented pairs."""
    V = np.random.default_rng(2).random((3, 4))
    pairs = contributing_pairs(V, zero_coupling(4),
                               [f"O{i}" for i in range(3)],
                               [f"d{i}" for i in range(4)])
    assert pairs == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:                                     # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
