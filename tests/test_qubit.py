from __future__ import annotations

"""ark42.qubit — statevector simulator correctness.

Physics ground truths a simulator must reproduce: normalized states, Bell
correlations, exact amplitude encoding, and measurement statistics that
converge to the encoded distribution.
"""

import numpy as np
import pytest

from ark42.qubit import QubitRegister, encode_distribution, rank_consistency_check


def test_bell_state_is_maximally_correlated():
    reg = QubitRegister(2, seed=7)
    reg.h(0)
    reg.cnot(0, 1)
    p = reg.probabilities()
    # |00> and |11> at 0.5 each; cross terms zero.
    assert p[0] == pytest.approx(0.5, abs=1e-12)
    assert p[3] == pytest.approx(0.5, abs=1e-12)
    assert p[1] == pytest.approx(0.0, abs=1e-12)
    assert p[2] == pytest.approx(0.0, abs=1e-12)
    shots = reg.sample(4000)
    assert set(np.unique(shots)) <= {0, 3}     # only correlated outcomes


def test_state_stays_normalized_through_circuit():
    reg = QubitRegister(3, seed=1)
    reg.h(0); reg.ry(0.7, 1); reg.cnot(0, 2); reg.cz(1, 2); reg.x(1)
    assert np.abs(reg.state).__pow__(2).sum() == pytest.approx(1.0, abs=1e-10)


def test_amplitude_encoding_reproduces_distribution_exactly():
    target = np.array([0.42, 0.23, 0.20, 0.10, 0.05])
    reg = encode_distribution(target, seed=3)
    got = reg.probabilities()[:5]
    assert np.allclose(got, target, atol=1e-10)


def test_measurement_converges_to_encoded_distribution():
    target = np.array([0.6, 0.25, 0.15])
    out = rank_consistency_check(target, draws=40000, seed=11)
    assert out["encoding_fidelity"] == pytest.approx(1.0, abs=1e-9)
    assert out["max_abs_divergence"] < 0.02    # sampling noise at 40k draws
    assert "no quantum hardware" in out["backend"]  # honesty label is load-bearing


def test_quantum_check_is_optional_and_off_by_default(monkeypatch):
    monkeypatch.delenv("ARK42_QUANTUM", raising=False)
    from ark42 import montecarlo, tensor
    # smallest possible tensor: 2 options × 1 criterion × 1 discipline
    t = tensor.ScoreTensor(
        options=["A", "NO_INTERVENTION"], criteria=["c"], disciplines=["d"],
        mean=np.array([[[0.7]], [[0.4]]]), std=np.array([[[0.1]], [[0.1]]]),
        conf=np.array([[[0.9]], [[0.9]]]),
        relevance=np.array([1.0]))
    out = montecarlo.simulate(t, n=500, seed=5)
    assert "quantum_rank_check" not in out
    monkeypatch.setenv("ARK42_QUANTUM", "1")
    out2 = montecarlo.simulate(t, n=500, seed=5)
    assert out2["quantum_rank_check"]["encoding_fidelity"] > 0.999
    # 진단은 고전 결과를 절대 바꾸지 않는다
    assert np.allclose(out["expected_utility"], out2["expected_utility"])
