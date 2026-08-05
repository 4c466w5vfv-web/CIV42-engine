"""Statevector qubit simulator — quantum computing without hardware.

WHAT THIS IS, HONESTLY
----------------------
A classical simulation of an n-qubit quantum register: the full complex
statevector (2^n amplitudes) with unitary gate operations and projective
measurement. This is the standard way quantum algorithms are developed and
verified worldwide; it IS quantum computation in the mathematical sense,
executed on classical hardware. It is NOT quantum hardware and claims no
speedup — the engine's honesty rules apply to its own internals.

ROLE IN THE ENGINE
------------------
Opt-in self-consistency check for the Monte Carlo ranking distribution
(montecarlo.simulate, env ARK42_QUANTUM=1): the option-rank probabilities are
amplitude-encoded into a quantum state through an explicit RY-rotation circuit
(Grover–Rudolph construction), then re-measured. If the re-measured
distribution diverges from the input beyond sampling noise, something is wrong
in the pipeline's probability handling. It never alters the classical results.

Stdlib + numpy only, deterministic under a seed — same rules as montecarlo.
"""
from __future__ import annotations

import numpy as np

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


class QubitRegister:
    """n-qubit register. Qubit 0 is the most significant bit of the index."""

    def __init__(self, n_qubits: int, seed: int | None = None):
        if not 1 <= n_qubits <= 20:      # 2^20 amplitudes ≈ 16MB — sane cap
            raise ValueError("n_qubits must be in 1..20")
        self.n = n_qubits
        self.state = np.zeros(2 ** n_qubits, dtype=complex)
        self.state[0] = 1.0
        self._rng = np.random.default_rng(seed)

    # ---- gates ---------------------------------------------------------
    def _apply1(self, gate: np.ndarray, q: int) -> None:
        s = self.state.reshape([2] * self.n)
        s = np.moveaxis(s, q, 0)
        s = (gate @ s.reshape(2, -1)).reshape([2] * self.n)
        self.state = np.moveaxis(s, 0, q).reshape(-1)

    def h(self, q: int) -> None:
        self._apply1(_H, q)

    def x(self, q: int) -> None:
        self._apply1(_X, q)

    def ry(self, theta: float, q: int) -> None:
        self._apply1(_ry(theta), q)

    def controlled_ry(self, theta: float, q: int, controls: dict) -> None:
        """RY on qubit q, applied only where each control qubit holds the
        given classical value ({qubit: 0|1}). Multi-controlled rotations are
        exactly what the Grover–Rudolph encoding circuit is built from."""
        idx = np.arange(2 ** self.n)
        mask = np.ones(2 ** self.n, dtype=bool)
        for cq, val in controls.items():
            mask &= ((idx >> (self.n - 1 - cq)) & 1) == val
        bit = ((idx >> (self.n - 1 - q)) & 1)
        lo = mask & (bit == 0)
        hi = mask & (bit == 1)
        pair = idx ^ (1 << (self.n - 1 - q))       # partner index (q flipped)
        g = _ry(theta)
        s = self.state
        new = s.copy()
        new[lo] = g[0, 0] * s[lo] + g[0, 1] * s[pair[lo]]
        new[hi] = g[1, 0] * s[pair[hi]] + g[1, 1] * s[hi]
        self.state = new

    def cnot(self, control: int, target: int) -> None:
        idx = np.arange(2 ** self.n)
        cbit = ((idx >> (self.n - 1 - control)) & 1) == 1
        pair = idx ^ (1 << (self.n - 1 - target))
        new = self.state.copy()
        new[cbit] = self.state[pair[cbit]]
        self.state = new

    def cz(self, a: int, b: int) -> None:
        idx = np.arange(2 ** self.n)
        both = (((idx >> (self.n - 1 - a)) & 1) == 1) & \
               (((idx >> (self.n - 1 - b)) & 1) == 1)
        self.state[both] *= -1

    # ---- measurement ---------------------------------------------------
    def probabilities(self) -> np.ndarray:
        p = np.abs(self.state) ** 2
        return p / p.sum()

    def sample(self, draws: int) -> np.ndarray:
        """Projective measurement in the computational basis, `draws` shots."""
        return self._rng.choice(2 ** self.n, size=draws, p=self.probabilities())


def encode_distribution(probs: np.ndarray, seed: int | None = None
                        ) -> QubitRegister:
    """Amplitude-encode a probability distribution via the Grover–Rudolph
    circuit: a tree of (controlled-)RY rotations, one level per qubit. The
    resulting state measures back to `probs` exactly (up to float error) —
    built from explicit gate operations, not by writing the vector directly.
    """
    p = np.asarray(probs, dtype=float)
    if p.ndim != 1 or len(p) < 2:
        raise ValueError("need a 1-D distribution with >= 2 outcomes")
    if not np.all(p >= 0):
        raise ValueError("probabilities must be non-negative")
    n = int(np.ceil(np.log2(len(p))))
    full = np.zeros(2 ** n)
    full[:len(p)] = p / p.sum()

    reg = QubitRegister(n, seed=seed)
    for level in range(n):
        block = 2 ** (n - level)                  # outcomes per branch node
        for node in range(2 ** level):
            seg = full[node * block:(node + 1) * block]
            total = seg.sum()
            if total <= 0:
                continue
            p_right = seg[block // 2:].sum() / total
            theta = 2 * np.arcsin(np.sqrt(np.clip(p_right, 0.0, 1.0)))
            controls = {q: (node >> (level - 1 - q)) & 1 for q in range(level)}
            reg.controlled_ry(theta, level, controls)
    return reg


def rank_consistency_check(p_rank1: np.ndarray, draws: int = 20000,
                           seed: int = 42) -> dict:
    """Encode the Monte Carlo rank-1 distribution into a quantum circuit,
    re-measure it, and report the divergence. A pipeline-health diagnostic —
    the classical numbers stay authoritative."""
    p = np.asarray(p_rank1, dtype=float)
    reg = encode_distribution(p, seed=seed)
    shots = reg.sample(draws)
    counts = np.bincount(shots, minlength=2 ** reg.n)[:len(p)]
    measured = counts / counts.sum()
    encoded = reg.probabilities()[:len(p)]
    target = p / p.sum()
    fidelity = float(np.sum(np.sqrt(encoded * target)) ** 2)
    return {
        "backend": "statevector-simulation (no quantum hardware; classical "
                   "simulation of the circuit — no speedup claimed)",
        "n_qubits": reg.n,
        "circuit": "Grover-Rudolph amplitude encoding (RY tree)",
        "p_rank1_input": target.tolist(),
        "p_rank1_measured": measured.tolist(),
        "encoding_fidelity": fidelity,           # 1.0 = exact encoding
        "max_abs_divergence": float(np.max(np.abs(measured - target))),
        "draws": draws,
    }
