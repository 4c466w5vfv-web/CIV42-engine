"""Evidence-gated learning of the discipline-interaction coupling matrix J.

The additive model predicts a utility/probability per option WITHOUT any
discipline-interaction term. This module learns, from RESOLVED OUTCOME cases
only, a symmetric zero-diagonal coupling matrix J such that the interaction
utility of the chosen option is  V[chosen] @ J @ V[chosen].

Honesty invariant (mirrors learning.py's "principle changes require verified
evidence"): a coupling coefficient J[d1,d2] stays 0 unless BOTH
  (a) enough real cases had both disciplines actually participating, AND
  (b) the coefficient IMPROVES held-out prediction (out-of-sample Brier).
Nothing here is invented. A missing or under-supported pair is 0, never a guess.

Storage (under <data_dir>/):
  J_updates.jsonl   append-only ledger, one line per learned-matrix update,
                    with before/after coefficients, the exact supporting case
                    ids per changed pair, a passed-in timestamp, and hashes —
                    so every nonzero coefficient traces back to the outcomes
                    that justified it. Matches weight_updates.jsonl style.

This module imports NOTHING from interaction.py (kept pure math, no cycle);
it only reuses the lineage primitives that learning.py / outcomes.py use.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .lineage import atomic_write_text, canonical_hash, make_id  # noqa: F401

# A discipline "participates" in a case when its score vector entry is nonzero.
_PARTICIPATION_TOL = 1e-12


# ----------------------------------------------------------------- helpers
def zero_coupling(n: int) -> np.ndarray:
    """The honest default: no learned coupling is all-zeros."""
    return np.zeros((n, n), dtype=float)


def _brier(pred: np.ndarray, y: np.ndarray) -> float:
    """Brier score (mean squared error of probabilities), same rule family as
    outcomes._summarize. Predictions are clipped to [0,1] — a probability
    outside the unit interval is not a valid forecast."""
    p = np.clip(np.asarray(pred, dtype=float), 0.0, 1.0)
    return float(np.mean((p - np.asarray(y, dtype=float)) ** 2))


def _fit_slope(x: np.ndarray, r: np.ndarray) -> float:
    """Least-squares slope of residual r on the CENTERED feature x (simple 1-D
    regression, no free intercept beyond centering). Zero-variance feature →
    slope 0 (no signal to fit)."""
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom <= 0.0:
        return 0.0
    return float((xc * (r - r.mean())).sum() / denom)


def _kfold_indices(n: int, k: int, seed: int) -> list[np.ndarray]:
    """Deterministic k-fold split under the given seed."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    return [f for f in np.array_split(idx, k) if len(f) > 0]


def _heldout_brier(x: np.ndarray, add_pred: np.ndarray, y: np.ndarray,
                   k: int, seed: int) -> tuple[float, float]:
    """Out-of-sample Brier of (add_pred + J*x) vs add_pred alone.

    For each held-out fold the slope J is fit on the OTHER folds only, then
    applied to the held-out cases as the deployed model would apply it
    (add_pred + J * raw_feature). Held-out predictions are pooled and scored
    once, so every case is weighted equally. This is deliberately conservative:
    the coefficient must earn its keep on data it never saw.
    """
    n = len(x)
    r = y - add_pred
    folds = _kfold_indices(n, min(k, n), seed)
    pred_before = np.empty(n, dtype=float)
    pred_after = np.empty(n, dtype=float)
    order = np.empty(n, dtype=int)
    pos = 0
    for f in folds:
        train = np.setdiff1d(np.arange(n), f, assume_unique=False)
        slope = _fit_slope(x[train], r[train]) if len(train) >= 2 else 0.0
        for j in f:
            pred_before[pos] = add_pred[j]
            pred_after[pos] = add_pred[j] + slope * x[j]
            order[pos] = j
            pos += 1
    ys = y[order]
    return _brier(pred_before, ys), _brier(pred_after, ys)


# ------------------------------------------------------------ core learning
def learn_coupling(cases: list[dict], disciplines: list[str],
                   min_cases: int = 8, seed: int = 0) -> dict:
    """Learn J from resolved-outcome cases, one coefficient per discipline pair.

    Each case must provide at least:
      v_chosen : length-D discipline score vector of the chosen option
      add_pred : additive model's predicted score/prob for the chosen option, [0,1]
      y        : realized outcome, [0,1]
      case_id  : (optional) provenance handle

    For pair (d1<d2): feature x = v_chosen[d1]*v_chosen[d2];
    residual r = y - add_pred (what the additive model failed to explain);
    J[d1,d2] = LS slope of r on centered x over supporting cases.

    EVIDENCE GATE — J[d1,d2] stays 0 unless BOTH hold:
      (a) n_support >= min_cases (cases where BOTH disciplines participated), and
      (b) held-out mean Brier of (add_pred + J*x) is STRICTLY less than that of
          add_pred alone (deterministic k-fold under `seed`).
    A pair with a raw slope that fails the gate is reported in `gated_out` and
    its coefficient is 0.
    """
    D = len(disciplines)
    J = zero_coupling(D)
    n_support = [[0] * D for _ in range(D)]
    gated_out: list[dict] = []
    learned_pairs: list[dict] = []

    # Vectorize the case fields once.
    V = np.array([np.asarray(c["v_chosen"], dtype=float) for c in cases]) \
        if cases else np.zeros((0, D))
    add = np.array([float(c["add_pred"]) for c in cases]) if cases else np.zeros(0)
    y = np.array([float(c["y"]) for c in cases]) if cases else np.zeros(0)
    ids = [c.get("case_id") for c in cases]

    for i in range(D):
        for jx in range(i + 1, D):
            if V.shape[0] == 0:
                continue
            participates = (np.abs(V[:, i]) > _PARTICIPATION_TOL) & \
                           (np.abs(V[:, jx]) > _PARTICIPATION_TOL)
            n = int(participates.sum())
            n_support[i][jx] = n_support[jx][i] = n
            if n < 2:
                continue

            x = V[participates, i] * V[participates, jx]
            ap = add[participates]
            yy = y[participates]
            r = yy - ap
            raw_slope = _fit_slope(x, r)

            support_ids = [ids[t] for t, p in enumerate(participates) if p]

            # Gate (a): enough real support.
            if n < min_cases:
                if raw_slope != 0.0:
                    gated_out.append({
                        "d1": disciplines[i], "d2": disciplines[jx],
                        "raw_slope": round(raw_slope, 6), "n": n,
                        "reason": f"under_supported (n={n} < min_cases={min_cases})",
                    })
                continue

            # Gate (b): must improve held-out prediction.
            brier_before, brier_after = _heldout_brier(x, ap, yy, k=5, seed=seed)
            improves = brier_after < brier_before

            if improves and raw_slope != 0.0:
                J[i][jx] = J[jx][i] = raw_slope
                learned_pairs.append({
                    "d1": disciplines[i], "d2": disciplines[jx],
                    "coupling": round(raw_slope, 6), "n": n,
                    "brier_before": round(brier_before, 6),
                    "brier_after": round(brier_after, 6),
                    "support_case_ids": support_ids,
                })
            elif raw_slope != 0.0:
                gated_out.append({
                    "d1": disciplines[i], "d2": disciplines[jx],
                    "raw_slope": round(raw_slope, 6), "n": n,
                    "brier_before": round(brier_before, 6),
                    "brier_after": round(brier_after, 6),
                    "reason": "no_heldout_improvement",
                })

    return {
        "J": J.tolist(),
        "n_support": n_support,
        "gated_out": gated_out,
        "learned_pairs": learned_pairs,
        "method": ("per-pair least-squares slope of (y - add_pred) on the "
                   "centered co-score feature v[d1]*v[d2], each coefficient "
                   "gated to 0 unless n_support >= min_cases AND it strictly "
                   "improves deterministic 5-fold held-out Brier; symmetric, "
                   "zero-diagonal. No coefficient is invented — under-supported "
                   "or non-generalizing pairs stay 0."),
        "disciplines": list(disciplines),
        "min_cases": min_cases,
        "seed": seed,
    }


# --------------------------------------------------------------- provenance
def record_coupling_update(data_dir, before_J, after_J, supporting_case_ids,
                           meta: dict, stamp: str) -> dict:
    """Append one line to J_updates.jsonl recording a coupling-matrix update.

    Append-only and atomic per-line, matching weight_updates.jsonl style. The
    timestamp is a PASSED-IN param (`stamp`) — determinism matters, so this
    function never reads the clock. `supporting_case_ids` maps each changed
    pair to the exact outcome case ids that justified its coefficient, so a
    reader can trace every nonzero coefficient back to real outcomes.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    before = np.asarray(before_J, dtype=float).tolist()
    after = np.asarray(after_J, dtype=float).tolist()

    changed_pairs = []
    a = np.asarray(after_J, dtype=float)
    b = np.asarray(before_J, dtype=float)
    D = a.shape[0]
    for i in range(D):
        for jx in range(i + 1, D):
            if a[i][jx] != b[i][jx]:
                changed_pairs.append([i, jx])

    hash_before = canonical_hash(before)
    hash_after = canonical_hash(after)
    entry = {
        "update_id": make_id("jupd", stamp, hash_before, hash_after),
        "type": "coupling_update",
        "stamp": stamp,
        "before_J": before,
        "after_J": after,
        "changed_pairs": changed_pairs,
        "supporting_case_ids": supporting_case_ids,
        "meta": meta,
        "hash_before": hash_before,
        "hash_after": hash_after,
    }
    entry["hash"] = canonical_hash({k: v for k, v in entry.items() if k != "hash"})

    ledger = data_dir / "J_updates.jsonl"
    with open(ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def learn_from_outcome(library_dir, run_dir, stamp: str,
                       min_cases: int = 8) -> dict:
    """PRODUCTION writer for J — the piece that was missing, leaving the
    coupling learner structurally inert.

    Called once per resolved outcome. It (1) builds this run's interaction
    training row from the persisted discipline vectors + the chosen option +
    the honest predicted P(1위) + the realized outcome grade, (2) appends it to
    the library's interaction_cases.jsonl (append-only, idempotent by run), and
    (3) re-learns J over ALL accumulated cases via the evidence-gated
    learn_coupling, recording a J_updates.jsonl entry when the matrix changes.

    Honest by construction: it reuses the same gate (min real cases + held-out
    Brier improvement), so J stays 0 until enough real outcomes justify a
    coefficient. Returns a status dict; never raises the run down — a missing
    input (e.g. an old run without discipline_vectors.json) is reported, not
    fatal.
    """
    library_dir, run_dir = Path(library_dir), Path(run_dir)
    dv_path = run_dir / "discipline_vectors.json"
    dec_path = run_dir / "decision.json"
    res_path = run_dir / "results.json"
    if not (dv_path.exists() and dec_path.exists() and res_path.exists()):
        return {"learned": False, "reason": "run missing vectors/decision/results"}
    dv = json.loads(dv_path.read_text())
    dec = json.loads(dec_path.read_text())
    res = json.loads(res_path.read_text())
    from . import outcomes
    y = outcomes.outcome_grade(run_dir)
    if y is None:
        return {"learned": False, "reason": "no resolved outcome grade"}

    chosen = dec.get("chosen_option") or dec.get("selected_option_id")
    options, disciplines = dv["options"], dv["disciplines"]
    if chosen not in options:
        return {"learned": False, "reason": f"chosen {chosen!r} not in options"}
    v_chosen = dv["V"][options.index(chosen)]
    # add_pred = the engine's HONEST (correlated) predicted P(1위) of the chosen
    # option — the same number the forecaster/snapshot now use.
    mc = res["monte_carlo"]
    p = mc.get("correlated", {}).get("p_rank1", mc["p_rank1"])
    add_pred = float(p[options.index(chosen)])

    row = {"case_id": run_dir.name, "disciplines": disciplines,
           "v_chosen": v_chosen, "add_pred": add_pred, "y": float(y)}

    # Append idempotently (one row per run).
    cases_path = library_dir / "interaction_cases.jsonl"
    existing = []
    if cases_path.exists():
        for line in cases_path.read_text().splitlines():
            if line.strip():
                c = json.loads(line)
                if c.get("case_id") != row["case_id"]:
                    existing.append(c)
    all_cases = existing + [row]
    atomic_write_text(cases_path,
        "\n".join(json.dumps(c, ensure_ascii=False) for c in all_cases) + "\n")

    # Canonical discipline space = union across all accumulated cases; project
    # each case's v_chosen into it (absent discipline → 0) so runs that selected
    # different disciplines can be learned together.
    canon = sorted({d for c in all_cases for d in c["disciplines"]})
    idx = {d: i for i, d in enumerate(canon)}
    proj = []
    for c in all_cases:
        vec = [0.0] * len(canon)
        for d, val in zip(c["disciplines"], c["v_chosen"]):
            vec[idx[d]] = float(val)
        proj.append({"v_chosen": vec, "add_pred": c["add_pred"], "y": c["y"],
                     "case_id": c["case_id"]})

    before = load_coupling(library_dir, canon)
    result = learn_coupling(proj, canon, min_cases=min_cases)
    after = np.asarray(result["J"], dtype=float)
    changed = not np.allclose(before, after)
    if changed:
        support = {f"{lp['d1']}|{lp['d2']}": lp.get("support_case_ids", [])
                   for lp in result.get("learned_pairs", [])}
        record_coupling_update(library_dir, before, after, support,
                               {"reason": "outcome", "n_cases": len(all_cases),
                                "disciplines": canon}, stamp=stamp)
    return {"learned": bool(np.any(after != 0.0)), "changed": changed,
            "n_cases": len(all_cases), "n_disciplines": len(canon),
            "learned_pairs": result.get("learned_pairs", [])}


def load_coupling(data_dir, disciplines: list[str]) -> np.ndarray:
    """Return the latest learned J from the ledger, or zero_coupling if none.

    Never invents: a missing ledger, an empty ledger, or a stored matrix whose
    shape does not match `disciplines` all yield all-zeros.
    """
    D = len(disciplines)
    ledger = Path(data_dir) / "J_updates.jsonl"
    if not ledger.exists():
        return zero_coupling(D)
    last = None
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if line.strip():
            last = line
    if last is None:
        return zero_coupling(D)
    try:
        after = np.asarray(json.loads(last)["after_J"], dtype=float)
    except (KeyError, ValueError, json.JSONDecodeError):
        return zero_coupling(D)
    if after.shape != (D, D):
        return zero_coupling(D)
    return after
