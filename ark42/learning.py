"""The learning loop: resolved cases become reusable, AUDITABLE knowledge.

Storage layout under <library_dir>/:
  cases.jsonl                resolved cases (precedents), synthetic-flagged
  reliability.json           PRODUCTION discipline weights (versioned store)
  reliability_sandbox.json   fixture/synthetic weights — never read by
                             production runs (invariant 10)
  weight_updates.jsonl       append-only ledger: every update (and every
                             no-change) with before/after states + hashes

Versioned store format:
  {"_meta": {"version": N, "hash": H, "applied_runs": {run_id: update_id},
             "updated_at": ts},
   "disciplines": {d: {"alpha": a, "beta": b, "n_cases": n}}}

Invariants enforced here:
  3. Re-submitting an outcome for an already-applied run records an
     explicit no-change entry and does NOT touch the weights.
  4. Updates run only off a validated outcome (score.json must exist).
  5. Every update links run_id → outcome_id → score_id and stores
     before/after snapshots with hashes.
  10. synthetic=True writes ONLY to the sandbox store.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from .lineage import (UPDATE_RULE, atomic_write_text, canonical_hash,
                      make_id, utcnow)
from .outcomes import outcome_grade

PRIOR_ALPHA = PRIOR_BETA = 5.0   # strong-ish prior: reliability moves slowly


# ------------------------------------------------------- versioned store
def _store_path(library_dir: Path, synthetic: bool) -> Path:
    return Path(library_dir) / ("reliability_sandbox.json" if synthetic
                                else "reliability.json")


def _load_store(path: Path) -> dict:
    if not path.exists():
        return {"_meta": {"version": 0, "hash": canonical_hash({}),
                          "applied_runs": {}, "updated_at": None},
                "disciplines": {}}
    doc = json.loads(path.read_text())
    if "_meta" not in doc:                        # migrate legacy flat format
        doc = {"_meta": {"version": 1, "hash": canonical_hash(doc),
                         "applied_runs": {}, "updated_at": None,
                         "migrated_from": "flat-v0"},
               "disciplines": doc}
    return doc


def _save_store(path: Path, doc: dict) -> None:
    doc["_meta"]["hash"] = canonical_hash(doc["disciplines"])
    doc["_meta"]["updated_at"] = utcnow()
    atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=2))


def reliability_state(library_dir: Path | None,
                      channel: str = "production") -> dict:
    """Multipliers + provenance for the next run. Explicit default when no
    learning exists (invariant 9)."""
    default = {"multipliers": {}, "weight_version": "default-v0",
               "weight_hash": canonical_hash({}), "source": "default",
               "latest_update_id": None, "channel": channel}
    if library_dir is None:
        return default
    path = _store_path(library_dir, synthetic=(channel == "sandbox"))
    if not path.exists():
        return default
    doc = _load_store(path)
    mult = {d: 0.5 + (e["alpha"] / (e["alpha"] + e["beta"]))
            for d, e in doc["disciplines"].items()}
    latest = (list(doc["_meta"]["applied_runs"].values())[-1]
              if doc["_meta"]["applied_runs"] else None)
    return {"multipliers": mult,
            "weight_version": doc["_meta"]["version"],
            "weight_hash": doc["_meta"]["hash"],
            "source": str(path), "latest_update_id": latest,
            "channel": channel}


def reliability_multipliers(library_dir: Path | None,
                            channel: str = "production") -> dict[str, float]:
    """Back-compat helper: multipliers only."""
    return reliability_state(library_dir, channel)["multipliers"]


# ---------------------------------------------------------------- library
def update_from_run(run_dir: Path, library_dir: Path,
                    synthetic: bool = False) -> dict:
    """Fold one resolved run into the library. Idempotent per run_id:
    a second call records a no-change ledger entry and changes nothing."""
    run_dir, library_dir = Path(run_dir), Path(library_dir)
    library_dir.mkdir(parents=True, exist_ok=True)
    grade = outcome_grade(run_dir)
    if grade is None:
        raise ValueError("no resolved outcomes for chosen option; nothing to learn from")
    if not (run_dir / "score.json").exists():
        raise ValueError("no score record; weight update requires a validated "
                         "outcome + score (invariant 4)")

    problem = json.loads((run_dir / "problem.json").read_text())
    decision = json.loads((run_dir / "decision.json").read_text())
    outcome = json.loads((run_dir / "outcome.json").read_text())
    score = json.loads((run_dir / "score.json").read_text())
    run_id = problem["problem_id"]

    store_path = _store_path(library_dir, synthetic)
    store = _load_store(store_path)
    ledger = library_dir / "weight_updates.jsonl"

    # invariant 3: no double learning from the same run
    if run_id in store["_meta"]["applied_runs"]:
        entry = {"update_id": make_id("wupd", run_id, "nochange", utcnow()),
                 "type": "no_change", "ts": utcnow(), "synthetic": synthetic,
                 "run_id": run_id, "outcome_id": outcome["latest_outcome_id"],
                 "reason": "run already applied; duplicate submission ignored",
                 "version_unchanged": store["_meta"]["version"],
                 "prior_update_id": store["_meta"]["applied_runs"][run_id]}
        _append(ledger, entry)
        return {"skipped": True, "reason": entry["reason"],
                "update_id": entry["prior_update_id"],
                "weight_version": store["_meta"]["version"],
                "reliability_changes": {}, "case": None}

    # ---- case record (precedent) -------------------------------------
    case = {
        "case_id": run_id, "recorded_at": utcnow(), "synthetic": synthetic,
        "statement": problem["statement"],
        "options": [{"option_id": o["option_id"], "title": o["title"]}
                    for o in problem["options"]],
        "chosen": decision["chosen_option"],
        "decision_id": decision.get("decision_id"),
        "outcome_id": outcome["latest_outcome_id"],
        "score_id": score["score_id"],
        "outcome_grade": round(grade, 3),
        "verdicts": {k: v["verdict"] for k, v in outcome["verdicts"].items()},
        "engine_p_rank1_of_chosen": _p_rank1_of(run_dir, decision["chosen_option"]),
        "brier": score["calculated_score"],
    }
    cases = [c for c in read_cases(library_dir) if c["case_id"] != run_id] + [case]
    atomic_write_text(library_dir / "cases.jsonl",
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n")

    # ---- weight update -----------------------------------------------
    before = json.loads(json.dumps(store["disciplines"]))
    hash_before, version_before = store["_meta"]["hash"], store["_meta"]["version"]
    contributions = _contributions(run_dir, decision["chosen_option"], grade)
    changes = {}
    for d, alignment in contributions.items():
        e = store["disciplines"].get(
            d, {"alpha": PRIOR_ALPHA, "beta": PRIOR_BETA, "n_cases": 0})
        prev_rel = e["alpha"] / (e["alpha"] + e["beta"])
        e["alpha"] += alignment
        e["beta"] += 1.0 - alignment
        e["n_cases"] += 1
        store["disciplines"][d] = e
        new_rel = e["alpha"] / (e["alpha"] + e["beta"])
        changes[d] = {"alignment": round(alignment, 4),
                      "reliability": round(new_rel, 4),
                      "delta": round(new_rel - prev_rel, 5)}

    update_id = make_id("wupd", run_id, outcome["latest_outcome_id"],
                        version_before + 1)
    store["_meta"]["version"] = version_before + 1
    store["_meta"]["applied_runs"][run_id] = update_id
    _save_store(store_path, store)

    entry = {
        "update_id": update_id, "type": "update", "ts": utcnow(),
        "synthetic": synthetic, "store": str(store_path),
        "run_id": run_id, "outcome_id": outcome["latest_outcome_id"],
        "score_id": score["score_id"], "outcome_grade": round(grade, 4),
        "update_rule": UPDATE_RULE,
        "learning_rate": "alpha += alignment; beta += (1 - alignment)",
        "prior": {"alpha": PRIOR_ALPHA, "beta": PRIOR_BETA},
        "clipping": "multiplier = 0.5 + reliability, bounded [0.5, 1.5]",
        "contributions": {d: c["alignment"] for d, c in changes.items()},
        "before": before, "after": store["disciplines"],
        "version_before": version_before,
        "version_after": store["_meta"]["version"],
        "hash_before": hash_before, "hash_after": store["_meta"]["hash"],
    }
    _append(ledger, entry)

    # Also learn the interaction coupling J from this real outcome — the
    # production writer that was missing (J was structurally inert). Best-effort
    # and non-synthetic only: a failure here must never take down the reliability
    # update or the run. J stays 0 until enough real outcomes pass its gate.
    coupling = None
    if not synthetic:
        try:
            from .interaction_learning import learn_from_outcome
            coupling = learn_from_outcome(library_dir, run_dir, stamp=utcnow())
        except Exception as e:                       # never load-bearing
            coupling = {"learned": False, "reason": f"error: {e!r}"}

    return {"skipped": False, "case": case, "update_id": update_id,
            "weight_version": store["_meta"]["version"],
            "reliability_changes": changes, "coupling": coupling}


def _contributions(run_dir: Path, chosen: str, grade: float) -> dict[str, float]:
    """Per-discipline alignment with reality, in [0,1]: 1 when the
    discipline's relative ranking of the chosen option matches the outcome
    grade. Deliberately rank-based and inspectable; its arbitrariness is a
    documented risk, not a hidden one."""
    out = {}
    for f in sorted((Path(run_dir) / "analyses").glob("*.json")):
        a = json.loads(f.read_text())
        per_option: dict[str, list[float]] = {}
        for s in a["assessments"]:
            per_option.setdefault(s["option_id"], []).append(float(s["score_mean"]))
        if chosen not in per_option or len(per_option) < 2:
            continue
        means = {o: float(np.mean(v)) for o, v in per_option.items()}
        ranked = sorted(means.values())
        # Average rank for ties: a discipline that rated the chosen option
        # equal-best must not be scored as ranking it last (bisect fix).
        import bisect
        lo = bisect.bisect_left(ranked, means[chosen])
        hi = bisect.bisect_right(ranked, means[chosen]) - 1
        pct = ((lo + hi) / 2) / (len(ranked) - 1) if len(ranked) > 1 else 0.5
        out[a["discipline"]] = 1.0 - abs(pct - grade)
    return out


def read_cases(library_dir: Path, include_synthetic: bool = True) -> list[dict]:
    path = Path(library_dir) / "cases.jsonl"
    if not path.exists():
        return []
    cases = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return cases if include_synthetic else [c for c in cases if not c.get("synthetic")]


def _p_rank1_of(run_dir: Path, option_id: str) -> float | None:
    try:
        results = json.loads((Path(run_dir) / "results.json").read_text())
        i = results["options"].index(option_id)
        mc = results["monte_carlo"]
        # Record the honest (correlated) P(1위), consistent with the forecaster
        # and the snapshot; fall back to independent only if no correlated block.
        p = mc.get("correlated", {}).get("p_rank1", mc["p_rank1"])
        return round(p[i], 4)
    except (ValueError, KeyError, FileNotFoundError):
        return None


def _append(path: Path, entry: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ------------------------------------------------------- precedent retrieval
def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[가-힣a-zA-Z0-9]{2,}", text.lower())}


def find_precedents(statement: str, library_dir: Path | None, k: int = 3,
                    include_synthetic: bool = False,
                    owner_filter=None) -> list[dict]:
    """Retrieve similar resolved cases to inject as evidence.

    SECURITY: `owner_filter` is a callable(case_id) -> bool. Precedents are
    injected into the analyst PROMPT, so without a filter one tenant's
    problem statement leaks into another tenant's analysis. The /library API
    filters by owner, but that filtering did not cover this path. Callers
    that serve multiple users MUST pass owner_filter.
    """
    if library_dir is None:
        return []
    cases = read_cases(library_dir, include_synthetic=include_synthetic)
    if owner_filter is not None:
        cases = [c for c in cases if owner_filter(c.get("case_id"))]
    if not cases:
        return []
    q = _tokens(statement)
    scored = []
    for c in cases:
        t = _tokens(c["statement"])
        jac = len(q & t) / len(q | t) if q | t else 0.0
        if jac > 0.05:
            scored.append((jac, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:k]]


def precedents_block(precedents: list[dict]) -> str:
    if not precedents:
        return ""
    lines = ["", "선례 — 현실 결과로 판정이 끝난 과거 사례 (증거로 참고하되, "
             "현재 문제와의 맥락 차이를 반드시 명시하라):"]
    for c in precedents:
        titles = {o["option_id"]: o["title"] for o in c["options"]}
        # Externally-mined public cases have no engine prediction of their own
        # (they were never run through the pipeline). Show a provenance tag and
        # omit the engine-P field rather than inventing one.
        p1 = c.get("engine_p_rank1_of_chosen")
        if c.get("provenance") == "external_public":
            tail = ("(외부 공개자료 — 엔진이 돌린 사례가 아님, 맥락·누출 주의; "
                    f"신뢰가중 {c.get('learning_weight', 0)})")
        else:
            tail = f"당시 엔진 P(1위) {p1}" if p1 is not None else "(엔진 P 기록 없음)"
        lines.append(
            f"- [{c['case_id']}] 문제: {c['statement'][:120]} / "
            f"선택: {titles.get(c['chosen'], c['chosen'])} / "
            f"결과 등급 {c['outcome_grade']:.2f} (1=예측대로 성공, 0=실패) / "
            + tail)
    return "\n".join(lines)
