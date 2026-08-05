"""Outcome recording, linkage, and prediction-vs-reality scoring.

Invariants enforced here:
  1. An outcome must link to an existing run, decision, and prediction
     snapshot (missing links → error, nothing written).
  2. The prediction snapshot is verified untouched before every outcome
     write; a mutated results/forecasts file aborts the recording.
  3. Scores are computed ONLY from the immutable snapshot's probabilities
     plus the validated outcome record — never from live files — and are
     persisted with their inputs (score.json) so they are recomputable.

Evidence honesty: verdicts default to evidence_state="self_reported".
This engine has no external evidence integration; that limitation is
recorded on every verdict rather than hidden.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from .lineage import SCORING_RULE, canonical_hash, make_id, utcnow
from .snapshots import (ensure_decision_lineage, freeze_prediction_snapshot,
                        verify_prediction_snapshot)

VERDICT_VALUE = {"true": 1.0, "false": 0.0, "partial": 0.5}
VALID_VERDICTS = ("true", "false", "partial", "unresolved")


def load_forecasts(run_dir: Path) -> list[dict]:
    fc = json.loads((Path(run_dir) / "forecasts.json").read_text())
    out = []
    for f in fc["forecasts"]:
        for i, p in enumerate(f["predictions"]):
            out.append({"option_id": f["option_id"], "prediction_index": i, **p})
    return out


def checkins(run_dir: Path) -> dict:
    """Which predictions are due for a verdict, given elapsed real time.
    Only the CHOSEN option's predictions (plus baseline) are actionable."""
    run_dir = Path(run_dir)
    dec_file = run_dir / "decision.json"
    chosen = (json.loads(dec_file.read_text())["chosen_option"]
              if dec_file.exists() else None)
    # Clock anchor: the prediction snapshot's created_at is written once and
    # never rewritten. problem.json's mtime was the old anchor and the
    # pipeline rewrites that file, so re-running an id reset every due date.
    started = None
    snap = run_dir / "prediction_snapshot.json"
    if snap.exists():
        try:
            started = time.mktime(time.strptime(
                json.loads(snap.read_text())["created_at"], "%Y-%m-%dT%H:%M:%SZ"))
            started -= time.timezone          # created_at is UTC
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            started = None
    if started is None:                        # legacy runs: best available
        started = (run_dir / "problem.json").stat().st_mtime
    months_elapsed = (time.time() - started) / (30.44 * 86400)
    recorded = _recorded_keys(run_dir)
    items = []
    for p in load_forecasts(run_dir):
        if chosen and p["option_id"] not in (chosen, "NO_INTERVENTION"):
            continue
        key = f"{p['option_id']}#{p['prediction_index']}"
        items.append({**p, "key": key,
                      "due": months_elapsed >= p["horizon_months"],
                      "already_recorded": key in recorded})
    return {"chosen_option": chosen, "months_elapsed": round(months_elapsed, 2),
            "items": items}


def record(run_dir: Path, verdicts: list[dict], recorded_by: str = "user") -> dict:
    """Append a batch of verdicts. Each: {key, verdict, actual?, observed_at?,
    evidence_reference?}. Returns the full outcome doc (with ids)."""
    run_dir = Path(run_dir)
    if not (run_dir / "problem.json").exists():
        raise FileNotFoundError(f"unknown run: {run_dir}")
    if not (run_dir / "decision.json").exists():
        raise ValueError("no decision recorded for this run; an outcome must "
                         "link to a decision (invariant 1)")
    decision = ensure_decision_lineage(run_dir)
    freeze_prediction_snapshot(run_dir)          # legacy runs: freeze on entry
    snap = verify_prediction_snapshot(run_dir)   # abort if sources mutated

    valid = {f"{p['option_id']}#{p['prediction_index']}": p
             for p in load_forecasts(run_dir)}
    errors = [v.get("key") for v in verdicts
              if v.get("key") not in valid or v.get("verdict") not in VALID_VERDICTS]
    if errors:
        raise ValueError(f"invalid verdicts: {errors}")

    recorded_at = utcnow()
    run_id = snap["run_id"]
    batch_verdicts = [{
        "key": v["key"],
        "verdict": v["verdict"],
        "observed_value": v.get("actual", v.get("observed_value", "")),
        "observed_at": v.get("observed_at") or recorded_at,
        "evidence_state": v.get("evidence_state", "self_reported"),
        "evidence_reference": v.get("evidence_reference", ""),
    } for v in verdicts]
    outcome_id = make_id("out", run_id, recorded_at, canonical_hash(batch_verdicts))

    path = run_dir / "outcome.json"
    doc = (json.loads(path.read_text()) if path.exists()
           else {"run_id": run_id, "decision_id": decision["decision_id"],
                 "prediction_snapshot_id": snap["prediction_snapshot_id"],
                 "checkins": [], "verdicts": {}})
    doc.setdefault("run_id", run_id)
    doc.setdefault("decision_id", decision["decision_id"])
    doc.setdefault("prediction_snapshot_id", snap["prediction_snapshot_id"])
    doc["checkins"].append({"outcome_id": outcome_id, "recorded_at": recorded_at,
                            "ts": recorded_at,          # legacy alias
                            "recorded_by": recorded_by,
                            "verdicts": batch_verdicts})
    for v in batch_verdicts:                     # latest verdict per key wins
        doc["verdicts"][v["key"]] = {
            "verdict": v["verdict"],
            "actual": v["observed_value"],       # legacy alias (report reads it)
            "observed_value": v["observed_value"],
            "observed_at": v["observed_at"],
            "evidence_state": v["evidence_state"],
            "evidence_reference": v["evidence_reference"],
            "outcome_id": outcome_id,
        }
    doc["latest_outcome_id"] = outcome_id
    doc["summary"] = _summarize(doc["verdicts"], snap)
    from .lineage import atomic_write_text
    atomic_write_text(path, json.dumps(doc, ensure_ascii=False, indent=2))

    _write_score(run_dir, snap, doc)
    return doc


def _summarize(verdict_map: dict, snap: dict) -> dict:
    """Brier over resolved verdicts, probabilities taken from the IMMUTABLE
    snapshot — not from live forecasts.json."""
    probs = snap["forecast_probabilities"]
    resolved, sq = [], []
    for key, v in verdict_map.items():
        if v["verdict"] == "unresolved" or key not in probs:
            continue
        o = VERDICT_VALUE[v["verdict"]]
        sq.append((probs[key] - o) ** 2)
        resolved.append(key)
    brier = sum(sq) / len(sq) if sq else None
    return {"n_predictions_tracked": len(probs), "n_resolved": len(resolved),
            "brier": brier,
            "brier_vs_chance": (0.25 - brier) if brier is not None else None}


def _write_score(run_dir: Path, snap: dict, doc: dict) -> dict:
    probs = snap["forecast_probabilities"]
    inputs = [{"key": k, "p": probs[k], "o": VERDICT_VALUE[v["verdict"]]}
              for k, v in doc["verdicts"].items()
              if v["verdict"] != "unresolved" and k in probs]
    score = {
        "score_id": make_id("score", snap["prediction_snapshot_id"],
                            doc["latest_outcome_id"]),
        "prediction_snapshot_id": snap["prediction_snapshot_id"],
        "outcome_id": doc["latest_outcome_id"],
        "scoring_rule": SCORING_RULE,
        "calculation_version": 1,
        "inputs": inputs,
        "calculated_score": doc["summary"]["brier"],
        "brier_vs_chance": doc["summary"]["brier_vs_chance"],
        "calculated_at": utcnow(),
    }
    from .lineage import atomic_write_text
    atomic_write_text(Path(run_dir) / "score.json",
                      json.dumps(score, ensure_ascii=False, indent=2))
    return score


def _recorded_keys(run_dir: Path) -> set[str]:
    path = Path(run_dir) / "outcome.json"
    if not path.exists():
        return set()
    return set(json.loads(path.read_text()).get("verdicts", {}).keys())


def outcome_grade(run_dir: Path) -> float | None:
    """0..1 grade of how well the CHOSEN option's predictions came true."""
    run_dir = Path(run_dir)
    if not (run_dir / "outcome.json").exists() or not (run_dir / "decision.json").exists():
        return None
    chosen = json.loads((run_dir / "decision.json").read_text())["chosen_option"]
    doc = json.loads((run_dir / "outcome.json").read_text())
    vals = [VERDICT_VALUE[v["verdict"]]
            for key, v in doc["verdicts"].items()
            if key.startswith(chosen + "#") and v["verdict"] != "unresolved"]
    return sum(vals) / len(vals) if vals else None
