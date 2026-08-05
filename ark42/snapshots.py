"""Prediction snapshots and decision records with full lineage.

freeze_prediction_snapshot: capture the engine's prediction state into an
immutable record BEFORE any reality is observed. Idempotent — a second call
returns the existing snapshot untouched. verify_prediction_snapshot detects
any post-freeze mutation of the source files and refuses further processing.

record_decision: persist the human choice with decision_id, timestamps,
the option set that was on the table, the run snapshot hash, and the
prediction_snapshot_id it was based on. A decision is locked once an
outcome exists.
"""
from __future__ import annotations

import json
from pathlib import Path

from .lineage import (ENGINE_VERSION, atomic_write_text, MC_SEED, SENS_SEED, canonical_hash,
                      file_hash, make_id, utcnow)


class SnapshotViolation(Exception):
    """Raised when a source file changed after its snapshot was frozen."""


def freeze_prediction_snapshot(run_dir: Path) -> dict:
    run_dir = Path(run_dir)
    path = run_dir / "prediction_snapshot.json"
    if path.exists():
        return json.loads(path.read_text())          # idempotent, never overwrite

    results = json.loads((run_dir / "results.json").read_text())
    forecasts_path = run_dir / "forecasts.json"
    forecasts = (json.loads(forecasts_path.read_text())
                 if forecasts_path.exists() else {"forecasts": []})
    run_id = json.loads((run_dir / "problem.json").read_text())["problem_id"]

    source_hashes = {"results.json": file_hash(run_dir / "results.json")}
    if forecasts_path.exists():
        source_hashes["forecasts.json"] = file_hash(forecasts_path)

    # Model provenance — WHICH models produced this prediction. Without it, a
    # later switch of LLM vendor silently breaks Brier/calibration time-series
    # comparability (you can no longer tell whether a score shift is the world
    # changing or the model changing). Recorded INSIDE the snapshot so it is
    # covered by snapshot_hash and cannot be backdated. Read from the run's
    # provider_map.json (written by the pipeline); absent → recorded honestly
    # as unknown, never invented.
    pmap_path = run_dir / "provider_map.json"
    _pm = (json.loads(pmap_path.read_text()) if pmap_path.exists() else {})
    providers = _pm.get("providers", {}) if isinstance(_pm, dict) else {}
    distinct = sorted({v for v in providers.values() if v})
    model_provenance = {
        "providers": providers,                       # discipline -> provider_id
        "distinct_providers": distinct,
        "n_providers": len(distinct),
        "multi_provider": len(distinct) > 1,
        "recorded": bool(providers),                  # False → older/keyless run
        "provenance_hash": canonical_hash(
            {"providers": providers, "engine_version": ENGINE_VERSION}),
    }

    mc = results["monte_carlo"]
    # HONESTY: the snapshot is the immutable record reality is scored against,
    # so its authoritative p_rank1 must be the CORRELATED (honest, widened)
    # figure the engine actually believes — not the independent (rho=0) number
    # its own independence measurement flags as overconfident. The independent
    # value is kept alongside for transparency, never as the headline. When
    # independence was unmeasurable the two are equal (rho=0), so this is safe.
    corr = mc.get("correlated", {})
    p_rank1 = corr.get("p_rank1", mc["p_rank1"])
    p_beats = corr.get("p_beats_baseline", mc["p_beats_baseline"])
    exp_u = corr.get("expected_utility", mc["expected_utility"])
    per_option = {
        o: {"expected_utility": exp_u[i],
            "p_rank1": p_rank1[i],                    # authoritative = correlated
            "p_rank1_independent": mc["p_rank1"][i],  # transparency only
            "p_beats_baseline": p_beats[i],
            "prob_basis": corr.get("rho_source", "independent_no_correlated_block")}
        for i, o in enumerate(results["options"])
    }
    # A dict comprehension over the forecasts LIST silently lets a second block
    # for the same option_id overwrite the first block's keys. The audit file
    # (forecasts.json) would then show 'we said 0.05' while the snapshot — the
    # thing reality actually scores — holds 0.99, and the hash check passes
    # because the divergence is created AT freeze time, not after it. Build the
    # map explicitly and refuse on collision. (Red-team finding F2, 2026-07-29:
    # measured Brier 0.2275 -> 0.0019 on the same problem and the same truth.)
    forecast_probabilities: dict[str, float] = {}
    collisions = []
    for f in forecasts["forecasts"]:
        for i, p in enumerate(f["predictions"]):
            key = f"{f['option_id']}#{i}"
            if key in forecast_probabilities:
                collisions.append(key)
                continue
            forecast_probabilities[key] = float(p["probability"])
    if collisions:
        raise ValueError(
            "forecasts.json에 중복된 예측 키가 있습니다: "
            f"{sorted(set(collisions))}. 같은 option_id 블록이 두 번 나오면 "
            "채점되는 확률이 감사 파일과 달라집니다 — 스냅샷을 만들지 않습니다.")
    snap = {
        "prediction_snapshot_id": make_id("psnap", run_id,
                                          source_hashes["results.json"],
                                          source_hashes.get("forecasts.json", "-")),
        "run_id": run_id,
        "created_at": utcnow(),
        "engine_version": ENGINE_VERSION,
        "model_provenance": model_provenance,
        "seeds": {"monte_carlo": MC_SEED, "sensitivity": SENS_SEED},
        "n_draws": mc["n_draws"],
        "source_hashes": source_hashes,
        "weights_used": results.get("weights_used",
                                    {"weight_version": "default-v0"}),
        "criterion_scores": results["point_estimate"]["criterion_scores"],
        "per_option": per_option,
        "forecast_probabilities": forecast_probabilities,
        "snapshot_hash": None,   # filled below over the content
    }
    snap["snapshot_hash"] = canonical_hash({k: v for k, v in snap.items()
                                            if k != "snapshot_hash"})
    atomic_write_text(path, json.dumps(snap, ensure_ascii=False, indent=2))
    return snap


def verify_prediction_snapshot(run_dir: Path) -> dict:
    """Re-hash the source files; raise if anything mutated post-freeze."""
    run_dir = Path(run_dir)
    snap = json.loads((run_dir / "prediction_snapshot.json").read_text())
    for fname, recorded in snap["source_hashes"].items():
        current = file_hash(run_dir / fname)
        if current != recorded:
            raise SnapshotViolation(
                f"{fname} changed after prediction snapshot was frozen "
                f"(recorded {recorded[:12]}…, now {current[:12]}…)")
    body_hash = canonical_hash({k: v for k, v in snap.items()
                                if k != "snapshot_hash"})
    if body_hash != snap["snapshot_hash"]:
        raise SnapshotViolation("prediction_snapshot.json itself was edited")
    return snap


class DecisionExists(Exception):
    """A decision is already recorded for this run (pass replace=True to
    supersede it; the prior decision is preserved in decision_history.jsonl)."""


def record_decision(run_dir: Path, option_id: str, decided_by: str,
                    note: str = "", replace: bool = False) -> dict:
    run_dir = Path(run_dir)
    if (run_dir / "outcome.json").exists():
        raise ValueError("decision is locked: an outcome is already recorded")
    existing_path = run_dir / "decision.json"
    if existing_path.exists():
        existing = json.loads(existing_path.read_text())
        if not replace:
            raise DecisionExists(
                f"decision already recorded: "
                f"{existing.get('selected_option_id', existing.get('chosen_option'))} "
                f"({existing.get('decision_id', 'legacy')}). "
                "Pass replace=true to supersede it.")
        # Superseding is allowed but never silent: the prior decision is
        # appended to an immutable history so the lineage keeps both.
        with open(run_dir / "decision_history.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({**existing, "superseded_at": utcnow()},
                               ensure_ascii=False) + "\n")
    results = json.loads((run_dir / "results.json").read_text())
    if option_id not in results["options"]:
        raise ValueError(f"unknown option {option_id!r}")
    run_id = json.loads((run_dir / "problem.json").read_text())["problem_id"]
    snap = freeze_prediction_snapshot(run_dir)
    decided_at = utcnow()
    run_snapshot_hash = canonical_hash({
        "problem": file_hash(run_dir / "problem.json"),
        **snap["source_hashes"],
    })
    decision = {
        "decision_id": make_id("dec", run_id, option_id, decided_at),
        "run_id": run_id,
        "selected_option_id": option_id,
        "chosen_option": option_id,          # legacy alias (report reads this)
        "decision_status": "decided",
        "decided_by": decided_by,
        "decided_at": decided_at,
        "note": note,
        "run_snapshot_hash": run_snapshot_hash,
        "options_at_decision": results["options"],
        "prediction_snapshot_id": snap["prediction_snapshot_id"],
        "human_authority": {"engine_recommendation_is_not_binding": True},
        "engine_recommendation_is_not_binding": True,   # legacy alias
    }
    if (run_dir / "decision_history.jsonl").exists():
        decision["supersedes"] = [
            json.loads(l)["decision_id"]
            for l in (run_dir / "decision_history.jsonl").read_text().splitlines()
            if l.strip() and "decision_id" in json.loads(l)]
    atomic_write_text(run_dir / "decision.json", json.dumps(decision, ensure_ascii=False, indent=2))
    return decision


def ensure_decision_lineage(run_dir: Path) -> dict:
    """Non-destructive upgrade of a legacy decision.json: preserves the
    original choice verbatim, adds missing lineage fields."""
    run_dir = Path(run_dir)
    d = json.loads((run_dir / "decision.json").read_text())
    if "decision_id" in d:
        return d
    run_id = json.loads((run_dir / "problem.json").read_text())["problem_id"]
    snap = freeze_prediction_snapshot(run_dir)
    d.setdefault("selected_option_id", d["chosen_option"])
    d.setdefault("run_id", run_id)
    d.setdefault("decision_status", "decided")
    d.setdefault("decided_at", utcnow() + " (legacy: enriched later)")
    d.setdefault("prediction_snapshot_id", snap["prediction_snapshot_id"])
    d.setdefault("options_at_decision", list(snap["per_option"].keys()))
    d["decision_id"] = make_id("dec", run_id, d["chosen_option"], "legacy")
    d["legacy_enriched"] = True
    atomic_write_text(run_dir / "decision.json", json.dumps(d, ensure_ascii=False, indent=2))
    return d
