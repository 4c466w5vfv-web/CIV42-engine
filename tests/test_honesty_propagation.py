from __future__ import annotations
"""Regression tests for two situation-report fixes:

  ① The honest (correlated) probability must reach the forecaster AND the
     immutable snapshot — not just the HTML report. Reality Brier-scores the
     snapshot, so it must carry the number the engine actually believes, not
     the independent (rho=0) figure its own instrumentation flags overconfident.
  ② The interaction coupling J must have a PRODUCTION writer wired to real
     outcomes, so it is not structurally inert.
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np                                    # noqa: E402
from fixture_data import write_recorded, PROBLEM      # noqa: E402
from ark42.llm import RecordedBackend                 # noqa: E402
from ark42.pipeline import Run, make_problem          # noqa: E402


def _run(rid, tmp):
    rec = write_recorded(tmp / "rec")
    p = make_problem(rid, PROBLEM["statement"], PROBLEM["context"], [])
    r = Run(p, RecordedBackend(rec), tmp / "runs",
            library_dir=tmp / "lib", library_channel="sandbox")
    r.frame_options()
    res = r.quantify(r.analyze(r.select_disciplines()))
    r.forecast(res)
    return r, res


# ---- ① honest probability propagation --------------------------------------

def test_snapshot_freezes_correlated_not_independent():
    tmp = Path(tempfile.mkdtemp())
    r, res = _run("h1", tmp)
    mc = res["monte_carlo"]
    snap = json.loads((r.dir / "prediction_snapshot.json").read_text())
    corr = mc["correlated"]["p_rank1"]
    for i, o in enumerate(res["options"]):
        po = snap["per_option"][o]
        # authoritative headline == correlated (honest)
        assert abs(po["p_rank1"] - corr[i]) < 1e-9
        # independent kept only for transparency, never as the headline
        assert abs(po["p_rank1_independent"] - mc["p_rank1"][i]) < 1e-9
        assert po["prob_basis"] in ("measured", "unmeasurable_fallback_independent")


def test_correlated_block_carries_the_honest_interval():
    tmp = Path(tempfile.mkdtemp())
    _, res = _run("h2", tmp)
    corr = res["monte_carlo"]["correlated"]
    assert "percentiles" in corr and "expected_utility" in corr
    assert "p5" in corr["percentiles"] and "p95" in corr["percentiles"]


def test_precedent_records_honest_p_rank1():
    from ark42.learning import _p_rank1_of
    tmp = Path(tempfile.mkdtemp())
    r, res = _run("h3", tmp)
    lead = res["options"][int(np.argmax(res["monte_carlo"]["p_rank1"]))]
    got = _p_rank1_of(r.dir, lead)
    corr = res["monte_carlo"]["correlated"]["p_rank1"][res["options"].index(lead)]
    assert abs(got - round(corr, 4)) < 1e-6            # honest, not independent


# ---- ② interaction J has a production writer -------------------------------

def test_quantify_persists_discipline_vectors():
    tmp = Path(tempfile.mkdtemp())
    r, _ = _run("h4", tmp)
    dv = json.loads((r.dir / "discipline_vectors.json").read_text())
    assert dv["options"] and dv["disciplines"] and dv["V"]
    assert len(dv["V"]) == len(dv["options"])
    assert len(dv["V"][0]) == len(dv["disciplines"])


def test_production_writer_flips_J_from_outcomes():
    """The core of finding ②: with enough real outcomes carrying a planted
    interaction, the PRODUCTION path (learn_from_outcome) must move J off zero
    and persist it where load_coupling reads it."""
    from ark42.interaction_learning import learn_from_outcome, load_coupling
    from ark42.lineage import utcnow
    import ark42.outcomes as oc

    lib = Path(tempfile.mkdtemp()) / "lib"; lib.mkdir(parents=True)
    disc = ["economics", "law_regulation", "finance"]
    rng = np.random.default_rng(1)
    oc.outcome_grade = lambda rd: json.loads(
        (Path(rd) / "outcome.json").read_text()).get("grade")

    def make_run(i):
        rd = Path(tempfile.mkdtemp()) / f"r{i}"; rd.mkdir(parents=True)
        v = rng.random(3).tolist()
        add = float(rng.uniform(0.3, 0.6))
        y = float(np.clip(add + 0.35 * v[0] * v[1] + rng.normal(0, 0.01), 0, 1))
        (rd / "discipline_vectors.json").write_text(json.dumps(
            {"options": ["A", "B"], "disciplines": disc, "V": [v, [0, 0, 0]]}))
        (rd / "decision.json").write_text(json.dumps({"chosen_option": "A"}))
        (rd / "results.json").write_text(json.dumps(
            {"options": ["A", "B"],
             "monte_carlo": {"p_rank1": [add, 0.1],
                             "correlated": {"p_rank1": [add, 0.1]}}}))
        (rd / "outcome.json").write_text(json.dumps({"grade": y}))
        return rd

    st = None
    for i in range(12):
        st = learn_from_outcome(lib, make_run(i), stamp=utcnow())
    assert st["learned"] and st["changed"]
    canon = sorted(disc)
    J = load_coupling(lib, canon)
    ij = (canon.index("economics"), canon.index("law_regulation"))
    assert abs(J[ij]) > 0.05                            # recovered the planted pair
    assert (lib / "J_updates.jsonl").exists()

    # under-supported: 3 cases → gate keeps J at 0
    lib2 = Path(tempfile.mkdtemp()) / "lib"; lib2.mkdir(parents=True)
    for i in range(3):
        learn_from_outcome(lib2, make_run(100 + i), stamp=utcnow())
    assert np.allclose(load_coupling(lib2, canon), 0.0)


def test_writer_is_idempotent_per_run():
    from ark42.interaction_learning import learn_from_outcome
    from ark42.lineage import utcnow
    import ark42.outcomes as oc
    oc.outcome_grade = lambda rd: 0.7
    lib = Path(tempfile.mkdtemp()) / "lib"; lib.mkdir(parents=True)
    rd = Path(tempfile.mkdtemp()) / "runX"; rd.mkdir(parents=True)
    (rd / "discipline_vectors.json").write_text(json.dumps(
        {"options": ["A"], "disciplines": ["economics", "finance"], "V": [[0.5, 0.4]]}))
    (rd / "decision.json").write_text(json.dumps({"chosen_option": "A"}))
    (rd / "results.json").write_text(json.dumps(
        {"options": ["A"], "monte_carlo": {"p_rank1": [0.6],
                                           "correlated": {"p_rank1": [0.6]}}}))
    (rd / "outcome.json").write_text("{}")
    learn_from_outcome(lib, rd, stamp=utcnow())
    learn_from_outcome(lib, rd, stamp=utcnow())          # same run twice
    rows = (lib / "interaction_cases.jsonl").read_text().strip().splitlines()
    assert len(rows) == 1                                # not double-counted
