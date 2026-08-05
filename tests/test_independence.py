from __future__ import annotations
"""Tests for the independence measurement — the product's central claim.

The audit found 42 tests, 29 of which covered implementation detail while
the things that would end the company had none. This file covers one of
those: the measurement that now sets the reported probability.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ark42.independence import MIN_SHARED_CELLS, RHO_CAP, measure  # noqa: E402
from ark42.montecarlo import simulate                              # noqa: E402
from ark42.tensor import ScoreTensor                               # noqa: E402


def tensor(D=6, O=4, C=8, mode="indep", seed=1, conf=0.8):
    rng = np.random.default_rng(seed)
    mean = np.zeros((O, C, D))
    base = rng.random((O, C))
    for d in range(D):
        if mode == "ident":
            mean[:, :, d] = base
        elif mode == "indep":
            mean[:, :, d] = rng.random((O, C))
        elif mode == "anti":
            mean[:, :, d] = base if d % 2 == 0 else 1 - base
        elif mode == "flat":
            mean[:, :, d] = 0.5
    return ScoreTensor([f"O{i}" for i in range(O)], [f"C{i}" for i in range(C)],
                       [f"d{i}" for i in range(D)], mean,
                       np.full((O, C, D), 0.1), np.full((O, C, D), conf),
                       np.ones(D))


def test_identical_disciplines_collapse_to_one_perspective():
    m = measure(tensor(D=8, mode="ident"))
    assert m["reliable"] is True
    assert m["mean_r"] == pytest.approx(1.0, abs=1e-6)
    assert m["n_effective"] == pytest.approx(1.0, abs=0.01)
    assert m["rho_used"] == RHO_CAP          # clamped, and says so
    assert m["rho_was_capped"] is True
    assert m["unanimous_top"] is True
    # Single provider (no providers passed) → the verdict names the cause.
    assert "같은 모델 하나" in m["verdict"]


def test_independent_disciplines_keep_full_credit():
    m = measure(tensor(D=6, mode="indep"))
    assert abs(m["mean_r"]) < 0.3
    assert m["n_effective"] == pytest.approx(6.0, abs=0.6)
    assert m["rho_used"] < 0.3


def test_negative_correlation_is_treated_as_full_independence():
    """Disciplines that actively disagree are the BEST case; n_eff must not
    exceed D and rho must not go negative (the copula requires rho >= 0)."""
    m = measure(tensor(D=6, mode="anti"))
    assert m["mean_r"] < 0
    assert m["n_effective"] == 6.0
    assert m["rho_used"] == 0.0


def test_single_discipline_is_unmeasurable_not_zero():
    m = measure(tensor(D=1))
    assert m["reliable"] is False
    assert m["mean_r"] is None          # never invent a number
    assert m["rho_used"] == 0.0
    assert "1개 이하" in m["reason"]


def test_too_few_shared_cells_is_unmeasurable():
    t = tensor(D=4, O=1, C=2)           # 2 cells < MIN_SHARED_CELLS
    m = measure(t)
    assert m["reliable"] is False
    assert m["mean_r"] is None
    assert str(MIN_SHARED_CELLS) in m["reason"]


def test_zero_variance_discipline_does_not_crash():
    m = measure(tensor(D=4, mode="flat"))
    assert m["reliable"] is False        # correlation undefined, not 0
    assert m["mean_r"] is None


def test_partial_coverage_uses_only_commonly_filled_cells():
    t = tensor(D=3, mode="indep")
    t.conf[0, :, 0] = 0.0               # discipline 0 skipped option 0
    m = measure(t)
    assert m["reliable"] is True
    assert m["shared_cells"] == (t.mean.shape[0] - 1) * t.mean.shape[1]


def test_higher_correlation_lowers_reported_p_rank1():
    """The whole point: measured dependence must reduce confidence, and the
    means must not move (rho changes spread, not location)."""
    t = tensor(D=6, mode="indep", seed=3)
    # make one option clearly better so there IS a leader to be confident about
    t.mean[0] += 0.25
    lo = simulate(t, n=4000, rho=0.0, seed=7)
    hi = simulate(t, n=4000, rho=0.9, seed=7)
    assert hi["p_rank1"].max() < lo["p_rank1"].max()
    assert hi["utility_std"].mean() > lo["utility_std"].mean()
    np.testing.assert_allclose(hi["expected_utility"], lo["expected_utility"],
                               atol=0.02)


def test_measurement_is_persisted_and_drives_the_simulation(tmp_path):
    """End-to-end through the pipeline: independence.json exists, and the
    rho actually used in results equals the measured one."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from fixture_data import execute_run, write_recorded
    rec = write_recorded(tmp_path / "rec")
    run, results = execute_run("indep-e2e", tmp_path / "runs", rec,
                               tmp_path / "lib", channel="sandbox")
    indep = json.loads((run.dir / "independence.json").read_text())
    assert indep["n_disciplines"] == 2
    assert results["independence"]["rho_used"] == indep["rho_used"]
    corr = results["monte_carlo"]["correlated"]
    assert corr["rho"] == indep["rho_used"]
    assert corr["rho_source"] in ("measured", "unmeasurable_fallback_independent")
    # If the fixture's two disciplines were unmeasurable, the label must say so
    # rather than implying a measurement happened.
    if not indep["reliable"]:
        assert corr["rho_source"] == "unmeasurable_fallback_independent"


# ---- multi-provider (독립 다중 주체) ----------------------------------------

def test_within_vs_cross_provider_split():
    """Same-provider pairs and cross-provider pairs are measured separately;
    the split is what lets the product claim (or disclaim) real independence."""
    t = tensor(D=4, mode="ident")           # all identical scores
    # But two providers: cross-provider pairs must still be flagged, even when
    # the scores happen to match (identity here just makes the number stable).
    providers = ["m:A", "m:A", "m:B", "m:B"]
    m = measure(t, providers=providers)
    assert m["n_providers"] == 2
    assert m["single_provider"] is False
    assert m["within_provider_mean_r"] is not None
    assert m["cross_provider_mean_r"] is not None
    # every pair is tagged
    assert all("same_provider" in p for p in m["pairs"])
    n_cross = sum(1 for p in m["pairs"] if not p["same_provider"])
    assert n_cross == 4          # 2×2 cross pairs among 4 disciplines


def test_single_provider_verdict_names_the_cause():
    t = tensor(D=6, mode="ident")
    m = measure(t)                          # no providers → single
    assert m["single_provider"] is True
    assert "같은 모델 하나" in m["verdict"]


def test_diverse_providers_raise_effective_perspectives():
    """The whole point of direction 1: genuinely different judgements across
    providers must lower overall correlation and raise n_eff vs one voice."""
    D = 6
    # panel A: one coherent view; panel B: a different coherent view
    ta = tensor(D=D, mode="ident", seed=1)
    tb = tensor(D=D, mode="ident", seed=999)
    # build a mixed tensor: half the disciplines take A's scores, half take B's
    mixed = ta.mean.copy()
    for d in range(D):
        if d % 2 == 1:
            mixed[:, :, d] = tb.mean[:, :, d]
    from ark42.tensor import ScoreTensor
    tm = ScoreTensor(ta.options, ta.criteria, ta.disciplines, mixed,
                     ta.std, ta.conf, ta.relevance)
    providers = ["m:A" if d % 2 == 0 else "m:B" for d in range(D)]
    single = measure(tm)                    # ignore providers
    multi = measure(tm, providers=providers)
    # cross-provider correlation is lower than within-provider
    assert multi["cross_provider_mean_r"] < multi["within_provider_mean_r"]
    # and the diversity is visible; single-provider view cannot see it
    assert multi["n_providers"] == 2 and single["n_providers"] == 1


def test_panel_backend_routes_and_records_providers():
    from ark42.backends import PanelBackend
    from ark42.llm import RecordedBackend
    import tempfile, pathlib, json as _json
    d = pathlib.Path(tempfile.mkdtemp())
    (d / "a").mkdir(); (d / "b").mkdir()
    for panel in ("a", "b"):
        (d / panel / "analysis_economics.json").write_text('{"x":1}')
        (d / panel / "analysis_law_regulation.json").write_text('{"x":1}')
    pb = PanelBackend([RecordedBackend(d / "a", provider_id="m:A"),
                       RecordedBackend(d / "b", provider_id="m:B")])
    assert pb.multi is True
    pb.complete("analysis_economics", "s", "p")
    pb.complete("analysis_law_regulation", "s", "p")
    pm = pb.provider_map()
    assert set(pm.values()) <= {"m:A", "m:B"}
    assert set(pm) == {"economics", "law_regulation"}
    # routing is stable: same key → same provider on repeat
    before = pb.provider_of["analysis_economics"]
    pb.complete("analysis_economics", "s", "p")
    assert pb.provider_of["analysis_economics"] == before
