#!/usr/bin/env python3
"""Power analysis for ARK-42's `beats_baseline` test — how many cases are enough?

WHAT THIS IS
------------
A Monte Carlo characterisation of the MEASURING INSTRUMENT, not of ARK-42's
skill. It answers one operational question: how many low-leakage backtest cases
must exist before `clean_calibration.beats_baseline` carries information?

It does that by generating synthetic (predicted_p, observed_y) samples with a
KNOWN amount of skill built in, and feeding them through the ENGINE'S OWN
scorers — :func:`ark42.backtest.naive_baseline_brier`, :func:`ark42.backtest._pool`
and :func:`ark42.backtest._brier` are imported, never reimplemented — combined
exactly the way :func:`ark42.backtest.backtest` combines them::

    pooled     = _pool(results)                              # micro sample
    mean_brier = _brier(pooled)
    naive      = naive_baseline_brier([y for _p, y in pooled])
    beats      = mean_brier is not None and naive is not None and mean_brier < naive

Nothing under ``ark42/`` is modified or monkeypatched.

SCORING SPACE (fixed 2026-07-29)
--------------------------------
One observation = one (option, prediction) forecast key. The event is "did this
arm materialise" (true=1 / false=0). Engine Brier and the naive baseline are
both means over the SAME pooled observation sample; the baseline is a constant
predictor at that sample's own base rate.

CASE SHAPES (from ark42.backtest.resolve_verdicts)
--------------------------------------------------
* ``winning_option_id`` (WIN shape): the winner's keys score y=1, every other
  option's keys score y=0. -> ``n_options`` observations, exactly one of them 1.
  Measured on real data: ~3 observations per case.
* ``failed_option_id`` (FAIL shape): only the failed arm's keys score y=0; every
  other option is ``unresolved`` and EXCLUDED from the sample. -> 1 observation,
  always 0. Measured: ~1 observation per case.

Consequence, and it is the point of section 5: the win/fail MIX mechanically
determines the pooled base rate. With ``k`` predictions per option and
``m`` options, a win-fraction ``w`` gives::

    base_rate = w / (w * m + (1 - w))          (k cancels)

so all-win/3-options -> 1/3, the measured pilot mix (9 win / 7 fail) -> 0.26,
all-fail -> 0.0. And the baseline Brier of a constant-at-base-rate predictor is
exactly ``r*(1-r)``, which goes to 0 as r goes to 0.

PREDICTOR MODELS
----------------
SKILLED(s, sigma, m): genuinely discriminative but imperfect.
    p = clip( 0.5 + (2y-1)*s/2 + N(0, sigma) )   then miscalibration
    p = clip( 0.5 + (p - 0.5) * m )              (m>1 overconfident, m<1 shrunk)
    s=0 is indistinguishable from noise; s=1 is a perfect separator.

NULL variants (zero information about WHICH arm materialises):
    null_const_baserate  p = a fixed constant at the assumed pooled base rate
    null_const_half      p = 0.5 everywhere
    null_noise_half      p = clip(0.5 + N(0, 0.15)), independent of y
    null_uniform         p ~ U(0, 1), independent of y
    null_case_random     one U(0,1) draw per CASE, applied to all its keys
    null_structural      p = 1 / n_options for every key of that case
                         — knows the SHAPE of the problem, knows nothing about
                         which arm wins. This is the dangerous one.

The false-positive rate of the nulls is the number that matters most.

USAGE
-----
    python3 tools/power_analysis.py               # full run (3000+ reps)
    python3 tools/power_analysis.py --quick       # fast smoke run
    python3 tools/power_analysis.py --reps 10000  # override replicate count
    python3 tools/power_analysis.py --json out.json

DETERMINISM
-----------
Every random draw comes from a numpy Generator descended from ``--seed``
(default 20260729) via :class:`numpy.random.SeedSequence` spawn keys, so a given
(seed, section, config) always produces the identical stream regardless of
execution order or how many other sections run.
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from pathlib import Path

import numpy as np

# Import the REAL scorers from the engine. Nothing here reimplements Brier.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ark42.backtest import _brier, _pool, naive_baseline_brier  # noqa: E402

# --------------------------------------------------------------------------
# Defaults measured from the 17-case pilot conversion.
# --------------------------------------------------------------------------
PILOT_WIN_FRACTION = 9 / 16      # 9 win-shaped, 7 fail-shaped (16 shape-bearing)
PILOT_BASE_RATE = 0.257          # 9 ones / 35 pooled observations
FIXTURE_BASE_RATE = 0.50         # the fixture set
DEFAULT_N_OPTIONS = 3            # ~3 observations per win-shaped case
DEFAULT_PREDS_PER_OPTION = 1

N_CASES_GRID = (1, 2, 3, 5, 8, 10, 15, 20, 30, 50)
SKILL_GRID = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9)
MIX_GRID = (0.0, 0.25, 0.4, PILOT_WIN_FRACTION, 0.75, 0.9, 1.0)
NULL_NAMES = ("null_const_baserate", "null_const_half", "null_noise_half",
              "null_uniform", "null_case_random", "null_structural")

EPS = 1e-6


# --------------------------------------------------------------------------
# 1. Case generator
# --------------------------------------------------------------------------
def gen_case(rng: np.random.Generator, win_fraction: float, n_options,
             preds_per_option: int) -> tuple[np.ndarray, int]:
    """Generate one case's observation y-vector.

    Returns ``(y, n_options)``. WIN shape emits ``n_options * preds_per_option``
    observations with ``preds_per_option`` ones (the winning arm's keys). FAIL
    shape emits only the failed arm's ``preds_per_option`` observations, all
    zero — every other option is `unresolved` and excluded by resolve_verdicts.

    ``n_options`` may be an int (every case has the same arm count) or a
    sequence, in which case each case draws its arm count uniformly from it.
    A VARYING arm count is what makes `null_structural` non-constant.
    """
    m = int(n_options) if np.isscalar(n_options) else int(rng.choice(n_options))
    if rng.random() < win_fraction:
        y = np.zeros(m * preds_per_option)
        winner = rng.integers(m)
        y[winner * preds_per_option:(winner + 1) * preds_per_option] = 1.0
        return y, m
    return np.zeros(preds_per_option), m


def gen_pool(rng: np.random.Generator, n_cases: int, win_fraction: float,
             n_options, preds_per_option: int) -> list[tuple[np.ndarray, int]]:
    return [gen_case(rng, win_fraction, n_options, preds_per_option)
            for _ in range(n_cases)]


# --------------------------------------------------------------------------
# 2. Predictor models
# --------------------------------------------------------------------------
def predict_skilled(rng, y, n_opt, skill: float, sigma: float, miscal: float):
    """Discriminative but imperfect. skill=0 -> pure noise, skill=1 -> perfect."""
    p = 0.5 + (2.0 * y - 1.0) * (skill / 2.0)
    if sigma > 0:
        p = p + rng.normal(0.0, sigma, size=y.shape)
    if miscal != 1.0:
        p = 0.5 + (p - 0.5) * miscal
    return np.clip(p, EPS, 1.0 - EPS)


def predict_null(rng, y, n_opt, variant: str, assumed_base_rate: float):
    """Zero information about WHICH arm materialises."""
    if variant == "null_const_baserate":
        return np.full(y.shape, assumed_base_rate)
    if variant == "null_const_half":
        return np.full(y.shape, 0.5)
    if variant == "null_noise_half":
        return np.clip(0.5 + rng.normal(0.0, 0.15, size=y.shape), EPS, 1 - EPS)
    if variant == "null_uniform":
        return np.clip(rng.random(y.shape), EPS, 1 - EPS)
    if variant == "null_case_random":
        # ONE random constant per case, applied to every key. Varies across
        # cases but carries no within-case discrimination. Contrast control for
        # null_structural: shows that varying across cases is not by itself
        # enough — the per-case constant has to TRACK the within-case rate.
        return np.full(y.shape, float(rng.random()))
    if variant == "null_structural":
        # knows there are n_opt arms and one of them wins; knows nothing about
        # which. Non-constant across cases whenever n_options varies, and its
        # per-case constant equals the within-case base rate exactly.
        return np.full(y.shape, 1.0 / n_opt)
    raise ValueError(f"unknown null variant {variant!r}")


# --------------------------------------------------------------------------
# 3. Score one replicate exactly the way backtest() does
# --------------------------------------------------------------------------
def score_replicate(pairs_per_case: list[list[tuple[float, float]]]) -> dict:
    """Run the engine's own aggregation over per-case (p, y) samples.

    Mirrors ark42.backtest.backtest()'s clean_calibration block verbatim:
    pool the per-case samples, take the micro Brier, and compare against the
    naive baseline computed on the identical observation vector.
    """
    results = [{"scored_pairs": pc} for pc in pairs_per_case]
    pooled = _pool(results)                                    # REAL
    mean_brier = _brier(pooled)                                # REAL
    ys = [y for _p, y in pooled]
    naive = naive_baseline_brier(ys)                           # REAL
    beats = (mean_brier is not None and naive is not None and mean_brier < naive)
    return {
        "n_observations": len(pooled),
        "mean_brier": mean_brier,
        "naive_baseline_brier": naive,
        "base_rate": (sum(ys) / len(ys)) if ys else None,
        "beats_baseline": beats,
    }


def run_cell(seed_seq: np.random.SeedSequence, reps: int, n_cases: int,
             win_fraction: float, n_options: int, preds_per_option: int,
             predictor, assumed_base_rate: float) -> dict:
    """`reps` Monte Carlo replicates of one (n_cases, model) configuration."""
    rng = np.random.default_rng(seed_seq)
    n_beat = 0
    briers, naives, rates, nobs, margins = [], [], [], [], []
    for _ in range(reps):
        cases = gen_pool(rng, n_cases, win_fraction, n_options, preds_per_option)
        per_case = []
        for y, n_opt in cases:
            p = predictor(rng, y, n_opt)
            per_case.append([(float(pi), float(yi)) for pi, yi in zip(p, y)])
        out = score_replicate(per_case)
        n_beat += bool(out["beats_baseline"])
        nobs.append(out["n_observations"])
        if out["mean_brier"] is not None:
            briers.append(out["mean_brier"])
            naives.append(out["naive_baseline_brier"])
            rates.append(out["base_rate"])
            # margin > 0 is exactly the `beats_baseline` condition; keeping the
            # distribution lets us derive a significance threshold (section 6).
            margins.append(out["naive_baseline_brier"] - out["mean_brier"])
    return {
        "reps": reps,
        "p_beats": n_beat / reps,
        "mean_brier": float(np.mean(briers)) if briers else None,
        "mean_naive": float(np.mean(naives)) if naives else None,
        "mean_base_rate": float(np.mean(rates)) if rates else None,
        "mean_n_obs": float(np.mean(nobs)) if nobs else None,
        "margins": np.asarray(margins, dtype=float),
    }


def wilson_ci(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion — honest error bars on p_beats."""
    if n == 0:
        return (0.0, 1.0)
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    h = z * ((ph * (1 - ph) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


# --------------------------------------------------------------------------
# 4. Sections
# --------------------------------------------------------------------------
def _spawn(root: np.random.SeedSequence, *tag) -> np.random.SeedSequence:
    """Deterministic child stream keyed by a tuple, order-independent.

    Uses zlib.crc32, NOT the builtin hash(): CPython randomises string hashing
    per process (PYTHONHASHSEED), which would make runs irreproducible.
    """
    key = tuple(zlib.crc32(str(t).encode("utf-8")) & 0x7FFFFFFF for t in tag)
    return np.random.SeedSequence(entropy=root.entropy, spawn_key=key)


def section_power(root, reps, n_grid, skill_grid, cfg) -> dict:
    """P(beats_baseline) for the SKILLED model across (n_cases, skill)."""
    out = {}
    for s in skill_grid:
        row = {}
        for n in n_grid:
            def pred(rng, y, n_opt, s=s):
                return predict_skilled(rng, y, n_opt, s, cfg["sigma"], cfg["miscal"])
            cell = run_cell(_spawn(root, "power", s, n), reps, n,
                            cfg["win_fraction"], cfg["n_options"],
                            cfg["preds_per_option"], pred, cfg["assumed_base_rate"])
            cell["ci"] = wilson_ci(cell["p_beats"] * reps, reps)
            row[n] = cell
        out[s] = row
    return out


def section_null(root, reps, n_grid, cfg) -> dict:
    """False-positive rate for every null variant across n_cases."""
    out = {}
    for name in NULL_NAMES:
        row = {}
        for n in n_grid:
            def pred(rng, y, n_opt, name=name):
                return predict_null(rng, y, n_opt, name, cfg["assumed_base_rate"])
            cell = run_cell(_spawn(root, "null", name, n), reps, n,
                            cfg["win_fraction"], cfg["n_options"],
                            cfg["preds_per_option"], pred, cfg["assumed_base_rate"])
            cell["ci"] = wilson_ci(cell["p_beats"] * reps, reps)
            row[n] = cell
        out[name] = row
    return out


def section_mix(root, reps, n_grid, mix_grid, cfg, skill: float) -> dict:
    """Sweep the win/fail mix from all-fail to all-win, skilled AND worst null."""
    out = {}
    for w in mix_grid:
        row = {}
        for n in n_grid:
            def pred_s(rng, y, n_opt):
                return predict_skilled(rng, y, n_opt, skill, cfg["sigma"], cfg["miscal"])

            def pred_n(rng, y, n_opt):
                return predict_null(rng, y, n_opt, "null_uniform", cfg["assumed_base_rate"])
            sk = run_cell(_spawn(root, "mix_s", w, n), reps, n, w,
                          cfg["n_options"], cfg["preds_per_option"], pred_s,
                          cfg["assumed_base_rate"])
            nl = run_cell(_spawn(root, "mix_n", w, n), reps, n, w,
                          cfg["n_options"], cfg["preds_per_option"], pred_n,
                          cfg["assumed_base_rate"])
            row[n] = {"skilled": sk, "null_uniform": nl}
        out[w] = row
    return out


def section_baserate(root, reps, n_grid, cfg, skill: float) -> dict:
    """Base rate as a free parameter, varied via n_options (all-win pools).

    n_options=2 -> base rate 0.50 (the fixture set); 3 -> 0.333; 4 -> 0.25;
    6 -> 0.167. Reported alongside the mix sweep so the two routes to a given
    base rate can be compared.
    """
    out = {}
    for m in (2, 3, 4, 6):
        row = {}
        for n in n_grid:
            def pred_s(rng, y, n_opt):
                return predict_skilled(rng, y, n_opt, skill, cfg["sigma"], cfg["miscal"])

            def pred_n(rng, y, n_opt):
                return predict_null(rng, y, n_opt, "null_uniform", cfg["assumed_base_rate"])
            sk = run_cell(_spawn(root, "br_s", m, n), reps, n, 1.0, m,
                          cfg["preds_per_option"], pred_s, cfg["assumed_base_rate"])
            nl = run_cell(_spawn(root, "br_n", m, n), reps, n, 1.0, m,
                          cfg["preds_per_option"], pred_n, cfg["assumed_base_rate"])
            row[n] = {"skilled": sk, "null_uniform": nl, "base_rate": 1.0 / m}
        out[m] = row
    return out


def section_threshold(root, reps, n_grid, cfg, skill_grid) -> dict:
    """Derive the significance threshold a bare inequality is missing.

    `beats_baseline` is `margin > 0` where ``margin = naive - mean_brier``. If a
    zero-skill predictor produces ``margin > 0`` more than 5% of the time, the
    bare inequality is not a 5%-level test. The fix is a threshold delta(n) such
    that ``P_null(margin > delta) <= 0.05``. We take delta(n) as the 95th
    percentile of the margin distribution under the WORST (most permissive) null
    at that n, then report the skilled model's power against that threshold.
    """
    out = {}
    for n in n_grid:
        null_margins, worst_name, worst_q = {}, None, -1e9
        for name in NULL_NAMES:
            def pred(rng, y, n_opt, name=name):
                return predict_null(rng, y, n_opt, name, cfg["assumed_base_rate"])
            c = run_cell(_spawn(root, "thr_null", name, n), reps, n,
                         cfg["win_fraction"], cfg["n_options"],
                         cfg["preds_per_option"], pred, cfg["assumed_base_rate"])
            q = float(np.quantile(c["margins"], 0.95)) if len(c["margins"]) else 0.0
            null_margins[name] = {"q95": q, "p_beats_bare": c["p_beats"]}
            if q > worst_q:
                worst_q, worst_name = q, name
        delta = max(0.0, worst_q)
        powers = {}
        for s in skill_grid:
            def pred_s(rng, y, n_opt, s=s):
                return predict_skilled(rng, y, n_opt, s, cfg["sigma"], cfg["miscal"])
            c = run_cell(_spawn(root, "thr_sk", s, n), reps, n,
                         cfg["win_fraction"], cfg["n_options"],
                         cfg["preds_per_option"], pred_s, cfg["assumed_base_rate"])
            m = c["margins"]
            powers[s] = {
                "bare": c["p_beats"],
                "thresholded": float(np.mean(m > delta)) if len(m) else 0.0,
            }
        out[n] = {"delta": delta, "worst_null": worst_name,
                  "nulls": null_margins, "skilled": powers}
    return out


def section_miscal(root, reps, n_grid, cfg, skill: float) -> dict:
    """Hold skill fixed, sweep the miscalibration multiplier."""
    out = {}
    for m in (0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        row = {}
        for n in n_grid:
            def pred(rng, y, n_opt, m=m):
                return predict_skilled(rng, y, n_opt, skill, cfg["sigma"], m)
            row[n] = run_cell(_spawn(root, "miscal", m, n), reps, n,
                              cfg["win_fraction"], cfg["n_options"],
                              cfg["preds_per_option"], pred, cfg["assumed_base_rate"])
        out[m] = row
    return out


# --------------------------------------------------------------------------
# 5. Summarisers
# --------------------------------------------------------------------------
def smallest_n_at_power(row: dict, target: float = 0.80):
    """Smallest n in the grid where p_beats >= target AND stays >= target after."""
    ns = sorted(row)
    for i, n in enumerate(ns):
        if all(row[k]["p_beats"] >= target for k in ns[i:]):
            return n
    return None


def settles_below(row: dict, target: float = 0.05):
    """Smallest n from which the rate is < target for that n and all larger n."""
    ns = sorted(row)
    for i, n in enumerate(ns):
        if all(row[k]["p_beats"] < target for k in ns[i:]):
            return n
    return None


def fmt_table(header: list[str], rows: list[list[str]]) -> str:
    line = "| " + " | ".join(header) + " |"
    sep = "|" + "|".join("---" for _ in header) + "|"
    body = ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join([line, sep] + body)


def report(res: dict, cfg: dict, reps: int, n_grid) -> str:
    L = []
    A = L.append
    A("=" * 78)
    A("ARK-42 backtest power analysis")
    A(f"reps={reps}  n_grid={list(n_grid)}  cfg={cfg}")
    A("=" * 78)

    A("\n[1] SKILLED power: P(beats_baseline) by skill x n_cases")
    hdr = ["skill"] + [str(n) for n in n_grid] + ["min n @0.80"]
    rows = []
    for s, row in sorted(res["power"].items()):
        cells = [f"{row[n]['p_beats']:.3f}" for n in n_grid]
        rows.append([f"{s:.2f}"] + cells + [str(smallest_n_at_power(row))])
    A(fmt_table(hdr, rows))

    A("\n[2] NULL false-positive rate by variant x n_cases")
    hdr = ["null"] + [str(n) for n in n_grid] + ["settles <0.05"]
    rows = []
    for name, row in res["null"].items():
        cells = [f"{row[n]['p_beats']:.3f}" for n in n_grid]
        rows.append([name] + cells + [str(settles_below(row))])
    A(fmt_table(hdr, rows))

    A("\n[3] MIX sweep (win fraction) — skilled s=%.2f / null_uniform"
      % res["_mix_skill"])
    hdr = ["win frac", "base rate"] + [f"S n={n}" for n in n_grid] \
        + [f"N n={n}" for n in n_grid]
    rows = []
    for w, row in sorted(res["mix"].items()):
        br = row[n_grid[-1]]["skilled"]["mean_base_rate"]
        rows.append([f"{w:.3f}", f"{br:.3f}"]
                    + [f"{row[n]['skilled']['p_beats']:.3f}" for n in n_grid]
                    + [f"{row[n]['null_uniform']['p_beats']:.3f}" for n in n_grid])
    A(fmt_table(hdr, rows))

    A("\n[4] BASE RATE via n_options (all-win pools)")
    hdr = ["n_options", "base rate"] + [f"S n={n}" for n in n_grid] \
        + [f"N n={n}" for n in n_grid]
    rows = []
    for m, row in sorted(res["baserate"].items()):
        rows.append([str(m), f"{1.0/m:.3f}"]
                    + [f"{row[n]['skilled']['p_beats']:.3f}" for n in n_grid]
                    + [f"{row[n]['null_uniform']['p_beats']:.3f}" for n in n_grid])
    A(fmt_table(hdr, rows))

    A("\n[5] MISCALIBRATION sweep at skill=%.2f" % res["_mix_skill"])
    hdr = ["miscal"] + [str(n) for n in n_grid]
    rows = [[f"{m:.2f}"] + [f"{row[n]['p_beats']:.3f}" for n in n_grid]
            for m, row in sorted(res["miscal"].items())]
    A(fmt_table(hdr, rows))

    A("\n[6] SIGNIFICANCE THRESHOLD delta(n) = 95th pct of worst-null margin")
    thr = res["threshold"]
    skills = sorted(next(iter(thr.values()))["skilled"])
    hdr = ["n", "delta", "worst null", "bare FPR"] \
        + [f"s={s:.2f} bare/thr" for s in skills]
    rows = []
    for n in n_grid:
        t = thr[n]
        bare = t["nulls"][t["worst_null"]]["p_beats_bare"]
        rows.append([str(n), f"{t['delta']:.4f}", str(t["worst_null"]),
                     f"{bare:.3f}"]
                    + [f"{t['skilled'][s]['bare']:.3f}/{t['skilled'][s]['thresholded']:.3f}"
                       for s in skills])
    A(fmt_table(hdr, rows))
    A("")
    return "\n".join(L)


def strip_margins(obj):
    """Recursively drop the raw margin arrays before JSON serialisation."""
    if isinstance(obj, dict):
        return {k: strip_margins(v) for k, v in obj.items() if k != "margins"}
    if isinstance(obj, (list, tuple)):
        return [strip_margins(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true",
                    help="fast smoke run (400 reps, reduced grids)")
    ap.add_argument("--reps", type=int, default=None, help="replicates per cell")
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--sigma", type=float, default=0.10,
                    help="predictor noise sd (default 0.10)")
    ap.add_argument("--miscal", type=float, default=1.0,
                    help="miscalibration multiplier for the main power sweep")
    ap.add_argument("--win-fraction", type=float, default=PILOT_WIN_FRACTION)
    ap.add_argument("--n-options", type=int, default=DEFAULT_N_OPTIONS)
    ap.add_argument("--n-options-mix", type=str, default=None,
                    help="comma-separated arm counts drawn per case, e.g. '2,3,4,5'. "
                         "Overrides --n-options. A VARYING arm count is what lets "
                         "null_structural (p=1/n_options) beat the pooled baseline.")
    ap.add_argument("--preds-per-option", type=int, default=DEFAULT_PREDS_PER_OPTION)
    ap.add_argument("--assumed-base-rate", type=float, default=PILOT_BASE_RATE,
                    help="constant the null_const_baserate predictor emits")
    ap.add_argument("--mix-skill", type=float, default=0.40,
                    help="skill level held fixed in the mix/base-rate/miscal sweeps")
    ap.add_argument("--json", type=str, default=None, help="write raw results JSON")
    a = ap.parse_args(argv)

    reps = a.reps if a.reps else (400 if a.quick else 3000)
    n_grid = (1, 2, 3, 5, 10, 20, 50) if a.quick else N_CASES_GRID
    skill_grid = (0.0, 0.1, 0.2, 0.4, 0.9) if a.quick else SKILL_GRID
    mix_grid = (0.0, 0.5, PILOT_WIN_FRACTION, 1.0) if a.quick else MIX_GRID

    n_opts = ([int(x) for x in a.n_options_mix.split(",")]
              if a.n_options_mix else a.n_options)
    cfg = {"sigma": a.sigma, "miscal": a.miscal, "win_fraction": a.win_fraction,
           "n_options": n_opts, "preds_per_option": a.preds_per_option,
           "assumed_base_rate": a.assumed_base_rate}
    root = np.random.SeedSequence(a.seed)

    res = {
        "power": section_power(root, reps, n_grid, skill_grid, cfg),
        "null": section_null(root, reps, n_grid, cfg),
        "mix": section_mix(root, reps, n_grid, mix_grid, cfg, a.mix_skill),
        "baserate": section_baserate(root, reps, n_grid, cfg, a.mix_skill),
        "miscal": section_miscal(root, reps, n_grid, cfg, a.mix_skill),
        "threshold": section_threshold(root, reps, n_grid, cfg, skill_grid),
        "_mix_skill": a.mix_skill,
    }
    print(report(res, cfg, reps, n_grid))

    if a.json:
        payload = {"config": cfg, "reps": reps, "seed": a.seed,
                   "n_grid": list(n_grid), "results": strip_margins(res)}
        Path(a.json).write_text(
            json.dumps(payload, indent=1, default=str), encoding="utf-8")
        print(f"wrote {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
