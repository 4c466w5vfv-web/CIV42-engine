"""Backtest driver: score the engine's EX-ANTE predictions against reality.

A backtest case carries a real decision problem together with its KNOWN
real-world outcome. We run each case through the ordinary pipeline (framing →
analysis → quantify → forecast), record a decision, then hand the case's real
outcome to the engine's EXISTING outcome/Brier machinery
(:mod:`ark42.outcomes`). Nothing about scoring is reimplemented here — Brier is
computed exactly once, server-side, from the immutable prediction snapshot.

Shared case JSON contract (produced by the ingest adapter)::

    {"case_id", "source",
     "problem": {"problem_id", "statement", "context",
                 "options": [{"option_id", "title", "description", "is_baseline"}]},
     "ground_truth": {"winning_option_id" OR "failed_option_id",
                      "outcome_scalar", "outcome_description",
                      "resolved_at", "basis"},
     "leakage": {...}, "provenance": {...}}

Honesty stance: the mapping from a real outcome to per-forecast verdicts is the
crux of an honest backtest, so :func:`resolve_verdicts` is deliberately small,
documented, and refuses to guess (unmappable keys become ``unresolved``). The
aggregate report ALWAYS splits results by leakage-risk bucket, and always
compares against :func:`naive_baseline_brier` — an engine that cannot beat the
base rate has learned nothing, and we never hide that.
"""
from __future__ import annotations

from pathlib import Path

from . import outcomes
from .ontology import DecisionOption, Problem


# --------------------------------------------------------------------------
# 1. The honest outcome → verdict mapping (the crux).
# --------------------------------------------------------------------------
def resolve_verdicts(run_dir: Path, ground_truth: dict) -> list[dict]:
    """Map a case's real-world outcome to per-forecast verdicts.

    The engine emits one forecast prediction per (option, metric); its key is
    ``'{option_id}#{prediction_index}'`` (see :func:`outcomes.load_forecasts`).
    A backtest learns at most ONE fact from reality, in one of two shapes:

    ``winning_option_id`` — a known winner:

    * every prediction attached to the winning option → ``'true'``
      (that arm materialised in the real world);
    * every prediction attached to any OTHER real option → ``'false'``
      (that arm did not materialise);

    ``failed_option_id`` — a known failure and no known winner (the common
    shape for a real historical case, where only the arm actually taken was
    ever observed, and it did not work out):

    * predictions on the failed option → ``'false'``;
    * predictions on every other option → ``'unresolved'``, because nobody
      took those arms and reality never reported on them. Crediting them with
      a win would be inventing a counterfactual.

    If neither id is found among the emitted forecasts, every key →
    ``'unresolved'`` — we refuse to guess. A `winning_option_id` that resolves
    takes precedence; `failed_option_id` is only consulted when it does not.

    The verdict strings are exactly :data:`outcomes.VALID_VERDICTS`, and the
    numeric outcome each maps to is :data:`outcomes.VERDICT_VALUE` (``true``→1,
    ``false``→0, ``partial``→0.5, ``unresolved`` excluded from Brier). No
    verdict token is invented here.

    HONESTY CAVEAT: collapsing a real, multi-arm outcome to a single winning
    option is an approximation. It cannot observe counterfactuals (what a
    non-chosen arm would have done), and it grades every non-winning prediction
    as fully falsified even when that arm may have partially succeeded in
    reality. Where that reduction is unsafe (genuinely simultaneous or
    partial-success outcomes) the case should carry a higher leakage/uncertainty
    flag upstream, and the leakage-bucket split in :func:`backtest` keeps such
    cases visible separately.
    """
    ground_truth = ground_truth or {}
    winner = ground_truth.get("winning_option_id")
    failed = ground_truth.get("failed_option_id")
    actual = ground_truth.get("outcome_description", "")
    forecasts = outcomes.load_forecasts(run_dir)
    option_ids = {f["option_id"] for f in forecasts}
    mappable = winner in option_ids            # refuse to guess if unmapped
    failed_mappable = failed in option_ids and not mappable

    verdicts = []
    for f in forecasts:
        key = f"{f['option_id']}#{f['prediction_index']}"
        if mappable:
            verdict = "true" if f["option_id"] == winner else "false"
        elif failed_mappable:
            # Known-failure case: reality showed the taken arm did not work
            # out, and said NOTHING about the arms nobody took. Grading those
            # as winners would invent a counterfactual.
            verdict = "false" if f["option_id"] == failed else "unresolved"
        else:
            verdict = "unresolved"
        assert verdict in outcomes.VALID_VERDICTS   # never invent a token
        verdicts.append({"key": key, "verdict": verdict, "actual": actual})
    return verdicts


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _problem_from_case(case: dict) -> Problem:
    p = case["problem"]
    options = [DecisionOption(
        option_id=o["option_id"], title=o.get("title", o["option_id"]),
        description=o.get("description", ""),
        is_baseline=bool(o.get("is_baseline", False)),
    ) for o in p.get("options", [])]
    return Problem(problem_id=p["problem_id"], statement=p["statement"],
                   context=p.get("context", ""), options=options)


def _leakage_risk(case: dict) -> str:
    """Best-effort leakage-risk bucket label for honest reporting.

    The ingest adapter owns the `leakage` schema; we read the common field
    names and fall back to 'unknown' rather than pretending a case is clean.
    """
    lk = case.get("leakage") or {}
    for k in ("risk", "bucket", "leakage_risk", "level"):
        if lk.get(k):
            return str(lk[k])
    return "unknown"


def _scored_pairs(run_dir: Path, doc: dict, m: int) -> list[dict]:
    """The observations the engine is graded on — ONE PER (case, ARM).

    Each observation is ``{"option_id", "p", "sq", "y", "m", "n_preds"}``.
    ``sq`` is the arm's score: the MEAN OF SQUARED ERRORS over that arm's
    predictions, not the squared error of their mean. ``p`` is the mean
    probability, kept for reporting only. ``m`` is how many arms the decision
    ranged over, passed in from the PROBLEM.

    Three red-team findings shaped this (2026-07-29), and the third was caused
    by fixing the first two:

    * **F1 — the engine set its own sample.** Per-(option, prediction) scoring
      let the engine choose the observation count and hence the pooled base
      rate. Same beliefs, same truth: one prediction per option scored 0.2275
      vs baseline 0.1875 (fail); padding the winning arm to four predictions
      scored 0.1986 vs 0.2449 (pass). Truth is defined per ARM, so an arm is
      one observation and padding is inert.
    * **F1b — the engine also picked how many ARMS.** ``m`` used to be counted
      from the frozen snapshot's forecast keys, and nothing requires the
      forecaster to emit one block per option. Dropping from 4 arms to 2 moved
      the same beliefs from −0.0400 (fail) to +0.0450 (pass). ``m`` now comes
      from the problem, which is what the decision actually ranged over, and
      arms the forecaster skipped are reported as missing coverage rather than
      quietly shrinking the denominator.
    * **F14 — averaging forgave the spread.** The first version of this fix
      scored ``(mean p − y)²``, and ``(mean p − y)² = mean((p−y)²) − Var(p)``.
      An arm predicted at 0.00 and 1.00 averages to 0.50 and was scored 0.0644
      instead of 0.2015 — flipping the verdict, with 0.1371 of penalty simply
      forgiven, while forecasts.json still read "확신도 98%". Averaging the
      SQUARED ERRORS keeps one observation per arm (so F1 stays closed) and
      charges the engine for every claim it actually published.
    """
    from .snapshots import verify_prediction_snapshot
    probs = verify_prediction_snapshot(run_dir)["forecast_probabilities"]

    by_option: dict[str, dict] = {}
    for key, v in (doc.get("verdicts") or {}).items():
        if v["verdict"] == "unresolved" or key not in probs:
            continue
        oid = key.split("#", 1)[0]
        slot = by_option.setdefault(oid, {"option_id": oid, "ps": [], "ys": set()})
        slot["ps"].append(float(probs[key]))
        slot["ys"].add(float(outcomes.VERDICT_VALUE[v["verdict"]]))

    out = []
    for oid, slot in sorted(by_option.items()):
        if len(slot["ys"]) != 1:      # one arm cannot both materialise and not
            raise ValueError(f"conflicting verdicts for option {oid!r} in "
                             f"{run_dir}: {sorted(slot['ys'])}")
        y = next(iter(slot["ys"]))
        ps = slot["ps"]
        out.append({"option_id": oid,
                    "p": sum(ps) / len(ps),
                    "sq": sum((p - y) ** 2 for p in ps) / len(ps),
                    "y": y, "m": max(1, int(m)), "n_preds": len(ps)})
    return out


# --------------------------------------------------------------------------
# 2. Run one case end-to-end and score it against reality.
# --------------------------------------------------------------------------
def run_case(case: dict, runs_dir: Path, backend, library_dir: Path | None = None,
             learn: bool = False, defer_learning: bool = False) -> dict:
    """Execute one backtest case through the full pipeline and score it.

    When `learn=True` AND the case is LOW-leakage, the scored outcome is also
    folded into the learning loop (reliability weights + interaction J) via the
    same evidence-gated path a real user outcome takes — this is how a batch of
    historical known-outcome decisions bootstraps calibration WITHOUT any users.
    High/medium-leakage cases are NEVER learned from: a memorized outcome would
    teach the engine from its own memory, not from prediction. Learning requires
    a real `library_dir`.

    `backend` is injected, so a RecordedBackend (tests/offline) or a real
    key-gated LLM backend (production) travel the SAME code path — the LLM step
    is key-gated, the driver is uniform. Returns a result dict with the engine's
    Brier (computed by :mod:`ark42.outcomes`, never here), the base-rate gap
    (`brier_vs_chance`), the number of resolved predictions, the leakage bucket,
    the engine's ex-ante probability for the winning option, and the real
    outcome scalar.
    """
    from .pipeline import Run

    problem = _problem_from_case(case)
    # The pipeline's quantify() reads the interaction/reliability stores by path
    # and does not accept a None library_dir. When no library is supplied we
    # point it at an empty scratch dir: a missing/empty store yields DEFAULT
    # weights and a zero (no-op) coupling, so this is semantically identical to
    # "no learning" while keeping the code path uniform.
    lib = library_dir if library_dir is not None else Path(runs_dir) / "_backtest_default_lib"
    run = Run(problem, backend, runs_dir, library_dir=lib)

    # framing is a no-op when the case already supplies options; kept so the
    # code path is identical to a from-scratch run.
    run.frame_options()
    chosen = run.select_disciplines()
    analyses = run.analyze(chosen)
    results = run.quantify(analyses)
    run.forecast(results)                    # writes forecasts.json + freezes snapshot

    ground_truth = case.get("ground_truth") or {}
    winner = ground_truth.get("winning_option_id")

    # A decision is required before an outcome can be recorded (invariant 1).
    # We record the engine's own top-ranked option — its ex-ante recommendation
    # — which is what a real user would have been advised to pick. The decision
    # does not affect the Brier score (Brier spans all resolved forecast keys).
    # Use the HONEST (correlated) p_rank1 for the recommendation, consistent
    # with the forecaster/snapshot; fall back to independent if absent.
    mc = results["monte_carlo"]
    p_rank1 = mc.get("correlated", {}).get("p_rank1", mc["p_rank1"])
    top_idx = max(range(len(p_rank1)), key=lambda i: p_rank1[i])
    run.record_decision(results["options"][top_idx], decided_by="backtest",
                        note=f"engine top pick (backtest of {case.get('case_id')})")

    doc = outcomes.record(run.dir, resolve_verdicts(run.dir, ground_truth),
                          recorded_by="backtest")
    summary = doc["summary"]

    # m is a property of the DECISION, not of what the forecaster chose to talk
    # about (red-team F1b: shrinking the arm count moved the same beliefs from
    # fail to pass).
    n_arms = len(results["options"])
    n_forecast_arms = len({f["option_id"] for f in outcomes.load_forecasts(run.dir)})
    obs = _scored_pairs(run.dir, doc, n_arms)

    # Feed the learning loop — ONLY for low-leakage cases (a memorized outcome
    # would teach from memory, not prediction) and only with a real library.
    leak = _leakage_risk(case)
    learned = None
    # `learn` here is a REQUEST, not the act. :func:`backtest` predicts and
    # scores every case first and only then learns, because a case learned
    # mid-batch writes its outcome into cases.jsonl, from which
    # learning.find_precedents pulls it into the ANALYST PROMPT of every later
    # case ("- [leak_first] ... 결과 등급 1.00"). That is a train-on-test leak
    # created by list order alone (red-team F6, 2026-07-29). Callers who invoke
    # run_case directly still get the old inline behaviour, which is correct for
    # a single case with nothing after it.
    if learn and not defer_learning and leak == "low" and library_dir is not None:
        try:
            from .learning import update_from_run
            learned = update_from_run(run.dir, library_dir, synthetic=False)
        except Exception as e:                       # never fail the backtest
            learned = {"error": repr(e)}

    predicted_p = None
    if winner in results["options"]:
        predicted_p = p_rank1[results["options"].index(winner)]

    return {
        "case_id": case.get("case_id"),
        "brier": summary["brier"],
        "brier_vs_chance": summary["brier_vs_chance"],
        "n_resolved": summary["n_resolved"],
        # The case's own (p, y) sample, so the aggregate can score the engine and
        # the baseline on identical observations (scoring-space invariant).
        "scored_pairs": obs,
        # arms the forecaster never spoke about. Silent shrinkage of the
        # denominator was F1b; make it a number in the report instead.
        "n_arms_problem": n_arms,
        "n_arms_forecast": n_forecast_arms,
        "arm_coverage_complete": n_forecast_arms == n_arms,
        "leakage_risk": leak,
        "predicted_p": predicted_p,
        "outcome_scalar": ground_truth.get("outcome_scalar"),
        "run_dir": str(run.dir),
        "learned": bool(learned and not learned.get("error")
                        and not learned.get("skipped")),
        "learn_excluded_reason": (None if (learn and leak == "low"
                                           and library_dir is not None)
                                  else ("not_low_leakage" if learn and leak != "low"
                                        else ("no_library" if learn else "learn_off"))),
    }


# --------------------------------------------------------------------------
# 3. The number the engine must beat.
# --------------------------------------------------------------------------
def stratified_baseline(observations: list[dict]) -> list[float]:
    """The no-information prediction for each observation: ``1/m`` for its case.

    Exactly one arm materialises per case, so a predictor that knows only how
    many arms were on the table — and nothing about which one wins — says
    ``1/m``. That is the floor the engine has to clear.

    Why not a single pooled constant (what this used to be): if ``m`` varies
    across cases, a predictor emitting ``1/m`` per arm carries ZERO information
    about which arm wins and still beats a global constant, because the pooled
    constant pays the between-case variance the per-case one does not. Measured:
    such a predictor passed `beats_baseline` on 41.5–46.7% of simulated runs at
    every sample size from n=2 to n=50, with no decay — it is an identity, not
    noise, so no amount of data fixes it. Stratifying makes that predictor tie
    exactly, which is what "no information" should score.
    """
    return [1.0 / max(1, int(o.get("m") or 1)) for o in observations]


def strongest_baseline(observations: list[dict]) -> dict:
    """The bar the engine must clear: the BETTER of two no-information predictors.

    Neither one alone is safe, and the red team proved both directions:

    * **1/m per case (structural).** Exactly one arm materialises, so a
      predictor knowing only the arm count says 1/m. Without this, a predictor
      emitting 1/m — zero information about WHICH arm wins — beat a global
      constant whenever arm counts varied, on 41.5–46.7% of simulated runs at
      every sample size from n=2 to n=50, with no decay. Not noise: an identity.
    * **The in-sample optimal constant (empirical base rate r).** Its Brier is
      r(1−r), and for any constant c the gap is exactly (c−r)² ≥ 0 — so while
      this is the bar, no constant predictor can ever pass. Replacing it with
      1/m alone THREW THAT GUARANTEE AWAY: on a fail-heavy sample (12 fail /
      2 win, base rate 0.10) a flat p=0.10 scored 0.0900 against a 1/m baseline
      of 0.1125 and passed by +0.0225. That was a regression introduced by the
      first fix, and it inverted the incentive from "drop failure cases" to
      "add failure cases".

    Taking the MINIMUM (lower Brier = stronger) keeps both guarantees: a constant can never pass, and neither
    can a predictor that merely counts arms. Returns both components so the
    report can show which one bound.
    """
    if not observations:
        return {"brier": None, "binding": None,
                "stratified": None, "constant": None, "base_rate": None}
    ys = [o["y"] for o in observations]
    r = sum(ys) / len(ys)
    strat = naive_baseline_brier(ys, stratified_baseline(observations))
    const = naive_baseline_brier(ys, r)
    # LOWER Brier = stronger competitor, so the bar is the MINIMUM. Taking the
    # max would mean "beat the weaker of the two", which is exactly the hole
    # this function exists to close.
    return {"brier": min(strat, const),
            "binding": "stratified_1_over_m" if strat <= const else "optimal_constant",
            "stratified": strat, "constant": const, "base_rate": r}


def significance(observations: list[dict], results: list[dict],
                 n_boot: int = 4000, seed: int = 20260729) -> dict:
    """Margin over the baseline with a case-level bootstrap interval.

    A bare ``mean_brier < baseline`` is a point estimate reported as a fact. On
    a measured 4-case configuration it said True while only 49.9% of resamples
    agreed — a coin flip presented as evidence, by the same engine whose whole
    pitch is not doing that. Resampling is at the CASE level because cases, not
    arms, are the independent unit.

    ``beats_baseline`` is True only when the margin is positive AND the 95%
    interval excludes zero. Everything needed to disagree with that call is in
    the returned dict.
    """
    import random
    if not observations or not results:
        return {"margin": None, "ci95": None, "p_positive": None,
                "n_cases": len(results), "beats_baseline": False,
                "note": "채점된 관측치가 없다"}
    per_case = [r.get("scored_pairs") or [] for r in results]
    per_case = [c for c in per_case if c]

    def margin_of(sample: list[list[dict]]) -> float | None:
        flat = [o for c in sample for o in c]
        if not flat:
            return None
        b = strongest_baseline(flat)["brier"]
        return b - _brier(flat)          # positive = engine better

    point = margin_of(per_case)
    rng = random.Random(seed)
    n = len(per_case)
    draws = []
    for _ in range(n_boot):
        s = [per_case[rng.randrange(n)] for _ in range(n)]
        mg = margin_of(s)
        if mg is not None:
            draws.append(mg)
    draws.sort()
    lo = draws[int(0.025 * len(draws))] if draws else None
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))] if draws else None
    p_pos = (sum(1 for d in draws if d > 0) / len(draws)) if draws else None
    passes = bool(point is not None and point > 0 and lo is not None and lo > 0)
    return {"margin": point, "ci95": [lo, hi], "p_positive": p_pos,
            "n_cases": n, "n_boot": len(draws), "beats_baseline": passes,
            "note": ("margin>0 이고 95% 구간이 0을 넘지 않을 때만 True. "
                     "구간이 0을 걸치면 점추정이 양수여도 판정하지 않는다.")}


def naive_baseline_brier(observed: list[float],
                         base_rate: float | list[float] | None = None) -> float | None:
    """Brier of the no-information predictor over the SAME observations.

    `observed` is the list of realized outcome values the engine was graded on —
    one entry per scored item, in whatever unit the caller scored in.
    `base_rate` may be a constant, a per-observation vector (the stratified
    case — pass :func:`stratified_baseline`), or None to use the empirical mean
    of `observed`. Either way it is scored on the identical sample. An engine
    whose Brier is not below this number has learned nothing beyond it.

    SCORING-SPACE INVARIANT (2026-07-29 — this is the bug this signature fixes):
    the engine's Brier and this baseline must be means over the *same* sample of
    the *same* random variable. Previously this function was handed the CASE
    list and scored 'how well did the winner do', while the engine was scored
    per (option, prediction) key on 'did this arm materialise' — two different
    events, averaged over two different sample sizes. The comparison, and every
    `beats_baseline` derived from it, was meaningless. Callers must now pass the
    engine's own observation vector; see :func:`backtest`.

    Returns ``None`` when there is nothing to score.
    """
    ys = [float(y) for y in observed if y is not None]
    if not ys:
        return None
    if isinstance(base_rate, (list, tuple)):
        if len(base_rate) != len(ys):
            raise ValueError("per-observation baseline must match the sample "
                             f"({len(base_rate)} vs {len(ys)})")
        return sum((float(r) - y) ** 2 for r, y in zip(base_rate, ys)) / len(ys)
    r = base_rate if base_rate is not None else sum(ys) / len(ys)
    return sum((r - y) ** 2 for y in ys) / len(ys)


# --------------------------------------------------------------------------
# 4. Aggregate backtest with an always-visible leakage split.
# --------------------------------------------------------------------------
def _mean(xs: list[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _pool(results: list[dict]) -> list[dict]:
    """Flatten per-case observations into the one sample both scores use."""
    return [o for r in results for o in (r.get("scored_pairs") or [])]


def _brier(obs: list[dict]) -> float | None:
    """Mean over arms of each arm's own mean squared error (see _scored_pairs)."""
    return (sum(o["sq"] for o in obs) / len(obs)) if obs else None


def backtest(cases: list[dict], runs_dir: Path, backend,
             library_dir: Path | None = None, learn: bool = False) -> dict:
    """Run every case, aggregate, and compare to the naive base rate.

    Results are ALWAYS broken down by leakage-risk bucket so that low-leakage
    performance is legible separately from high-leakage performance; the split
    is never collapsed away. Returns a summary dict.

    When `learn=True`, low-leakage cases also FEED the learning loop (this is
    the bootstrap-from-history path); the report states how many cases actually
    moved the engine vs were excluded, so learning is never silent.
    """
    # PASS 1 — predict and score every case with the library frozen as it was
    # when the batch started. No case may see another case's outcome.
    results = [run_case(c, runs_dir, backend, library_dir=library_dir,
                        learn=learn, defer_learning=True)
               for c in cases]

    # PASS 2 — only now fold the low-leakage outcomes into the learning loop.
    # Every prediction above is already frozen in its snapshot, so nothing
    # learned here can have influenced anything scored.
    if learn and library_dir is not None:
        from .learning import update_from_run
        for r, c in zip(results, cases):
            if r["leakage_risk"] != "low":
                continue
            try:
                got = update_from_run(Path(r["run_dir"]), library_dir,
                                      synthetic=False)
                r["learned"] = bool(got and not got.get("error")
                                    and not got.get("skipped"))
                r["learn_excluded_reason"] = None
            except Exception as e:                   # never fail the backtest
                r["learned"] = False
                r["learn_excluded_reason"] = f"learn_error: {e!r}"
    n_learned = sum(1 for r in results if r.get("learned"))

    scored = [r for r in results if r["brier"] is not None]
    mean_bvc = _mean([r["brier_vs_chance"] for r in scored])

    # Engine vs baseline are computed on the SAME pooled (p, y) sample — the one
    # the engine was actually graded on. `mean_brier` is that pooled (micro)
    # Brier, NOT the mean of per-case means: a case emitting 12 predictions and a
    # case emitting 3 must not carry equal weight against a per-observation
    # baseline. The per-case macro average is reported separately as
    # `mean_brier_per_case` for continuity; it is not the comparator.
    pooled = _pool(scored)
    mean_brier = _brier(pooled)
    baseline = strongest_baseline(pooled)
    naive = baseline["brier"]
    mean_brier_per_case = _mean([r["brier"] for r in scored])
    verdict = significance(pooled, scored)

    by_bucket: dict[str, dict] = {}
    for r in results:
        b = by_bucket.setdefault(r["leakage_risk"],
                                 {"n": 0, "n_scored": 0, "briers": [], "bvc": []})
        b["n"] += 1
        if r["brier"] is not None:
            b["n_scored"] += 1
            b["briers"].append(r["brier"])
            b["bvc"].append(r["brier_vs_chance"])
    by_leakage_risk = {
        bucket: {"n": b["n"], "n_scored": b["n_scored"],
                 "mean_brier": _mean(b["briers"]),
                 "mean_brier_vs_chance": _mean(b["bvc"])}
        for bucket, b in by_bucket.items()
    }

    # The headline mean_brier/naive blend ALL leakage buckets, so "beats
    # baseline" on it can be carried by memorized (high/medium) cases (audit
    # finding 4). The only calibration number that means anything is the
    # LOW-leakage subset scored against its OWN baseline. Compute that
    # explicitly and never let it be inferred from the blended figure.
    low_cases = [c for c in cases if _leakage_risk(c) == "low"]
    low_scored = [r for r in scored if r["leakage_risk"] == "low"]
    low_pooled = _pool(low_scored)
    low_mean = _brier(low_pooled)
    low_baseline = strongest_baseline(low_pooled)
    low_naive = low_baseline["brier"]
    low_verdict = significance(low_pooled, low_scored)
    clean_calibration = {
        "n_cases": len(low_cases),
        "n_scored": len(low_scored),
        "n_observations": len(low_pooled),
        "mean_brier": low_mean,
        "naive_baseline_brier": low_naive,
        "base_rate": low_baseline["base_rate"],
        "beats_baseline": low_verdict["beats_baseline"],
        "margin": low_verdict["margin"],
        "ci95": low_verdict["ci95"],
        "p_positive": low_verdict["p_positive"],
        "baseline_binding": low_baseline["binding"],
        "baseline_components": {"stratified_1_over_m": low_baseline["stratified"],
                                "optimal_constant": low_baseline["constant"]},
        "note": ("이 값만 진짜 보정 지표다 — 저누출 사례를 자기 기준선과 비교. "
                 "headline mean_brier/beats_baseline는 고·중누출을 섞은 값이라 "
                 "보정 근거로 쓰면 안 된다. 엔진·기준선 모두 동일한 "
                 "(선택지,예측) 관측치 위에서 계산된다."),
    }

    return {
        "n_cases": len(cases),
        "n_scored": len(scored),
        "n_observations": len(pooled),          # the sample both scores share
        "mean_brier": mean_brier,               # pooled/micro — the comparator
        "mean_brier_per_case": mean_brier_per_case,   # macro, reported not compared
        "mean_brier_vs_chance": mean_bvc,
        "naive_baseline_brier": naive,
        "base_rate": baseline["base_rate"],
        # what the baseline actually predicted, so the comparison is auditable
        "baseline_kind": "min(케이스별 1/m, 표본 최적 상수) — Brier가 더 낮은 쪽 = 더 센 상대",
        "arm_counts": sorted({o["m"] for o in pooled}),
        "beats_baseline": verdict["beats_baseline"],
        "margin": verdict["margin"],
        "ci95": verdict["ci95"],
        "p_positive": verdict["p_positive"],
        "baseline_binding": baseline["binding"],
        "baseline_components": {"stratified_1_over_m": baseline["stratified"],
                                "optimal_constant": baseline["constant"]},
        "arm_coverage_incomplete": [r["case_id"] for r in results
                                    if not r.get("arm_coverage_complete", True)],
        "headline_is_leakage_contaminated": any(
            b != "low" for b in by_leakage_risk),
        "scoring": {
            "unit": "(케이스, 선택지) 관측치 1개 — 팔의 예측들에 대한 제곱오차 평균",
            "event": "이 선택지가 실제로 실현되었는가 (true=1 / false=0)",
            "note": ("mean_brier와 naive_baseline_brier는 동일한 관측 표본 위에서 "
                     "계산된다. 기준선은 케이스별 1/m(무정보 예측기)로 층화돼 "
                     "있어, 선택지 개수만 아는 예측기는 이기지 못하고 정확히 "
                     "비긴다. mean_brier_vs_chance는 p=0.5 기준(0.25-Brier)이라 "
                     "이 표본과 무관하다 — 기준선 비교에 쓰지 말 것."),
        },
        "clean_calibration": clean_calibration,   # low-leakage only, own baseline
        "by_leakage_risk": by_leakage_risk,   # honest split, never hidden
        "learning": {
            "requested": bool(learn),
            "n_fed_learning": n_learned,           # low-leakage cases that learned
            "n_excluded_high_or_medium": sum(
                1 for r in results if learn and r["leakage_risk"] != "low"),
            "note": ("backtest가 학습을 부트스트랩한 건수 — 저누출만 학습에 반영, "
                     "고·중누출은 기억 오염 방지를 위해 제외." if learn else
                     "learn=False: 채점만, 학습 반영 없음."),
        },
        "results": results,
    }
