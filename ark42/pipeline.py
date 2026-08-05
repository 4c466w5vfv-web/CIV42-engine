"""Run orchestrator. One Run = one problem through the full workflow,
with every intermediate artifact persisted under runs/<run_id>/:

  problem.json          the immutable problem snapshot
  selection.json        which disciplines were chosen and why (and rejected)
  analyses/<disc>.json  each discipline's independent, validated analysis
  tensor.npz            the raw score tensor
  results.json          aggregation + sensitivity + Monte Carlo numbers
  report.html           human-facing visualization

Reality remains the final validator: results.json contains predictions;
outcome recording (compare-to-reality) appends to outcome.json later.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .llm import LLMBackend
from .montecarlo import simulate
from .ontology import DecisionOption, Problem, analysis_from_dict
from .prompts import (ANALYST_SYSTEM, FORECASTER_SYSTEM, FRAMER_SYSTEM,
                      SELECTOR_SYSTEM, analyst_prompt, forecaster_prompt,
                      framer_prompt, selector_prompt)
from .tensor import aggregate, build_tensor, sensitivity


class Run:
    def __init__(self, problem: Problem, backend: LLMBackend, runs_dir: Path,
                 library_dir: Path | None = None,
                 library_channel: str = "production",
                 precedent_owner_filter=None):
        self.problem = problem
        self.backend = backend
        self.library_dir = Path(library_dir) if library_dir else None
        self.library_channel = library_channel
        # Multi-tenant safety: only precedents this caller may see are
        # injected into analyst prompts. None = no filter (CLI/single-tenant).
        self.precedent_owner_filter = precedent_owner_filter
        self.dir = Path(runs_dir) / problem.problem_id
        self.dir.mkdir(parents=True, exist_ok=True)
        if problem.options:                      # options supplied by the user
            problem.ensure_baseline()
        self._write_problem()

    def _write_problem(self) -> None:
        self._write("problem.json", {
            "problem_id": self.problem.problem_id,
            "statement": self.problem.statement,
            "context": self.problem.context,
            "options": [asdict(o) for o in self.problem.options],
        })

    # ---- stage 0: option generation (only when none were given) ------
    def frame_options(self) -> list[DecisionOption]:
        """Generate competing decision options from the bare problem."""
        if self.problem.options:
            return self.problem.options
        raw = self.backend.complete("framing", FRAMER_SYSTEM,
                                    framer_prompt(self.problem))
        framing = json.loads(raw)
        opts = framing["options"]
        if not (3 <= len(opts) <= 5):
            raise ValueError(f"framer returned {len(opts)} options; expected 3-5")
        ids = [o["option_id"] for o in opts]
        if len(set(ids)) != len(ids) or "NO_INTERVENTION" in ids:
            raise ValueError("framer produced duplicate or reserved option ids")
        self.problem.options = [DecisionOption(**o) for o in opts]
        self.problem.ensure_baseline()
        self._write("framing.json", framing)
        self._write_problem()
        return self.problem.options

    # ---- stage 1: discipline selection -------------------------------
    def select_disciplines(self) -> list[dict]:
        """Selected disciplines are capped: each one costs an LLM call, so an
        uncapped selector made the variable cost of a single credit unbounded
        (relevance>=0.5 could admit all 20). Cap keeps cost per run bounded and
        predictable; the dropped disciplines are recorded, not hidden."""
        import os
        cap = int(os.environ.get("ARK42_MAX_DISCIPLINES", "6"))
        raw = self.backend.complete("selection", SELECTOR_SYSTEM,
                                    selector_prompt(self.problem))
        sel = json.loads(raw)
        eligible = sorted((s for s in sel["selected"]
                           if float(s["relevance"]) >= 0.5),
                          key=lambda s: -float(s["relevance"]))
        chosen, dropped = eligible[:cap], eligible[cap:]
        if dropped:
            sel["dropped_by_cap"] = [
                {"discipline": s["discipline"], "relevance": s["relevance"],
                 "reason": f"exceeded ARK42_MAX_DISCIPLINES={cap}"}
                for s in dropped]
        sel["cap_applied"] = cap
        self._write("selection.json", sel)
        return chosen

    # ---- stage 2: independent per-discipline analysis ----------------
    def analyze(self, chosen: list[dict]) -> list:
        from .learning import find_precedents, precedents_block
        prec = precedents_block(find_precedents(
            self.problem.statement, self.library_dir,
            owner_filter=self.precedent_owner_filter))
        if prec:
            self._write("precedents.json", {"injected": prec})
        analyses = []
        option_ids = {o.option_id for o in self.problem.options}
        errors_all = []
        for s in chosen:
            d = s["discipline"]
            raw = self.backend.complete(
                f"analysis_{d}",
                ANALYST_SYSTEM.format(discipline=d),
                analyst_prompt(self.problem, d, float(s["relevance"]), s.get("why", ""),
                               precedents=prec),
            )
            a = analysis_from_dict(json.loads(raw))
            errs = a.validate(option_ids)
            if errs:
                errors_all.extend(errs)
            (self.dir / "analyses").mkdir(exist_ok=True)
            self._write(f"analyses/{d}.json", json.loads(raw))
            analyses.append(a)
        if errors_all:
            raise ValueError("ontology validation failed:\n" + "\n".join(errors_all))
        # Provenance: which provider answered each discipline. Independence is
        # only real across distinct providers, so this map is what makes the
        # cross-provider correlation split possible downstream.
        pmap = {}
        if hasattr(self.backend, "provider_map"):
            pmap = self.backend.provider_map()
        else:
            pid = getattr(self.backend, "provider_id", "unknown")
            pmap = {s["discipline"]: pid for s in chosen}
        self._provider_map = pmap
        self._write("provider_map.json", {
            "providers": pmap,
            "distinct_providers": sorted(set(pmap.values())),
            "multi_provider": len(set(pmap.values())) > 1,
        })
        return analyses

    # ---- stage 3-4: tensor + Monte Carlo (deterministic code) --------
    def quantify(self, analyses: list, n_draws: int = 20000) -> dict:
        from .learning import reliability_state
        wstate = reliability_state(self.library_dir, self.library_channel)
        t = build_tensor(self.problem, analyses,
                         reliability=wstate["multipliers"])
        weights_used = {
            "weight_version": wstate["weight_version"],
            "weight_hash": wstate["weight_hash"],
            "source": wstate["source"],
            "channel": wstate["channel"],
            "latest_update_id": wstate["latest_update_id"],
            "multipliers_applied": {d: round(wstate["multipliers"].get(d, 1.0), 6)
                                    for d in t.disciplines},
        }
        self._write("weights_used.json", weights_used)
        np.savez(self.dir / "tensor.npz", mean=t.mean, std=t.std, conf=t.conf,
                 relevance=t.relevance,
                 options=np.array(t.options), criteria=np.array(t.criteria),
                 disciplines=np.array(t.disciplines))
        # Independence is MEASURED, not assumed. The correlation between the
        # disciplines' judgements sets the copula's rho — a hardcoded guess
        # (this used to be 0.5) was optimistic against the first real run,
        # which measured 0.934.
        from .independence import measure
        providers = [getattr(self, "_provider_map", {}).get(d) for d in t.disciplines]
        indep = measure(t, providers=providers)
        self._write("independence.json", indep)
        rho_measured = indep["rho_used"]

        agg = aggregate(t)
        sens = sensitivity(t)

        # Cross-discipline interaction (genuine tensor contraction). The
        # additive point estimate treats disciplines as separable; a real
        # decision can have synergy/conflict BETWEEN dimensions (e.g. legal
        # risk × cash position). We model that as a bilinear contraction
        # U_int[o] = V[o]·J·V[o] over a learned coupling matrix J.
        # HONESTY: J is LEARNED from verified outcomes (interaction_learning),
        # never invented. With no outcomes yet it is all-zeros, so the
        # contraction is an EXACT no-op and the additive/MC numbers below are
        # unchanged. The block is shipped ready; it bends results only once
        # reality supplies evidence.
        from .interaction import (discipline_vectors, combined_utility,
                                   contributing_pairs)
        from .interaction_learning import load_coupling, zero_coupling
        # No library → no learned coupling yet: zero (exact no-op). Guarding
        # here (not just in callers) keeps quantify() robust when run without a
        # library_dir, e.g. one-off/backtest runs.
        J = (load_coupling(self.library_dir, t.disciplines)
             if self.library_dir else zero_coupling(len(t.disciplines)))
        V = discipline_vectors(t)
        inter = combined_utility(np.asarray(agg["utility"]), V, J)
        J_learned = bool(np.any(np.asarray(J) != 0.0))
        # Persist the per-option discipline vectors so the interaction learner
        # can, at OUTCOME time, build this run's training row (v_chosen) and
        # actually learn J from reality. Without this the coupling learner has
        # no production input and stays structurally inert.
        self._write("discipline_vectors.json",
                    {"options": t.options, "disciplines": list(t.disciplines),
                     "V": np.asarray(V).tolist()})

        mc = simulate(t, n=n_draws)                       # rho=0 (independent)
        mc_corr = simulate(t, n=n_draws, rho=rho_measured)   # measured, honest
        results = {
            "options": t.options,
            "criteria": t.criteria,
            "disciplines": t.disciplines,
            "weights_used": weights_used,
            "point_estimate": {
                "utility": agg["utility"].tolist(),
                "criterion_scores": np.nan_to_num(agg["criterion_scores"]).tolist(),
                "disagreement": agg["disagreement"].tolist(),
            },
            "sensitivity": {"p_rank1_weights_only": sens["p_rank1_weights_only"].tolist()},
            "monte_carlo": {
                "n_draws": mc["n_draws"],
                "expected_utility": mc["expected_utility"].tolist(),
                "utility_std": mc["utility_std"].tolist(),
                "percentiles": {k: v.tolist() for k, v in mc["percentiles"].items()},
                "p_rank1": mc["p_rank1"].tolist(),
                "p_beats_baseline": mc["p_beats_baseline"].tolist(),
                "expected_regret": mc["expected_regret"].tolist(),
                # Honest range: p_rank1 under independent (rho=0) vs correlated
                # (rho = the MEASURED cross-discipline correlation) cell
                # sampling. The correlated figure is the honest lower bound;
                # the gap between them is the overconfidence that independent
                # averaging would otherwise hide.
                "rho": 0.0,
                "correlated": {
                    "rho": rho_measured,
                    "rho_source": ("measured" if indep.get("reliable")
                                   else "unmeasurable_fallback_independent"),
                    "p_rank1": mc_corr["p_rank1"].tolist(),
                    "utility_std": mc_corr["utility_std"].tolist(),
                    "p_beats_baseline": mc_corr["p_beats_baseline"].tolist(),
                    # The honest interval too, so the forecaster and the frozen
                    # snapshot can be anchored on the correlated (widened)
                    # numbers rather than the narrow independent ones. When
                    # independence was unmeasurable, rho=0 and these equal the
                    # independent values, so this is always safe to use.
                    "expected_utility": mc_corr["expected_utility"].tolist(),
                    "percentiles": {k: v.tolist()
                                    for k, v in mc_corr["percentiles"].items()},
                },
            },
            "independence": indep,
            # Cross-discipline interaction. additive == the point estimate
            # above; adjusted == additive + the learned bilinear contraction.
            # When J_learned is false, adjusted == additive exactly (no-op).
            "interaction": {
                "learned": J_learned,
                "method": ("학문 간 결합 텐서 J와의 이차형식 수축 "
                           "U_int[o]=V[o]·J·V[o]. J는 검증된 결과에서만 학습되며 "
                           "결과가 없으면 0(항등)."),
                "additive_utility": np.asarray(inter["additive"]).tolist(),
                "adjusted_utility": np.asarray(inter["adjusted"]).tolist(),
                "delta": np.asarray(inter["delta"]).tolist(),
                "contributing_pairs": (contributing_pairs(V, J, t.options,
                                       t.disciplines) if J_learned else []),
                "note": ("" if J_learned else
                         "상호작용 계수 아직 미학습 — 검증된 결과가 쌓이면 활성화됩니다. "
                         "현재는 가산모델과 동일(항등)."),
            },
        }
        self._write("results.json", results)
        np.save(self.dir / "samples.npy", mc["utility_samples_head"])
        self._tensor = t
        return results

    # ---- cost accounting ---------------------------------------------
    def record_cost(self, in_price_per_mtok: float = 3.0,
                    out_price_per_mtok: float = 15.0,
                    krw_per_usd: float = 1450.0,
                    credit_value_krw: float | None = None,
                    ceiling_frac: float = 0.5) -> dict | None:
        """Persist the run's ACTUAL token usage and cost, when the backend
        reports it. Replaces guessing unit economics from prompt lengths.

        Also computes REALIZED MARGIN for this run — the engine's own
        "measure, don't assume" principle applied to its P&L. The price sheet's
        73% margin is a projection from one run; this records the actual margin
        of THIS run against the (conservative) value of the 1 credit it
        consumed, and flags when a run's cost eats more than `ceiling_frac` of
        that credit. Aggregated across runs (billing.realized_margin_report),
        this turns the assumed margin into a measured distribution with a tail.

        `credit_value_krw`: KRW value of the 1 credit a run consumes. Defaults
        to env ARK42_CREDIT_VALUE_KRW, else the CONSERVATIVE lowest per-run
        price on the sheet (biggest-bundle discount), so realized margin is a
        worst-case not a flattering one.
        """
        usage = getattr(self.backend, "usage", None)
        if not usage:
            return None
        # The defaults are ANTHROPIC's list price in USD. Route through another
        # provider (or a KRW-billed gateway like Cafe24 LLM Router) and these
        # produce a fictional cost, which then produces a fictional margin.
        # Env overrides let the operator state the real rate; `price_source`
        # records which one was used so a margin report can never be read as
        # measured when it was assumed.
        src = "default_anthropic_list_usd"
        env_in = os.environ.get("ARK42_PRICE_IN_PER_MTOK")
        env_out = os.environ.get("ARK42_PRICE_OUT_PER_MTOK")
        env_krw = os.environ.get("ARK42_KRW_PER_USD")
        if env_in or env_out:
            in_price_per_mtok = float(env_in) if env_in else in_price_per_mtok
            out_price_per_mtok = float(env_out) if env_out else out_price_per_mtok
            src = "env_override"
        if env_krw:
            krw_per_usd = float(env_krw)
        # A KRW-priced gateway: state won per million tokens directly and skip
        # the USD leg entirely, instead of inventing an exchange rate.
        krw_in = os.environ.get("ARK42_PRICE_IN_KRW_PER_MTOK")
        krw_out = os.environ.get("ARK42_PRICE_OUT_KRW_PER_MTOK")
        ti = sum(u["input_tokens"] for u in usage)
        to = sum(u["output_tokens"] for u in usage)
        # 프롬프트 캐싱: 캐시 쓰기는 입력가의 1.25배, 읽기는 0.1배로 청구된다.
        # 이 두 항을 빼고 계산하면 장부가 실제보다 싸게 나온다 — 절감을
        # 자랑하려고 원가를 과소계상하는 것은 이 엔진이 금지하는 일이다.
        tcw = sum(u.get("cache_creation_input_tokens", 0) or 0 for u in usage)
        tcr = sum(u.get("cache_read_input_tokens", 0) or 0 for u in usage)
        billable_in = ti + tcw * 1.25 + tcr * 0.10
        if krw_in or krw_out:
            krw = ((billable_in * float(krw_in or 0) / 1e6)
                   + (to * float(krw_out or 0) / 1e6))
            usd = krw / krw_per_usd if krw_per_usd else 0.0
            src = "env_krw_per_mtok"
        else:
            usd = billable_in * in_price_per_mtok / 1e6 + to * out_price_per_mtok / 1e6
            krw = usd * krw_per_usd
        if credit_value_krw is None:
            env = os.environ.get("ARK42_CREDIT_VALUE_KRW")
            if env:
                credit_value_krw = float(env)
            else:
                try:                     # conservative: lowest per-run price
                    from .billing import PRICING
                    credit_value_krw = min(w / c for c, w in PRICING.items() if c)
                except Exception:
                    credit_value_krw = 3900.0
        margin = (1.0 - krw / credit_value_krw) if credit_value_krw else None
        doc = {"calls": len(usage), "input_tokens": ti, "output_tokens": to,
               "cache_write_tokens": tcw, "cache_read_tokens": tcr,
               # 캐시가 실제로 절감한 금액. 캐시 없이 같은 입력을 정가로
               # 보냈을 때와의 차액이며, 0 이면 캐시가 작동하지 않은 것이다.
               "cache_saved_krw": round(
                   (tcr * 0.90 - tcw * 0.25) * in_price_per_mtok
                   / 1e6 * krw_per_usd) if not (krw_in or krw_out) else None,
               "usd": round(usd, 4), "krw": round(krw),
               "price_source": src,
               "provider_ids": sorted({u.get("provider_id") for u in usage
                                       if u.get("provider_id")}) or None,
               "credit_value_krw": round(credit_value_krw),
               "realized_margin": round(margin, 4) if margin is not None else None,
               # A run whose COGS exceeds ceiling_frac of one credit is a margin
               # warning; > 1.0 credit means the run LOST money outright.
               "cost_ceiling_frac": ceiling_frac,
               "cost_ceiling_exceeded": bool(margin is not None
                                             and krw > credit_value_krw * ceiling_frac),
               "unprofitable": bool(margin is not None and margin < 0),
               "price_assumption": {"in_per_mtok": in_price_per_mtok,
                                    "out_per_mtok": out_price_per_mtok,
                                    "krw_per_usd": krw_per_usd},
               "provider_map_ref": "provider_map.json",
               "per_call": usage}
        self._write("cost.json", doc)
        return doc

    # ---- stage 5: outcome forecasts (falsifiable predictions) --------
    def forecast(self, results: dict) -> dict:
        mc = results["monte_carlo"]
        # HONESTY: the falsifiable predictions this produces are what reality
        # Brier-scores, so they must be anchored on the CORRELATED (widened)
        # probabilities the engine actually believes — not the independent
        # (rho=0) numbers, which the engine's own independence measurement
        # flags as overconfident. The correlated block equals the independent
        # one when independence was unmeasurable, so this is always safe.
        corr = mc.get("correlated", {})
        p_rank1 = corr.get("p_rank1", mc["p_rank1"])
        p_beats = corr.get("p_beats_baseline", mc["p_beats_baseline"])
        exp_u = corr.get("expected_utility", mc["expected_utility"])
        pct = corr.get("percentiles", mc["percentiles"])
        lines = []
        for i, o in enumerate(results["options"]):
            lines.append(
                f"- {o}: 기대효용 {exp_u[i]:.3f} "
                f"(90% 구간 {pct['p5'][i]:.3f}~{pct['p95'][i]:.3f}), "
                f"P(1위) {p_rank1[i]:.1%}, 기준안 대비 우위 확률 {p_beats[i]:.1%}")
        raw = self.backend.complete("forecasts", FORECASTER_SYSTEM,
                                    forecaster_prompt(self.problem, "\n".join(lines)))
        fc = json.loads(raw)
        option_ids = {o.option_id for o in self.problem.options}
        errors = []
        seen_options = set()
        for f in fc["forecasts"]:
            if f["option_id"] not in option_ids:
                errors.append(f"forecast for unknown option {f['option_id']!r}")
            # One block per option. Two blocks for the same option collide on
            # the '<option>#<index>' key the snapshot is built from, so the
            # scored probability would silently differ from the audit file.
            if f["option_id"] in seen_options:
                errors.append(f"duplicate forecast block for {f['option_id']!r} "
                              f"— one block per option")
            seen_options.add(f["option_id"])
            for p in f["predictions"]:
                if not (0.0 <= float(p["probability"]) <= 1.0):
                    errors.append(f"{f['option_id']}: probability out of [0,1]")
                if not p.get("falsified_if"):
                    errors.append(f"{f['option_id']}: prediction without falsified_if")
        if errors:
            raise ValueError("forecast validation failed:\n" + "\n".join(errors))
        self._write("forecasts.json", fc)
        from .snapshots import freeze_prediction_snapshot
        freeze_prediction_snapshot(self.dir)     # freeze BEFORE any reality
        return fc

    # ---- stage 6: human decision (recorded, never automated) ---------
    def record_decision(self, option_id: str, decided_by: str, note: str = "") -> dict:
        from .snapshots import record_decision
        return record_decision(self.dir, option_id, decided_by, note)

    # ---- stage 7: outcome → learning → honest reward -----------------
    def record_outcome(self, verdicts: list[dict], recorded_by: str = "user",
                       synthetic: bool = False) -> dict:
        """Record reality, fold it into the library, return the reward payload.

        The reward is honest by construction: it reports the person's real
        contribution (predictions resolved, calibration measured, weights
        updated) and never depends on WHICH verdict was given — reporting
        a failed prediction earns exactly the same credit as a success.
        """
        from . import outcomes
        from .learning import read_cases, update_from_run
        doc = outcomes.record(self.dir, verdicts, recorded_by=recorded_by)
        learn = None
        if self.library_dir and outcomes.outcome_grade(self.dir) is not None:
            learn = update_from_run(self.dir, self.library_dir, synthetic=synthetic)
        s = doc["summary"]
        n_cases = len(read_cases(self.library_dir)) if self.library_dir else 0
        reward = {
            "outcome_id": doc["latest_outcome_id"],
            "decision_id": doc.get("decision_id"),
            "prediction_snapshot_id": doc.get("prediction_snapshot_id"),
            "weight_update_id": (learn or {}).get("update_id"),
            "weight_version": (learn or {}).get("weight_version"),
            "update_skipped": (learn or {}).get("skipped", False),
            "newly_recorded": len(verdicts),
            "resolved_total": s["n_resolved"],
            "tracked_total": s["n_predictions_tracked"],
            "brier": s["brier"],
            "brier_vs_chance": s["brier_vs_chance"],
            "calibration_note": _calibration_note(s),
            "reliability_changes": (learn or {}).get("reliability_changes", {}),
            "library_cases_total": n_cases,
            "contribution": (
                "판정이 기록되었습니다. 이 사례는 이미 학습에 반영되어 있어 "
                "가중치는 중복 갱신되지 않았습니다 (no-change 기록됨)."
                if learn and learn.get("skipped") else
                f"이 판정으로 사례 라이브러리가 {n_cases}건이 되었고, "
                f"{len((learn or {}).get('reliability_changes', {}))}개 학문의 "
                f"신뢰도 가중치가 갱신되었습니다 (v{(learn or {}).get('weight_version')}). "
                "다음 분석부터 반영됩니다."
                if learn else
                "판정이 기록되었습니다. 선택한 옵션의 예측이 판정되면 학습에 반영됩니다."),
        }
        self._write("reward.json", reward)
        return reward

    def _write(self, name: str, obj: dict) -> None:
        from .lineage import atomic_write_text
        atomic_write_text(self.dir / name,
                          json.dumps(obj, ensure_ascii=False, indent=2))


def _calibration_note(s: dict) -> str:
    if s["brier"] is None:
        return "아직 판정된 예측이 없어 보정 점수를 계산할 수 없습니다."
    if s["n_resolved"] < 10:
        return (f"Brier {s['brier']:.3f} (판정 {s['n_resolved']}건). "
                "10건 미만이라 통계적 의미는 약합니다 — 축적이 곧 검증입니다.")
    verdict = "무작위 추측(0.25)보다 좋음" if s["brier"] < 0.25 else "무작위 추측 수준 이하 — 엔진 확률을 신뢰하지 마십시오"
    return f"Brier {s['brier']:.3f}, {verdict} (판정 {s['n_resolved']}건)."


def make_problem(problem_id: str, statement: str, context: str,
                 options: list[dict]) -> Problem:
    return Problem(
        problem_id=problem_id, statement=statement, context=context,
        options=[DecisionOption(**o) for o in options],
    )
