"""Operator entrypoint to run a backtest end-to-end.

    python -m ark42.backtest_run --cases cases.json [--learn] [--out report.json]
    python -m ark42.backtest_run --fetch --cutoff 2024-01-01 [--max 200] [--learn]

Three input modes:
  --cases FILE   Load a prebuilt backtest-case list (the shared case-JSON
                 contract) and score/learn from it. Runs fully offline given a
                 recorded backend; needs LLM keys for a real prediction step.
  --pilot [FILE] Convert the mined narrative dataset (data/pilot_cases.json) to
                 backtest cases. Prints every dropped case and the leakage
                 split, and warns loudly when no LOW-leakage case exists —
                 which is the case for published sources, so this mode measures
                 RECALL of known outcomes, not calibration.
  --field [FILE] Owner field cases (data/field_cases.json): decisions the owner
                 lived through and never published. The ONLY source that can be
                 LOW leakage without waiting for a forward prediction to
                 resolve. Prints hindsight/asymmetry flags per case.
  --fetch        Pull COMPLETED ClinicalTrials.gov trials that posted results
                 on/after --cutoff, ingest them to cases (with leakage tags),
                 then run. NETWORK: the fetch happens in YOUR environment; this
                 session is robots-blocked, so --fetch is for the operator.

The LLM backend comes from build_backend (env: ARK42_PANELS / ANTHROPIC_API_KEY
/ ARK42_FAILOVER / ARK42_RECORDED_DIR / ...). With --learn, LOW-leakage cases
also bootstrap the learning loop (reliability weights + J) — history training
the engine WITHOUT any users. High/medium-leakage cases are scored but never
learned from. The report ALWAYS separates the clean (low-leakage) calibration
number from the leakage-contaminated headline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_cases(args) -> list[dict]:
    if args.pilot:
        from .ingest.pilot_narrative import to_backtest_cases
        r = to_backtest_cases(args.pilot)
        for d in r["dropped"]:
            print(f"drop {d['case_id']}: {d['why'][0]}", file=sys.stderr)
        print(f"pilot → {len(r['cases'])} cases, leakage {r['by_leakage']}",
              file=sys.stderr)
        if not r["n_low"]:
            print("경고: 저누출 케이스 0건 — 이 데이터셋으로는 clean "
                  "calibration을 산출할 수 없습니다. 채점은 '엔진이 이미 아는 "
                  "결과를 재현하는가'만 측정합니다.", file=sys.stderr)
        return r["cases"]
    if args.field:
        from .ingest.owner_field import convert
        r = convert(args.field)
        for d in r["dropped"]:
            print(f"drop {d['case_id']}: {d['why'][0]}", file=sys.stderr)
        for f in r["framing_flagged"]:
            print(f"후견지명 플래그 {f['case_id']}: "
                  f"{sorted(f['flags'])}", file=sys.stderr)
        print(f"field → {len(r['cases'])} cases, leakage {r['by_leakage']}, "
              f"저누출 {r['n_low']}건, 당시 논쟁 있던 케이스 {r['n_contested']}건",
              file=sys.stderr)
        return r["cases"]
    if args.cases:
        doc = json.loads(Path(args.cases).read_text())
        return doc["cases"] if isinstance(doc, dict) and "cases" in doc else doc
    # --fetch: operator environment only (network).
    from .ingest.clinicaltrials import fetch_trials, studies_to_cases
    studies = fetch_trials(cutoff=args.cutoff, max_studies=args.max)
    cases = studies_to_cases(studies, cutoff=args.cutoff)
    print(f"fetched {len(studies)} studies → {len(cases)} cases "
          f"({len(studies) - len(cases)} dropped: leakage/ex-ante)", file=sys.stderr)
    return cases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ark42.backtest_run")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--cases", help="path to a backtest-case JSON file")
    src.add_argument("--fetch", action="store_true",
                     help="fetch trials from ClinicalTrials.gov (operator env)")
    src.add_argument("--field", nargs="?", const="data/field_cases.json",
                     help="owner 현장 사례(미공개·1차 확인)를 백테스트 케이스로 "
                          "변환 (기본 경로: data/field_cases.json)")
    src.add_argument("--pilot", nargs="?", const="data/pilot_cases.json",
                     help="convert the mined pilot dataset to backtest cases "
                          "(default path: data/pilot_cases.json)")
    ap.add_argument("--cutoff", default="2024-01-01",
                    help="results-first-posted cutoff (leakage boundary)")
    ap.add_argument("--max", type=int, default=200, help="max studies to fetch")
    ap.add_argument("--learn", action="store_true",
                    help="feed low-leakage cases into the learning loop")
    ap.add_argument("--runs", default="runs", help="runs directory")
    ap.add_argument("--library", default="library", help="learning library dir")
    ap.add_argument("--out", help="write the full report JSON here")
    args = ap.parse_args(argv)

    from .backtest import backtest
    from .backends import build_backend

    cases = _load_cases(args)
    if not cases:
        print("no cases to run", file=sys.stderr)
        return 2
    backend = build_backend()
    rep = backtest(cases, Path(args.runs), backend,
                   library_dir=Path(args.library), learn=args.learn)

    # Human-readable summary to stderr; full JSON to --out (or stdout).
    cc = rep["clean_calibration"]
    print(f"\n=== backtest: {rep['n_cases']} cases, {rep['n_scored']} scored ===",
          file=sys.stderr)
    print(f"CLEAN calibration (low-leakage only, the number that means anything):",
          file=sys.stderr)
    print(f"  n={cc['n_cases']} cases / {cc['n_observations']} observations "
          f"(base_rate={cc['base_rate']})", file=sys.stderr)
    print(f"  mean_brier={cc['mean_brier']} baseline={cc['naive_baseline_brier']} "
          f"beats_baseline={cc['beats_baseline']}", file=sys.stderr)
    print(f"  (엔진·기준선 모두 같은 관측 표본에서 계산됨 — 단위: "
          f"{rep['scoring']['unit']})", file=sys.stderr)
    print(f"  headline blended (do NOT trust for calibration): "
          f"mean_brier={rep['mean_brier']} contaminated={rep['headline_is_leakage_contaminated']}",
          file=sys.stderr)
    if args.learn:
        L = rep["learning"]
        print(f"learning: {L['n_fed_learning']} low-leakage cases bootstrapped the "
              f"engine; {L['n_excluded_high_or_medium']} excluded (memory guard)",
              file=sys.stderr)

    out = json.dumps(rep, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(out, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
