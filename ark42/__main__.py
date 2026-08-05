from __future__ import annotations
"""CLI: python -m ark42 <problem.json> [--recorded DIR] [--draws N]

problem.json: {"problem_id", "statement", "context", "options":[...]}
With --recorded, LLM responses are replayed from DIR (no API key needed).
"""
import argparse
import json
import sys
from pathlib import Path

from .backends import build_backend
from .llm import RecordedBackend
from .pipeline import Run, make_problem
from .report import render_report


def main() -> int:
    ap = argparse.ArgumentParser(prog="ark42")
    ap.add_argument("problem_file")
    ap.add_argument("--recorded", help="replay LLM responses from this directory")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--client", action="store_true",
                    help="고객용 리포트(client_report.html)도 함께 생성")
    args = ap.parse_args()

    spec = json.loads(Path(args.problem_file).read_text(encoding="utf-8"))
    problem = make_problem(spec["problem_id"], spec["statement"],
                           spec.get("context", ""), spec.get("options", []))
    # build_backend honours the same env precedence as the server (Anthropic →
    # cafe24 → Gemini → Groq), so the operator's local key — whichever vendor —
    # just works for concierge runs.
    backend = (RecordedBackend(Path(args.recorded)) if args.recorded
               else build_backend(log_dir=Path(args.runs_dir) / spec["problem_id"] / "analyses"))
    run = Run(problem, backend, Path(args.runs_dir))
    opts = run.frame_options()
    print(f"[0/5] 옵션: {len(opts)}개 — {[o.option_id for o in opts]}"
          + (" (엔진 생성)" if not spec.get("options") else " (사용자 제공)"))
    chosen = run.select_disciplines()
    print(f"[1/5] 학문 선별: {len(chosen)}개 — {[c['discipline'] for c in chosen]}")
    analyses = run.analyze(chosen)
    print(f"[2/5] 독립 분석 완료: {sum(len(a.assessments) for a in analyses)}개 온톨로지 셀, 검증 통과")
    results = run.quantify(analyses, n_draws=args.draws)
    print(f"[3/5] 텐서 {len(results['options'])}×{len(results['criteria'])}×{len(results['disciplines'])}"
          f" · 몬테카를로 {args.draws:,} draws 완료")
    fc = run.forecast(results)
    print(f"[4/5] 결과 예상치: {sum(len(f['predictions']) for f in fc['forecasts'])}개 예측 생성·검증")
    out = render_report(run.dir)
    print(f"[5/5] 리포트: {out}")
    if args.client:
        from .client_report import render_client_report
        cout = render_client_report(run.dir)
        print(f"[+] 고객용 리포트: {cout}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
