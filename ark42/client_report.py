# -*- coding: utf-8 -*-
"""고객용 리포트 — 맥킨지 피라미드 원칙, 10초 이해 규율.

내부 리포트(report.py)와의 분리 원칙
------------------------------------
- 첫 화면이 답이다: 권고 → 이유 3줄 → 뒤집힘 조건 1줄. 읽는 순서가 곧 논증.
- **내부 지수는 비공개다.** 기대효용·P(rank1)·ρ·표준편차 같은 숫자는 이 파일
  밖으로 나가지 않는다. 고객에게는 순위, ●◐○ 등급, 평문 문장만 보인다.
  (숫자를 숨기는 이유는 포장이 아니라 정직함이다 — 맥락 없는 0.606은
  정보가 아니라 권위 연출이 된다.)
- 예측만은 숫자를 유지한다 — 반증 가능성이 상품의 차별점이므로, 확률과
  판정 조건은 고객 언어로 그대로 보여주고 3개월 뒤 공개 채점을 예고한다.
- 서면은 오프라인에서 열려야 한다: 외부 참조 없는 단일 HTML.

사용: render_client_report(run_dir) → run_dir/client_report.html
CLI:  python -m ark42 problem.json --recorded DIR --client
"""
from __future__ import annotations

import html
import json
from pathlib import Path

#: 기준 8개의 고객 언어 라벨 — 온톨로지 키는 노출하지 않는다.
CRITERION_KO = {
    "demand": "수요가 있는가",
    "feasibility": "지금 실행 가능한가",
    "cost_efficiency": "비용 부담이 낮은가",
    "revenue_potential": "수익으로 이어지는가",
    "defensibility": "따라 하기 어려운가",
    "risk_resilience": "위험이 낮은가",
    "time_to_validation": "결과를 빨리 아는가",
    "sustainability": "오래 지속 가능한가",
}


def _grade_row(scores: list[float]) -> list[str]:
    """한 기준에서 옵션들을 상대 비교해 ●◐○로 변환. 절대 수치는 버린다."""
    top = max(scores)
    out = []
    for s in scores:
        if s >= top - 0.05:
            out.append("●")
        elif s >= top - 0.15:
            out.append("◐")
        else:
            out.append("○")
    return out


def _confidence_phrase(mc: dict, winner_idx: int) -> str:
    corr = mc.get("correlated") or mc
    p1 = corr["p_rank1"][winner_idx]
    if p1 >= 0.6:
        return "검토한 관점들이 같은 방향을 가리킵니다 — 판단 신뢰도 높음."
    if p1 >= 0.35:
        return ("우세하지만 확정적이지 않습니다 — 아래 '뒤집힘 조건'을 함께 "
                "보셔야 합니다.")
    return "선택지 간 차이가 크지 않습니다 — 뒤집힘 조건이 사실상 결론입니다."


def _esc(s) -> str:
    return html.escape(str(s))


def render_client_report(run_dir) -> Path:
    d = Path(run_dir)
    results = json.loads((d / "results.json").read_text(encoding="utf-8"))
    problem = json.loads((d / "problem.json").read_text(encoding="utf-8"))
    forecasts = json.loads((d / "forecasts.json").read_text(encoding="utf-8"))

    options = results["options"]
    criteria = results["criteria"]
    cs = results["point_estimate"]["criterion_scores"]      # [opt][crit]
    mc = results["monte_carlo"]
    corr = mc.get("correlated") or mc

    titles = {o["option_id"]: o.get("title", o["option_id"])
              for o in problem.get("options", [])}
    titles.setdefault("NO_INTERVENTION", "아무것도 하지 않음")
    descs = {o["option_id"]: o.get("description", "")
             for o in problem.get("options", [])}
    descs.setdefault("NO_INTERVENTION",
                     "현재 상태를 유지합니다. 비용도 없지만 학습도 없습니다.")

    order = sorted(range(len(options)), key=lambda i: -corr["p_rank1"][i])
    win = order[0]
    win_id = options[win]

    # 권고 이유 3개: 승자가 ●를 받은 기준 중 대안과의 격차가 큰 순
    reasons = []
    for ci, c in enumerate(criteria):
        col = [cs[oi][ci] for oi in range(len(options))]
        others = [v for i, v in enumerate(col) if i != win]
        margin = col[win] - (sum(others) / len(others))
        reasons.append((margin, c))
    reasons.sort(reverse=True)
    reason_lines = [
        f"‘{CRITERION_KO.get(c, c)}’ 측면에서 검토한 {len(options)}개 안 중 "
        f"가장 유리합니다." for _, c in reasons[:3]]

    # 뒤집힘 조건: 승자 예측 중 최고확률 건의 falsified_if
    win_preds = next((f["predictions"] for f in forecasts["forecasts"]
                      if f["option_id"] == win_id), [])
    win_preds = sorted(win_preds, key=lambda p: -p.get("probability", 0))
    flip = (win_preds[0]["falsified_if"] if win_preds
            else "핵심 가정이 2주 내 반증되는 경우")

    grade_rows = []
    for ci, c in enumerate(criteria):
        col = [cs[oi][ci] for oi in range(len(options))]
        grade_rows.append((CRITERION_KO.get(c, c), _grade_row(col)))

    def option_cards():
        out = []
        for rank, oi in enumerate(order, 1):
            oid = options[oi]
            badge = ("권고" if oi == win else f"{rank}위")
            out.append(
                f'<div class="opt {"win" if oi == win else ""}">'
                f'<span class="rank">{badge}</span>'
                f'<h3>{_esc(titles.get(oid, oid))}</h3>'
                f'<p>{_esc(descs.get(oid, ""))}</p></div>')
        return "\n".join(out)

    pred_items = "".join(
        f'<li><strong>{_esc(p["metric"])}</strong> — {_esc(p["prediction"])} '
        f'<span class="odds">우리 확률 {round(p["probability"]*100)}%</span>'
        f'<br><span class="falsify">틀렸다고 판정되는 조건: '
        f'{_esc(p["falsified_if"])}</span></li>'
        for p in win_preds[:3])

    head_cells = "".join(
        f"<th>{_esc(titles.get(options[oi], options[oi]))}</th>" for oi in order)
    body_rows = "".join(
        "<tr><td>" + _esc(label) + "</td>"
        + "".join(f'<td class="g">{grades[oi]}</td>' for oi in order)
        + "</tr>" for label, grades in grade_rows)

    doc = f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>결정 검토 리포트 — CIV42</title>
<style>
  :root {{ --ink:#1e1820; --plum:#6d5876; --deep:#4a3a52; --paper:#f6f4f2;
           --lav:#ece8ee; --line:#dbd5d9; --slate:#544a58; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
    font:15px/1.75 -apple-system, "Apple SD Gothic Neo", "Noto Sans KR",
    sans-serif; }}
  .wrap {{ max-width:760px; margin:0 auto; padding:48px 22px 90px; }}
  .brand {{ font-family:Georgia, "Noto Serif KR", serif; letter-spacing:.05em;
    font-size:14px; color:var(--deep); }}
  .answer {{ background:#221b26; color:#f6f4f2; border-radius:14px;
    padding:30px 28px; margin:26px 0 34px; }}
  .answer .k {{ font-size:11px; letter-spacing:.16em; color:#b7a9be; }}
  .answer h1 {{ font-family:Georgia, "Noto Serif KR", serif; margin:8px 0 14px;
    font-size:27px; line-height:1.35; }}
  .answer ul {{ margin:0 0 14px; padding-left:20px; color:#e6dfe9; }}
  .answer .flip {{ border-top:1px solid #3a2f3e; padding-top:12px; margin:0;
    font-size:13px; color:#cfc6d3; }}
  .conf {{ font-size:13px; color:#b7a9be; margin:10px 0 0; }}
  h2 {{ font-family:Georgia, "Noto Serif KR", serif; font-size:19px;
    margin:40px 0 12px; }}
  .opt {{ border:1px solid var(--line); border-radius:12px; background:#fff;
    padding:16px 18px; margin:10px 0; }}
  .opt.win {{ border-color:var(--plum); box-shadow:0 0 0 1px var(--plum); }}
  .opt .rank {{ font-size:11px; letter-spacing:.1em; color:var(--plum);
    font-weight:700; }}
  .opt h3 {{ margin:4px 0 6px; font-size:16px; }}
  .opt p {{ margin:0; color:var(--slate); font-size:13.5px; }}
  table {{ width:100%; border-collapse:collapse; background:#fff;
    border:1px solid var(--line); border-radius:12px; overflow:hidden; }}
  th, td {{ padding:9px 10px; border-bottom:1px solid var(--line);
    font-size:13px; text-align:center; }}
  td:first-child, th:first-child {{ text-align:left; }}
  th {{ background:var(--lav); font-size:12px; }}
  td.g {{ font-size:15px; color:var(--deep); }}
  .legend {{ font-size:12px; color:var(--slate); margin:8px 0 0; }}
  ol.preds {{ padding-left:20px; }} ol.preds li {{ margin:12px 0; }}
  .odds {{ color:var(--plum); font-weight:700; font-size:13px; }}
  .falsify {{ color:var(--slate); font-size:12.5px; }}
  .foot {{ margin-top:52px; padding-top:16px; border-top:1px solid var(--line);
    font-size:12px; color:var(--slate); }}
</style></head><body><div class="wrap">
  <p class="brand">CIV42 · PRIVATE DECISION ADVISORY — 결정 검토 리포트</p>

  <div class="answer">
    <p class="k">권고 — 한 줄 답부터</p>
    <h1>{_esc(titles.get(win_id, win_id))}</h1>
    <ul>{"".join(f"<li>{_esc(r)}</li>" for r in reason_lines)}</ul>
    <p class="flip"><strong>이 권고가 뒤집히는 조건:</strong> {_esc(flip)}</p>
    <p class="conf">{_esc(_confidence_phrase(mc, win))}</p>
  </div>

  <h2>무엇을 검토했나</h2>
  <p>{_esc(problem.get("statement", ""))}</p>
  {option_cards()}

  <h2>선택지 비교 — 8가지 질문</h2>
  <table><thead><tr><th>질문</th>{head_cells}</tr></thead>
  <tbody>{body_rows}</tbody></table>
  <p class="legend">● 이 질문에서 가장 유리 · ◐ 경합 · ○ 열위 — 등급은
  세 개 독립 관점의 종합 상대평가입니다.</p>

  <h2>우리가 건 예측 — 3개월 뒤 공개 채점됩니다</h2>
  <p>이 리포트는 의견으로 끝나지 않습니다. 아래 예측은 등록 시점에 고정되며,
  3개월 뒤 실제 결과와 대조해 맞았는지 틀렸는지 그대로 알려드립니다.</p>
  <ol class="preds">{pred_items}</ol>

  <h2>첫 실행 — 가장 작게 시작하는 법</h2>
  <p>권고안을 통째로 믿고 큰 약속을 하지 마십시오. 위 예측 중 판정 기한이
  가장 짧은 것 하나를 골라, 되돌릴 수 있는 최소 행동으로 그 조건만 먼저
  확인하십시오. 그 결과가 이 리포트의 첫 채점이 됩니다.</p>

  <div class="foot">
    <p>이 리포트는 정보 제공이며 결정을 대신하지 않습니다 — 최종 선택과 책임은
    의뢰인에게 있습니다. 새로운 관점이 하나도 없었다고 판단되시면 수령 후
    48시간 내 이유 한 줄과 함께 회신해 주십시오. 전액 환불합니다.</p>
    <p>위 예측은 공개 대장(civ42.com/ledger)에 등록되어 기한이 오면 채점됩니다.
    투자·금융 상품에 대한 판단은 다루지 않습니다.</p>
    <p>CIV42 · civ42.com · 수익의 10%는 유니세프한국위원회에 기부됩니다</p>
  </div>
</div></body></html>"""
    out = d / "client_report.html"
    out.write_text(doc, encoding="utf-8")
    return out
