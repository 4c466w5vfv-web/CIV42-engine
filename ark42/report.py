"""Self-contained HTML report for one run.

Follows the dataviz method: form first, validated categorical palette
(slots 1-3 + muted gray for the NO_INTERVENTION baseline), thin marks,
4px rounded data-ends, hairline gridlines, legend + selective direct
labels, hover tooltips, a full table view, light/dark from one source.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np


def _esc_tree(obj, protect=("problem_id",)):
    """Recursively HTML-escape every string leaf so user- and LLM-supplied
    text (statements, option titles, forecasts, verdicts) can never inject
    markup into the report. Keys in `protect` are left verbatim because they
    are used in JS/URL contexts and are constrained slugs with no HTML chars.
    """
    if isinstance(obj, dict):
        return {k: (v if k in protect else _esc_tree(v, protect))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [_esc_tree(v, protect) for v in obj]
    if isinstance(obj, str):
        return html.escape(obj, quote=True)
    return obj

# Reference palette (light, dark) — see dataviz references/palette.md
SERIES = [("#2a78d6", "#3987e5"), ("#eb6834", "#d95926"), ("#1baf7a", "#199e70"),
          ("#eda100", "#c98500"), ("#e87ba4", "#d55181"), ("#008300", "#008300"),
          ("#e34948", "#e66767")]
# Baseline uses categorical slot 7 (violet): gray failed the palette
# validator's chroma/CVD checks when adjacent to aqua; violet passes both modes.
BASELINE_COLOR = ("#4a3aa7", "#9085e9")
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
       "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

CRIT_KO = {
    "demand": "실수요", "feasibility": "실행 가능성", "cost_efficiency": "원가 효율",
    "revenue_potential": "수익 잠재력", "defensibility": "방어 가능성",
    "risk_resilience": "위험 내성", "time_to_validation": "검증 속도",
    "sustainability": "지속 가능성",
}

# ---- 표지 구성요소 (web/COMPONENTS.md 1·2·9) --------------------------------
# 장식이므로 aria-hidden. currentColor를 쓰니 다크모드 자동 대응.
# 길이가 다른 막대 넷 = 불확실성이 서로 다른 선택지들(不均斉).
MARK_SVG = (
    '<svg class="mark" width="22" height="22" viewBox="0 0 22 22" aria-hidden="true"'
    ' fill="none" stroke="currentColor" stroke-width="1.5">'
    '<line x1="2" y1="4" x2="20" y2="4"/>'
    '<line x1="2" y1="9" x2="13" y2="9"/>'
    '<line x1="2" y1="14" x2="17" y2="14"/>'
    '<line x1="2" y1="19" x2="7" y2="19"/></svg>')

# 파이프라인 6단계. cx = 16 + i*129.6 (COMPONENTS.md 2번의 좌표를 그대로 쓴다).
FLOW_STEPS = [("01", "질문 입력"), ("02", "선택지 생성"), ("03", "관련 분야 고르기"),
              ("04", "독립 분석"), ("05", "시뮬레이션"), ("06", "결론·예측")]
_FLOW_CX = [16, 145.6, 275.2, 404.8, 534.4, 664]

# Design tokens have ONE declaration: web/tokens.css. The report inlines it
# rather than linking it, because a delivered report has to open from a local
# file with no server — but it must never carry a second, drifting copy of the
# palette. If the file is missing we say so in the output instead of silently
# falling back to a stale hard-coded block.
_TOKENS_PATH = Path(__file__).resolve().parent.parent / "web" / "tokens.css"


def _tokens_css() -> str:
    try:
        return "\n".join("  " + ln if ln.strip() else ln
                         for ln in _TOKENS_PATH.read_text().splitlines())
    except OSError as e:                      # honest, not silent
        return (f"  /* web/tokens.css를 읽지 못했습니다 ({e!r}) — 색 토큰이 "
                f"비어 있어 브라우저 기본색으로 렌더됩니다. */")
FLOW_LABEL = "문제 입력에서 결론까지의 6단계 처리 과정"
# 넓은 화면: 헤어라인 축 위의 점 여섯. 시리즈 색은 쓰지 않는다 — 데이터가 아니다.
FLOW_SVG = (
    f'<svg class="flow" viewBox="0 0 680 96" role="img" aria-label="{FLOW_LABEL}">'
    '<line x1="16" y1="40" x2="664" y2="40" stroke="var(--axis)" stroke-width="1"/>'
    '<g fill="var(--surface)" stroke="var(--ink)" stroke-width="1.5">'
    + "".join(f'<circle cx="{cx:g}" cy="40" r="4"/>' for cx in _FLOW_CX)
    + '</g><g class="flow-n">'
    + "".join(f'<text x="{cx:g}" y="24">{n}</text>'
              for cx, (n, _) in zip(_FLOW_CX, FLOW_STEPS))
    + '</g><g class="flow-t">'
    + "".join(f'<text x="{cx:g}" y="62">{t}</text>'
              for cx, (_, t) in zip(_FLOW_CX, FLOW_STEPS))
    + '</g></svg>')
# 좁은 화면(<900px): 같은 6단계를 세로 목록 + 좌측 세로 헤어라인으로 내린다.
# 680 너비 도식을 390px에 욱여넣으면 12.5px 라벨이 6px로 찌그러지므로, 축소가
# 아니라 배치를 바꾼다(COMPONENTS.md 2번이 허용하는 대체안). 목록은 그 자체로
# 접근 가능한 구조라 role="img"를 붙이지 않는다 — 숨겨진 쪽은 낭독되지 않는다.
FLOW_LIST = ('<ol class="flowlist">'
             + "".join(f'<li><span class="n">{n}</span><span class="t">{t}</span></li>'
                       for n, t in FLOW_STEPS)
             + '</ol>')
FLOW_NOTE = "이 리포트는 위 6단계의 산출물이며, 모든 수치는 예측입니다."


def _load(run_dir: Path) -> dict:
    d = {}
    d["problem"] = json.loads((run_dir / "problem.json").read_text())
    d["selection"] = json.loads((run_dir / "selection.json").read_text())
    d["results"] = json.loads((run_dir / "results.json").read_text())
    d["samples"] = np.load(run_dir / "samples.npy")
    d["analyses"] = {}
    for p in sorted((run_dir / "analyses").glob("*.json")):
        d["analyses"][p.stem] = json.loads(p.read_text())
    for opt in ("forecasts", "framing", "outcome", "reward", "decision"):
        f = run_dir / f"{opt}.json"
        d[opt] = json.loads(f.read_text()) if f.exists() else None
    # Escape every string leaf (except numeric arrays in results/samples,
    # which carry no free text) before it reaches the HTML template.
    for k in ("problem", "selection", "analyses", "forecasts", "framing",
              "outcome", "reward", "decision", "results"):
        if d.get(k) is not None:
            d[k] = _esc_tree(d[k])
    return d


def _colors(options: list[str]) -> dict[str, tuple[str, str]]:
    out, i = {}, 0
    for o in options:
        if o == "NO_INTERVENTION":
            out[o] = BASELINE_COLOR
        else:
            if i >= len(SERIES):    # never cycle hues — that breaks identity
                raise ValueError("more options than categorical slots; fold or facet")
            out[o] = SERIES[i]
            i += 1
    return out


def _seq_color(v: float) -> str:
    return SEQ[min(len(SEQ) - 1, max(0, int(v * len(SEQ))))]


def _pct(v: float) -> str:
    """Honest probability display: never round to a false certainty."""
    if v >= 0.9995:
        return ">99.9%"
    if v <= 0.0005 and v > 0:
        return "<0.1%"
    return f"{v:.1%}"


def _ink_for(hexcolor: str) -> str:
    """Readable text colour on a filled swatch. The two literals are the ink
    and page values from web/tokens.css; they are literals (not var()) because
    this is used inside SVG fills where a custom property would not resolve
    against the swatch. Keep in step with tokens.css if the palette moves."""
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return "#0b0b0c" if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else "#ffffff"


def _trunc(s: str, max_chars: int = 15) -> str:
    """SVG axis labels never clip mid-character — truncate with an ellipsis
    instead and carry the full text in a tooltip (data-tip) so nothing is
    lost, per the dataviz skill's no-clip rule."""
    return s if len(s) <= max_chars else s[:max_chars - 1].rstrip() + "…"


def _sec(label: str, inner: str, sid: str = "", turn: bool = False) -> str:
    """One row of the asymmetric (不均斉) grid: an 11px label rail on the left,
    the content column on the right, nothing but a hairline between sections.
    `turn` widens the gap where the document changes register (모델 → 증거,
    분석 → 행동) so the rhythm is meaning-driven, not mechanical."""
    cls = "sheet turn" if turn else "sheet"
    idattr = f' id="{sid}"' if sid else ""
    return (f'<section class="{cls}"{idattr}><div class="rail">{label}</div>'
            f'<div class="body">{inner}</div></section>')


def _overlap_ko(r: float) -> str:
    """쌍 상관 하나를 사람이 읽는 말로. verdict_ko와 같은 문턱(0.2/0.5/0.8)을 쓴다."""
    if r < 0:
        return "반대 방향"
    if r <= 0.2:
        return "거의 겹치지 않음"
    if r <= 0.5:
        return "부분 겹침"
    if r <= 0.8:
        return "상당히 겹침"
    return "사실상 동일"


def _neff_meter(D: int, n_eff: float) -> str:
    """1..D 축 위에 유효 독립 관점 수를 찍는 헤어라인 눈금 하나.

    범주형 데이터가 아니므로 시리즈 색을 쓰지 않는다 (DESIGN.md 색 용법):
    축·눈금은 --axis, 표식은 --ink, n_eff/D < 0.25일 때만 상태색 --bad.
    모든 선은 1px, 모든 표식은 radius 0 (사각형 — 원은 곡률을 갖는다).
    """
    W, x0, x1, y = 760, 216, 700, 34
    span = max(D - 1, 1)
    n_eff = min(max(float(n_eff), 1.0), float(D))

    def px(v: float) -> float:
        return x0 + (min(max(v, 1.0), D) - 1) / span * (x1 - x0)

    ratio = n_eff / D if D else 1.0
    mark = "var(--bad)" if ratio < 0.25 else "var(--ink)"
    ticks = "".join(
        f'<line x1="{px(k):.1f}" y1="{y - 4}" x2="{px(k):.1f}" y2="{y + 4}" class="axis"/>'
        f'<text x="{px(k):.1f}" y="{y + 20}" class="tick" text-anchor="middle">{k}</text>'
        for k in range(1, D + 1))
    xe = px(n_eff)
    lab = f"유효 {n_eff:.2f} / 명목 {D}"
    anchor, lx = ("end", xe - 8) if xe > x1 - 150 else ("start", xe + 8)
    aria = (f"유효 독립 관점 수 {n_eff:.2f}개. 축은 1개(관점 하나)부터 "
            f"{D}개(투입한 분야 수, 완전 독립)까지.")
    return (f'<div class="fig"><svg viewBox="0 0 {W} {y + 30}" role="img" aria-label="{aria}">'
            f'<text x="{x0 - 10}" y="{y + 4}" class="axl" text-anchor="end">유효 독립 관점</text>'
            f'<line x1="{x0}" y1="{y}" x2="{x1}" y2="{y}" class="axis"/>{ticks}'
            # 명목 D: 속이 빈 사각 표식 (측정값이 아니라 겉보기 상한)
            f'<rect x="{px(D) - 4:.1f}" y="{y - 4}" width="8" height="8" rx="0" '
            f'fill="var(--surface)" stroke="var(--axis)" stroke-width="1" '
            f'data-tip="투입한 분야 {D}개 — 상관이 0일 때만 도달하는 상한"/>'
            # 측정된 n_eff: 채운 사각 표식
            f'<rect x="{xe - 3:.1f}" y="{y - 9}" width="6" height="18" rx="0" fill="{mark}" '
            f'data-tip="측정된 유효 독립 관점 {n_eff:.2f}개 (명목 {D}개의 {ratio:.0%})"/>'
            f'<text x="{lx:.1f}" y="{y - 14}" class="dl" text-anchor="{anchor}">{lab}</text>'
            f'</svg></div>')


def _provider_block(indep: dict) -> str:
    """분석에 참여한 공급자(모델) 구성 + 모델 내/간 상관.

    독립성은 서로 다른 모델 사이에서만 진짜다. 그래서 (1) 몇 개의 모델이 썼는지,
    (2) 단일 모델이면 모델 간 독립성이 아예 검증되지 않았다는 사실을 상태색으로
    강조하고, (3) 다중 모델이면 모델 내(같은 모델, 높게 나오는 게 정상)와 모델 간
    (진짜 신호) 상관을 나눠 보여준다.

    공급자 필드(n_providers)가 없는 예전 계측에서는 빈 문자열을 반환한다 — 구버전
    리포트가 오늘과 완전히 동일하게 렌더되도록 하기 위한 하위호환 가드다.
    """
    n_prov = indep.get("n_providers")
    if not isinstance(n_prov, int):
        return ""
    provs = [str(p) for p in (indep.get("providers") or [])]
    names = ", ".join(provs) if provs else "식별자 기록 없음"
    head = (f'<p class="note">이 분석에 쓰인 서로 다른 모델(공급자): '
            f'<b>{n_prov}개</b> — {names}.</p>')

    if indep.get("single_provider"):
        # 오늘의 정직한 기본값: 한 모델뿐이면 모델 간 독립성은 검증 자체가 불가능하다.
        return (head + '<div class="stats"><div class="stat">'
                '<div class="lbl">모델 간 독립성</div>'
                '<div class="val" style="color:var(--bad)">검증 안 됨</div>'
                '<div class="sub warn">이 실행은 한 개 모델만 썼습니다 — 서로 다른 '
                '모델 사이의 독립성은 검증되지 않았습니다. 위 상관은 모두 같은 모델 '
                '내부의 값입니다.</div></div></div>')

    within = indep.get("within_provider_mean_r")
    cross = indep.get("cross_provider_mean_r")
    within_v = f"{within:.3f}" if isinstance(within, (int, float)) else "—"
    if isinstance(cross, (int, float)):
        hi = cross > 0.7
        val_style = ' style="color:var(--bad)"' if hi else ''
        sub_cls = ' warn' if hi else ''
        cross_msg = ('모델을 바꿔도 판단이 크게 겹칩니다' if hi
                     else '진짜 독립성을 나타내는 유일한 값')
        cross_tile = (
            '<div class="stat"><div class="lbl">모델 간 상관 (cross)</div>'
            f'<div class="val"{val_style}>{cross:.3f}</div>'
            f'<div class="sub{sub_cls}">{cross_msg}</div></div>')
    else:
        cross_tile = (
            '<div class="stat"><div class="lbl">모델 간 상관 (cross)</div>'
            '<div class="val">해당 없음</div>'
            '<div class="sub">모델 간 비교 쌍이 없어 측정되지 않았습니다 — '
            '숫자를 만들지 않습니다.</div></div>')

    return (head + '<div class="stats">'
            '<div class="stat"><div class="lbl">모델 내 상관 (within)</div>'
            f'<div class="val">{within_v}</div>'
            '<div class="sub">같은 모델끼리라 높게 나오는 게 정상</div></div>'
            + cross_tile + '</div>'
            '<p class="note after-fig">독립성으로 인정되는 건 오직 '
            '<b>모델 간(cross) 상관</b>뿐입니다 — 모델 내 상관이 아무리 높아도 '
            '그것은 같은 목소리의 반복일 뿐 독립과 무관합니다.</p>')


def _independence_sections(indep: dict | None) -> list[str]:
    """독립성 계측 절 (+ 쌍별 상관 표). 계측이 없는 예전 런에서는 빈 목록."""
    if not isinstance(indep, dict) or not indep:
        return []
    D = indep.get("n_disciplines")
    method = indep.get("method", "")
    method_p = f'<p class="note">측정 방법: {method}</p>' if method else ""

    # ---- 측정 불가: 이유를 그대로 적고 숫자는 하나도 만들지 않는다 ----
    if not indep.get("reliable"):
        reason = indep.get("reason") or "사유가 기록되지 않았습니다"
        return [_sec("독립성", f"""<h2>분야 간 독립성 — 재지 못했습니다</h2>
          <p class="note">이 실행에서는 분야들의 판단이 실제로 독립적이었는지 측정할 수
          없었습니다. 측정 시도의 결과를 그대로 적습니다 — 대신 쓸 추정치를 만들지
          않습니다.</p>
          <p class="note"><b>사유:</b> {reason}</p>
          <p class="note">따라서 이 리포트에는 평균 상관·유효 독립 관점 수·쌍별 상관표가
          없습니다. "{D}개 분야"는 이름을 센 수치이며, 그것이 몇 개의 독립된 관점이었는지는
          알 수 없습니다.</p>
          {method_p}""")]

    mean_r = indep.get("mean_r")
    n_eff = indep.get("n_effective")
    if not isinstance(mean_r, (int, float)) or not isinstance(n_eff, (int, float)):
        return []                      # 형태가 낯선 계측이면 조용히 비운다
    lo, hi = indep.get("min_r"), indep.get("max_r")
    rng = (f"쌍별 {lo:.3f} ~ {hi:.3f}"
           if isinstance(lo, (int, float)) and isinstance(hi, (int, float))
           else "쌍별 범위 기록 없음")
    ratio = n_eff / D if D else 1.0
    thin = ratio < 0.25
    verdict = indep.get("verdict", "")
    uni = indep.get("unanimous_top")
    if uni is True:
        uni_txt = (f"<b>{D}개 분야 전부가 같은 선택지를 1등으로 꼽았습니다.</b> "
                   "가장 단순한 독립성 검사에서 다양성이 관측되지 않았습니다 — "
                   "합의가 아니라 중복일 수 있습니다.")
    elif uni is False:
        uni_txt = ("분야마다 1등으로 꼽은 선택지가 <b>갈렸습니다</b> — 적어도 최종 순위에서는 "
                   "서로 다른 판단이 남아 있습니다.")
    else:
        uni_txt = "최상위 선택의 만장일치 여부는 판정할 수 없었습니다."
    cap_txt = ""
    if indep.get("rho_was_capped"):
        cap_txt = (f'<p class="note">시뮬레이션에는 측정값 대신 ρ={indep.get("rho_used")}'
                   f'(상한 {indep.get("rho_cap")})가 쓰였습니다 — 코퓰라의 수치 안정성 '
                   f'때문이며, 위에 적힌 측정값 자체는 깎지 않았습니다.</p>')

    out = [_sec("독립성", f"""<h2>독립성 계측 — 이 분석은 몇 개의 관점이었나</h2>
      <p class="note">{verdict}</p>
      {_provider_block(indep)}
      <div class="stats">
        <div class="stat"><div class="lbl">투입한 분야</div><div class="val">{D}</div>
          <div class="sub">이름을 센 수 = 겉보기 시각의 수</div></div>
        <div class="stat"><div class="lbl">측정 평균 상관</div><div class="val">{mean_r:.3f}</div>
          <div class="sub">{rng}</div></div>
        <div class="stat"><div class="lbl">유효 독립 관점 수</div><div class="val">{n_eff:.2f}</div>
          <div class="sub{' warn' if thin else ''}">명목 {D}개의 {ratio:.0%}
            {'— 독립 다관점이라 부를 수 없는 수준' if thin else '— n_eff = D / (1 + (D-1)·r)'}</div></div>
      </div>
      {_neff_meter(int(D), float(n_eff))}
      <p class="note after-fig">축 왼쪽 끝 1 = 하나의 관점을 {D}번 확인한 것,
      오른쪽 끝 {D} = 분야 수만큼 서로 다른 시각. 채운 표식이 실제로 잰 값, 빈 표식이 상한.</p>
      <p class="note">{uni_txt}</p>
      <p class="note">비교에 쓴 공통 (옵션×기준) 셀 {indep.get('shared_cells')}개
      (최소 {indep.get('min_shared_cells')}개 미달이면 측정을 포기합니다).</p>
      {cap_txt}{method_p}""")]

    # ---- 쌍별 상관표: 평균 하나가 아니라 "누가 누구와" 겹쳤는지 ----
    # 히트맵 대신 표를 쓴다: SEQ는 0~1 단일 색상 램프인데 r은 음수까지 가므로
    # 이 자료를 SEQ로 칠하면 부호를 숨기게 되고, 학문 id(behavioral_psychology 등)를
    # 8열 매트릭스의 11px 헤더에 넣으면 잘린다. 표가 더 정직하다.
    pairs = [p for p in (indep.get("pairs") or [])
             if isinstance(p.get("r"), (int, float))]
    if pairs:
        # 공급자 필드가 있는 런에서만 '모델' 열을 붙인다. 예전 계측(same_provider·
        # n_providers 없음)은 열을 추가하지 않아 오늘과 동일하게 렌더된다.
        has_prov = isinstance(indep.get("n_providers"), int)

        def _sp_cell(p: dict) -> str:
            # 같은 모델 내 중복은 예상된 것(음영 처리), 모델 간 중복이 진짜 우려다.
            if p.get("same_provider"):
                return '<td class="fals">모델 내</td>'
            return '<td>모델 간</td>'

        prs = "".join(
            f"<tr><td>{p['a']}</td><td>{p['b']}</td><td>{p['r']:.3f}</td>"
            + (_sp_cell(p) if has_prov else "")
            + f"<td class=\"fals\">{_overlap_ko(float(p['r']))}</td></tr>"
            for p in pairs)
        prov_th = "<th>모델</th>" if has_prov else ""
        prov_note = ("<span class=\"fals\">모델 내</span>(같은 모델, 예상된 중복)와 "
                     "모델 간(서로 다른 모델, 겹치면 우려)을 구분해 표시했다. "
                     if has_prov else "")
        out.append(_sec("쌍별 상관", f"""<h2>어느 분야끼리 겹쳤나</h2>
          <p class="note">공통으로 채운 (옵션×기준) 셀에 대한 피어슨 상관, 높은 쌍부터.
          1에 가까운 쌍은 라벨만 다른 같은 판단이라는 뜻이다. {prov_note}위에 적힌 측정 평균 상관
          {mean_r:.3f}은 이 표의 {len(pairs)}개 쌍을 평균한 값이다.</p>
          <div class="fig"><table><thead><tr><th>분야 A</th><th>분야 B</th>
          <th>상관 r</th>{prov_th}<th>겹침</th></tr></thead><tbody>{prs}</tbody></table></div>"""))
    return out


def _interaction_section(inter: dict | None) -> list[str]:
    """학문 간 상호작용(텐서 수축) 절. 계수는 검증된 결과에서만 학습되며
    학습 전에는 '미학습'을 정직하게 표시하고 숫자를 만들지 않는다."""
    if not isinstance(inter, dict) or not inter:
        return []
    # 미학습: 구조는 있으나 결과 데이터가 없어 항등(가산모델과 동일)임을 밝힌다.
    if not inter.get("learned"):
        return [_sec("상호작용", f"""<h2>분야 간 상호작용 — 아직 배우지 못했습니다</h2>
          <p class="note">이 엔진은 분야들을 더하기만 하지 않고, 두 차원이 서로를
          증폭·상쇄하는 상호작용(예: 법적 위험 × 현금 사정)을 텐서 수축
          U<sub>int</sub>[o]=V[o]·J·V[o]으로 모델링할 수 있습니다.</p>
          <p class="note"><b>단, 결합계수 J는 검증된 실제 결과에서만 학습합니다.</b>
          아직 축적된 결과가 없어 J는 전부 0이고, 따라서 이 수축은 <b>항등</b>입니다 —
          위의 가산 점추정과 정확히 같은 값을 냅니다. 지어낸 상호작용 숫자는 없습니다.
          결과가 쌓이면 이 절에 어떤 분야 쌍이 결과를 얼마나 바꿨는지, 그리고 그 근거가
          된 사례 ID가 나타납니다.</p>""")]
    # 학습됨: 조정치와 기여 쌍(근거 추적)을 보인다.
    pairs = inter.get("contributing_pairs") or []
    add = inter.get("additive_utility") or []
    adj = inter.get("adjusted_utility") or []
    delta = inter.get("delta") or []
    dmax = max((abs(float(x)) for x in delta), default=0.0)
    rows = "".join(
        f"<tr><td>{p.get('d1')}</td><td>{p.get('d2')}</td>"
        f"<td>{float(p.get('coupling',0)):+.3f}</td>"
        f"<td class=\"fals\">{'증폭' if p.get('sign')=='amplify' else '감쇠'}</td></tr>"
        for p in pairs)
    tbl = (f"""<div class="fig"><table><thead><tr><th>분야 A</th><th>분야 B</th>
          <th>결합계수 J</th><th>방향</th></tr></thead><tbody>{rows}</tbody></table></div>"""
           if rows else '<p class="note">기여한 쌍이 기록되지 않았습니다.</p>')
    return [_sec("상호작용", f"""<h2>분야 간 상호작용 — 결과에서 학습된 결합</h2>
      <p class="note">가산 점추정에 더해, 검증된 결과에서 학습한 결합 텐서 J로
      분야 쌍 사이의 상호작용을 반영했습니다. 최대 조정폭 |Δ| ≈ {dmax:.3f}.
      아래 계수는 모두 <code>J_updates.jsonl</code>의 실제 결과 사례로 역추적됩니다.</p>
      {tbl}""")]


def render_report(run_dir: Path) -> Path:
    run_dir = Path(run_dir)
    d = _load(run_dir)
    res, prob = d["results"], d["problem"]
    options = res["options"]
    titles = {o["option_id"]: o["title"] for o in prob["options"]}
    colors = _colors(options)
    mc = res["monte_carlo"]
    eu = mc["expected_utility"]
    pct = mc["percentiles"]
    # Two views of the same ranking question:
    #   p1_ind  — every cell sampled independently (rho=0). Averages away most
    #             of the input uncertainty, so it reads overconfident.
    #   p1_corr — Gaussian-copula draw with within-option correlation rho>0.
    #             Honest (wider) ranking distribution. Absent on older runs.
    # Anything the report presents to the reader AS "1위 확률" uses p1 below;
    # the independent figure stays visible, but always labelled as such.
    p1_ind = mc["p_rank1"]
    corr = mc.get("correlated")
    p1_corr = corr["p_rank1"] if corr else None
    p1 = p1_corr if p1_corr is not None else p1_ind
    rho_c = corr.get("rho") if corr else None
    # ρ는 측정값이다 (independence.measure → pipeline). 측정이 불가능했던 런에서는
    # 같은 자리에 대체값이 들어오므로, 라벨이 '측정'인지 '가정'인지를 밝혀야 한다.
    # rho_source가 아예 없는 런은 측정으로 간주하지 않는다 — 모르는 것은 가정이다.
    rho_measured = bool(corr and corr.get("rho_source") == "measured")
    rho_kind = "측정" if rho_measured else "가정"
    if isinstance(rho_c, (int, float)):
        rho_txt = f"{rho_kind} ρ={rho_c:g}"       # 문장 안에 들어가는 형태
        rho_lab = f"{rho_kind} 상관 ρ={rho_c:g}"  # 범례·열 머리에 들어가는 형태
    else:
        rho_txt = rho_lab = "상관 반영"
    rho_caveat = ("" if rho_measured else
                  " 이 ρ는 분야 간 상관을 측정할 수 없어 쓰인 가정값이며, 측정 결과가 아니다.")
    # P(beats baseline) is a ranking probability too — same rule applies.
    pbb_corr = corr.get("p_beats_baseline") if corr else None
    pbb = pbb_corr if pbb_corr is not None else mc["p_beats_baseline"]
    disagreement = res["point_estimate"]["disagreement"]
    sens = res["sensitivity"]["p_rank1_weights_only"]
    crit_scores = np.array(res["point_estimate"]["criterion_scores"])
    criteria = res["criteria"]
    best = int(np.argmax(p1))

    parts: list[str] = []
    add = parts.append

    # ---------- 요약: hero + hairline-separated stat band (no boxes) ----------
    warn = max(disagreement) > 0.12
    # Honest headline: correlated (lower) figure leads, independent in parens.
    if p1_corr is not None:
        head_prob = (f"이 옵션 집합 내 1위 확률 {_pct(p1_corr[best])}"
                     f"~{_pct(p1_ind[best])} (분야 간 {rho_lab}~독립 가정 ρ=0)")
    else:
        head_prob = f"이 옵션 집합 내 1위 확률 {_pct(p1_ind[best])} (독립 가정)"
    add(_sec("요약", f"""<div class="hero">
        <div class="lbl">시뮬레이션 1위 확률이 가장 높은 옵션</div>
        <div class="opt">{titles[options[best]]}</div>
        <div class="prob">{head_prob} · 기준안 대비 우위 {_pct(pbb[best])}</div>
      </div>
      <div class="stats">
        <div class="stat"><div class="lbl">몬테카를로 표본</div><div class="val">{mc['n_draws']:,}</div>
          <div class="sub">점수·기준가중·분야가중 동시 샘플링</div></div>
        <div class="stat"><div class="lbl">참여 분야</div><div class="val">{len(res['disciplines'])} / 20</div>
          <div class="sub">선별 사유는 아래 근거 절 참조</div></div>
        <div class="stat"><div class="lbl">분야 간 불일치 최대</div><div class="val">{max(disagreement):.3f}</div>
          <div class="sub{' warn' if warn else ''}">{'⚠ 0.12 초과 — 결론이 분야 간 의견 차이 위에 서 있음' if warn else '허용 범위 — 분야 간 의견 차이 크지 않음'}</div></div>
      </div>"""))

    # ---------- 독립성 계측: 차트보다 앞선다 ----------
    # 제품의 전제("여러 학문이 독립적으로 분석한다")가 실제로 성립했는지는
    # 측정된 사실이고, 그 사실이 아래 모든 확률의 의미를 바꾼다. 그래서 요약
    # 바로 다음, 어떤 차트보다 앞에 놓는다. 계측이 없는 예전 런에서는
    # _independence_sections()가 빈 목록을 주므로 절 자체가 생기지 않는다.
    parts.extend(_independence_sections(res.get("independence")))
    # 학문 간 상호작용(텐서 수축). 미학습이면 정직하게 '항등'을 밝히고, 학습됐으면
    # 조정치와 근거 사례를 보인다. interaction 키가 없는 예전 런에서는 빈 목록.
    parts.extend(_interaction_section(res.get("interaction")))

    # ---------- chart: expected utility with p5-p95 interval ----------
    # Left gutter matches the heatmap's (216) so every chart's category
    # column starts on the same vertical line as the surrounding text.
    W, LX = 760, 216
    rows = []
    span_lo = min(pct["p5"]) - 0.04
    span_hi = max(pct["p95"]) + 0.04

    def x(v: float) -> float:
        return LX + (v - span_lo) / (span_hi - span_lo) * (W - LX - 60)

    grid = "".join(
        f'<line x1="{x(g):.1f}" y1="8" x2="{x(g):.1f}" y2="{30 * len(options) + 10}" class="grid"/>'
        f'<text x="{x(g):.1f}" y="{30 * len(options) + 26}" class="tick" text-anchor="middle">{g:.2f}</text>'
        for g in np.arange(np.ceil(span_lo * 10) / 10, span_hi, 0.1))
    for i, o in enumerate(options):
        y = 24 + 30 * i
        c = colors[o][0]
        rows.append(
            f'<text x="{LX - 10}" y="{y + 4}" class="axl" text-anchor="end" '
            f'data-tip="{titles[o]}">{_trunc(titles[o])}</text>'
            f'<line x1="{x(pct["p5"][i]):.1f}" y1="{y}" x2="{x(pct["p95"][i]):.1f}" y2="{y}" '
            f'stroke="{c}" stroke-width="2" stroke-linecap="round" opacity="0.45"/>'
            f'<circle cx="{x(eu[i]):.1f}" cy="{y}" r="5" fill="{c}" class="ring" '
            f'data-tip="{titles[o]} — 기대효용 {eu[i]:.3f} · 90% 구간 [{pct["p5"][i]:.3f}, {pct["p95"][i]:.3f}]"/>'
            f'<text x="{x(pct["p95"][i]) + 8:.1f}" y="{y + 4}" class="dl">{eu[i]:.3f}</text>')
    # The percentiles are the independent-sampling (rho=0) run's — say so.
    eu_note = ("점 = 기대효용, 선 = p5–p95 구간. 구간이 겹치면 순위는 확정이 아니다. "
               "이 구간은 <b>독립 가정(ρ=0)</b> 시뮬레이션의 percentile이다")
    eu_note += (f" — 분야 간 상관({rho_txt})을 반영하면 실제 구간은 이보다 넓다. "
                "기대효용(점)은 상관 여부와 무관하게 같다."
                if p1_corr is not None else ".")
    add(_sec("기대효용", f"""<h2>옵션별 기대효용 (몬테카를로, 90% 구간)</h2>
      <p class="note">{eu_note}</p>
      <div class="fig"><svg viewBox="0 0 {W} {30 * len(options) + 34}" role="img">{grid}{''.join(rows)}</svg></div>"""))

    # ---------- chart: P(rank1) — 상관 반영 vs 독립 가정 (+ 가중치만 민감도) ----------
    # Shade encodes the assumption, not the quantity: same hue per option, one
    # bar per sampling assumption, each one labelled. Older runs (no correlated
    # block) keep the original two-bar 전체 불확실성 / 가중치만 pattern.
    if p1_corr is not None:
        series = [(p1_corr, rho_lab, "1"),
                  (p1_ind, "독립 가정 (ρ=0)", "0.5"),
                  (sens, "가중치만 민감도", "0.22")]
    else:
        series = [(p1_ind, "전체 불확실성 (독립 가정)", "1"),
                  (sens, "가중치만 민감도", "0.45")]
    rows = []
    bw, gap, g2 = 11, 2, 26
    grp = len(series) * bw + (len(series) - 1) * gap
    for i, o in enumerate(options):
        y0 = 18 + i * (grp + g2)
        c = colors[o][0]
        for j, (vals, lab, op) in enumerate(series):
            v = vals[i]
            y = y0 + j * (bw + gap)
            wpx = v * (W - LX - 90)
            rows.append(
                f'<rect x="{LX}" y="{y}" width="{max(wpx, 0.1):.1f}" height="{bw}" fill="{c}" '
                f'opacity="{op}" rx="0" data-tip="{titles[o]} — {lab}: P(1위) {_pct(v)}"/>'
                f'<rect x="{LX + max(wpx - 4, 0):.1f}" y="{y}" width="4" height="{bw}" fill="{c}" opacity="{op}" rx="3"/>'
                f'<text x="{LX + wpx + 8:.1f}" y="{y + bw - 2}" class="dl">{_pct(v)}</text>')
        rows.append(f'<text x="{LX - 10}" y="{y0 + grp / 2 + 4:.1f}" class="axl" text-anchor="end" '
                    f'data-tip="{titles[o]}">{_trunc(titles[o])}</text>')
    H = 18 + len(options) * (grp + g2)
    shade_key = "".join(
        f'<span class="key"><span class="sw" style="background:var(--ink2);opacity:{op}"></span>{lab}</span>'
        for _, lab, op in series)
    if p1_corr is not None:
        p1_note = (f"진한 막대 = 분야 간 점수 상관({rho_txt})을 반영한 1위 확률 — <b>이 리포트가 "
                   f"'1위 확률'로 제시하는 값</b>. 중간 막대 = 모든 셀을 독립으로 가정했을 때의 값으로, "
                   f"상관을 무시하면 불확실성이 평균으로 상쇄돼 과신이 된다. 두 막대의 격차가 곧 독립 "
                   f"가정이 숨기는 과신의 크기다. 옅은 막대 = 기준 가중치 섭동만 반영한 민감도 분석 "
                   f"(가중치 선택에 결론이 휘둘리는지 보는 별개 질문).{rho_caveat}")
    else:
        p1_note = ("진한 막대 = 점수·가중치 전체 불확실성 반영(몬테카를로, 셀 독립 가정). 옅은 막대 = 기준 "
                   "가중치 섭동만 반영한 민감도 분석. 두 값이 비슷하면 결론이 가중치 선택에 휘둘리지 "
                   "않는다는 뜻. (이 런에는 분야 간 상관 반영 시뮬레이션이 없어 독립 가정 값만 표시된다.)")
    add(_sec("1위 확률", f"""<h2>1위 확률 P(rank 1)</h2>
      <p class="note">{p1_note}</p>
      <div class="legend">{shade_key}</div>
      <div class="fig"><svg viewBox="0 0 {W} {H}" role="img">{''.join(rows)}</svg></div>"""))

    # ---------- heatmap: option × criterion ----------
    # Wider cells / narrower label gutter than the bar charts: the 11px column
    # labels need the room now that ticks sit on the type scale (no clipping).
    cell_w, cell_h = 62, 30
    hx0 = 216
    rows = [f'<text x="{hx0 + j * (cell_w + 2) + cell_w / 2}" y="14" class="tick" text-anchor="middle">'
            f'{CRIT_KO.get(c, c)}</text>' for j, c in enumerate(criteria)]
    for i, o in enumerate(options):
        y = 24 + i * (cell_h + 2)
        rows.append(f'<text x="{hx0 - 10}" y="{y + cell_h / 2 + 4}" class="axl" text-anchor="end" '
                    f'data-tip="{titles[o]}">{_trunc(titles[o])}</text>')
        for j, c in enumerate(criteria):
            v = crit_scores[i][j]
            bg = _seq_color(v)
            rows.append(
                f'<rect x="{hx0 + j * (cell_w + 2)}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{bg}" rx="0" '
                f'data-tip="{titles[o]} × {CRIT_KO.get(c, c)}: {v:.2f}"/>'
                f'<text x="{hx0 + j * (cell_w + 2) + cell_w / 2}" y="{y + cell_h / 2 + 4}" '
                f'class="cell" fill="{_ink_for(bg)}" text-anchor="middle">{v:.2f}</text>')
    H = 24 + len(options) * (cell_h + 2) + 12
    add(_sec("점수 텐서", f"""<h2>옵션 × 기준 점수 텐서 (분야 가중 평균)</h2>
      <p class="note">각 셀은 참여 분야의 relevance×confidence 가중 평균. 진할수록 유리. 0~1 단일 색상 램프.</p>
      <div class="fig"><svg viewBox="0 0 {max(W, hx0 + len(criteria) * (cell_w + 2) + 20)} {H}" role="img">{''.join(rows)}</svg></div>"""))

    # ---------- distribution (KDE-lite via histogram smoothing) ----------
    S = d["samples"]
    lo, hi = S.min() - 0.02, S.max() + 0.02
    bins = np.linspace(lo, hi, 60)
    PH, PW = 220, W
    paths, labels = [], []
    for i, o in enumerate(options):
        h, edges = np.histogram(S[:, i], bins=bins, density=True)
        h = np.convolve(h, np.ones(5) / 5, mode="same")
        mx = h.max() if h.max() > 0 else 1
        pts = " ".join(
            f"{LX + (edges[k] - lo) / (hi - lo) * (PW - LX - 40):.1f},{PH - 24 - h[k] / mx * 0 - h[k] * (PH - 44) / (mx * 1.0):.1f}"
            for k in range(len(h)))
        c = colors[o][0]
        paths.append(f'<polyline points="{pts}" fill="none" stroke="{c}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
    # Direct labels intentionally omitted: the non-baseline curves converge,
    # so stacked labels would collide — the legend + table carry identity.
    labels = []
    legend = "".join(
        f'<span class="key"><span class="sw" style="background:{colors[o][0]}"></span>{titles[o]}</span>'
        for o in options)
    add(_sec("분포", f"""<h2>효용 분포 (표본 {len(S):,}개)</h2>
      <div class="legend">{legend}</div>
      <div class="fig"><svg viewBox="0 0 {PW} {PH}" role="img">
        <line x1="{LX}" y1="{PH - 24}" x2="{PW - 30}" y2="{PH - 24}" class="axis"/>
        {''.join(paths)}{''.join(labels)}</svg></div>
      <p class="note after-fig">분포가 넓게 겹칠수록 "1위"는 확률적 진술일 뿐이다. 겹침 자체가 결과의 일부다.</p>"""))

    # ---------- disciplines & provenance ----------
    sel_rows = "".join(
        f"<tr><td>{s['discipline']}</td><td>{float(s['relevance']):.2f}</td><td>{s['why']}</td></tr>"
        for s in d["selection"]["selected"])
    rej_rows = "".join(
        f"<tr><td>{s['discipline']}</td><td>—</td><td>{s['why_not']}</td></tr>"
        for s in d["selection"].get("rejected_examples", []))
    add(_sec("분야 선별", f"""<h2>분야 선별 근거 (20개 풀 → {len(res['disciplines'])}개)</h2>
      <div class="fig"><table><thead><tr><th>분야</th><th>관련도</th><th>사유</th></tr></thead>
      <tbody>{sel_rows}{rej_rows}</tbody></table></div>""", turn=True))

    fiu = []
    for name, a in d["analyses"].items():
        unk = "".join(f"<li>{u}</li>" for u in a.get("unknowns", []))
        fiu.append(f"""<details><summary><b>{name}</b> — {a.get('rationale', '')[:120]}…</summary>
          <div class="cols"><div><h4>사실 {len(a.get('facts', []))}</h4><ul>{''.join(f'<li>{x}</li>' for x in a.get('facts', []))}</ul></div>
          <div><h4>추론 {len(a.get('inferences', []))}</h4><ul>{''.join(f'<li>{x}</li>' for x in a.get('inferences', []))}</ul></div>
          <div><h4>미지 {len(a.get('unknowns', []))}</h4><ul>{unk}</ul></div></div></details>""")
    add(_sec("독립 분석",
             f"<h2>분야별 독립 분석 — 사실 / 추론 / 미지 분리 보존</h2>{''.join(fiu)}"))

    # ---------- outcome forecasts + user selection ----------
    if d["framing"]:
        add(_sec("생성 근거", f"""<h2>옵션 생성 근거 (엔진 생성)</h2>
        <p class="note">{d['framing'].get('framing_note', '')}</p>"""))
    if d["forecasts"]:
        verdicts = (d["outcome"] or {}).get("verdicts", {})
        chosen_opt = (d["decision"] or {}).get("chosen_option")
        fc_by_opt = {f["option_id"]: f["predictions"] for f in d["forecasts"]["forecasts"]}
        V_KO = {"true": "적중", "false": "빗나감", "partial": "부분 적중",
                "unresolved": "판정 보류"}
        cards = []
        for o in options:
            preds = fc_by_opt.get(o, [])
            rows = []
            for pi, p in enumerate(preds):
                key = f"{o}#{pi}"
                v = verdicts.get(key)
                if v:
                    cell = f"<b>{V_KO.get(v['verdict'], v['verdict'])}</b>"
                    if v.get("actual"):
                        cell += f'<div class="fals">{v["actual"]}</div>'
                elif chosen_opt and o in (chosen_opt, "NO_INTERVENTION"):
                    cell = (f'<span class="vbtns" data-key="{key}">'
                            f'<button class="vb" data-v="true">적중</button>'
                            f'<button class="vb" data-v="partial">부분</button>'
                            f'<button class="vb" data-v="false">빗나감</button></span>')
                else:
                    cell = '<span class="fals">—</span>'
                rows.append(
                    f"""<tr><td>{p['metric']}</td><td>{p['prediction']}</td>
                    <td>{float(p['probability']):.0%}</td><td>{p['horizon_months']}개월</td>
                    <td class="fals">{p['falsified_if']}</td><td>{cell}</td></tr>""")
            rows_html = "".join(rows)
            sw = colors[o][0]
            cards.append(f"""<details {'open' if o == options[best] else ''}>
              <summary><span class="sw" style="background:{sw}"></span> <b>{titles[o]}</b>
              — 예측 {len(preds)}개</summary>
              <div class="fig"><table><thead><tr><th>지표</th><th>예측</th><th>확률</th><th>기한</th>
              <th>반증 조건</th><th>현실 판정</th></tr></thead><tbody>{rows_html}</tbody></table></div></details>""")
        add(_sec("예상치", f"""<h2>옵션별 결과 예상치 (반증 가능한 예측)</h2>
          <p class="note">각 예측은 기한 내에 참/거짓 판정이 가능하도록 작성됐다. 선택한 옵션의 예측에는
          기한이 되면 판정 버튼이 활성화된다 — 빗나감을 기록해도 적중과 똑같이 학습에 기여한다.
          정직한 판정만이 엔진을 개선한다.</p>
          {''.join(cards)}<div id="vmsg" class="note"></div>"""))

    # ---------- learning contribution (the earned reward, if any) ----------
    if d["reward"]:
        r = d["reward"]
        rel_rows = "".join(
            f"<tr><td>{k}</td><td>{v['alignment']:.2f}</td><td>{v['reliability']:.3f}</td>"
            f"<td>{v['delta']:+.4f}</td></tr>"
            for k, v in r.get("reliability_changes", {}).items())
        rel_tbl = (f'<div class="fig"><table><thead><tr><th>분야</th>'
                   f'<th>현실 정합도</th><th>신뢰도</th><th>변화</th></tr></thead>'
                   f'<tbody>{rel_rows}</tbody></table></div>'
                   if rel_rows else "")
        add(_sec("학습 기여", f"""<h2>학습 기여 — 이 사례가 엔진을 바꾼 것</h2>
          <div class="stats">
            <div class="stat"><div class="lbl">판정 완료</div>
              <div class="val">{r['resolved_total']} / {r['tracked_total']}</div>
              <div class="sub">추적 중인 예측</div></div>
            <div class="stat"><div class="lbl">보정 점수 (Brier)</div>
              <div class="val">{f"{r['brier']:.3f}" if r['brier'] is not None else "—"}</div>
              <div class="sub">낮을수록 좋음 · 0.25 = 무작위</div></div>
            <div class="stat"><div class="lbl">누적 사례</div>
              <div class="val">{r['library_cases_total']}</div>
              <div class="sub">라이브러리의 판정 완료 사례</div></div>
          </div>
          <p class="note after-fig">{r['contribution']}</p>
          <p class="note">{r['calibration_note']}</p>{rel_tbl}"""))

    # 선택 UI: API로 서빙될 때 POST /runs/{id}/decision 호출. 정적 파일이면 안내.
    btns = "".join(
        f"""<button class="pick" data-opt="{o}" style="--c:{colors[o][0]}">
        {titles[o]}</button>""" for o in options)
    add(_sec("선택", f"""<h2>선택 — 결정 권한은 사람에게 있다</h2>
      <p class="note">엔진의 확률은 참고 자료다. 아래에서 선택하면 결정이 decision.json에 기록되고,
      이후 현실 결과와 비교된다.</p>
      <div class="picks">{btns}</div><div id="pickmsg" class="note"></div>""",
             sid="decide", turn=True))

    # ---------- table view (accessibility fallback) ----------
    # Both ranking-probability views get their own column, each headed with the
    # assumption it was computed under — never merged into one "P(1위)".
    if p1_corr is not None:
        p1_head = f"<th>P(1위) {rho_txt}</th><th>P(1위) 독립가정 ρ=0</th>"
        pbb_head = f"<th>P(기준안 우위) {rho_txt}</th>"
        tbl_note = (f"P(1위) <b>{rho_txt}</b> = 분야 간 점수 상관을 ρ={rho_c:g}로 반영한 값이고, "
                    f"<b>독립가정 ρ=0</b> = 모든 셀을 독립으로 샘플링한 값이다. 같은 양을 두 가정으로 계산한 "
                    f"결과이며, 리포트 본문·상단 타일의 '1위 확률'과 P(기준안 우위)는 {rho_lab} 값을 쓴다. "
                    f"p5/p50/p95와 기대 후회는 독립 가정(ρ=0) 시뮬레이션 값이다.{rho_caveat} ")
    else:
        p1_head = "<th>P(1위) 독립가정</th>"
        pbb_head = "<th>P(기준안 우위)</th>"
        tbl_note = ("이 런에는 상관 반영 시뮬레이션이 없어 모든 확률이 독립 가정(ρ=0) 값이다 — "
                    "독립 가정은 1위 확률을 과신하는 쪽으로 치우친다는 점을 감안해 읽어야 한다. ")

    def _p1_cells(i: int) -> str:
        if p1_corr is not None:
            return f"<td>{_pct(p1_corr[i])}</td><td>{_pct(p1_ind[i])}</td>"
        return f"<td>{_pct(p1_ind[i])}</td>"

    trs = "".join(
        f"<tr><td>{titles[o]}</td><td>{eu[i]:.3f}</td><td>{pct['p5'][i]:.3f}</td><td>{pct['p50'][i]:.3f}</td>"
        f"<td>{pct['p95'][i]:.3f}</td>{_p1_cells(i)}<td>{_pct(pbb[i])}</td>"
        f"<td>{mc['expected_regret'][i]:.3f}</td><td>{disagreement[i]:.3f}</td></tr>"
        for i, o in enumerate(options))
    add(_sec("수치", f"""<h2>수치 전체 (표)</h2>
      <div class="fig"><table><thead><tr><th>옵션</th><th>기대효용</th><th>p5</th><th>p50</th><th>p95</th>
      {p1_head}{pbb_head}<th>기대 후회</th><th>불일치</th></tr></thead>
      <tbody>{trs}</tbody></table></div>
      <p class="note after-fig">{tbl_note}이 수치는 예측이다. 최종 검증자는 현실이며, 선택·결과는 decision.json /
      outcome.json에 기록되어 다음 판단의 증거가 된다.</p>"""))

    html = _shell(prob, parts)
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


def _shell(prob: dict, parts: list[str]) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ARK-42 — {prob['problem_id']}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Hahmlet:wght@400;500;600;700;800&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css">
<style>
{_tokens_css()}
  * {{ box-sizing:border-box; }}
  /* --- 타입 스케일: 11 / 12.5 / 14 / 15 / 19 / 26. 이 6단계 외 금지 --- */
  body {{ margin:0; font-family:var(--font); background:var(--page); color:var(--ink);
    font-size:14px; font-weight:400; line-height:1.75; word-break:keep-all;
    -webkit-text-size-adjust:100%; }}
  .viz-root {{ padding:44px 0 96px; min-height:100vh; }}
  main {{ max-width:1080px; margin:0 auto; padding:0 24px; }}
  h1 {{ font-size:19px; font-weight:500; letter-spacing:-.015em; line-height:1.4;
    margin:0 0 14px; text-wrap:pretty; }}
  h2 {{ font-size:19px; font-weight:500; letter-spacing:-.015em; line-height:1.4;
    margin:0 0 14px; text-wrap:pretty; }}
  h4 {{ font-size:11px; font-weight:500; letter-spacing:.08em; line-height:1.5;
    color:var(--muted); margin:0 0 8px; }}
  a {{ color:var(--accent); text-underline-offset:3px; text-decoration-thickness:1px; }}
  :focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
  /* --- 비대칭 그리드: 132px 레이블 레일 + 600px 본문, 오른쪽은 間 --- */
  .sheet {{ display:grid; grid-template-columns:1fr; gap:6px 32px; }}
  .rail, .body {{ grid-column:1; min-width:0; }}
  .sheet + .sheet {{ border-top:1px solid var(--grid); margin-top:32px; padding-top:32px; }}
  .sheet + .sheet.turn {{ margin-top:64px; }}
  @media (min-width:900px) {{
    .sheet {{ grid-template-columns:132px minmax(0,600px); gap:0 32px; }}
    .rail {{ grid-column:1; }} .body {{ grid-column:2; }}
  }}
  .rail {{ font-size:11px; font-weight:500; letter-spacing:.08em; line-height:1.5;
    color:var(--muted); padding-top:7px; }}
  .masthead .body {{ border-bottom:1px solid var(--grid); padding-bottom:28px; }}
  .masthead + .sheet {{ border-top:0; margin-top:0; padding-top:44px; }}
  /* --- 표지: 브랜드 마크 + 파이프라인 도식 (COMPONENTS.md 1·2·9) --- */
  .masthead h1 {{ display:flex; align-items:flex-start; gap:12px; }}
  .mark {{ color:var(--ink); flex:0 0 auto; width:22px; height:22px; margin-top:3px; }}
  .masthead h1 > span {{ min-width:0; }}
  /* 좁은 화면에선 세로 목록, ≥900px에선 가로 도식. 도식은 1:1(680px)을 넘겨
     확대하지 않아 라벨이 타입스케일(11 / 12.5)에서 벗어나지 않는다. */
  .flowwrap {{ margin:22px 0 0; }}
  .flow {{ display:none; }}
  .flow-n text {{ font-size:11px; font-weight:500; letter-spacing:.08em;
    fill:var(--muted); text-anchor:middle; }}
  .flow-t text {{ font-size:12.5px; fill:var(--ink2); text-anchor:middle; }}
  .flow-n text:first-child, .flow-t text:first-child {{ text-anchor:start; }}
  .flow-n text:last-child, .flow-t text:last-child {{ text-anchor:end; }}
  .flowlist {{ list-style:none; margin:0; padding:2px 0 2px 18px;
    border-left:1px solid var(--axis); }}
  .flowlist li {{ position:relative; padding:0 0 9px; }}
  .flowlist li:last-child {{ padding-bottom:0; }}
  .flowlist li::before {{ content:""; position:absolute; left:-21.5px; top:7px;
    width:7px; height:7px; border-radius:999px; background:var(--surface);
    border:1.5px solid var(--ink); }}
  .flowlist .n {{ font-size:11px; font-weight:500; letter-spacing:.08em;
    line-height:1.5; color:var(--muted); margin-right:10px; }}
  .flowlist .t {{ font-size:12.5px; line-height:1.7; color:var(--ink2); }}
  .flownote {{ font-size:12.5px; line-height:1.7; color:var(--ink2); margin:12px 0 0; }}
  @media (min-width:900px) {{
    .flowwrap {{ width:min(680px, calc(100vw - 232px)); }}
    .flow {{ display:block; }}
    .flowlist {{ display:none; }}
  }}
  .stmt {{ font-size:15px; letter-spacing:-.005em; line-height:1.6; margin:0 0 12px; }}
  .runline {{ font-size:12.5px; line-height:1.7; color:var(--muted); margin:0; }}
  /* --- 요약: 상자 없이 히어로 + 헤어라인 밴드 --- */
  .hero {{ margin:0 0 32px; }}
  .lbl {{ font-size:11px; font-weight:500; letter-spacing:.08em; line-height:1.5;
    color:var(--muted); }}
  .hero .opt {{ font-family:var(--font-display); font-size:26px; font-weight:500; letter-spacing:-.02em; line-height:1.3;
    margin:10px 0 12px; text-wrap:pretty; }}
  .hero .prob {{ font-size:15px; letter-spacing:-.005em; line-height:1.6; color:var(--ink2); }}
  .stats {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:0 24px;
    border-top:1px solid var(--grid); border-bottom:1px solid var(--grid); padding:16px 0; }}
  .stat .val {{ font-size:19px; font-weight:500; letter-spacing:-.015em; line-height:1.4;
    margin:6px 0 4px; font-variant-numeric:tabular-nums; }}
  .stat .sub {{ font-size:12.5px; line-height:1.7; color:var(--muted); }}
  .stat .sub.warn {{ color:var(--bad); }}
  .note {{ font-size:12.5px; line-height:1.7; color:var(--ink2); margin:0 0 16px; }}
  .note.after-fig {{ margin:16px 0 0; }}
  /* --- 도판: 본문 열보다 넓게 오른쪽 여백으로 흘려보낸다(왼쪽 정렬 유지) --- */
  .fig {{ margin:0; overflow-x:auto; }}
  @media (min-width:900px) {{ .fig {{ width:min(760px, calc(100vw - 232px)); }} }}
  svg {{ width:100%; height:auto; display:block; }}
  .grid {{ stroke:var(--grid); stroke-width:1; }} .axis {{ stroke:var(--axis); stroke-width:1; }}
  .tick {{ font-size:11px; font-weight:500; letter-spacing:.04em; fill:var(--muted); }}
  .axl {{ font-size:12.5px; fill:var(--ink2); }}
  .dl {{ font-size:11px; font-weight:500; fill:var(--ink2); font-variant-numeric:tabular-nums; }}
  .cell {{ font-size:11px; font-weight:500; font-variant-numeric:tabular-nums; }}
  .ring {{ stroke:var(--surface); stroke-width:2; }}
  .legend {{ display:flex; gap:16px; flex-wrap:wrap; font-size:12.5px; line-height:1.7;
    color:var(--ink2); margin:0 0 12px; }}
  .key {{ display:inline-flex; align-items:center; gap:6px; }}
  .sw {{ width:10px; height:10px; border-radius:0; display:inline-block; flex:0 0 auto; }}
  /* --- 표: 세로 괘선 없음, 11px 레이블 헤더 --- */
  table {{ width:100%; border-collapse:collapse; font-size:12.5px; line-height:1.7; }}
  th {{ text-align:left; font-size:11px; font-weight:500; letter-spacing:.08em;
    line-height:1.5; color:var(--muted); border-bottom:1px solid var(--axis);
    padding:8px 10px; white-space:nowrap; }}
  td {{ border-bottom:1px solid var(--grid); padding:8px 10px;
    font-variant-numeric:tabular-nums; vertical-align:top; }}
  th:first-child, td:first-child {{ padding-left:0; }}
  th:last-child, td:last-child {{ padding-right:0; }}
  details {{ border-top:1px solid var(--grid); padding:14px 0; }}
  details:last-of-type {{ border-bottom:1px solid var(--grid); }}
  summary {{ cursor:pointer; font-size:14px; line-height:1.75; }}
  summary:hover {{ color:var(--ink2); }}
  details .fig {{ margin-top:12px; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:24px; font-size:12.5px;
    line-height:1.7; margin-top:14px; }}
  .cols ul {{ margin:0; padding-left:15px; }} .cols li {{ margin-bottom:6px; }}
  .fals {{ color:var(--muted); }}
  /* --- 버튼: 주요는 잉크 채움, 작은 버튼은 11px 레이블, radius 0 --- */
  .vbtns {{ display:inline-flex; gap:6px; }}
  .vb {{ font-family:inherit; font-size:11px; font-weight:500; letter-spacing:.08em;
    line-height:1.5; padding:6px 12px; border-radius:0; cursor:pointer;
    background:transparent; color:var(--ink2); border:1px solid var(--axis);
    transition:background .15s, color .15s, border-color .15s; }}
  .vb:hover {{ background:var(--ink); color:var(--surface); border-color:var(--ink); }}
  .picks {{ display:flex; gap:12px; flex-wrap:wrap; margin:0 0 14px; }}
  .pick {{ font-family:inherit; font-size:14px; font-weight:500; line-height:1.4;
    padding:12px 22px; border:0; border-left:4px solid var(--c); border-radius:0;
    cursor:pointer; background:var(--ink); color:var(--surface);
    transition:opacity .15s; text-align:left; }}
  .pick:hover {{ opacity:.82; }}
  .pick.chosen {{ font-weight:700; }}
  .pick.chosen::after {{ content:" ✓"; }}
  #tip {{ position:fixed; pointer-events:none; background:var(--ink); color:var(--surface);
    font-size:12.5px; line-height:1.7; padding:7px 10px; border-radius:0; opacity:0;
    transition:opacity .15s; max-width:300px; z-index:9; word-break:keep-all; }}
  @media (max-width:700px) {{
    .stats {{ grid-template-columns:1fr; gap:16px; }}
    .cols {{ grid-template-columns:1fr; }}
  }}
</style></head>
<body><div class="viz-root"><main>
<header class="sheet masthead"><div class="rail">리포트</div><div class="body">
<h1>{MARK_SVG}<span>ARK-42 Decision Lab — 분석 리포트</span></h1>
<p class="stmt">{prob['statement']}</p>
<p class="runline">run: {prob['problem_id']} · 이 리포트의 모든 수치는 예측이며, 결정 권한은 사람에게 있다.</p>
<div class="flowwrap">{FLOW_SVG}{FLOW_LIST}
<p class="flownote">{FLOW_NOTE}</p></div>
</div></header>
{''.join(parts)}
<div id="tip"></div>
<script>
const tip = document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el => {{
  el.addEventListener('mousemove', e => {{ tip.textContent = el.dataset.tip;
    tip.style.left = Math.min(e.clientX + 14, innerWidth - 310) + 'px';
    tip.style.top = (e.clientY + 14) + 'px'; tip.style.opacity = 1; }});
  el.addEventListener('mouseleave', () => tip.style.opacity = 0);
}});
const RUN_ID = {json.dumps(prob['problem_id'])};
const msg = document.getElementById('pickmsg');
document.querySelectorAll('.pick').forEach(btn => btn.addEventListener('click', async () => {{
  if (location.protocol === 'file:') {{
    msg.textContent = '정적 파일로 열려 있어 기록할 수 없습니다. API 서버(uvicorn ark42.api:app)로 열면 선택이 decision.json에 기록됩니다. 선택하신 값: ' + btn.dataset.opt;
    return;
  }}
  try {{
    const r = await fetch(`/runs/${{RUN_ID}}/decision`, {{ method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{option_id: btn.dataset.opt, decided_by: 'report-ui', note: ''}}) }});
    if (r.ok) {{
      document.querySelectorAll('.pick').forEach(b => b.classList.remove('chosen'));
      btn.classList.add('chosen');
      msg.textContent = '기록되었습니다: ' + btn.dataset.opt + ' (decision.json)';
    }} else {{ msg.textContent = '기록 실패: ' + (await r.text()); }}
  }} catch (e) {{ msg.textContent = '기록 실패: ' + e; }}
}}));
const vmsg = document.getElementById('vmsg');
document.querySelectorAll('.vb').forEach(btn => btn.addEventListener('click', async () => {{
  const key = btn.closest('.vbtns').dataset.key;
  if (location.protocol === 'file:') {{
    if (vmsg) vmsg.textContent = '정적 파일로 열려 있어 기록할 수 없습니다. API 서버로 열면 판정이 저장·학습됩니다. (' + key + ' → ' + btn.dataset.v + ')';
    return;
  }}
  try {{
    const r = await fetch(`/runs/${{RUN_ID}}/outcome`, {{ method: 'POST',
      headers: {{'content-type': 'application/json'}},
      body: JSON.stringify({{verdicts: [{{key: key, verdict: btn.dataset.v}}], recorded_by: 'report-ui'}}) }});
    const j = await r.json();
    if (r.ok) {{
      const rw = j.reward;
      if (vmsg) vmsg.textContent = `기록 완료 — 판정 ${{rw.resolved_total}}/${{rw.tracked_total}}` +
        (rw.brier !== null ? ` · Brier ${{rw.brier.toFixed(3)}}` : '') + ` · ${{rw.contribution}}`;
      setTimeout(() => location.reload(), 1600);
    }} else {{ if (vmsg) vmsg.textContent = '기록 실패: ' + JSON.stringify(j.error); }}
  }} catch (e) {{ if (vmsg) vmsg.textContent = '기록 실패: ' + e; }}
}}));
</script>
</main></div></body></html>"""
