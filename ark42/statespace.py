"""42차원 문명 상태공간 — 좌표계 계층 (디지털 트윈의 전 단계).

정직한 명칭 규율 (2026-08-04 비전 문서)
--------------------------------------
- 현재: **42차원 상태공간 모델** — 상태를 표현할 좌표계와 갱신 규약.
- 중기: 의사결정 시뮬레이션 모델 — 결정이 상태를 어떻게 바꾸는지 추적.
- 장기: 문명 디지털 트윈 — 아래 요건을 전부 갖춘 뒤에만 그렇게 부른다.

디지털 트윈 요건 체크리스트 (전부 충족 전에는 트윈이라 부르지 않는다):
  [x] 42개 변수의 명확한 정의            ← 이 모듈 (REGISTRY)
  [x] 각 변수의 측정 방법                ← 이 모듈 (measurement 필드)
  [ ] 실제 데이터 입력                   ← record_observation (수집은 운영 몫)
  [x] 시간에 따른 상태 갱신              ← update_state / 이력 append
  [x] 예측과 현실 비교                   ← 엔진 기존 (forecasts→outcomes, Brier)
  [x] 모델 보정 기록                     ← 엔진 기존 (learning.reliability) +
                                          이 모듈의 calibration 레코드
  [ ] 시나리오별 결과 검증(외부 재현)    ← 공개 채점 데이터가 쌓인 뒤

엔진 스파인과의 관계: 이 모듈은 스파인(옵션→시뮬레이션→결정→실험→피드백)을
대체하지 않는다. 결정이 만지는 축을 태깅하고, 실험 결과가 상태 이력으로
쌓이는 **피드백 좌표계**다. 파이프라인 결합은 opt-in이며 기본 동작을 바꾸지
않는다.

저장: 다른 모듈과 같은 규약 — 로컬은 파일, SUPABASE_* 설정 시 kv 미러
(ark42.supa). 상태 이력은 append-only.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

#: 7개 도메인 × 6개 변수 = 42축. 각 변수: 정의 1문장 + 측정 방법 1문장.
#: 측정은 정직하게 — 자기보고는 자기보고라고 쓴다. 지어낸 정밀도 금지.
_DOMAINS = {
    "resource": "자원·경제",
    "relation": "관계·사회자본",
    "vitality": "건강·에너지",
    "knowledge": "지식·기술",
    "attention": "시간·주의",
    "resilience": "리스크·회복력",
    "meaning": "의미·방향",
}

REGISTRY: list[dict] = [
    # ---- resource 자원·경제 -------------------------------------------
    dict(id="resource.liquidity", name="현금 유동성",
         definition="즉시 동원 가능한 현금이 월 고정지출을 몇 달 버티는가.",
         measurement="가용현금 ÷ 월 고정지출 (개월 수, 장부 기반).", unit="개월"),
    dict(id="resource.income_diversity", name="수입 다변성",
         definition="수입이 단일 원천에 얼마나 의존하는가.",
         measurement="최대 단일 수입원의 비중(%). 낮을수록 다변화.", unit="%"),
    dict(id="resource.fixed_burden", name="고정비 부담",
         definition="수입 중 고정지출이 차지하는 비율.",
         measurement="월 고정지출 ÷ 월 평균수입 (%).", unit="%"),
    dict(id="resource.capital_access", name="자본 접근성",
         definition="필요 시 외부 자본(대출·투자·후원)에 닿을 수 있는 정도.",
         measurement="30일 내 조달 가능 금액의 보수적 추정(자기평가 근거 명시).",
         unit="KRW"),
    dict(id="resource.asset_flexibility", name="자산 탄력성",
         definition="자산을 손실 없이 현금화할 수 있는 정도.",
         measurement="7일 내 90% 이상 가치로 처분 가능한 자산 비중(%).", unit="%"),
    dict(id="resource.debt_health", name="부채 건전성",
         definition="부채 상환이 현금흐름을 압박하는 정도.",
         measurement="월 상환액 ÷ 월 수입 (%). 연체 여부 별도 기록.", unit="%"),
    # ---- relation 관계·사회자본 ---------------------------------------
    dict(id="relation.trust_network", name="신뢰 네트워크",
         definition="중요한 일을 맡기거나 맡아줄 수 있는 사람의 수.",
         measurement="실제로 부탁해본/받아본 이력이 있는 인원 수(명부 기반).",
         unit="명"),
    dict(id="relation.interdependence", name="상호의존 건강도",
         definition="핵심 관계가 일방 의존이 아니라 상호적인가.",
         measurement="핵심 관계 5개의 주고/받음 균형 자기평가(1-5), 근거 메모.",
         unit="1-5"),
    dict(id="relation.reputation", name="평판 자본",
         definition="제3자가 나/조직을 신뢰할 근거가 공개되어 있는 정도.",
         measurement="검증 가능한 공개 실적물 수(리포트·기록·추천).", unit="건"),
    dict(id="relation.leverage", name="협상 레버리지",
         definition="관계에서 대안(BATNA)을 가진 정도.",
         measurement="핵심 협상 상대별 실행 가능한 대안 수.", unit="개"),
    dict(id="relation.belonging", name="커뮤니티 소속",
         definition="정기적으로 상호작용하는 공동체의 수와 깊이.",
         measurement="월 1회 이상 기여하는 공동체 수.", unit="개"),
    dict(id="relation.weak_ties", name="약한 연결 다양성",
         definition="다른 분야·집단으로 통하는 다리 연결의 폭.",
         measurement="최근 90일 새로운 분야 접촉 수(기록 기반).", unit="건"),
    # ---- vitality 건강·에너지 -----------------------------------------
    dict(id="vitality.energy", name="신체 에너지",
         definition="하루 중 고품질 작업이 가능한 에너지 총량.",
         measurement="주간 평균 자기보고(1-5) — 자기보고임을 명시.", unit="1-5"),
    dict(id="vitality.sleep", name="수면 회복",
         definition="수면이 회복 기능을 하고 있는가.",
         measurement="주간 평균 수면시간 + 기상 시 회복감(1-5).", unit="시간/1-5"),
    dict(id="vitality.stress_load", name="스트레스 부하",
         definition="만성 스트레스 원천의 수와 강도.",
         measurement="지속 2주 이상 스트레스 원천 수 + 강도 자기보고.", unit="건/1-5"),
    dict(id="vitality.focus_span", name="주의 지속력",
         definition="끊기지 않는 깊은 집중이 가능한 시간.",
         measurement="실측 최장 집중 블록(분), 주간 기록.", unit="분"),
    dict(id="vitality.mood_stability", name="정서 안정성",
         definition="정서 변동이 판단을 흔드는 빈도.",
         measurement="주간 자기보고(1-5) + 결정 연기/번복 횟수.", unit="1-5"),
    dict(id="vitality.health_risk", name="건강 리스크 노출",
         definition="방치된 건강 신호와 검진 공백.",
         measurement="미조치 이상 신호 수 + 최근 검진 경과 개월.", unit="건/개월"),
    # ---- knowledge 지식·기술 ------------------------------------------
    dict(id="knowledge.core_depth", name="핵심 역량 깊이",
         definition="시장이 값을 지불하는 핵심 기술의 깊이.",
         measurement="유료 검증 이력(판매·고용·인용) 있는 역량 수.", unit="개"),
    dict(id="knowledge.learning_rate", name="학습 속도",
         definition="새 영역을 실전 수준까지 끌어올리는 속도.",
         measurement="최근 12개월 실전 투입까지 간 신규 역량 수.", unit="개"),
    dict(id="knowledge.stack_currency", name="기술 스택 현대성",
         definition="도구·방법이 현재 생태계 대비 낡지 않았는가.",
         measurement="핵심 도구의 최신 안정 버전 대비 지연(세대).", unit="세대"),
    dict(id="knowledge.documentation", name="암묵지 문서화",
         definition="머릿속 지식이 남에게 전달 가능한 형태로 존재하는가.",
         measurement="핵심 프로세스 중 문서화된 비율(%).", unit="%"),
    dict(id="knowledge.info_quality", name="정보 접근 품질",
         definition="판단에 쓰는 정보원의 신뢰도와 다양성.",
         measurement="1차 출처 비율 + 상충 관점 확보 여부(체크리스트).", unit="%"),
    dict(id="knowledge.experiment_capacity", name="실험 역량",
         definition="가설을 싸고 빠르게 검증하는 능력.",
         measurement="최근 90일 완료한 최소 실험 수(판정 기준 있는 것만).",
         unit="건"),
    # ---- attention 시간·주의 ------------------------------------------
    dict(id="attention.free_hours", name="가처분 시간",
         definition="의무가 아닌 곳에 쓸 수 있는 주간 시간.",
         measurement="주간 실측(캘린더 기반).", unit="시간/주"),
    dict(id="attention.autonomy", name="시간 자율성",
         definition="내 시간표를 내가 정하는 정도.",
         measurement="타인이 고정한 시간 비율(%). 낮을수록 자율.", unit="%"),
    dict(id="attention.deep_blocks", name="집중 블록 가용성",
         definition="90분 이상 방해 없는 블록을 확보할 수 있는가.",
         measurement="주간 확보된 90분+ 블록 수.", unit="개/주"),
    dict(id="attention.commitment_density", name="약속 밀도",
         definition="이행해야 할 약속이 처리 용량을 초과하는 정도.",
         measurement="미이행 약속 수 ÷ 주간 처리 용량(자기추정).", unit="비율"),
    dict(id="attention.rhythm", name="리듬 안정성",
         definition="작업·휴식 리듬이 예측 가능한가.",
         measurement="주간 계획 대비 실행 일치율(%).", unit="%"),
    dict(id="attention.priority_clarity", name="우선순위 명료성",
         definition="지금 가장 중요한 것 하나를 즉답할 수 있는가.",
         measurement="주간 점검: 즉답 가능 여부 + 상위 3개의 안정성.", unit="예/아니오"),
    # ---- resilience 리스크·회복력 -------------------------------------
    dict(id="resilience.reversibility", name="가역성 여유",
         definition="현재 진행 중인 약속들을 되돌릴 수 있는 정도.",
         measurement="진행 중 약속 중 30일 내 철회 가능 비율(%).", unit="%"),
    dict(id="resilience.buffer", name="완충 자원",
         definition="충격 흡수용으로 비워둔 자원(돈·시간·에너지).",
         measurement="용도 미지정 예비 자원의 명시적 목록과 규모.", unit="목록"),
    dict(id="resilience.single_points", name="단일 실패점",
         definition="하나가 무너지면 전체가 무너지는 지점의 수.",
         measurement="대체 불가능한 의존(사람·도구·수입원·건강) 수.", unit="개"),
    dict(id="resilience.coverage", name="보험 커버리지",
         definition="치명적 손실이 이전 가능한가.",
         measurement="주요 리스크별 보험/계약 커버 여부 체크리스트.", unit="비율"),
    dict(id="resilience.adaptation_history", name="적응 이력",
         definition="과거 충격에서 실제로 회복해 본 경험.",
         measurement="회복 완료된 충격 사례 수와 회복 기간(기록).", unit="건"),
    dict(id="resilience.stress_testing", name="스트레스 테스트 빈도",
         definition="최악 시나리오를 미리 검토하는 습관.",
         measurement="분기당 명시적 최악 시나리오 점검 횟수.", unit="회/분기"),
    # ---- meaning 의미·방향 --------------------------------------------
    dict(id="meaning.goal_clarity", name="목표 명료성",
         definition="1년 뒤 도달점을 측정 가능하게 말할 수 있는가.",
         measurement="측정 가능한 형태의 목표 문장 존재 여부 + 최근 갱신일.",
         unit="예/아니오"),
    dict(id="meaning.value_alignment", name="가치 일치도",
         definition="시간을 쓰는 곳과 중요하다고 말하는 것의 일치.",
         measurement="상위 3개 가치와 주간 시간 배분의 대응 점검.", unit="1-5"),
    dict(id="meaning.motivation_durability", name="동기 지속성",
         definition="외부 보상이 없어도 지속되는 동기의 존재.",
         measurement="보상 공백기(무반응 기간)에 지속한 작업 이력.", unit="주"),
    dict(id="meaning.narrative_coherence", name="서사 일관성",
         definition="과거-현재-다음 계획이 하나의 이야기로 이어지는가.",
         measurement="제3자에게 3분 내 일관되게 설명 가능한지(녹취 점검).",
         unit="예/아니오"),
    dict(id="meaning.contribution", name="사회적 기여감",
         definition="내 산출이 남에게 닿는다는 검증 가능한 신호.",
         measurement="외부 수혜 증거(사용·감사·기부 영수증) 건수.", unit="건"),
    dict(id="meaning.horizon", name="장기 지평선",
         definition="며칠이 아니라 몇 년 단위로 사고하는 정도.",
         measurement="현재 진행 결정 중 지평선 1년+ 비율(%).", unit="%"),
]

for _v in REGISTRY:                       # 도메인 파생 필드 주입
    _v["domain"] = _v["id"].split(".")[0]
    _v["domain_ko"] = _DOMAINS[_v["domain"]]
    _v["levels"] = ["individual", "organization"]


def validate_registry() -> None:
    """좌표계 무결성 — 42축, 유일 id, 필수 필드. import 시가 아니라 호출
    시 검증한다(테스트와 CLI가 부른다)."""
    ids = [v["id"] for v in REGISTRY]
    assert len(REGISTRY) == 42, f"expected 42 axes, got {len(REGISTRY)}"
    assert len(set(ids)) == 42, "duplicate axis ids"
    domains = {v["domain"] for v in REGISTRY}
    assert domains == set(_DOMAINS), f"unknown domains: {domains - set(_DOMAINS)}"
    for v in REGISTRY:
        for field in ("name", "definition", "measurement", "unit"):
            assert v.get(field), f"{v['id']}: missing {field}"
    counts = {d: sum(1 for v in REGISTRY if v["domain"] == d) for d in _DOMAINS}
    assert all(c == 6 for c in counts.values()), f"domain imbalance: {counts}"


# ---------------------------------------------------------------- state I/O

def _dir() -> Path:
    from .paths import data_dir
    d = data_dir() / "statespace"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _kv_key(subject_id: str) -> str:
    return f"statespace:{subject_id}"


def load_history(subject_id: str) -> list[dict]:
    """대상의 상태 이력(시간 오름차순). 없으면 빈 목록."""
    from . import supa
    if supa.enabled():
        return supa.kv_get(_kv_key(subject_id), []) or []
    p = _dir() / f"{subject_id}.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []


def record_observation(subject_id: str, values: dict, source: str,
                       note: str = "") -> dict:
    """관측 1건을 이력에 append. values = {axis_id: number|str}.
    미지의 축은 거부한다 — 좌표계 밖의 관측은 좌표계 오류이므로 조용히
    받지 않는다. 반환: 저장된 스냅샷(관측 시각·이전 값 대비 델타 포함)."""
    known = {v["id"] for v in REGISTRY}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"unknown axes: {sorted(unknown)}")
    if not values:
        raise ValueError("empty observation")
    history = load_history(subject_id)
    prev = history[-1]["values"] if history else {}
    snapshot = {
        "t": time.time(),
        "source": source,                  # 예: "self-report", "ledger", "run:xyz"
        "note": note,
        "values": {**prev, **values},      # 최신 상태 = 이전 상태 + 이번 관측
        "observed": sorted(values),        # 이번에 실제로 관측된 축만 구분
        "delta": {k: {"from": prev.get(k), "to": values[k]}
                  for k in values if prev.get(k) != values[k]},
    }
    history.append(snapshot)
    from . import supa
    if supa.enabled():
        supa.kv_put(_kv_key(subject_id), history)
    else:
        from .lineage import atomic_write_text
        atomic_write_text(_dir() / f"{subject_id}.json",
                          json.dumps(history, ensure_ascii=False, indent=1))
    return snapshot


def coverage(subject_id: str) -> dict:
    """좌표계 대비 실제 관측된 축의 비율 — '데이터 입력' 요건의 정직한
    진행률. 42축 전부가 채워지기 전에는 모델이 아니라 좌표계다."""
    history = load_history(subject_id)
    seen = set()
    for snap in history:
        seen.update(snap.get("observed", []))
    return {"observed_axes": len(seen), "total_axes": 42,
            "coverage": len(seen) / 42,
            "missing": sorted({v["id"] for v in REGISTRY} - seen)}
