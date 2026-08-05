from __future__ import annotations
"""The fixed pool of 20 disciplines the selector chooses from.

The selector must pick ONLY the disciplines whose core explanatory
mechanisms bear on the problem — selection is part of the audit trail.
"""

DISCIPLINES: dict[str, str] = {
    "economics": "가격, 유인, 시장 구조, 거래비용, 수요·공급",
    "behavioral_psychology": "인지 편향, 의사결정 휴리스틱, 채택 행동, 신뢰 형성",
    "sociology": "집단 규범, 확산, 사회적 자본, 조직 간 관계",
    "political_science": "권력 구조, 공공 정책, 정부 자원 배분, 규제 정치",
    "law_regulation": "계약, 책임, 지식재산, 규제 준수, 데이터 법제",
    "game_theory": "전략적 상호작용, 경쟁 대응, 협상, 커미트먼트",
    "statistics_data": "추정, 불확실성 정량화, 표본 편향, 검정력",
    "systems_complexity": "피드백 루프, 창발, 경로 의존성, 임계점",
    "operations_management": "프로세스 설계, 병목, 용량, 품질, 공급망",
    "finance_accounting": "현금흐름, 원가 구조, 자본 조달, 단위 경제성",
    "marketing": "세분화, 포지셔닝, 채널, 고객 획득 비용",
    "engineering_software": "아키텍처, 확장성, 기술 부채, 보안, 운영 신뢰성",
    "design_hci": "사용성, 인지 부하, 신뢰 UI, 정보 시각화",
    "ethics": "책임 귀속, 공정성, 투명성, 이해상충",
    "history": "유사 사례의 전개, 기술 채택사, 제도 변화의 선례",
    "anthropology": "문화적 맥락, 현장 관행, 의미 체계",
    "ecology_environment": "자원 제약, 외부효과, 지속가능성",
    "education_learning": "역량 이전, 학습 곡선, 온보딩",
    "communication_media": "메시지 전달, 평판, 담론 형성",
    "organizational_theory": "조직 구조, 대리인 문제, 거버넌스, 제도화",
}

assert len(DISCIPLINES) == 20, "discipline pool must be exactly 20"
