"""저장 위치의 단일 출처.

왜 별도 모듈인가
----------------
`data_dir()` 이 auth.py 와 billing.py 에 각각 있었고, 엔진 모듈(statespace,
verify_ledger)이 그 중 하나를 골라 임포트했다. 그 결과 **엔진이 웹 계층을
참조하는 역방향 결합**이 생겼다 — 엔진만 떼어 쓰려면 계정·결제 코드까지
끌고 와야 한다는 뜻이다.

경로 결정은 저수준 관심사이므로 여기로 모은다. 이제 엔진은 웹 계층을
전혀 모르고, 웹 계층이 엔진을 쓴다(한 방향).
"""
from __future__ import annotations

import os
from pathlib import Path


def data_dir() -> Path:
    """데이터 루트. env ARK42_DATA_DIR, 없으면 <repo>/data.

    Vercel 같은 서버리스에서는 저장소 트리가 읽기 전용이고 인스턴스마다
    분리되므로 기본값이 /tmp 로 내려간다 — 영속성은 이 경로가 아니라
    ark42.supa 의 미러가 담당한다.
    """
    default = (Path("/tmp/ark42/data") if os.environ.get("VERCEL")
               else Path(__file__).resolve().parent.parent / "data")
    d = Path(os.environ.get("ARK42_DATA_DIR", default))
    d.mkdir(parents=True, exist_ok=True)
    return d
