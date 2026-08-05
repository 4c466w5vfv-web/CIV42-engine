#!/usr/bin/env python3
"""키가 실제로 동작하는지 최소 비용으로 확인한다.

    python3 tools/check_llm_key.py

토큰 몇 개짜리 호출 한 번만 보냅니다. 백엔드 선택은 운영 코드와 동일한
ark42.backends.build_backend 를 그대로 씁니다 — 여기서 통과하면 실제 run 도
같은 백엔드로 갑니다. 실패하면 원인을 그대로 보여줍니다(키 오류/네트워크/
모델명 오류를 구분할 수 있게).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    from ark42.backends import build_backend
    try:
        b = build_backend()
    except Exception as e:
        print(f"백엔드를 만들지 못했습니다: {e}", file=sys.stderr)
        print("\n환경변수 확인: ANTHROPIC_API_KEY / CAFE24_LLM_API_KEY / "
              "ARK42_PANELS / ARK42_FAILOVER", file=sys.stderr)
        return 2

    pid = getattr(b, "provider_id", type(b).__name__)
    url = getattr(b, "base_url", "(anthropic)")
    print(f"백엔드 : {pid}")
    print(f"엔드포인트: {url}")
    print("최소 호출 1회 보냅니다…")
    try:
        out = b.complete("ping", 'Reply with JSON only.',
                         'Return exactly {"ok":true}. No other text.')
    except Exception as e:
        print(f"\n실패: {type(e).__name__}: {e}", file=sys.stderr)
        print("\n흔한 원인:\n"
              "  Tunnel connection failed → 프록시/방화벽이 막음 (키 문제 아님)\n"
              "  401/403 → 키가 틀렸거나 만료\n"
              "  404     → 모델 id 가 그 게이트웨이에 없음 "
              "(ARK42_CAFE24_MODEL 확인)\n"
              "  타임아웃 → 서버에서 외부 HTTPS 아웃바운드가 막힘",
              file=sys.stderr)
        return 1

    print(f"응답: {out[:120]}")
    u = getattr(b, "usage", [])
    if u:
        ti = sum(x.get("input_tokens", 0) for x in u)
        to = sum(x.get("output_tokens", 0) for x in u)
        print(f"토큰: 입력 {ti} / 출력 {to}")
        print("\n이 숫자로 1건당 원가를 역산할 수 있습니다. 게이트웨이의 "
              "모델별 단가를 확인해 아래를 설정하면 cost.json 의 원가·마진이 "
              "가정값이 아니라 실제값이 됩니다:")
        print("  ARK42_PRICE_IN_KRW_PER_MTOK=<입력 100만 토큰당 원>")
        print("  ARK42_PRICE_OUT_KRW_PER_MTOK=<출력 100만 토큰당 원>")
    else:
        print("사용량을 보고하지 않는 백엔드입니다 (recorded 등).")
    print("\nOK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
