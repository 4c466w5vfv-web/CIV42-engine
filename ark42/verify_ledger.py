"""Ledger integrity check — `python3 -m ark42.verify_ledger [data_dir]`.

Why this exists: `_read_jsonl` deliberately skips unparsable lines so one
corrupt line can never take the whole service down. That is right for
serving, and dangerous for backups — a truncated copy reads as valid and
its balances are silently wrong. This command makes the corruption loud.

Exit 0 = clean, 1 = problems found (so backup.sh can refuse to keep it).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def verify(data_dir: Path) -> tuple[int, list[str]]:
    problems: list[str] = []
    checked = 0

    for name in ("credits.jsonl", "payments.jsonl"):
        path = data_dir / name
        if not path.exists():
            continue
        raw = path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            problems.append(f"{name}: 파일이 개행으로 끝나지 않음 — "
                            "쓰기 중 복사되었거나 마지막 줄이 잘렸을 수 있음")
        for i, line in enumerate(raw.decode("utf-8", "replace").splitlines(), 1):
            if not line.strip():
                continue
            checked += 1
            try:
                json.loads(line)
            except json.JSONDecodeError as e:
                problems.append(f"{name}:{i} JSON 파싱 실패 ({e.msg}) — "
                                "서비스는 이 줄을 조용히 건너뛰므로 잔액이 틀어집니다")

    # 잔액 음수 검사: 정상 운영에서는 나올 수 없다.
    ledger = data_dir / "credits.jsonl"
    if ledger.exists():
        bal: dict[str, int] = {}
        keys: dict[str, int] = {}
        for line in ledger.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.strip():
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = e.get("user_id", "?")
            bal[uid] = bal.get(uid, 0) + int(e.get("delta", 0))
            k = e.get("idempotency_key")
            if k:
                keys[k] = keys.get(k, 0) + 1
        for uid, v in bal.items():
            if v < 0:
                problems.append(f"credits.jsonl: user {uid} 잔액 {v} (< 0) — "
                                "이중 차감 또는 누락된 적립")
        for k, n in keys.items():
            if n > 1:
                problems.append(f"credits.jsonl: idempotency_key {k!r} 가 {n}회 "
                                "등장 — 멱등성 위반(다중 프로세스 실행 의심)")

    users = data_dir / "users.json"
    if users.exists():
        try:
            u = json.loads(users.read_text())
            checked += len(u)
        except json.JSONDecodeError as e:
            problems.append(f"users.json: 파싱 실패 ({e.msg}) — 계정 전체 손실 위험")

    return checked, problems


def main() -> int:
    d = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if d is None:
        from .paths import data_dir
        d = data_dir()
    if not d.exists():
        print(f"데이터 디렉터리를 찾을 수 없습니다: {d}", file=sys.stderr)
        return 1
    checked, problems = verify(d)
    print(f"검사 대상 {checked}건 · 문제 {len(problems)}건 ({d})")
    for p in problems:
        print(f"  - {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
