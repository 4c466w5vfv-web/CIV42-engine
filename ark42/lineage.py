"""Lineage primitives: content hashes, deterministic IDs, versions.

Every record in the learning loop carries an ID derived from its content
and links to its upstream record, so the chain
  run_b → weight_version → weight_update → score → outcome
        → prediction_snapshot → decision → run_a → source_hash
is reconstructible from files alone.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path


def atomic_write_text(path, text: str) -> None:
    """Write via a temp file in the same dir then os.replace — readers never
    see a half-written file, and a crash mid-write leaves the old file intact.
    Use for every JSON record in the learning loop / run store."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)

ENGINE_VERSION = "0.2.0"
SCORING_RULE = "brier_v1"
UPDATE_RULE = "beta_v1"
MC_SEED = 42            # fixed seeds; recorded in every prediction snapshot
SENS_SEED = 7


def canonical_hash(obj) -> str:
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_id(prefix: str, *parts) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{h[:12]}"


def utcnow() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
