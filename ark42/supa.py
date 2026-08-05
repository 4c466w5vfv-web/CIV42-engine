"""Supabase key-value persistence for serverless deployments.

WHY THIS EXISTS
---------------
The file-backed stores (users.json, credits.jsonl, runs/<id>/*) assume one
long-lived process with one disk. On Vercel serverless, every request may land
on a different instance with its own ephemeral /tmp, and an instance can be
frozen right after responding — so accounts evaporate, and a run created by
one request 404s when polled by the next (observed in production 2026-08-03).

This module is the ONE seam that fixes that: a single Postgres table
(`kv_store`, key text PK / value jsonb) accessed over Supabase's PostgREST
HTTP API with the service-role key. Callers keep their existing file logic for
local dev and tests; when SUPABASE_URL + SUPABASE_SERVICE_KEY are set,
`enabled()` turns true and the file seams in auth/billing/api mirror through
here instead.

Design constraints, same as the rest of the codebase: stdlib urllib only, no
new dependencies. Transient HTTP failures retry with backoff (mirrors
backends.OpenAIBackend). RLS is ON with no policies — the publishable key can
read nothing; only the service key (server-side env) passes.

Table DDL (run once in the Supabase SQL editor):

    create table if not exists kv_store (
      key text primary key,
      value jsonb not null,
      updated_at timestamptz not null default now()
    );
    alter table kv_store enable row level security;
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

_TABLE = "kv_store"
_TIMEOUT = 15
_RETRIES = 2


def enabled() -> bool:
    return bool(os.environ.get("SUPABASE_URL")
                and os.environ.get("SUPABASE_SERVICE_KEY"))


def _base() -> str:
    return os.environ["SUPABASE_URL"].rstrip("/") + "/rest/v1/" + _TABLE


def _headers() -> dict:
    k = os.environ["SUPABASE_SERVICE_KEY"]
    return {"apikey": k, "Authorization": f"Bearer {k}",
            "Content-Type": "application/json"}


def _send(req: urllib.request.Request) -> bytes:
    delay = 1.0
    for attempt in range(_RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=_TIMEOUT) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503, 504) or attempt == _RETRIES:
                raise
        except urllib.error.URLError:
            if attempt == _RETRIES:
                raise
        time.sleep(delay)
        delay *= 2
    raise RuntimeError("unreachable")


def kv_get(key: str, default=None):
    """Value for key, or default. Raises only on transport-level failure so a
    dead Supabase never silently reads as an empty store (which would let a
    signup overwrite every existing user)."""
    q = urllib.parse.urlencode({"key": f"eq.{key}", "select": "value"})
    req = urllib.request.Request(f"{_base()}?{q}", headers=_headers())
    rows = json.loads(_send(req) or b"[]")
    return rows[0]["value"] if rows else default


def kv_put(key: str, value) -> None:
    """Upsert — atomic per key on the Postgres side."""
    body = json.dumps({"key": key, "value": value}).encode()
    headers = {**_headers(),
               "Prefer": "resolution=merge-duplicates,return=minimal"}
    req = urllib.request.Request(_base() + "?on_conflict=key", data=body,
                                 headers=headers, method="POST")
    _send(req)


def kv_put_new(key: str, value) -> bool:
    """Insert only-if-absent — Postgres primary-key insert as a cross-instance
    compare-and-set. True = we won the key; False = it already existed. Used
    to make run execution single-shot when two /execute calls race."""
    body = json.dumps({"key": key, "value": value}).encode()
    req = urllib.request.Request(_base(), data=body,
                                 headers={**_headers(),
                                          "Prefer": "return=minimal"},
                                 method="POST")
    try:
        _send(req)
        return True
    except urllib.error.HTTPError as e:
        if e.code == 409:
            return False
        raise


def kv_delete(key: str) -> None:
    q = urllib.parse.urlencode({"key": f"eq.{key}"})
    req = urllib.request.Request(f"{_base()}?{q}", headers=_headers(),
                                 method="DELETE")
    _send(req)


def kv_keys(prefix: str) -> list[str]:
    """Key names only (no values) for a prefix — cheap existence checks."""
    q = urllib.parse.urlencode({"key": f"like.{prefix}*", "select": "key"})
    req = urllib.request.Request(f"{_base()}?{q}", headers=_headers())
    rows = json.loads(_send(req) or b"[]")
    return [r["key"] for r in rows]


def kv_list(prefix: str) -> dict:
    """{key: value} for every key starting with prefix. PostgREST `like`
    uses * as the wildcard; % must be escaped away from user input — our
    prefixes are code-controlled, never user text."""
    q = urllib.parse.urlencode({"key": f"like.{prefix}*",
                                "select": "key,value"})
    req = urllib.request.Request(f"{_base()}?{q}", headers=_headers())
    rows = json.loads(_send(req) or b"[]")
    return {r["key"]: r["value"] for r in rows}
