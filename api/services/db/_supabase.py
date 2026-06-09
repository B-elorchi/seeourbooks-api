"""
Supabase REST backend for the unified DB interface.
Translates find/insert/upsert/update into Supabase PostgREST HTTP calls.
"""
from typing import Any
from urllib.parse import quote
import httpx
from api.config.settings import settings


def _enc(val: Any) -> str:
    """URL-encode a filter value so characters like '+' ':' in ISO timestamps
    aren't mangled by the server (PostgREST decodes a raw '+' as a space)."""
    return quote(str(val), safe="")

_HEADERS: dict = {}
_BASE: str = ""


def _init() -> None:
    global _HEADERS, _BASE
    _HEADERS = {
        "apikey":        settings.SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
        "Content-Type":  "application/json",
    }
    _BASE = f"{settings.SUPABASE_URL}/rest/v1"


def _build_path(table: str, filters: dict | None, select: str, order: str | None,
                limit: int | None, offset: int | None = None) -> str:
    """Build a Supabase REST query path from structured parameters."""
    parts: list[str] = [f"select={select}"]

    if filters:
        for col, val in filters.items():
            if isinstance(val, tuple) and val[0] in ("in", "gte", "lte", "gt", "lt", "neq"):
                op, raw = val[0], val[1]
                if op == "in":
                    joined = ",".join(_enc(v) for v in raw)
                    parts.append(f"{col}=in.({joined})")
                else:
                    parts.append(f"{col}={op}.{_enc(raw)}")
            else:
                parts.append(f"{col}=eq.{_enc(val)}")

    if order:
        # Accept "col DESC" or "col ASC" — convert to Supabase "col.desc"
        norm = order.strip().replace(" ", ".").lower()
        parts.append(f"order={norm}")

    if limit is not None:
        parts.append(f"limit={limit}")

    if offset is not None and offset > 0:
        parts.append(f"offset={offset}")

    return f"{table}?{'&'.join(parts)}"


async def startup() -> None:
    """Validate Supabase credentials are present."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "DB_BACKEND=supabase but SUPABASE_URL or SUPABASE_SERVICE_KEY is missing"
        )
    _init()


async def shutdown() -> None:
    pass  # Nothing to close for HTTP-based backend


async def find(
    table: str,
    *,
    filters: dict[str, Any] | None = None,
    select: str = "*",
    order: str | None = None,
    limit: int | None = None,
    offset: int | None = None,
) -> list[dict]:
    _init()
    path = _build_path(table, filters, select, order, limit, offset)
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{_BASE}/{path}", headers=_HEADERS)
        r.raise_for_status()
        return r.json()


def _strip_nul(value):
    """Recursively strip NUL bytes from dicts/lists/strings before they
    hit PostgREST.  Postgres rejects \\x00 in TEXT / JSONB."""
    if value is None or isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        return value.replace("\x00", "") if "\x00" in value else value
    if isinstance(value, dict):
        return {k: _strip_nul(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_strip_nul(v) for v in value]
    return value


def _raise_with_body(r: httpx.Response, op: str, table: str) -> None:
    """
    Replace the opaque `raise_for_status()` error with the actual PostgREST
    error body so callers can see WHY the request was rejected (missing
    column, not-null violation, type mismatch, etc.).
    """
    if r.status_code < 400:
        return
    body = (r.text or "")[:600]
    raise RuntimeError(
        f"Supabase {op} on {table!r} failed with HTTP {r.status_code}: {body}"
    )


async def insert(table: str, data: dict) -> dict:
    _init()
    data = _strip_nul(data)
    h = {**_HEADERS, "Prefer": "return=representation"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{_BASE}/{table}", headers=h, json=data)
        _raise_with_body(r, "INSERT", table)
        rows = r.json()
        return rows[0] if isinstance(rows, list) else rows


async def upsert(table: str, data: dict, conflict: str) -> dict:
    _init()
    data = _strip_nul(data)
    h = {**_HEADERS, "Prefer": "resolution=merge-duplicates,return=representation"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{_BASE}/{table}?on_conflict={conflict}", headers=h, json=data
        )
        _raise_with_body(r, f"UPSERT (on_conflict={conflict})", table)
        rows = r.json()
        return rows[0] if isinstance(rows, list) else rows


async def update(table: str, filters: dict, data: dict) -> None:
    _init()
    data = _strip_nul(data)
    parts = [f"{col}=eq.{_enc(val)}" for col, val in filters.items()]
    path = f"{table}?{'&'.join(parts)}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.patch(f"{_BASE}/{path}", headers=_HEADERS, json=data)
        _raise_with_body(r, "UPDATE", table)


async def delete(table: str, filters: dict) -> None:
    _init()
    parts = [f"{col}=eq.{_enc(val)}" for col, val in filters.items()]
    path = f"{table}?{'&'.join(parts)}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.delete(f"{_BASE}/{path}", headers=_HEADERS)
        _raise_with_body(r, "DELETE", table)
