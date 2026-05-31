"""
Supabase REST backend for the unified DB interface.
Translates find/insert/upsert/update into Supabase PostgREST HTTP calls.
"""
from typing import Any
import httpx
from api.config.settings import settings

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


def _build_path(table: str, filters: dict | None, select: str, order: str | None, limit: int | None) -> str:
    """Build a Supabase REST query path from structured parameters."""
    parts: list[str] = [f"select={select}"]

    if filters:
        for col, val in filters.items():
            if isinstance(val, tuple) and val[0] in ("in", "gte", "lte", "gt", "lt", "neq"):
                op, raw = val[0], val[1]
                if op == "in":
                    joined = ",".join(str(v) for v in raw)
                    parts.append(f"{col}=in.({joined})")
                else:
                    parts.append(f"{col}={op}.{raw}")
            else:
                parts.append(f"{col}=eq.{val}")

    if order:
        # Accept "col DESC" or "col ASC" — convert to Supabase "col.desc"
        norm = order.strip().replace(" ", ".").lower()
        parts.append(f"order={norm}")

    if limit is not None:
        parts.append(f"limit={limit}")

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
) -> list[dict]:
    _init()
    path = _build_path(table, filters, select, order, limit)
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.get(f"{_BASE}/{path}", headers=_HEADERS)
        r.raise_for_status()
        return r.json()


async def insert(table: str, data: dict) -> dict:
    _init()
    h = {**_HEADERS, "Prefer": "return=representation"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{_BASE}/{table}", headers=h, json=data)
        r.raise_for_status()
        rows = r.json()
        return rows[0] if isinstance(rows, list) else rows


async def upsert(table: str, data: dict, conflict: str) -> dict:
    _init()
    h = {**_HEADERS, "Prefer": "resolution=merge-duplicates,return=representation"}
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(
            f"{_BASE}/{table}?on_conflict={conflict}", headers=h, json=data
        )
        r.raise_for_status()
        rows = r.json()
        return rows[0] if isinstance(rows, list) else rows


async def update(table: str, filters: dict, data: dict) -> None:
    _init()
    parts = [f"{col}=eq.{val}" for col, val in filters.items()]
    path = f"{table}?{'&'.join(parts)}"
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.patch(f"{_BASE}/{path}", headers=_HEADERS, json=data)
        r.raise_for_status()
