"""
Supabase REST backend for the unified DB interface.
Translates find/insert/upsert/update into Supabase PostgREST HTTP calls.

Uses a single shared, connection-pooled httpx client with automatic retries on
transient network errors. Creating a fresh AsyncClient per call (the old
behaviour) opened/closed a TCP+TLS connection every request — under heavy
concurrent load (e.g. a 100-book batch) that exhausts sockets and produces
ConnectTimeout storms that cascade into "insert failed" / "could not fetch
chunks" / job-killing errors. The shared pool fixes that.
"""
import asyncio
import logging
import random
from typing import Any
from urllib.parse import quote
import httpx
from api.config.settings import settings

log = logging.getLogger(__name__)


def _enc(val: Any) -> str:
    """URL-encode a filter value so characters like '+' ':' in ISO timestamps
    aren't mangled by the server (PostgREST decodes a raw '+' as a space)."""
    return quote(str(val), safe="")

_HEADERS: dict = {}
_BASE: str = ""
_CLIENT: httpx.AsyncClient | None = None

# Transient errors raised BEFORE the request reaches the server — the write
# definitely did not happen, so retrying is always safe (even for POST inserts).
_CONNECT_ERRORS = (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout)
# Errors raised AFTER the request was sent — the server may have processed it,
# so we only retry these for idempotent methods to avoid duplicate writes.
_READ_ERRORS = (httpx.ReadTimeout, httpx.WriteTimeout, httpx.RemoteProtocolError)
# 5xx / rate-limit responses that are worth retrying for idempotent calls.
_RETRY_STATUS = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 4


def _init() -> None:
    global _HEADERS, _BASE
    _HEADERS = {
        "apikey":        settings.SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
        "Content-Type":  "application/json",
    }
    _BASE = f"{settings.SUPABASE_URL}/rest/v1"


def _get_client() -> httpx.AsyncClient:
    """Return the shared pooled client, building it lazily on first use."""
    global _CLIENT
    if _CLIENT is None or _CLIENT.is_closed:
        limits = httpx.Limits(
            max_connections=50,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        )
        # pool=30s — when every connection is busy, wait up to 30s for one to
        # free up instead of failing instantly with a PoolTimeout.
        timeout = httpx.Timeout(connect=10.0, read=30.0, write=30.0, pool=30.0)
        _CLIENT = httpx.AsyncClient(limits=limits, timeout=timeout)
    return _CLIENT


def _backoff(attempt: int) -> float:
    """Exponential backoff with jitter: ~0.5, 1, 2s (capped at 8s)."""
    return min(8.0, 0.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.3)


async def _request(method: str, url: str, *, idempotent: bool, **kwargs) -> httpx.Response:
    """
    Issue an HTTP request on the shared client with transient-failure retries.

    idempotent — True for GET/PATCH/DELETE and merge-duplicate upserts (safe to
    repeat). False for plain inserts: those are still retried on pre-send
    connection errors (which guarantee the write never happened) but NOT on
    read timeouts or 5xx, to avoid creating duplicate rows.
    """
    client = _get_client()
    last_exc: Exception | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        try:
            r = await client.request(method, url, **kwargs)
            if r.status_code in _RETRY_STATUS and idempotent and attempt < _MAX_ATTEMPTS:
                last_exc = RuntimeError(f"Supabase {method} {url!r} → HTTP {r.status_code}")
                await asyncio.sleep(_backoff(attempt))
                continue
            return r
        except _CONNECT_ERRORS as exc:
            last_exc = exc  # never reached the server — safe to retry any method
        except _READ_ERRORS as exc:
            if not idempotent:
                raise       # may have been processed — don't risk a duplicate
            last_exc = exc
        if attempt < _MAX_ATTEMPTS:
            log.warning("Supabase %s %r transient error (attempt %d/%d): %r — retrying",
                        method, url, attempt, _MAX_ATTEMPTS, last_exc)
            await asyncio.sleep(_backoff(attempt))
            continue
        break
    assert last_exc is not None
    raise last_exc


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
            elif isinstance(val, tuple) and val[0] == "is":
                # ("is", None) → is.null ; ("is", True/False) → is.true/false
                token = "null" if val[1] is None else str(val[1]).lower()
                parts.append(f"{col}=is.{token}")
            elif isinstance(val, tuple) and val[0] == "not_is":
                token = "null" if val[1] is None else str(val[1]).lower()
                parts.append(f"{col}=not.is.{token}")
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
    """Validate Supabase credentials are present and warm the shared client."""
    if not settings.SUPABASE_URL or not settings.SUPABASE_SERVICE_KEY:
        raise RuntimeError(
            "DB_BACKEND=supabase but SUPABASE_URL or SUPABASE_SERVICE_KEY is missing"
        )
    _init()
    _get_client()


async def shutdown() -> None:
    global _CLIENT
    if _CLIENT is not None and not _CLIENT.is_closed:
        await _CLIENT.aclose()
    _CLIENT = None


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
    r = await _request("GET", f"{_BASE}/{path}", headers=_HEADERS, idempotent=True)
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
    # idempotent=False: a plain insert that the server may have already applied
    # must not be blindly repeated. Connection-phase errors are still retried
    # inside _request (those guarantee the row was never written).
    r = await _request("POST", f"{_BASE}/{table}", headers=h, json=data, idempotent=False)
    _raise_with_body(r, "INSERT", table)
    rows = r.json()
    return rows[0] if isinstance(rows, list) else rows


async def upsert(table: str, data: dict, conflict: str) -> dict:
    _init()
    data = _strip_nul(data)
    h = {**_HEADERS, "Prefer": "resolution=merge-duplicates,return=representation"}
    # Upsert is idempotent (merge-duplicates on the conflict key), so it is safe
    # to retry on read timeouts / 5xx as well.
    r = await _request(
        "POST", f"{_BASE}/{table}?on_conflict={conflict}",
        headers=h, json=data, idempotent=True,
    )
    _raise_with_body(r, f"UPSERT (on_conflict={conflict})", table)
    rows = r.json()
    return rows[0] if isinstance(rows, list) else rows


async def update(table: str, filters: dict, data: dict) -> None:
    _init()
    data = _strip_nul(data)
    parts = [f"{col}=eq.{_enc(val)}" for col, val in filters.items()]
    path = f"{table}?{'&'.join(parts)}"
    r = await _request("PATCH", f"{_BASE}/{path}", headers=_HEADERS, json=data, idempotent=True)
    _raise_with_body(r, "UPDATE", table)


async def delete(table: str, filters: dict) -> None:
    _init()
    parts = [f"{col}=eq.{_enc(val)}" for col, val in filters.items()]
    path = f"{table}?{'&'.join(parts)}"
    r = await _request("DELETE", f"{_BASE}/{path}", headers=_HEADERS, idempotent=True)
    _raise_with_body(r, "DELETE", table)
