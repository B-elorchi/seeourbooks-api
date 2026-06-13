"""
API Key authentication for SeeOurBook.

Keys are generated as:
    sob_live_<32-random-hex-chars>

Only the SHA-256 hash is stored in the database (api_keys.key_hash).
The prefix (first 8 chars) is stored plain for display in the admin UI.

Lookup: O(1) — one SELECT by hash.
Cache:  valid keys are cached in-process for 60 seconds to avoid a DB
        round-trip on every request.

Usage (FastAPI dependency):
    from api.auth.apikey import require_api_key, ApiKeyUser

    @router.get("/protected")
    async def handler(user: ApiKeyUser = Depends(require_api_key)):
        ...
"""
from __future__ import annotations

import hashlib
import logging
import os
import secrets
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from api.services.db import find

log = logging.getLogger(__name__)

_PREFIX = "sob_live_"
_CACHE: dict[str, tuple[float, dict | None]] = {}  # hash -> (ts, row)
_CACHE_TTL = 60  # seconds


# ── Key generation ────────────────────────────────────────────────────────────

def generate_api_key() -> tuple[str, str, str]:
    """
    Returns (full_key, prefix, sha256_hash).
    full_key is shown to the user ONCE — we only store prefix + hash.
    """
    token    = secrets.token_hex(32)
    full_key = _PREFIX + token
    prefix   = full_key[:12]          # "sob_live_xxxx"  (first 12 chars)
    h        = _hash(full_key)
    return full_key, prefix, h


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


# ── Lookup ────────────────────────────────────────────────────────────────────

async def _lookup(key_hash: str) -> dict | None:
    """DB lookup with 60-second positive/negative cache."""
    now = time.monotonic()
    cached = _CACHE.get(key_hash)
    if cached and (now - cached[0]) < _CACHE_TTL:
        return cached[1]

    try:
        rows = await find(
            "api_keys",
            filters={"key_hash": key_hash, "is_active": "true"},
            limit=1,
        )
        row = rows[0] if rows else None
    except Exception as exc:
        log.warning("api_key lookup failed: %s", exc)
        row = None

    _CACHE[key_hash] = (now, row)
    return row


def _invalidate(key_hash: str) -> None:
    _CACHE.pop(key_hash, None)


# ── Dataclass returned to routes ──────────────────────────────────────────────

@dataclass(frozen=True)
class ApiKeyUser:
    key_id:  str
    user_id: str
    email:   str
    role:    str     # "admin" | "editor" | "viewer"

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def can_write(self) -> bool:
        return self.role in ("admin", "editor")


# ── FastAPI dependency ────────────────────────────────────────────────────────

def _extract_key(request: Request) -> str | None:
    """Read X-API-Key header (or ?api_key= query param for browser downloads)."""
    key = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
    if key:
        return key.strip() or None
    return (request.query_params.get("api_key") or "").strip() or None


async def get_api_key_user(request: Request) -> ApiKeyUser | None:
    """Soft dependency — returns None when no key provided or auth disabled."""
    from api.config.settings import settings  # noqa: PLC0415
    if not getattr(settings, "API_KEY_AUTH_ENABLED", True):
        return None

    raw = _extract_key(request)
    if not raw:
        return None

    h   = _hash(raw)
    row = await _lookup(h)
    if not row:
        return None

    # Update last_used_at asynchronously (fire-and-forget, best effort)
    import asyncio  # noqa: PLC0415
    from api.services.db import update as db_update  # noqa: PLC0415
    from datetime import datetime, timezone  # noqa: PLC0415
    asyncio.create_task(
        db_update(
            "api_keys",
            {"id": row["id"]},
            {"last_used_at": datetime.now(timezone.utc).isoformat()},
        )
    )

    return ApiKeyUser(
        key_id  = row["id"],
        user_id = row.get("user_id") or "",
        email   = row.get("email") or "",
        role    = row.get("role") or "viewer",
    )


async def require_api_key(request: Request) -> ApiKeyUser:
    """Hard dependency — raises 401 when key is missing or invalid."""
    from api.config.settings import settings  # noqa: PLC0415

    # If API key auth is disabled (dev mode), return a dummy admin
    if not getattr(settings, "API_KEY_AUTH_ENABLED", True):
        return ApiKeyUser(key_id="dev", user_id="dev", email="dev@local", role="admin")

    raw = _extract_key(request)
    if not raw:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key — pass X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    h   = _hash(raw)
    row = await _lookup(h)
    if not row:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or revoked API key",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Check expiry
    exp = row.get("expires_at")
    if exp:
        from datetime import datetime, timezone  # noqa: PLC0415
        try:
            exp_dt = datetime.fromisoformat(exp.replace("Z", "+00:00"))
            if datetime.now(timezone.utc) > exp_dt:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key expired",
                )
        except ValueError:
            pass

    return ApiKeyUser(
        key_id  = row["id"],
        user_id = row.get("user_id") or "",
        email   = row.get("email") or "",
        role    = row.get("role") or "viewer",
    )


async def require_admin_key(request: Request) -> ApiKeyUser:
    """Hard dependency — raises 403 when key is valid but role is not admin."""
    user = await require_api_key(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin API key required",
        )
    return user
