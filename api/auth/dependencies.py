"""
WordPress JWT Auth — FastAPI dependencies.

Verifies access tokens issued by the WordPress JWT Authentication plugin.
Uses the `WP_JWT_SECRET` configured in `.env` (which matches the WordPress `JWT_AUTH_SECRET_KEY`).

Auth is automatically DISABLED when WP_JWT_SECRET is missing — `require_user` / `require_admin` 
return a dummy admin so the API stays usable for local dev without WordPress configured.

Public surface
──────────────
    auth_enabled() -> bool
    get_current_user(request) -> AuthUser | None
    require_user(request) -> AuthUser
    require_admin(request) -> AuthUser

Token sources (checked in order)
────────────────────────────────
    1. Authorization: Bearer <token>
    2. ?access_token=<token>          (for direct downloads)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, replace

import jwt
from jwt.exceptions import (
    InvalidTokenError, ExpiredSignatureError,
    InvalidSignatureError, InvalidAudienceError,
)
from fastapi import HTTPException, Request, status

from api.config.settings import settings

log = logging.getLogger(__name__)


# Tolerance for clock skew between this server and WP.
_CLOCK_SKEW_LEEWAY_SEC = 60


# ── Public dataclass for FastAPI dependency injection ───────────────────────

@dataclass(frozen=True)
class AuthUser:
    """Decoded fields from a WordPress access token."""
    id:    str          # WordPress User ID
    email: str          # WordPress Email (can be empty if not in token payload)
    role:  str          # "authenticated"
    is_admin: bool      # computed from ADMIN_EMAILS or app_users.role
    # Application role from app_users.role: "admin" | "editor" | "viewer".
    app_role: str = "viewer"

    @property
    def jwt_subject(self) -> str:
        """Alias for `id` — useful when caller speaks JWT terminology."""
        return self.id

    def owner_filter(self) -> dict:
        """PostgREST-style filter for row ownership."""
        return {} if self.is_admin else {"user_id": self.id}


# ── Helpers ──────────────────────────────────────────────────────────────────

def auth_enabled() -> bool:
    """True when WP_JWT_SECRET is configured."""
    return bool((settings.WP_JWT_SECRET or "").strip())


def _admin_email_set() -> set[str]:
    raw = (settings.ADMIN_EMAILS or "").strip()
    if not raw:
        return set()
    return {e.strip().lower() for e in raw.split(",") if e.strip()}


def _is_admin_email(email: str | None) -> bool:
    if not email:
        return False
    admins = _admin_email_set()
    if not admins:
        # Single-tenant mode — every authenticated user is an admin
        return True
    return email.lower() in admins


# ── DB-backed role map (app_users.role) ──────────────────────────────────────
_DB_ROLE_CACHE: dict[str, object] = {"t": 0.0, "roles": {}}
_DB_ROLE_TTL_SEC = 60.0


async def _db_role_map() -> dict[str, str]:
    """Lower-cased email → role for active app_users. Cached 60s."""
    now = time.time()
    if (now - float(_DB_ROLE_CACHE["t"])) < _DB_ROLE_TTL_SEC:
        return _DB_ROLE_CACHE["roles"]  # type: ignore[return-value]
    roles: dict[str, str] = {}
    try:
        from api.services.db import find
        rows = await find(
            "app_users",
            filters={"is_active": True},
            select="email, role",
        )
        roles = {
            (r.get("email") or "").lower(): (r.get("role") or "viewer")
            for r in rows if r.get("email")
        }
    except Exception as exc:
        log.debug("auth: could not load DB roles: %s", exc)
        return _DB_ROLE_CACHE["roles"]  # type: ignore[return-value]
    _DB_ROLE_CACHE["t"] = now
    _DB_ROLE_CACHE["roles"] = roles
    return roles


async def _resolve_app_role(email: str | None) -> str:
    if _is_admin_email(email):
        return "admin"
    if not email:
        return "viewer"
    return (await _db_role_map()).get(email.lower(), "viewer")


def _extract_token(request: Request) -> str | None:
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() or None

    qs = request.query_params.get("access_token")
    return qs.strip() if qs else None


def _decode(token: str) -> dict | None:
    secret = (settings.WP_JWT_SECRET or "").strip()
    if not secret:
        log.warning("auth: WP_JWT_SECRET is not set.")
        return None
    
    try:
        return jwt.decode(
            token, secret, algorithms=["HS256"],
            options={"verify_aud": False},
            leeway=_CLOCK_SKEW_LEEWAY_SEC,
        )
    except ExpiredSignatureError:
        log.warning("auth: WP token expired")
        return None
    except InvalidSignatureError:
        log.warning("auth: HS256 signature invalid — your WP_JWT_SECRET doesn't match.")
        return None
    except InvalidTokenError as exc:
        log.warning("auth: WP token rejected: %s", exc)
        return None


def _payload_to_user(payload: dict) -> AuthUser:
    """
    WordPress JWT standard payload looks like:
    {'iss': 'https://...', 'iat': 1784233677, 'nbf': 1784233677, 'exp': 1784838477, 'data': {'user': {'id': '1'}}}
    We extract the WP User ID. We can extract email if added later.
    """
    data = payload.get("data", {})
    user_data = data.get("user", {})
    
    wp_id = str(user_data.get("id") or "")
    # Default to an empty email since standard WP JWT might not include it by default
    email = user_data.get("email") or ""
    
    return AuthUser(
        id       = wp_id,
        email    = email,
        role     = "authenticated",
        is_admin = _is_admin_email(email),
    )


async def _apply_app_role(user: AuthUser) -> AuthUser:
    app_role = await _resolve_app_role(user.email)
    return replace(user, app_role=app_role, is_admin=user.is_admin or app_role == "admin")


# ── FastAPI dependencies ─────────────────────────────────────────────────────

async def get_current_user(request: Request) -> AuthUser | None:
    if not auth_enabled():
        return None
    token = _extract_token(request)
    if not token:
        return None
    payload = _decode(token)
    if not payload:
        return None
    return await _apply_app_role(_payload_to_user(payload))


async def require_user(request: Request) -> AuthUser:
    if not auth_enabled():
        return AuthUser(id="dev", email="dev@local", role="dev", is_admin=True, app_role="admin")

    token = _extract_token(request)
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="missing access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = _decode(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="invalid or expired access token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return await _apply_app_role(_payload_to_user(payload))


async def require_admin(request: Request) -> AuthUser:
    user = await require_user(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin access required",
        )
    return user
