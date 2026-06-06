"""
Supabase Auth — FastAPI dependencies.

Verifies access tokens issued by Supabase Auth.  Supports BOTH:

    - **Legacy HS256** projects — verified with `SUPABASE_JWT_SECRET`
      (Supabase Dashboard → Project Settings → API → JWT Secret).

    - **Modern JWKS** projects — Supabase issues ES256/RS256 tokens that
      are verified using public keys fetched from the project's JWKS
      endpoint at `{SUPABASE_URL}/auth/v1/.well-known/jwks.json`.

The verifier inspects the token header to decide which method to use.
JWKS keys are cached in-process for 1 hour.

Auth is automatically DISABLED when both SUPABASE_JWT_SECRET and SUPABASE_URL
are missing — `require_user` / `require_admin` return a dummy admin so the
API stays usable for local dev without any Supabase project configured.

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
from dataclasses import dataclass

import httpx
import jwt
from jwt.exceptions import (
    InvalidTokenError, ExpiredSignatureError,
    InvalidSignatureError, InvalidAudienceError,
)
from fastapi import HTTPException, Request, status

from api.config.settings import settings

log = logging.getLogger(__name__)


# ── JWKS cache ───────────────────────────────────────────────────────────────
# Supabase rarely rotates signing keys, so a long cache TTL is fine.
_JWKS_CACHE: dict[str, tuple[float, dict]] = {}
_JWKS_TTL_SEC = 3600   # 1 hour


# ── Public dataclass for FastAPI dependency injection ───────────────────────

@dataclass(frozen=True)
class AuthUser:
    """Decoded fields from a Supabase access token."""
    id:    str          # auth.users.id (uuid)
    email: str          # auth.users.email
    role:  str          # "authenticated" by default; "service_role" for backend
    is_admin: bool      # computed from ADMIN_EMAILS

    @property
    def jwt_subject(self) -> str:
        """Alias for `id` — useful when caller speaks JWT terminology."""
        return self.id


# ── Helpers ──────────────────────────────────────────────────────────────────

def auth_enabled() -> bool:
    """
    True when ANY supported verification method is configured:
      - HS256 via SUPABASE_JWT_SECRET, OR
      - JWKS via SUPABASE_URL (modern projects).
    """
    return bool(
        (settings.SUPABASE_JWT_SECRET or "").strip()
        or (settings.SUPABASE_URL or "").strip()
    )


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


def _extract_token(request: Request) -> str | None:
    """Pull the bearer token from the Authorization header or ?access_token=."""
    auth = request.headers.get("Authorization") or request.headers.get("authorization")
    if auth:
        parts = auth.split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer":
            return parts[1].strip() or None

    qs = request.query_params.get("access_token")
    return qs.strip() if qs else None


def _fetch_jwks(force_refresh: bool = False) -> dict | None:
    """
    Fetch the JWKS document for the configured Supabase project.

    Caches in-process for 1 hour.  Returns None when SUPABASE_URL is missing
    or the endpoint is unreachable.
    """
    base = (settings.SUPABASE_URL or "").strip().rstrip("/")
    if not base:
        return None

    now = time.time()
    cached = _JWKS_CACHE.get(base)
    if cached and not force_refresh and (now - cached[0]) < _JWKS_TTL_SEC:
        return cached[1]

    url = f"{base}/auth/v1/.well-known/jwks.json"
    try:
        r = httpx.get(url, timeout=8)
        r.raise_for_status()
        jwks = r.json()
    except Exception as exc:
        log.warning("Could not fetch JWKS from %s: %s", url, exc)
        return cached[1] if cached else None   # serve stale on transient failure

    _JWKS_CACHE[base] = (now, jwks)
    return jwks


def _get_jwks_public_key(kid: str | None):
    """Find the key matching `kid` in the JWKS and return a PyJWT-ready PublicKey."""
    if not kid:
        return None
    jwks = _fetch_jwks()
    if not jwks:
        return None
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            try:
                return jwt.PyJWK(key).key
            except Exception as exc:
                log.warning("JWKS key for kid=%s could not be parsed: %s", kid, exc)
                return None
    # Cache miss — force a refresh in case keys rotated
    jwks = _fetch_jwks(force_refresh=True)
    if jwks:
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                try:
                    return jwt.PyJWK(key).key
                except Exception:
                    return None
    return None


def _decode(token: str) -> dict | None:
    """
    Decode + verify a Supabase access token.  Returns the payload or None.

    Strategy:
        1. Read the token header (without verifying) to learn `alg` and `kid`.
        2. If `alg` is HS256 → verify with `SUPABASE_JWT_SECRET`.
        3. If `alg` is ES256 / RS256 / etc → fetch the JWKS key by `kid`
           and verify with the public key.

    Logs at WARNING level when a token is rejected — admins debugging
    "Admin access required" need to see WHY.  The reason is also returned
    indirectly: None means "rejected" (caller raises 401).
    """
    try:
        header = jwt.get_unverified_header(token)
    except Exception as exc:
        log.warning("auth: could not parse token header: %s", exc)
        return None

    alg = header.get("alg")
    kid = header.get("kid")

    # ── Decode kwargs that work for both audience-on and audience-off projects
    def _try_decode(key, algorithms):
        # First try with the standard "authenticated" audience that Supabase uses.
        try:
            return jwt.decode(
                token, key, algorithms=algorithms, audience="authenticated",
            )
        except InvalidAudienceError:
            # Some projects disable audience verification — retry without it.
            return jwt.decode(
                token, key, algorithms=algorithms, options={"verify_aud": False},
            )

    # ── HS256 (legacy projects) ──────────────────────────────────────────────
    if alg == "HS256":
        secret = (settings.SUPABASE_JWT_SECRET or "").strip()
        if not secret:
            log.warning(
                "auth: token uses HS256 but SUPABASE_JWT_SECRET is not set. "
                "Add it from Supabase Dashboard → Settings → API → JWT Secret."
            )
            return None
        try:
            return _try_decode(secret, ["HS256"])
        except ExpiredSignatureError:
            log.warning("auth: token expired")
            return None
        except InvalidSignatureError:
            log.warning(
                "auth: HS256 signature invalid — your SUPABASE_JWT_SECRET "
                "doesn't match the project that issued the token. "
                "Verify it in Supabase Dashboard → Settings → API → JWT Secret."
            )
            return None
        except InvalidTokenError as exc:
            log.warning("auth: HS256 token rejected: %s", exc)
            return None

    # ── Asymmetric (ES256, RS256, etc.) via JWKS — modern projects ──────────
    if alg and alg != "none":
        key = _get_jwks_public_key(kid)
        if key is None:
            log.warning(
                "auth: token uses %s with kid=%s, but no matching key in JWKS. "
                "Make sure SUPABASE_URL is correct: %s",
                alg, kid, settings.SUPABASE_URL or "<not set>",
            )
            return None
        try:
            return _try_decode(key, [alg])
        except ExpiredSignatureError:
            log.warning("auth: token expired")
            return None
        except InvalidSignatureError:
            log.warning(
                "auth: %s signature invalid — JWKS key didn't verify the token. "
                "This usually means your SUPABASE_URL points at a different project "
                "than the one that issued the token.", alg,
            )
            return None
        except InvalidTokenError as exc:
            log.warning("auth: %s token rejected: %s", alg, exc)
            return None

    log.warning("auth: unsupported / missing algorithm in token header: %s", alg)
    return None


def _payload_to_user(payload: dict) -> AuthUser:
    user_meta = payload.get("user_metadata") or {}
    email = (
        payload.get("email")
        or user_meta.get("email")
        or ""
    )
    return AuthUser(
        id       = payload.get("sub") or "",
        email    = email,
        role     = payload.get("role") or "authenticated",
        is_admin = _is_admin_email(email),
    )


# ── FastAPI dependencies ─────────────────────────────────────────────────────

async def get_current_user(request: Request) -> AuthUser | None:
    """Soft auth — returns the user if a valid token is present, None otherwise."""
    if not auth_enabled():
        return None
    token = _extract_token(request)
    if not token:
        return None
    payload = _decode(token)
    if not payload:
        return None
    return _payload_to_user(payload)


async def require_user(request: Request) -> AuthUser:
    """Hard auth — raises 401 when no valid token is present."""
    if not auth_enabled():
        # Auth disabled → no-op AuthUser. Lets dev environments run without Supabase.
        return AuthUser(id="dev", email="dev@local", role="dev", is_admin=True)

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
    return _payload_to_user(payload)


async def require_admin(request: Request) -> AuthUser:
    """Hard auth + admin check — raises 403 when authenticated but not an admin."""
    user = await require_user(request)
    if not user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="admin access required",
        )
    return user
