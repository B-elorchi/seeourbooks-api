"""
Supabase Auth Admin API client.

Lets the backend provision real login accounts (email + password) using the
service-role key, so an admin can create a user in one step instead of asking
them to self-sign-up.

Uses the GoTrue admin endpoint:
    POST {SUPABASE_URL}/auth/v1/admin/users

Requires SUPABASE_URL + SUPABASE_SERVICE_KEY. Raises AuthAdminError on failure
(missing config, duplicate email, weak password, network error) so routes can
map it to a clean 4xx/5xx for the admin UI.
"""
from __future__ import annotations

import logging

import httpx

from api.config.settings import settings

log = logging.getLogger(__name__)


class AuthAdminError(Exception):
    """Raised when the Supabase Admin API call fails."""

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


def _admin_headers() -> dict[str, str]:
    key = (settings.SUPABASE_SERVICE_KEY or "").strip()
    if not key:
        raise AuthAdminError(
            "SUPABASE_SERVICE_KEY is not set — cannot create login accounts.",
            status_code=500,
        )
    return {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}


def _base_url() -> str:
    base = (settings.SUPABASE_URL or "").strip().rstrip("/")
    if not base:
        raise AuthAdminError(
            "SUPABASE_URL is not set — cannot create login accounts.",
            status_code=500,
        )
    return base


async def create_auth_user(email: str, password: str, *, name: str | None = None,
                           email_confirm: bool = True) -> dict:
    """
    Create a Supabase Auth user. Returns the created user object (incl. `id`).

    `email_confirm=True` marks the email confirmed so the user can sign in
    immediately without clicking a confirmation link.
    """
    url = f"{_base_url()}/auth/v1/admin/users"
    body: dict = {
        "email":         email,
        "password":      password,
        "email_confirm": email_confirm,
    }
    if name:
        body["user_metadata"] = {"name": name}

    try:
        async with httpx.AsyncClient(timeout=20) as client:
            r = await client.post(url, headers=_admin_headers(), json=body)
    except Exception as exc:
        raise AuthAdminError(f"Supabase Admin API unreachable: {exc}") from exc

    if r.status_code in (200, 201):
        return r.json()

    # Surface a useful message — GoTrue returns {"msg": ...} or {"error_description": ...}
    detail = ""
    try:
        data = r.json()
        detail = data.get("msg") or data.get("error_description") or data.get("message") or r.text
    except Exception:
        detail = r.text
    # 422 = duplicate / weak password → caller should show as 400
    status = 400 if r.status_code in (400, 409, 422) else 502
    raise AuthAdminError(f"Could not create login account: {detail}", status_code=status)
