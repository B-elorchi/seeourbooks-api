"""
User + API key management routes (admin only).

  POST   /api/users                      create user
  GET    /api/users                      list users
  GET    /api/users/{user_id}            get user
  PATCH  /api/users/{user_id}            update user (role, name, is_active)
  DELETE /api/users/{user_id}            delete user + all their keys

  POST   /api/users/{user_id}/api-keys   create key → returns full key ONCE
  GET    /api/users/{user_id}/api-keys   list keys (prefix only, no raw)
  DELETE /api/users/{user_id}/api-keys/{key_id}   revoke key

  GET    /api/users/{user_id}/costs      cost breakdown for that user
  GET    /api/costs                      costs for all users
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from api.auth.apikey import generate_api_key, _hash
from api.auth import AuthUser, require_admin
from api.services.auth_admin import AuthAdminError, create_auth_user
from api.services.db import find, insert, update as db_update, delete as db_delete

log = logging.getLogger(__name__)
router = APIRouter(prefix="/users", tags=["users"])


# ── Pydantic models ──────────────────────────────────────────────────────────

class CreateUserReq(BaseModel):
    email:    str
    name:     str = ""
    role:     str = "viewer"    # admin | editor | viewer
    # When provided, a real Supabase Auth login is created so the user can sign
    # in immediately. When omitted, only a metadata/role row is created.
    password: Optional[str] = None


class UpdateUserReq(BaseModel):
    name:      Optional[str]  = None
    role:      Optional[str]  = None
    is_active: Optional[bool] = None


class CreateKeyReq(BaseModel):
    name:       str = "Default Key"
    role:       str = "viewer"
    expires_at: Optional[str] = None   # ISO datetime or None


# ── Users ────────────────────────────────────────────────────────────────────

@router.post("", status_code=201)
async def create_user(
    body: CreateUserReq,
    admin: AuthUser = Depends(require_admin),
) -> dict:
    if body.role not in ("admin", "editor", "viewer"):
        raise HTTPException(400, "role must be admin | editor | viewer")

    existing = await find("app_users", filters={"email": body.email}, limit=1)
    if existing:
        raise HTTPException(409, f"User with email {body.email!r} already exists")

    new_row: dict = {
        "email":     body.email,
        "name":      body.name,
        "role":      body.role,
        "is_active": True,
    }

    # When a password is supplied, create a real Supabase Auth login first and
    # pin app_users.id to the auth uid so the JWT `sub`, app_users.id, and the
    # user_id stamped on jobs/documents are all the same value.
    if body.password:
        try:
            auth_user = await create_auth_user(
                body.email, body.password, name=body.name, email_confirm=True,
            )
        except AuthAdminError as exc:
            raise HTTPException(exc.status_code, str(exc)) from exc
        auth_id = auth_user.get("id")
        if auth_id:
            new_row["id"] = auth_id

    row = await insert("app_users", new_row)
    return row or {}


@router.get("")
async def list_users(
    admin: AuthUser = Depends(require_admin),
    limit: int = 100,
) -> list:
    return await find("app_users", filters={}, limit=limit)


@router.get("/keys")
async def list_all_keys(
    admin: AuthUser = Depends(require_admin),
) -> list:
    """List all API keys across all users (no key_hash exposed)."""
    rows = await find("api_keys", filters={}, limit=500)
    return [{k: v for k, v in r.items() if k != "key_hash"} for r in rows]


@router.get("/costs/all")
async def all_user_costs(
    admin: AuthUser = Depends(require_admin),
) -> list:
    """All users with their aggregate spend."""
    return await find("user_costs", filters={}, limit=500)


@router.get("/{user_id}")
async def get_user(
    user_id: str,
    admin: AuthUser = Depends(require_admin),
) -> dict:
    rows = await find("app_users", filters={"id": user_id}, limit=1)
    if not rows:
        raise HTTPException(404, "User not found")
    return rows[0]


@router.patch("/{user_id}")
async def update_user(
    user_id: str,
    body: UpdateUserReq,
    admin: AuthUser = Depends(require_admin),
) -> dict:
    data: dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
    if body.name is not None:
        data["name"] = body.name
    if body.role is not None:
        if body.role not in ("admin", "editor", "viewer"):
            raise HTTPException(400, "role must be admin | editor | viewer")
        data["role"] = body.role
    if body.is_active is not None:
        data["is_active"] = body.is_active

    await db_update("app_users", {"id": user_id}, data)
    updated = await find("app_users", filters={"id": user_id}, limit=1)
    if not updated:
        raise HTTPException(404, "User not found")
    return updated[0]


@router.delete("/{user_id}")
async def delete_user(
    user_id: str,
    admin: AuthUser = Depends(require_admin),
) -> Response:
    await db_delete("app_users", {"id": user_id})
    return Response(status_code=204)


# ── API keys ─────────────────────────────────────────────────────────────────

@router.post("/{user_id}/api-keys", status_code=201)
async def create_api_key(
    user_id: str,
    body: CreateKeyReq,
    admin: AuthUser = Depends(require_admin),
) -> dict:
    if body.role not in ("admin", "editor", "viewer"):
        raise HTTPException(400, "role must be admin | editor | viewer")

    users = await find("app_users", filters={"id": user_id}, limit=1)
    if not users:
        raise HTTPException(404, "User not found")

    full_key, prefix, h = generate_api_key()

    row_data: dict = {
        "user_id":    user_id,
        "name":       body.name,
        "key_prefix": prefix,
        "key_hash":   h,
        "role":       body.role,
        "is_active":  True,
    }
    if body.expires_at:
        row_data["expires_at"] = body.expires_at

    saved = await insert("api_keys", row_data)

    return {
        **(saved or {}),
        "key": full_key,   # returned ONCE — not stored in DB
        "warning": "Save this key now — it will not be shown again.",
    }


@router.get("/{user_id}/api-keys")
async def list_api_keys(
    user_id: str,
    admin: AuthUser = Depends(require_admin),
) -> list:
    rows = await find("api_keys", filters={"user_id": user_id}, limit=200)
    # Strip key_hash from the response — never expose it
    return [
        {k: v for k, v in r.items() if k != "key_hash"}
        for r in rows
    ]


@router.delete("/{user_id}/api-keys/{key_id}")
async def revoke_api_key(
    user_id: str,
    key_id:  str,
    admin: AuthUser = Depends(require_admin),
) -> Response:
    await db_delete("api_keys", {"id": key_id, "user_id": user_id})
    return Response(status_code=204)


# ── Costs ─────────────────────────────────────────────────────────────────────

@router.get("/{user_id}/costs")
async def user_costs(
    user_id: str,
    admin: AuthUser = Depends(require_admin),
) -> dict:
    """Aggregate spend for one user from the user_costs view."""
    rows = await find("user_costs", filters={"user_id": user_id}, limit=1)
    if not rows:
        return {"user_id": user_id, "total_cost_usd": 0, "total_jobs": 0, "total_calls": 0}
    return rows[0]
