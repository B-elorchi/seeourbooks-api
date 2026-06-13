"""
Auth helpers exposed to the frontend.
  GET /api/auth/me     → returns the current user (or 401)
  GET /api/auth/status → returns whether auth is enabled on this deployment
"""
from fastapi import APIRouter, Depends

from api.auth import AuthUser, auth_enabled, require_user

router = APIRouter(prefix="/auth", tags=["auth"])


@router.get("/status")
async def auth_status() -> dict:
    """Public endpoint — tells the frontend whether to render the login UI."""
    return {"enabled": auth_enabled()}


@router.get("/me")
async def me(user: AuthUser = Depends(require_user)) -> dict:
    return {
        "id":       user.id,
        "email":    user.email,
        "role":     user.role,
        "is_admin": user.is_admin,
        "app_role": user.app_role,
    }
