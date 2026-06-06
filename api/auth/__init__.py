"""Supabase Auth dependencies — see auth.dependencies for the FastAPI surface."""
from api.auth.dependencies import (
    AuthUser,
    get_current_user,
    require_user,
    require_admin,
    auth_enabled,
)

__all__ = [
    "AuthUser",
    "get_current_user",
    "require_user",
    "require_admin",
    "auth_enabled",
]
