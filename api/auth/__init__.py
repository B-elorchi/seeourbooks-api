"""Auth dependencies — Supabase JWT + API key auth."""
from api.auth.dependencies import (
    AuthUser,
    get_current_user,
    require_user,
    require_admin,
    auth_enabled,
)
from api.auth.apikey import (
    ApiKeyUser,
    generate_api_key,
    get_api_key_user,
    require_api_key,
    require_admin_key,
)

__all__ = [
    "AuthUser",
    "get_current_user",
    "require_user",
    "require_admin",
    "auth_enabled",
    "ApiKeyUser",
    "generate_api_key",
    "get_api_key_user",
    "require_api_key",
    "require_admin_key",
]
