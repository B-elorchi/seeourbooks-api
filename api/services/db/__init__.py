"""
Unified database interface.

All application code imports from here:
    from api.services.db import find, insert, upsert, update

The backend is selected at startup via the DB_BACKEND env var:
    DB_BACKEND=supabase  (default) — Supabase REST / PostgREST
    DB_BACKEND=postgres             — Direct PostgreSQL via asyncpg

Switching backends requires only a config change — no code changes anywhere else.
"""
from api.config.settings import settings

if settings.DB_BACKEND == "postgres":
    from api.services.db._postgres import find, insert, upsert, update, startup, shutdown
else:
    from api.services.db._supabase import find, insert, upsert, update, startup, shutdown

__all__ = ["find", "insert", "upsert", "update", "startup", "shutdown"]
