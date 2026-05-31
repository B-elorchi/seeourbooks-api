"""
PostgreSQL direct backend for the unified DB interface.
Uses asyncpg with a connection pool initialized at server startup.

JSONB columns: asyncpg does NOT auto-convert dicts/lists by default.
We register a custom codec on every new connection so that:
  - Python dicts/lists are auto-serialized to JSON strings before writing
  - JSON strings are auto-deserialized back to Python objects on read
This means callers can pass plain Python dicts; no manual json.dumps needed.
"""
import json
from typing import Any
import asyncpg
from api.config.settings import settings

_pool: asyncpg.Pool | None = None


async def _init_connection(conn: asyncpg.Connection) -> None:
    """Register JSON/JSONB codecs so asyncpg handles Python dicts transparently."""
    await conn.set_type_codec(
        "jsonb",
        encoder=lambda v: json.dumps(v, ensure_ascii=False),
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )
    await conn.set_type_codec(
        "json",
        encoder=lambda v: json.dumps(v, ensure_ascii=False),
        decoder=json.loads,
        schema="pg_catalog",
        format="text",
    )


async def startup() -> None:
    global _pool
    if not settings.DATABASE_URL:
        raise RuntimeError(
            "DB_BACKEND=postgres but DATABASE_URL is missing"
        )
    _pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=2,
        max_size=10,
        command_timeout=30,
        init=_init_connection,
    )


async def shutdown() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


def _pool_or_raise() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Postgres pool not initialised — startup() was not called")
    return _pool


# ── WHERE clause builder ──────────────────────────────────────────────────────

def _build_where(filters: dict | None) -> tuple[str, list]:
    """Return (WHERE clause string, positional values list)."""
    if not filters:
        return "", []

    clauses: list[str] = []
    values:  list[Any] = []

    _SQL_OPS = {"gte": ">=", "lte": "<=", "gt": ">", "lt": "<", "neq": "!="}

    for col, val in filters.items():
        if isinstance(val, tuple) and val[0] == "in":
            placeholders = ", ".join(f"${len(values) + i + 1}" for i in range(len(val[1])))
            clauses.append(f"{col} IN ({placeholders})")
            values.extend(val[1])
        elif isinstance(val, tuple) and val[0] in _SQL_OPS:
            clauses.append(f"{col} {_SQL_OPS[val[0]]} ${len(values) + 1}")
            values.append(val[1])
        else:
            clauses.append(f"{col} = ${len(values) + 1}")
            values.append(val)

    return "WHERE " + " AND ".join(clauses), values


# ── Column selector sanitiser ─────────────────────────────────────────────────

def _safe_select(select: str) -> str:
    """Validate that the select clause only contains safe characters."""
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_,* ")
    if any(c not in allowed for c in select):
        return "*"
    return select


# ── Unified interface ─────────────────────────────────────────────────────────

async def find(
    table: str,
    *,
    filters: dict[str, Any] | None = None,
    select: str = "*",
    order: str | None = None,
    limit: int | None = None,
) -> list[dict]:
    where, values = _build_where(filters)
    cols = _safe_select(select)

    # Accept "col DESC" / "col ASC" (SQL style) — used directly
    order_clause = f"ORDER BY {order}" if order else ""
    limit_clause = f"LIMIT {limit}" if limit is not None else ""

    sql = f"SELECT {cols} FROM {table} {where} {order_clause} {limit_clause}".strip()

    async with _pool_or_raise().acquire() as conn:
        rows = await conn.fetch(sql, *values)
    return [dict(r) for r in rows]


async def insert(table: str, data: dict) -> dict:
    cols   = ", ".join(data.keys())
    params = ", ".join(f"${i + 1}" for i in range(len(data)))
    sql    = f"INSERT INTO {table} ({cols}) VALUES ({params}) RETURNING *"

    async with _pool_or_raise().acquire() as conn:
        row = await conn.fetchrow(sql, *data.values())
    return dict(row)


async def upsert(table: str, data: dict, conflict: str) -> dict:
    cols          = ", ".join(data.keys())
    params        = ", ".join(f"${i + 1}" for i in range(len(data)))
    conflict_cols = conflict.replace(" ", "")
    updates       = ", ".join(
        f"{col} = EXCLUDED.{col}"
        for col in data.keys()
        if col not in conflict_cols.split(",")
    )
    sql = (
        f"INSERT INTO {table} ({cols}) VALUES ({params}) "
        f"ON CONFLICT ({conflict_cols}) DO UPDATE SET {updates} "
        f"RETURNING *"
    )

    async with _pool_or_raise().acquire() as conn:
        row = await conn.fetchrow(sql, *data.values())
    return dict(row)


async def update(table: str, filters: dict, data: dict) -> None:
    set_parts:   list[str] = []
    values:      list[Any] = []

    for i, (col, val) in enumerate(data.items()):
        set_parts.append(f"{col} = ${i + 1}")
        values.append(val)

    where_parts: list[str] = []
    for col, val in filters.items():
        where_parts.append(f"{col} = ${len(values) + 1}")
        values.append(val)

    sql = (
        f"UPDATE {table} "
        f"SET {', '.join(set_parts)} "
        f"WHERE {' AND '.join(where_parts)}"
    )

    async with _pool_or_raise().acquire() as conn:
        await conn.execute(sql, *values)
