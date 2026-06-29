import pathlib
p = pathlib.Path('api/jobs/store.py')
text = p.read_text('utf-8')

# Signature
text = text.replace(
    'async def list_jobs(limit: int = 50, offset: int = 0, status: str | None = None) -> list[dict]:',
    'async def list_jobs(limit: int = 50, offset: int = 0, status: str | None = None, date_filter: str | None = None) -> list[dict]:'
)

# Postgres WHERE building
old_pg = '''    _STATUS_CLAUSES: dict[str, str] = {
        "running": "status IN ('running', 'queued')",
        "done":    "status = 'done'",
        "failed":  "status IN ('failed', 'partial', 'cancelled')",
    }
    where = _STATUS_CLAUSES.get(status or "", "") if status else ""

    if settings.DB_BACKEND == "postgres":
        from api.services.db._postgres import _pool_or_raise
        where_clause = f"WHERE {where}" if where else ""'''

new_pg = '''    _STATUS_CLAUSES: dict[str, str] = {
        "running": "status IN ('running', 'queued')",
        "done":    "status = 'done'",
        "failed":  "status IN ('failed', 'partial', 'cancelled')",
    }
    where_parts = []
    if status and status in _STATUS_CLAUSES:
        where_parts.append(_STATUS_CLAUSES[status])
    if date_filter:
        where_parts.append(f"DATE(created_at) = '{date_filter}'")
    where_str = " AND ".join(where_parts)

    if settings.DB_BACKEND == "postgres":
        from api.services.db._postgres import _pool_or_raise
        where_clause = f"WHERE {where_str}" if where_str else ""'''
text = text.replace(old_pg, new_pg)

# Supabase WHERE building
old_sb = '''    else:
        # Supabase fallback
        filters = {}
        if where:'''

new_sb = '''    else:
        # Supabase fallback
        filters = {}
        if date_filter:
            filters["created_at"] = ("gte", f"{date_filter}T00:00:00Z")
        if status:'''
text = text.replace(old_sb, new_sb)

p.write_text(text, 'utf-8')
print("OK")
