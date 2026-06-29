import pathlib
p = pathlib.Path('api/jobs/store.py')
text = p.read_text('utf-8')

old_fn = '''def is_cancelled(job_id: str) -> bool:
    """Return True if a cancel has been requested for this job."""
    return job_id in _cancelled_jobs'''

new_fn = '''async def is_cancelled(job_id: str) -> bool:
    """Return True if a cancel has been requested for this job."""
    if job_id in _cancelled_jobs:
        return True
    
    from api.config.settings import settings
    if settings.DB_BACKEND == "postgres":
        from api.services.db._postgres import _pool_or_raise
        sql = "SELECT status FROM pipeline_jobs WHERE id = "
        try:
            async with _pool_or_raise().acquire() as conn:
                row = await conn.fetchrow(sql, job_id)
                if row and row["status"] == "cancelled":
                    _cancelled_jobs.add(job_id)
                    return True
        except Exception:
            pass
    else:
        from api.services.db._supabase import find
        try:
            res = await find("pipeline_jobs", filters={"id": job_id}, select="status")
            if res and res[0].get("status") == "cancelled":
                _cancelled_jobs.add(job_id)
                return True
        except Exception:
            pass
            
    return False'''

text = text.replace(old_fn, new_fn)
p.write_text(text, 'utf-8')

# Update orchestrator.py
p_orch = pathlib.Path('api/services/pipeline/orchestrator.py')
text_orch = p_orch.read_text('utf-8')
text_orch = text_orch.replace('if is_cancelled(job_id):', 'if await is_cancelled(job_id):')
p_orch.write_text(text_orch, 'utf-8')

# Update pipeline_v2.py
p_v2 = pathlib.Path('api/routes/pipeline_v2.py')
text_v2 = p_v2.read_text('utf-8')
text_v2 = text_v2.replace('if is_cancelled(job_id):', 'if await is_cancelled(job_id):')
p_v2.write_text(text_v2, 'utf-8')

print("OK")
