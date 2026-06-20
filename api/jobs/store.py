"""
Pipeline job store — backed by either Supabase or Postgres (see DB_BACKEND).
All functions use the unified db interface: find / insert / upsert / update / delete.
"""
from datetime import datetime, timedelta, timezone
import uuid

from api.services.db import find, insert, update, delete
from api.services.config.runtime import get_config_value

MAX_RETRIES = 8  # auto-retry attempts after first failure (QA + network + model errors)

# In-memory set of job IDs requested to cancel.
# Checked by the orchestrator after every checkpoint (between steps).
_cancelled_jobs: set[str] = set()


def is_cancelled(job_id: str) -> bool:
    """Return True if a cancel has been requested for this job."""
    return job_id in _cancelled_jobs


async def create_job(book_id: str, input_data: dict, user_id: str | None = None) -> str:
    # Generate the UUID client-side because the `id` column on `pipeline_jobs`
    # is declared `TEXT PRIMARY KEY` with no DEFAULT — inserting without an
    # explicit id triggers a NOT NULL violation and PostgREST returns 400.
    job_id = uuid.uuid4().hex
    max_ret = int(await get_config_value("PIPELINE_MAX_RETRIES", str(MAX_RETRIES)))
    data: dict = {
        "id":          job_id,
        "book_id":     book_id,
        "status":      "queued",
        "input":       input_data,
        "retry_count": 0,
        "max_retries": max_ret,
    }
    if user_id:
        data["user_id"] = user_id
    row = await insert("pipeline_jobs", data)
    return str(row.get("id") or job_id)


async def set_running(job_id: str) -> None:
    await update("pipeline_jobs", {"id": job_id}, {"status": "running"})


async def set_done(job_id: str, result: dict) -> None:
    await update("pipeline_jobs", {"id": job_id}, {"status": "done", "result": result})


async def set_failed(job_id: str, error: str) -> None:
    await update("pipeline_jobs", {"id": job_id}, {"status": "failed", "error_msg": error})


async def set_cancelled(job_id: str) -> None:
    """Mark job as cancelled in DB and in the in-memory set the orchestrator checks.
    
    CRITICAL: Preserves the existing 'result' field so that if the job had
    checkpointed partial progress, a subsequent retry can resume from it.
    """
    _cancelled_jobs.add(job_id)
    # Fetch existing result so we don't overwrite it with null
    job = await get_job(job_id)
    existing_result = job.get("result") if job else None
    update_data: dict = {"status": "cancelled", "error_msg": "Cancelled by admin"}
    if existing_result is not None:
        update_data["result"] = existing_result
    await update("pipeline_jobs", {"id": job_id}, update_data)


async def set_partial(job_id: str, result: dict) -> None:
    await update("pipeline_jobs", {"id": job_id}, {"status": "partial", "result": result})


# ── Retry helpers ─────────────────────────────────────────────────────────────

def can_retry(job: dict) -> bool:
    """Return True if the job has auto-retry attempts remaining."""
    return (job.get("retry_count") or 0) < (job.get("max_retries") or MAX_RETRIES)


async def increment_retry(job_id: str) -> int:
    """
    Bump retry_count by 1 and reset status to 'queued'.
    Returns the NEW retry_count so the caller can compute backoff delay.
    """
    job = await get_job(job_id)
    new_count = (job.get("retry_count") or 0) + 1
    # Clear cancelled flag on auto-retry (edge case: job was cancelled then recovered)
    _cancelled_jobs.discard(job_id)
    await update(
        "pipeline_jobs",
        {"id": job_id},
        {"retry_count": new_count, "status": "queued", "error_msg": None},
    )
    return new_count


async def reset_for_manual_retry(job_id: str) -> None:
    """
    Admin-triggered retry: zero out retry_count so the job gets a full
    3 fresh auto-retries again, and put it back in the queue.

    We intentionally do NOT clear 'result' here — the caller passes the
    existing result as previous_result so _run_job can merge the retry
    outputs into it (only failed steps are re-run).
    """
    # CRITICAL: Remove from the in-memory cancelled set so the job can actually run.
    # Without this, a previously-cancelled job would be immediately cancelled again.
    _cancelled_jobs.discard(job_id)
    await update(
        "pipeline_jobs",
        {"id": job_id},
        {"retry_count": 0, "status": "queued", "error_msg": None},
    )


# ── Queries ─────────────────────────────────────────────────────────────────--

async def get_job(job_id: str) -> dict | None:
    rows = await find("pipeline_jobs", filters={"id": job_id}, limit=1)
    return rows[0] if rows else None


async def get_step_results(job_id: str) -> list[dict]:
    """Return all persisted step results for a job (from pipeline_step_results table)."""
    rows = await find(
        "pipeline_step_results",
        filters={"job_id": job_id},
        select="step, status, output_url, error_msg, duration_sec",
    )
    return rows or []


async def get_output(book_id: str) -> dict | None:
    rows = await find(
        "pipeline_jobs",
        filters={"book_id": book_id, "status": ("in", ["done", "partial"])},
        order="created_at DESC",
        limit=1,
    )
    return rows[0] if rows else None


async def list_jobs(limit: int = 50, offset: int = 0, status: str | None = None) -> list[dict]:
    """
    Return a lightweight job list — only the columns needed by the sidebar.
    On Postgres: extracts result->'metadata' (title, author) instead of the
    full multi-MB result JSONB, making pagination fast even on large datasets.

    status filter values (mapped to DB columns):
      "running"  → status IN ('running', 'queued')
      "done"     → status = 'done'
      "failed"   → status IN ('failed', 'partial', 'cancelled')
      None / "all" → no filter (paginated)
    """
    from api.config.settings import settings
    # When a status filter is active fetch up to 2000 rows (all matching).
    # Pagination is disabled server-side so the client sees the full filtered set.
    if status and status != "all":
        limit = 2000
        offset = 0
    else:
        limit = min(max(limit, 1), 500)
        offset = max(offset, 0)

    _STATUS_CLAUSES: dict[str, str] = {
        "running": "status IN ('running', 'queued')",
        "done":    "status = 'done'",
        "failed":  "status IN ('failed', 'partial', 'cancelled')",
    }
    where = _STATUS_CLAUSES.get(status or "", "") if status else ""

    if settings.DB_BACKEND == "postgres":
        from api.services.db._postgres import _pool_or_raise
        where_clause = f"WHERE {where}" if where else ""
        sql = f"""
            SELECT id, book_id, status, created_at, input,
                   result->'metadata' AS metadata
            FROM pipeline_jobs
            {where_clause}
            ORDER BY created_at DESC
            LIMIT $1 OFFSET $2
        """
        async with _pool_or_raise().acquire() as conn:
            rows = await conn.fetch(sql, limit, offset)
        return [dict(r) for r in rows]

    # Supabase fallback
    filters = {}
    if where:
        # Map simplified status labels to Supabase filter format
        if status == "running":
            filters["status"] = ("in", ["running", "queued"])
        elif status == "done":
            filters["status"] = "done"
        elif status == "failed":
            filters["status"] = ("in", ["failed", "partial", "cancelled"])
    return await find("pipeline_jobs", filters=filters or None, order="created_at DESC",
                      limit=limit, offset=offset or None)


async def delete_job(job_id: str) -> None:
    """Permanently delete a job and its step results."""
    _cancelled_jobs.discard(job_id)
    await delete("pipeline_step_results", {"job_id": job_id})
    await delete("pipeline_jobs", {"id": job_id})


async def timeout_stuck_jobs(max_age_minutes: int = 60) -> list[str]:
    """
    Mark jobs that have been 'running' for longer than `max_age_minutes` as failed.
    Returns the list of timed-out job IDs.
    """
    cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, max_age_minutes))
    try:
        rows = await find(
            "pipeline_jobs",
            filters={"status": "running", "updated_at": ("lte", cutoff.isoformat())},
            select="id",
            limit=1000,
        )
    except Exception:
        return []

    timed_out: list[str] = []
    for r in rows:
        jid = r.get("id")
        if not jid:
            continue
        await update(
            "pipeline_jobs",
            {"id": jid},
            {
                "status": "failed",
                "error_msg": f"Timed out after running for more than {max_age_minutes} minutes",
            },
        )
        timed_out.append(jid)

    return timed_out
