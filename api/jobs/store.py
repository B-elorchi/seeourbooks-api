"""
Pipeline job store — backed by either Supabase or Postgres (see DB_BACKEND).
All functions use the unified db interface: find / insert / upsert / update.
"""
from api.services.db import find, insert, update

MAX_RETRIES = 3


async def create_job(book_id: str, input_data: dict) -> str:
    row = await insert("pipeline_jobs", {
        "book_id":     book_id,
        "status":      "queued",
        "input":       input_data,
        "retry_count": 0,
        "max_retries": MAX_RETRIES,
    })
    return str(row["id"])


async def set_running(job_id: str) -> None:
    await update("pipeline_jobs", {"id": job_id}, {"status": "running"})


async def set_done(job_id: str, result: dict) -> None:
    await update("pipeline_jobs", {"id": job_id}, {"status": "done", "result": result})


async def set_failed(job_id: str, error: str) -> None:
    await update("pipeline_jobs", {"id": job_id}, {"status": "failed", "error_msg": error})


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
    await update(
        "pipeline_jobs",
        {"id": job_id},
        {"retry_count": 0, "status": "queued", "error_msg": None},
    )


# ── Queries ───────────────────────────────────────────────────────────────────

async def get_job(job_id: str) -> dict | None:
    rows = await find("pipeline_jobs", filters={"id": job_id}, limit=1)
    return rows[0] if rows else None


async def get_output(book_id: str) -> dict | None:
    rows = await find(
        "pipeline_jobs",
        filters={"book_id": book_id, "status": ("in", ["done", "partial"])},
        order="created_at DESC",
        limit=1,
    )
    return rows[0] if rows else None


async def list_jobs(limit: int = 50) -> list[dict]:
    return await find("pipeline_jobs", order="created_at DESC", limit=limit)
