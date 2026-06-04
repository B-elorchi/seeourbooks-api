"""
Legacy job-status endpoint:
  GET /api/job/{job_id} → row from the `summary_jobs` table (the SSE pipeline's
                          legacy tracking table, not `pipeline_jobs`).

Prefer `GET /api/pipeline/status/{job_id}` for new clients — that returns the
modern pipeline-engine job state with per-step status and retry info.
"""
import logging

import httpx
from fastapi import APIRouter, HTTPException

from api.services.db.supabase import sg

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/job/{job_id}")
async def get_job(job_id: int):
    """Return the legacy summary_jobs row for the given numeric id, or 404."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            rows = await sg(client, f"summary_jobs?id=eq.{job_id}&select=*&limit=1")
    except httpx.HTTPStatusError as exc:
        log.warning("get_job(%s): DB returned %s — %s",
                    job_id, exc.response.status_code, exc.response.text[:300])
        raise HTTPException(
            status_code=502,
            detail=f"Database error while fetching job {job_id}",
        ) from exc
    except httpx.RequestError as exc:
        log.error("get_job(%s): DB unreachable — %s", job_id, exc)
        raise HTTPException(status_code=503, detail="Database unreachable") from exc
    except Exception as exc:
        log.exception("get_job(%s) unexpected failure", job_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if not rows:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return rows[0]
