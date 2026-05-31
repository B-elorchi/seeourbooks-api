import httpx
from fastapi import APIRouter, HTTPException
from api.services.db.supabase import sg

router = APIRouter()


@router.get("/job/{job_id}")
async def get_job(job_id: int):
    """Return job status."""
    async with httpx.AsyncClient(timeout=30) as client:
        rows = await sg(client, f"summary_jobs?id=eq.{job_id}&select=*&limit=1")
    if not rows:
        raise HTTPException(404, "Job not found")
    return rows[0]
