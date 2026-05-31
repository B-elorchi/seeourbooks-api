"""
Admin routes:
  GET  /api/admin/config              → all current provider settings (flat key:value dict)
  POST /api/admin/config              → { key, value } — update one setting
  GET  /api/admin/metrics             → job counts and timing stats
  GET  /api/admin/jobs                → all pipeline jobs (alias with full detail)
  POST /api/admin/jobs/{job_id}/retry → manually retry a failed/partial job
  GET  /api/admin/costs               → aggregated cost breakdown for the last N days
"""
from collections import Counter
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from api.services.config.runtime import get_all_config, set_config_key
from api.services.db import find
from api.jobs.store import get_job, reset_for_manual_retry
from api.models.requests import PipelineReq

router = APIRouter(prefix="/admin", tags=["admin"])


class ConfigUpdate(BaseModel):
    key:   str
    value: str


@router.get("/config")
async def admin_get_config() -> dict:
    """Return all provider settings as a flat key → value dict."""
    return await get_all_config()


@router.post("/config")
async def admin_set_config(body: ConfigUpdate) -> dict:
    """Update a single provider setting. Takes effect on the next pipeline job."""
    if not body.key:
        raise HTTPException(400, "key is required")
    await set_config_key(body.key, body.value)
    return {"ok": True, "key": body.key, "value": body.value}


@router.get("/metrics")
async def admin_metrics() -> dict:
    """Return job counts and aggregate stats for the last 500 jobs."""
    jobs = await find(
        "pipeline_jobs",
        select="status, created_at",
        order="created_at DESC",
        limit=500,
    )

    counts = Counter(j["status"] for j in jobs)
    return {
        "total":   len(jobs),
        "done":    counts.get("done", 0),
        "partial": counts.get("partial", 0),
        "failed":  counts.get("failed", 0),
        "running": counts.get("running", 0),
        "queued":  counts.get("queued", 0),
    }


@router.get("/jobs")
async def admin_jobs(limit: int = 100) -> list:
    """List all pipeline jobs ordered newest-first."""
    return await find("pipeline_jobs", order="created_at DESC", limit=limit)


@router.get("/costs")
async def admin_costs(days: int = 30) -> dict:
    """
    Aggregated cost breakdown for the last N days.

    Returns totals plus three groupings — by provider, by step, by model —
    each sorted by cost descending.  All amounts are USD estimates derived
    from the rate table in `usage_logger.py`.
    """
    if days < 1:
        days = 1
    if days > 365:
        days = 365

    since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        rows = await find(
            "usage_logs",
            filters={"created_at": ("gte", since_iso)},
            order="created_at DESC",
            limit=20000,
        )
    except Exception:
        # Table absent or query rejected — return an empty (but valid) shape
        rows = []

    by_provider: dict[str, dict] = {}
    by_step:     dict[str, dict] = {}
    by_model:    dict[str, dict] = {}
    total_cost  = 0.0

    for r in rows:
        cost  = float(r.get("cost_usd") or 0)
        units = float(r.get("units")    or 0)
        total_cost += cost

        p = r.get("provider") or "unknown"
        prov = by_provider.setdefault(p, {"provider": p, "calls": 0, "cost_usd": 0.0, "units": 0.0})
        prov["calls"]    += 1
        prov["cost_usd"] += cost
        prov["units"]    += units

        s = r.get("step") or "unknown"
        st = by_step.setdefault(s, {"step": s, "calls": 0, "cost_usd": 0.0})
        st["calls"]    += 1
        st["cost_usd"] += cost

        m = r.get("model") or "unknown"
        md = by_model.setdefault(m, {"model": m, "provider": p, "calls": 0, "cost_usd": 0.0, "units": 0.0,
                                      "unit_type": r.get("unit_type") or ""})
        md["calls"]    += 1
        md["cost_usd"] += cost
        md["units"]    += units

    def _round(rows_: list[dict]) -> list[dict]:
        for x in rows_:
            x["cost_usd"] = round(x["cost_usd"], 4)
            if "units" in x:
                x["units"] = round(x["units"], 2)
        return sorted(rows_, key=lambda x: x["cost_usd"], reverse=True)

    return {
        "days":           days,
        "total_calls":    len(rows),
        "total_cost_usd": round(total_cost, 4),
        "by_provider":    _round(list(by_provider.values())),
        "by_step":        _round(list(by_step.values())),
        "by_model":       _round(list(by_model.values())),
    }


@router.post("/jobs/{job_id}/retry", status_code=202)
async def admin_retry_job(job_id: str, background_tasks: BackgroundTasks) -> dict:
    """
    Manually retry a pipeline job from the admin panel.

    Smart retry — only re-runs steps that failed, merges results with
    the previous partial output so successful steps are not wasted.

    - Resets retry_count to 0 so the job gets a full 3 fresh auto-retries.
    - Works on any status (failed, partial).
    """
    # Lazy imports — avoids circular dependency (admin ↔ pipeline)
    from api.routes.pipeline import _run_job, _failed_steps   # noqa: PLC0415

    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    raw_input = job.get("input")
    if not raw_input:
        raise HTTPException(422, "Job has no stored input — cannot retry")

    try:
        req = PipelineReq.model_validate(raw_input)
    except Exception as exc:
        raise HTTPException(422, f"Stored input is invalid: {exc}") from exc

    # Carry the previous (partial) result so _run_job can merge after retry
    previous_result = job.get("result")

    # Tell the user which steps will be re-run
    failed = _failed_steps(previous_result)

    await reset_for_manual_retry(job_id)
    background_tasks.add_task(_run_job, job_id, req, previous_result)

    return {
        "ok":            True,
        "job_id":        job_id,
        "status":        "queued",
        "retrying_steps": failed or "all",   # "all" when there's no prior result
        "status_url":    f"/api/pipeline/status/{job_id}",
    }
