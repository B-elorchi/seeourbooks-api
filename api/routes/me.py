"""
Self-scoped endpoints for the logged-in user (any role).

These mirror a subset of the admin endpoints but restrict results to rows the
caller owns (admins still see everything). They deliberately expose **no cost
or spend data** — that stays on the admin-only surface.

    GET /api/me/jobs       → the caller's pipeline jobs (newest first)
    GET /api/me/metrics     → status counts for the caller's jobs (no cost)
    GET /api/me/documents   → the caller's documents
"""
from __future__ import annotations

import logging
from collections import Counter

from fastapi import APIRouter, Depends

from api.auth import AuthUser, require_user
from api.services.db import find

log = logging.getLogger(__name__)
router = APIRouter(prefix="/me", tags=["me"])


# Editors create summary_jobs (via the Summary page) — that's their "jobs".
def _norm_status(s: str) -> str:
    """Map summary_jobs statuses onto the dashboard's status buckets."""
    s = (s or "").lower()
    if s in ("done", "completed", "complete", "success"):
        return "done"
    if s in ("running", "processing"):
        return "running"
    if s in ("failed", "error"):
        return "failed"
    if s in ("queued", "pending"):
        return "queued"
    if s == "partial":
        return "partial"
    return s or "queued"


@router.get("/jobs")
async def my_jobs(
    limit: int = 100,
    user: AuthUser = Depends(require_user),
) -> list:
    """Summary jobs owned by the caller (all jobs for admins)."""
    try:
        return await find(
            "summary_jobs",
            filters=user.owner_filter(),
            order="created_at DESC",
            limit=limit,
        )
    except Exception as exc:
        log.warning("my_jobs: DB unreachable — %s", exc)
        return []


@router.get("/metrics")
async def my_metrics(user: AuthUser = Depends(require_user)) -> dict:
    """Status counts for the caller's jobs. No cost/spend figures."""
    try:
        jobs = await find(
            "summary_jobs",
            filters=user.owner_filter(),
            select="status, created_at",
            order="created_at DESC",
            limit=500,
        )
    except Exception as exc:
        log.warning("my_metrics: DB unreachable — %s", exc)
        return {"total": 0, "done": 0, "partial": 0, "failed": 0, "running": 0, "queued": 0}

    counts = Counter(_norm_status(j["status"]) for j in jobs)
    return {
        "total":   len(jobs),
        "done":    counts.get("done", 0),
        "partial": counts.get("partial", 0),
        "failed":  counts.get("failed", 0),
        "running": counts.get("running", 0),
        "queued":  counts.get("queued", 0),
    }


@router.get("/documents")
async def my_documents(
    limit: int = 100,
    user: AuthUser = Depends(require_user),
) -> dict:
    """Documents owned by the caller (all for admins)."""
    try:
        rows = await find(
            "documents",
            filters=user.owner_filter(),
            select="id, original_filename, status, progress, page_count, language, created_at",
            order="created_at DESC",
            limit=limit,
        )
    except Exception as exc:
        log.warning("my_documents: DB unreachable — %s", exc)
        return {"count": 0, "documents": [], "error": str(exc)[:200]}
    return {"count": len(rows), "documents": rows}
