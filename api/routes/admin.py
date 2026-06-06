"""
Admin routes:
  GET  /api/admin/config              → all current provider settings (flat key:value dict)
  POST /api/admin/config              → { key, value } — update one setting
  GET  /api/admin/metrics             → job counts and timing stats
  GET  /api/admin/jobs                → all pipeline jobs (alias with full detail)
  POST /api/admin/jobs/{job_id}/retry → manually retry a failed/partial job
  GET  /api/admin/costs               → aggregated cost breakdown for the last N days
  GET  /api/admin/openrouter-models   → cached, optionally-filtered live OpenRouter model list
"""
import logging
import time
from collections import Counter
from datetime import datetime, timedelta, timezone

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_admin
from api.services.config.runtime import get_all_config, set_config_key
from api.services.db import find
from api.jobs.store import get_job, reset_for_manual_retry
from api.models.requests import PipelineReq

log = logging.getLogger(__name__)

# Auth: every route in this router requires an admin user when SUPABASE_JWT_SECRET
# is configured.  In dev (no JWT secret) the dependency returns a dummy admin
# so the panel stays usable without a Supabase project.
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


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
    try:
        jobs = await find(
            "pipeline_jobs",
            select="status, created_at",
            order="created_at DESC",
            limit=500,
        )
    except Exception as exc:
        log.warning("admin_metrics: DB unreachable — %s", exc)
        # Return empty metrics rather than 500 so the admin panel still loads
        return {"total": 0, "done": 0, "partial": 0, "failed": 0, "running": 0, "queued": 0}

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
    try:
        return await find("pipeline_jobs", order="created_at DESC", limit=limit)
    except Exception as exc:
        log.warning("admin_jobs: DB unreachable — %s", exc)
        return []


# ── Catalog inspector — read-only proxy to a whitelist of tables ──────────────
#
# Powers the "Catalog" tab in the admin UI.  Useful for verifying that:
#   - production books / chunks / covers / etc. are visible to the API
#   - the pipeline is writing into the right tables
#   - per-book data is laid out as expected
#
# Whitelisted to prevent leaking unrelated DB tables through a generic API.
_CATALOG_TABLES: dict[str, dict] = {
    # ── client production tables ──
    "books":                  {"order": "book_id ASC",     "book_id_col": "book_id"},
    "ai_batches":             {"order": "system_created_at DESC", "book_id_col": "book_id"},
    "chunks":                 {"order": "chunk_index ASC", "book_id_col": "book_id"},
    "covers":                 {"order": "created_at DESC", "book_id_col": "bookId"},
    "reviews":                {"order": "updated_at DESC", "book_id_col": "book_id"},
    "audio":                  {"order": "updated_at DESC", "book_id_col": "book_id"},
    # ── seeourbook operational tables ──
    "pipeline_jobs":          {"order": "created_at DESC", "book_id_col": "book_id"},
    "pipeline_step_results":  {"order": "created_at DESC", "book_id_col": None},
    "book_summaries":         {"order": "created_at DESC", "book_id_col": "book_id"},
    "chunk_summaries":        {"order": "created_at DESC", "book_id_col": "book_id"},
    "usage_logs":             {"order": "created_at DESC", "book_id_col": None},
    "provider_config":        {"order": "updated_at DESC", "book_id_col": None},
    "uploaded_documents":     {"order": "created_at DESC", "book_id_col": None},
    "summary_jobs":           {"order": "created_at DESC", "book_id_col": "book_id"},
    # ── new documents pipeline ──
    "documents":              {"order": "created_at DESC", "book_id_col": None},
    "document_pages":         {"order": "page_number ASC", "book_id_col": None},
    "document_summaries":     {"order": "created_at DESC", "book_id_col": None},
    "knowledge_chunks":       {"order": "chunk_index ASC", "book_id_col": None},
}


@router.get("/catalog/tables")
async def admin_catalog_tables() -> dict:
    """List the tables exposed by /catalog/{table} along with their hints."""
    return {
        "tables": [
            {
                "name":            name,
                "default_order":   meta["order"],
                "supports_book_id": meta["book_id_col"] is not None,
            }
            for name, meta in _CATALOG_TABLES.items()
        ],
    }


@router.get("/catalog/{table}")
async def admin_catalog(
    table: str,
    limit:  int = 50,
    offset: int = 0,
    book_id: str | None = None,
) -> dict:
    """
    Return up to `limit` rows from the given table.

    Optional filters:
        book_id — when the table has a book_id-like column.  We map this to the
                  actual column name (some legacy tables use bookId / book_id).

    This endpoint is intentionally read-only and whitelisted.
    """
    if table not in _CATALOG_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"table {table!r} is not in the catalog whitelist. "
                   f"Allowed: {sorted(_CATALOG_TABLES.keys())}",
        )

    if limit < 1 or limit > 500:
        limit = max(1, min(500, limit))
    if offset < 0:
        offset = 0

    meta    = _CATALOG_TABLES[table]
    filters: dict | None = None

    if book_id and meta["book_id_col"]:
        # The production books / chunks / audio tables use INTEGER book_id.
        # Try numeric coercion first so eq.<int> matches.
        bid: object = book_id
        try:
            bid = int(book_id)
        except ValueError:
            pass
        filters = {meta["book_id_col"]: bid}

    try:
        rows = await find(
            table,
            filters=filters,
            order=meta["order"],
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        # Surface DB errors with a clean message instead of generic 500
        raise HTTPException(
            status_code=502,
            detail=f"DB read failed for {table!r}: {exc}",
        ) from exc

    return {
        "table":  table,
        "limit":  limit,
        "offset": offset,
        "count":  len(rows),
        "rows":   rows,
    }


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

    try:
        job = await get_job(job_id)
    except Exception as exc:
        raise HTTPException(503, f"Database unreachable: {exc}") from exc
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


# ── OpenRouter live model list (cached proxy) ────────────────────────────────
#
# OpenRouter's public /api/v1/models endpoint requires no auth and lists every
# model currently routable through them — including newly-added image, chat,
# and vision models.  We proxy + cache it for the admin Providers tab so the
# dropdowns always reflect what's actually available without us pushing code.

_OR_MODELS_CACHE: dict[str, tuple[float, list[dict]]] = {}
_OR_MODELS_CACHE_TTL_SEC = 3600   # 1 hour


async def _fetch_openrouter_models_raw() -> list[dict]:
    """Fetch the full OpenRouter model list (cached).  Returns stale cache on failure."""
    now    = time.time()
    cached = _OR_MODELS_CACHE.get("all")
    if cached and (now - cached[0]) < _OR_MODELS_CACHE_TTL_SEC:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get("https://openrouter.ai/api/v1/models")
            r.raise_for_status()
            models = (r.json() or {}).get("data") or []
    except Exception as exc:
        log.warning("OpenRouter /models fetch failed: %s", exc)
        # Return stale cache if any, else empty
        return cached[1] if cached else []

    _OR_MODELS_CACHE["all"] = (now, models)
    return models


@router.get("/openrouter-models")
async def openrouter_models(modality: str = "all") -> dict:
    """
    Live OpenRouter model list, optionally filtered by modality.

    `modality` values:
      - "all"     every model OpenRouter routes
      - "image"   models with image OUTPUT (for cover gen)
      - "vision"  models with image INPUT and text OUTPUT (for alt-text)
      - "chat"    models with text OUTPUT (for summarization / mindmap)

    Cached server-side for 1 hour.  Safe to call on every admin tab mount.
    """
    raw = await _fetch_openrouter_models_raw()

    def _out_mods(m: dict) -> list[str]:
        arch = m.get("architecture") or {}
        return arch.get("output_modalities") or []

    def _in_mods(m: dict) -> list[str]:
        arch = m.get("architecture") or {}
        return arch.get("input_modalities") or []

    if modality == "image":
        filtered = [m for m in raw if "image" in _out_mods(m)]
    elif modality == "vision":
        filtered = [m for m in raw if "image" in _in_mods(m) and "text" in _out_mods(m)]
    elif modality == "chat":
        filtered = [m for m in raw if "text" in _out_mods(m) and "image" not in _out_mods(m)]
    else:
        filtered = raw

    # Sort: by name when present, else by id — stable and predictable in the dropdown.
    filtered.sort(key=lambda m: (m.get("name") or m["id"]).lower())

    return {
        "modality": modality,
        "count":    len(filtered),
        "models": [
            {
                "id":      m["id"],
                "name":    m.get("name") or m["id"],
                "context": m.get("context_length"),
            }
            for m in filtered
        ],
    }
