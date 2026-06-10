import asyncio
import copy
import json

from fastapi import APIRouter, BackgroundTasks, HTTPException

from api.jobs.store import (
    create_job, set_running, set_done, set_failed, set_partial, set_cancelled,
    get_job, get_output, list_jobs,
    can_retry, increment_retry,
)
from api.models.requests import PipelineReq
from api.services.pipeline.orchestrator import run_pipeline, JobCancelledError

router = APIRouter(prefix="/pipeline")

# Backoff delays per attempt: retry 1 → 5 s, retry 2 → 10 s, retry 3 → 20 s
_BACKOFF = [5, 10, 20]


# ── Result helpers ────────────────────────────────────────────────────────────

def _failed_steps(result: dict | str | None) -> list[str]:
    """
    Extract the names of steps that need to be (re)run from a previous result.

    Includes:
      - "failed"  — step ran but errored
      - "running" — step was in progress when the server was killed (restart recovery)
      - "pending" — step was queued but never started (also lost on restart)

    Returns empty list when result is missing or all steps are done/skipped
    (→ retry everything from scratch).
    """
    if not result:
        return []
    if isinstance(result, str):
        try:
            result = json.loads(result)
        except Exception:
            return []
    steps = result.get("steps") or {}
    return [s for s, status in steps.items() if status in ("failed", "running", "pending")]


def _merge_results(old: dict | str | None, new: dict) -> dict:
    """
    Merge a retry's partial result back into the previous result,
    keeping every step that already succeeded and replacing only what
    the retry actually ran.

    Rules
    ─────
    • Steps: keep old status unless the new run ran that step (≠ skipped)
    • metadata: take any non-None value from new over old
    • summaries / audio: dict-merge, new keys win
    • mindmap / quick_summary / chapters: take new if present, else keep old
    • errors: drop errors for steps that now succeeded; keep others; add new
    • status: recalculate from the merged step map
    """
    if not old:
        return new
    if isinstance(old, str):
        try:
            old = json.loads(old)
        except Exception:
            return new

    merged = copy.deepcopy(old)

    # ── Steps ──────────────────────────────────────────────────────────────────
    # Drop stale "running"/"pending" statuses that were left in the checkpoint
    # when the server restarted mid-job — they'll be replaced by the retry result.
    old_step_map = merged.get("steps") or {}
    for step, status in list(old_step_map.items()):
        if status in ("running", "pending"):
            old_step_map[step] = "failed"   # treat as failed so retry covers them

    new_steps = new.get("steps") or {}
    for step, status in new_steps.items():
        if status != "skipped":                     # skipped = wasn't attempted
            merged.setdefault("steps", {})[step] = status

    # ── Metadata (cover_url, alt_text, etc.) ──────────────────────────────────
    old_meta  = merged.get("metadata") or {}
    new_meta  = new.get("metadata") or {}
    merged["metadata"] = {**old_meta, **{k: v for k, v in new_meta.items() if v is not None}}

    # ── Summaries dict e.g. {"10min_ar": {...}} ───────────────────────────────
    merged["summaries"] = {**(merged.get("summaries") or {}), **(new.get("summaries") or {})}

    # ── Audio dict e.g. {"full_en": {...}} ───────────────────────────────────
    merged["audio"] = {**(merged.get("audio") or {}), **(new.get("audio") or {})}

    # ── Scalar assets — prefer new if present, keep old otherwise ───────────
    for key in ("mindmap", "epub", "video"):
        if new.get(key):
            merged[key] = new[key]
    if new.get("quick_summary"):
        merged["quick_summary"] = new["quick_summary"]
    if new.get("summary_qa"):
        merged["summary_qa"] = new["summary_qa"]
    if new.get("chapters"):
        merged["chapters"] = new["chapters"]
    if new.get("processing_time"):
        merged["processing_time"] = new["processing_time"]

    # ── files index — deep-merge so per-step URLs are preserved ──────────────
    old_files = merged.get("files") or {}
    new_files  = new.get("files") or {}
    merged_files = {**old_files, **{k: v for k, v in new_files.items() if v is not None}}
    # chapters list: prefer new if non-empty
    if new_files.get("chapters"):
        merged_files["chapters"] = new_files["chapters"]
    merged["files"] = merged_files

    # ── Errors — remove resolved, keep unretried, add new ────────────────────
    old_errors   = old.get("errors") or {}
    new_errors   = new.get("errors") or {}
    merged_errs  = {}
    for step, err in old_errors.items():
        new_status = new_steps.get(step)
        if new_status is None or new_status == "skipped":
            merged_errs[step] = err          # wasn't retried — error still stands
    merged_errs.update(new_errors)
    merged["errors"] = merged_errs

    # ── Overall status — recalculate ─────────────────────────────────────────
    all_steps    = merged.get("steps") or {}
    active       = [s for s, st in all_steps.items() if st != "skipped"]
    failed_count = sum(1 for s in active if all_steps[s] == "failed")
    if failed_count == 0:
        merged["status"] = "done"
    elif failed_count == len(active):
        merged["status"] = "failed"
    else:
        merged["status"] = "partial"

    return merged


# ── Step discovery ────────────────────────────────────────────────────────────

@router.get("/steps")
async def pipeline_steps():
    """
    Return the list of valid pipeline steps and their dependencies.
    Clients can use this to render a UI without hardcoding step names.
    """
    return {
        "steps": [
            {"name": "summarize",      "depends_on": [],            "description": "3-pass text summary (Haiku → Sonnet → Review)"},
            {"name": "audio_full",     "depends_on": ["summarize"], "description": "TTS of the full summary → one MP3"},
            {"name": "audio_chapters", "depends_on": ["summarize"], "description": "TTS per chapter → one MP3 each"},
            {"name": "cover",          "depends_on": [],            "description": "AI-generated cover image"},
            {"name": "alt_text",       "depends_on": ["cover"],     "description": "AI cover description for accessibility + SEO"},
            {"name": "mindmap",        "depends_on": ["summarize"], "description": "AI Mermaid mind-map → SVG"},
        ],
        "rules": [
            "Send `steps: []` (or omit) to run all steps.",
            "Send `steps: [\"summarize\", \"cover\"]` to run only those.",
            "Dependencies are auto-added — e.g. requesting `audio_full` implicitly runs `summarize`.",
            "Unknown step names return HTTP 422 with the list of valid steps.",
        ],
    }


# ── Job runner ────────────────────────────────────────────────────────────────

@router.post("/run", status_code=202)
async def pipeline_run(req: PipelineReq, background_tasks: BackgroundTasks):
    """
    Accept a book request and start the pipeline.

    Step selection
    ──────────────
    The `steps` field in the payload controls which parts of the pipeline run:
      • Omit it (or send [])           → run all 6 steps
      • Send a subset (any 1+ steps)   → run only those (+ their dependencies)

    Examples
    ────────
      {"book_id": "1342", "steps": ["summarize", "audio_full"]}
      {"book_id": "x", "steps": ["cover", "alt_text", "mindmap"]}

    Returns 202 immediately with a job_id.
    Poll /api/pipeline/status/{job_id} to check progress.
    """
    job_id = await create_job(req.book_id, req.model_dump())
    background_tasks.add_task(_run_job, job_id, req)
    return {"job_id": job_id, "status": "queued", "status_url": f"/api/pipeline/status/{job_id}"}


async def _run_job(
    job_id: str,
    req: PipelineReq,
    previous_result: dict | str | None = None,
    force_steps: bool = False,
) -> None:
    """
    Execute one pipeline attempt.

    previous_result — the stored result from the last partial/failed run.
                      When provided, only failed steps are re-run and the
                      new outputs are merged back with the old successes.
    force_steps     — when True, req.steps is used exactly as-is (admin rerun).
                      When False (default), failed steps from previous_result
                      take precedence over req.steps (auto-retry behaviour).
    """
    # If we have a previous result and we're NOT in forced-step mode,
    # narrow req.steps to only the failed ones (standard auto-retry behaviour).
    if not force_steps:
        failed = _failed_steps(previous_result)
        if failed:
            req = req.model_copy(update={"steps": failed})

    # Tag all downstream usage logs with this job_id (read by usage_logger via contextvars).
    from api.services.usage_logger import set_job_context  # noqa: PLC0415
    set_job_context(job_id)

    await set_running(job_id)
    try:
        result = await run_pipeline(req, job_id=job_id, previous_result=previous_result)

        # Merge new outputs with the previous partial result (if any)
        if previous_result:
            result = _merge_results(previous_result, result)

        if result["status"] == "done":
            await set_done(job_id, result)
        elif result["status"] == "partial":
            await set_partial(job_id, result)
        else:
            error_msg = str(result.get("errors") or "pipeline returned failed status")
            await _handle_failure(job_id, req, error_msg, result)
    except JobCancelledError:
        # Cancellation is intentional — don't retry, don't log as error.
        await set_cancelled(job_id)
    except Exception as e:
        await _handle_failure(job_id, req, str(e), previous_result)


async def _handle_failure(
    job_id: str,
    req: PipelineReq,
    error_msg: str,
    current_result: dict | str | None = None,
) -> None:
    """
    Called whenever a job attempt fails or stays partial.
    If retries remain → sleep (exponential backoff) then re-run only failed steps.
    Otherwise → mark permanently failed/partial.
    """
    job = await get_job(job_id)
    if job and can_retry(job):
        attempt = await increment_retry(job_id)   # bumps count, resets status → queued
        delay   = _BACKOFF[min(attempt - 1, len(_BACKOFF) - 1)]
        await asyncio.sleep(delay)
        # Pass the current (partial) result so the next attempt only retries failures
        await _run_job(job_id, req, previous_result=current_result)
    else:
        await set_failed(job_id, error_msg)


@router.get("/status/{job_id}")
async def pipeline_status(job_id: str):
    """Return current job status and result (once done)."""
    try:
        job = await get_job(job_id)
    except Exception as exc:
        raise HTTPException(503, f"Database unreachable: {exc}") from exc
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@router.post("/jobs/{job_id}/cancel", status_code=200)
async def cancel_job(job_id: str):
    """
    Request cancellation of a running job.
    The orchestrator checks this flag after every step and stops at the next
    safe boundary (between steps — not mid-LLM-call).
    """
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job["status"] not in ("running", "queued"):
        return {"ok": True, "message": f"Job already in terminal state: {job['status']}"}
    await set_cancelled(job_id)
    return {"ok": True, "job_id": job_id, "message": "Cancellation requested — job will stop at the next step boundary"}


@router.get("/output/{book_id}")
async def pipeline_output(book_id: str):
    """Return the latest completed pipeline output for a book."""
    try:
        job = await get_output(book_id)
    except Exception as exc:
        raise HTTPException(503, f"Database unreachable: {exc}") from exc
    if not job:
        raise HTTPException(404, "No completed pipeline job found for this book")
    return job.get("result", {})


@router.get("/jobs")
async def pipeline_jobs(limit: int = 50):
    """List all pipeline jobs (newest first)."""
    try:
        return await list_jobs(limit=limit)
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning("pipeline_jobs: DB unreachable — %s", exc)
        return []
