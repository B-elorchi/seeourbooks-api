from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import summarize, books, jobs, health, pipeline, pipeline_v2, admin, document, documents, auth
from api.services.db import startup as db_startup, shutdown as db_shutdown
from api.services.config.migrations import run_migrations
from api.services.documents.errors import DocumentError


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_startup()       # connect pool (postgres) or validate creds (supabase)
    await run_migrations()   # fix any stale provider_config values automatically
    await _recover_stuck_jobs()
    yield
    await db_shutdown()      # graceful close


async def _recover_stuck_jobs() -> None:
    """
    On startup, recover jobs orphaned by a previous server instance.

    Both 'running' and 'queued' jobs are lost on restart because their
    background task lived only in the old process's event loop. We RE-DISPATCH
    them as fresh asyncio tasks here so they actually resume — simply flipping
    the DB status to 'queued' is not enough (nothing polls the queue).

    The job's partial result is passed back as previous_result, so the
    merge-aware retry skips steps that already finished.
    """
    import asyncio
    import logging
    from api.services.db import find, update
    from api.jobs.store import can_retry, _cancelled_jobs
    from api.routes.pipeline import _run_job          # local import avoids cycle
    from api.models.requests import PipelineReq
    log = logging.getLogger(__name__)

    try:
        stuck = []
        for status in ("running", "queued"):
            try:
                stuck += await find("pipeline_jobs", filters={"status": status}, limit=100)
            except Exception as exc:
                log.warning("Could not query '%s' jobs on startup: %s", status, exc)

        if not stuck:
            return

        log.warning("Recovering %d orphaned job(s) from previous server instance", len(stuck))
        for job in stuck:
            job_id = job["id"]
            try:
                # Clear any stale cancelled flag so the job can actually run
                _cancelled_jobs.discard(job_id)

                if not can_retry(job):
                    await update(
                        "pipeline_jobs",
                        filters={"id": job_id},
                        data={
                            "status": "failed",
                            "error_msg": "Server restarted — max retries exceeded",
                        },
                    )
                    log.warning("Job %s marked failed (max retries exceeded)", job_id)
                    continue

                req = PipelineReq.model_validate(job.get("input") or {})
                # Re-dispatch: resumes from the stored partial result.
                asyncio.create_task(
                    _run_job(job_id, req, job.get("result"), False)
                )
                log.info("Re-dispatched job %s (was %s)", job_id, job.get("status"))
            except Exception as exc:
                log.error("Could not recover job %s: %s", job_id, exc)
    except Exception as exc:
        log.warning("Could not check for stuck jobs on startup: %s", exc)


app = FastAPI(title="SeeOurBook Summarizer API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Any DocumentError that escapes a route maps to the typed HTTP status it
# carries (e.g. OCRMissingError → 503).  Background-task failures never hit
# this handler — they're caught inside the processor and written to the
# `documents.error_message` column for clients to poll.
@app.exception_handler(DocumentError)
async def _document_error_handler(_request: Request, exc: DocumentError) -> JSONResponse:
    body: dict = {"code": exc.code, "message": str(exc)}
    if exc.detail:
        body["detail"] = exc.detail
    return JSONResponse(status_code=exc.http_status, content=body)


app.include_router(summarize.router, prefix="/api")
app.include_router(books.router,     prefix="/api")
app.include_router(jobs.router,      prefix="/api")
app.include_router(health.router,    prefix="/api")
app.include_router(pipeline.router,    prefix="/api")
app.include_router(pipeline_v2.router, prefix="/api")
app.include_router(admin.router,     prefix="/api")
app.include_router(document.router,  prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(auth.router,      prefix="/api")
