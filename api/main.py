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
    On startup, handle jobs that were left in 'running' state from a previous
    server instance. 
    
    - If job has retries remaining: mark as 'queued' to auto-retry
    - If job has no retries left: mark as 'failed' with cancellation message
    
    The job's partial result is preserved so successful steps don't need to re-run.
    """
    import logging
    from api.services.db import find, update
    from api.jobs.store import can_retry
    log = logging.getLogger(__name__)
    try:
        stuck = await find("pipeline_jobs", filters={"status": "running"}, limit=100)
        if stuck:
            log.warning("Recovering %d stuck 'running' job(s) from previous server instance", len(stuck))
            for job in stuck:
                try:
                    job_id = job["id"]
                    if can_retry(job):
                        # Job can retry - put it back in queue
                        # The retry logic will use the partial result to skip successful steps
                        await update(
                            "pipeline_jobs",
                            filters={"id": job_id},
                            data={
                                "status": "queued", 
                                "error_msg": "Server restarted — will retry automatically"
                            },
                        )
                        log.info("Job %s queued for auto-retry (retry_count=%s)", job_id, job.get("retry_count", 0))
                    else:
                        # No retries left - mark as failed/cancelled
                        await update(
                            "pipeline_jobs",
                            filters={"id": job_id},
                            data={
                                "status": "failed", 
                                "error_msg": "Server restarted while job was running — max retries exceeded"
                            },
                        )
                        log.warning("Job %s marked as failed (max retries exceeded)", job_id)
                except Exception as exc:
                    log.error("Could not recover job %s: %s", job["id"], exc)
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
