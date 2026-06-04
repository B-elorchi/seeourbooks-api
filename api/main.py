from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routes import summarize, books, jobs, health, pipeline, admin, document, documents
from api.services.db import startup as db_startup, shutdown as db_shutdown
from api.services.config.migrations import run_migrations
from api.services.documents.errors import DocumentError


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db_startup()       # connect pool (postgres) or validate creds (supabase)
    await run_migrations()   # fix any stale provider_config values automatically
    yield
    await db_shutdown()      # graceful close


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
app.include_router(pipeline.router,  prefix="/api")
app.include_router(admin.router,     prefix="/api")
app.include_router(document.router,  prefix="/api")
app.include_router(documents.router, prefix="/api")
