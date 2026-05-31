from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes import summarize, books, jobs, health, pipeline, admin, document
from api.services.db import startup as db_startup, shutdown as db_shutdown
from api.services.config.migrations import run_migrations


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

app.include_router(summarize.router, prefix="/api")
app.include_router(books.router,     prefix="/api")
app.include_router(jobs.router,      prefix="/api")
app.include_router(health.router,    prefix="/api")
app.include_router(pipeline.router,  prefix="/api")
app.include_router(admin.router,     prefix="/api")
app.include_router(document.router,  prefix="/api")
