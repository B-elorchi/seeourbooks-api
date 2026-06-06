"""
Documents pipeline REST API.

    POST /api/documents/upload          → upload a PDF, get document_id
    POST /api/documents/{id}/process    → kick off the background pipeline
    GET  /api/documents/{id}/status     → status + progress
    GET  /api/documents/{id}/text       → all extracted pages
    GET  /api/documents/{id}/summary    → AI summary
    GET  /api/documents/{id}/json       → structured AI analysis

All work past upload runs as a FastAPI BackgroundTask — the request returns
immediately and the client polls /status until status == 'completed'.

Authentication is intentionally NOT enforced here — wire it at the router
prefix (e.g. via a Depends() guard) once an auth scheme is chosen for the
admin surface.  See the test report for context.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    HTTPException,
    UploadFile,
)
from pydantic import BaseModel, Field

from api.config.settings import settings
from api.services.documents import repository as repo
from api.services.documents.processor import process_document

router = APIRouter(prefix="/documents", tags=["documents"])
log = logging.getLogger(__name__)


# ── List endpoint (diagnostic / admin) ──────────────────────────────────────

@router.get("")
async def list_documents(limit: int = 50) -> dict[str, Any]:
    """
    Quick diagnostic — return every document row the API can see.

    Use this to confirm the API is actually writing to the database you
    expect.  If this returns rows but PgAdmin shows none, your PgAdmin is
    connected to a different database than the API uses.
    """
    from api.services.db import find as _find
    try:
        rows = await _find(
            "documents",
            select="id, original_filename, status, progress, page_count, language, created_at",
            order="created_at DESC",
            limit=limit,
        )
    except Exception as exc:
        log.warning("list_documents: DB unreachable — %s", exc)
        return {"count": 0, "documents": [], "error": str(exc)[:200]}
    return {"count": len(rows), "documents": rows}


@router.get("/health")
async def documents_health() -> dict[str, Any]:
    """
    Self-test for the documents pipeline schema + backend connection.

    Pings each of the four tables (documents, document_pages,
    document_summaries, knowledge_chunks) and reports their row counts.
    A failure on any table means the schema isn't applied or the API can't
    reach the database it's configured for.

    Use after `db/schema.sql` is applied to your Supabase project:
        curl https://your-api/api/documents/health
    """
    from api.config.settings import settings
    from api.services.db import find as _find

    tables = ["documents", "document_pages", "document_summaries", "knowledge_chunks"]
    results: dict[str, Any] = {}
    overall_ok = True

    for table in tables:
        try:
            rows = await _find(table, select="id", limit=1)
            count_rows = await _find(table, select="id", limit=10_000)
            results[table] = {"ok": True, "rows": len(count_rows), "sample_ok": True if not rows else True}
        except Exception as exc:
            overall_ok = False
            results[table] = {"ok": False, "error": str(exc)[:300]}

    return {
        "ok":         overall_ok,
        "backend":    settings.DB_BACKEND,
        "supabase_url": (settings.SUPABASE_URL or "").split("//")[-1][:60] if settings.DB_BACKEND == "supabase" else None,
        "tables":     results,
    }


# ── Response models ─────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    documentId: str
    status:     str = Field(..., description="uploaded | processing | …")


class ProcessResponse(BaseModel):
    documentId: str
    status:     str = "processing"


class StatusResponse(BaseModel):
    documentId:    str
    status:        str
    progress:      int
    page_count:    int | None = None
    language:      str | None = None
    error_message: str | None = None


class PageOut(BaseModel):
    page:    int
    content: str


class TextResponse(BaseModel):
    documentId: str
    pages:      list[PageOut]


class SummaryResponse(BaseModel):
    documentId: str
    summary:    str
    provider:   str | None = None
    model:      str | None = None


class JsonResponse(BaseModel):
    documentId: str
    title:    str
    summary:  str
    topics:   list[Any]
    keywords: list[Any]
    authors:  list[Any]
    entities: list[Any]
    chapters: list[Any]
    raw:      dict[str, Any]


# ── Helpers ─────────────────────────────────────────────────────────────────

_PDF_MAGIC = b"%PDF-"


def _validate_pdf_bytes(raw: bytes) -> None:
    """Magic-byte + size guard.  Raises HTTPException directly."""
    if len(raw) > settings.DOC_MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"file too large — max {settings.DOC_MAX_UPLOAD_BYTES // 1024 // 1024} MB",
        )
    if not raw.startswith(_PDF_MAGIC):
        raise HTTPException(
            status_code=415,
            detail="not a PDF (missing %PDF- header)",
        )


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, status_code=201)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """
    Upload a PDF and create a `documents` row.  Returns the document id;
    call POST /documents/{id}/process next to start the pipeline.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="only .pdf files are accepted")

    raw = await file.read()
    _validate_pdf_bytes(raw)

    document_id = str(uuid.uuid4())
    target_dir  = Path(settings.DOCUMENTS_DIR) / document_id
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"could not create document storage dir at {target_dir}: {exc}",
        ) from exc

    target_path = target_dir / "original.pdf"
    target_path.write_bytes(raw)

    try:
        await repo.create_document({
            "id":                 document_id,
            "original_filename":  file.filename,
            "original_file_path": str(target_path),
            "status":             "uploaded",
            "progress":           0,
        })
    except Exception as exc:
        log.exception("upload_document: DB insert failed")
        # Clean up the orphan PDF — DB row is the source of truth.
        try:
            target_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise HTTPException(status_code=500, detail=f"could not persist document: {exc}") from exc

    return UploadResponse(documentId=document_id, status="uploaded")


@router.post("/{document_id}/process", response_model=ProcessResponse, status_code=202)
async def start_processing(
    document_id: str,
    background_tasks: BackgroundTasks,
) -> ProcessResponse:
    """
    Kick off the OCR → extract → AI → chunk pipeline as a background task.
    Returns 202 immediately.  Poll /status for progress.
    """
    try:
        doc = await repo.get_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}") from exc
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")

    # Reject double-processing — the processor is idempotent but we'd rather
    # surface a clear 409 than silently noop.
    if doc.get("status") == "processing":
        raise HTTPException(status_code=409, detail="document is already processing")

    background_tasks.add_task(process_document, document_id)
    return ProcessResponse(documentId=document_id, status="processing")


@router.get("/{document_id}/status", response_model=StatusResponse)
async def get_status(document_id: str) -> StatusResponse:
    try:
        doc = await repo.get_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}") from exc
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    return StatusResponse(
        documentId    = document_id,
        status        = doc.get("status") or "uploaded",
        progress      = int(doc.get("progress") or 0),
        page_count    = doc.get("page_count"),
        language      = doc.get("language"),
        error_message = doc.get("error_message"),
    )


@router.get("/{document_id}/text", response_model=TextResponse)
async def get_text(document_id: str) -> TextResponse:
    try:
        doc = await repo.get_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}") from exc
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    pages = await repo.get_pages(document_id)
    return TextResponse(
        documentId=document_id,
        pages=[PageOut(page=p["page"], content=p["content"]) for p in pages],
    )


@router.get("/{document_id}/summary", response_model=SummaryResponse)
async def get_summary(document_id: str) -> SummaryResponse:
    try:
        doc = await repo.get_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}") from exc
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    row = await repo.get_summary(document_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail="summary not yet generated — wait for status == ai_processed or completed",
        )
    return SummaryResponse(
        documentId = document_id,
        summary    = row.get("summary") or "",
        provider   = row.get("provider"),
        model      = row.get("model"),
    )


@router.get("/{document_id}/json", response_model=JsonResponse)
async def get_structured_json(document_id: str) -> JsonResponse:
    try:
        doc = await repo.get_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Database unreachable: {exc}") from exc
    if not doc:
        raise HTTPException(status_code=404, detail="document not found")
    row = await repo.get_summary(document_id)
    if not row:
        raise HTTPException(
            status_code=404,
            detail="structured analysis not yet generated",
        )
    j = row.get("structured_json") or {}
    if not isinstance(j, dict):
        j = {}

    def _list(key: str) -> list:
        v = j.get(key)
        return v if isinstance(v, list) else []

    return JsonResponse(
        documentId = document_id,
        title      = str(j.get("title") or ""),
        summary    = str(j.get("summary") or row.get("summary") or ""),
        topics     = _list("topics"),
        keywords   = _list("keywords"),
        authors    = _list("authors"),
        entities   = _list("entities"),
        chapters   = _list("chapters"),
        raw        = j,
    )
