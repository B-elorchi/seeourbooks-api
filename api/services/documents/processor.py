"""
Documents pipeline orchestrator.

`process_document(document_id)` runs the full flow:

    uploaded  →  processing  →  ocr_completed  →  text_extracted
              →  ai_processed →  completed

Each stage:
  - reads the document row
  - does its work
  - persists output
  - advances the status + progress

Resumable: if you re-invoke process_document on a row in `text_extracted`
status, it will skip OCR + extraction and start at AI analysis.  This makes
manual retries cheap.

Failures:
  - Any uncaught exception sets status = 'failed' + writes error_message,
    so the row tells the client what went wrong.
  - Step-level errors include their `code` for programmatic mapping.
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from api.config.settings import settings
from api.services.config.runtime import get_config_value
from api.services.documents import repository as repo
from api.services.documents.ai import get_provider
from api.services.documents.chunking import chunk_text
from api.services.documents.embeddings import embed_texts
from api.services.documents.errors import (
    DocumentError,
    DocumentNotFound,
    InvalidPDFError,
)
from api.services.documents.extract import detect_language, extract_pages
from api.services.documents.ocr import needs_ocr, run_ocrmypdf
from api.services.usage_logger import set_job_context, set_step

log = logging.getLogger(__name__)


# Status progression — used to decide what to skip when resuming.
_STATUS_ORDER = [
    "uploaded",
    "processing",
    "ocr_completed",
    "text_extracted",
    "ai_processed",
    "completed",
]


def _stage_done(current: str, target: str) -> bool:
    """True if the document has already passed (or is at) `target`."""
    try:
        return _STATUS_ORDER.index(current) >= _STATUS_ORDER.index(target)
    except ValueError:
        return False


async def process_document(document_id: str) -> None:
    """
    Entry point for the background worker. Idempotent — safe to retry.
    """
    # Tag every usage_logs row written during this run with the document_id
    # so the Costs tab can group spend by document.
    set_job_context(f"doc:{document_id}")

    doc = await repo.get_document(document_id)
    if not doc:
        raise DocumentNotFound(f"document {document_id}")

    try:
        await _run_stages(doc)
    except DocumentError as exc:
        log.exception("process_document(%s) failed: %s", document_id, exc)
        await repo.set_status(
            document_id,
            "failed",
            error_message=f"{exc.code}: {exc}",
        )
        raise
    except Exception as exc:
        log.exception("process_document(%s) unexpected failure", document_id)
        await repo.set_status(
            document_id,
            "failed",
            error_message=f"unexpected: {exc}",
        )
        raise


async def _run_stages(doc: dict) -> None:
    document_id   = doc["id"]
    original_path = doc.get("original_file_path") or ""
    current       = doc.get("status") or "uploaded"

    if not original_path or not Path(original_path).exists():
        raise InvalidPDFError(
            f"Original PDF missing at {original_path!r}",
        )

    # Move from 'uploaded' → 'processing' immediately so the client sees activity
    if current in ("uploaded", "failed"):
        await repo.set_status(document_id, "processing", progress=5, error_message="")
        current = "processing"

    # ── Stage 1 — OCR (skip when PDF already has a text layer or already done) ──
    set_step("ocr")
    ocr_path = doc.get("ocr_file_path")
    if not _stage_done(current, "ocr_completed"):
        loop = asyncio.get_running_loop()
        try:
            needs = await loop.run_in_executor(None, needs_ocr, original_path)
        except DocumentError:
            raise
        except Exception as exc:
            raise InvalidPDFError(f"text-layer detection failed: {exc}") from exc

        if needs:
            ocr_path = str(Path(original_path).parent / "ocr.pdf")
            languages = (
                await get_config_value("DOC_OCR_LANGUAGES", settings.DOC_OCR_LANGUAGES)
                or settings.DOC_OCR_LANGUAGES
            )
            log.info("doc=%s: running OCR with languages=%s", document_id, languages)
            await run_ocrmypdf(original_path, ocr_path, languages=languages)
            await repo.update_document(document_id, {"ocr_file_path": ocr_path})
        else:
            log.info("doc=%s: PDF already has text — skipping OCR", document_id)
            ocr_path = original_path

        await repo.set_status(document_id, "ocr_completed", progress=30)
        current = "ocr_completed"

    # ── Stage 2 — Text extraction ──────────────────────────────────────────────
    set_step("extract")
    pages_data: list[dict] = []
    total_pages = doc.get("page_count") or 0
    if not _stage_done(current, "text_extracted"):
        source = ocr_path or original_path
        loop = asyncio.get_running_loop()
        pages_data, total_pages = await loop.run_in_executor(None, extract_pages, source)
        language = detect_language(pages_data)
        await repo.save_pages(document_id, pages_data)
        await repo.update_document(document_id, {
            "page_count": total_pages,
            "language":   language,
        })
        await repo.set_status(document_id, "text_extracted", progress=55)
        current = "text_extracted"

    # Reload pages from DB if we resumed past the extraction stage
    if not pages_data:
        pages_data = await repo.get_pages(document_id)
        if not pages_data:
            raise InvalidPDFError("no pages found after extraction stage")

    full_text = "\n\n".join(p["content"] for p in pages_data)
    language  = doc.get("language") or detect_language(pages_data)

    # ── Stage 3 — AI summary + structured JSON ─────────────────────────────────
    set_step("ai_analysis")
    if not _stage_done(current, "ai_processed"):
        provider = await get_provider()
        log.info("doc=%s: running AI analysis with provider=%s model=%s",
                 document_id, provider.name, provider.model)

        # Summary and structured analysis happen in parallel — they're
        # independent prompts on the same text.
        summary_task     = asyncio.create_task(provider.generate_summary(full_text, language))
        structured_task  = asyncio.create_task(provider.generate_structured_json(full_text, language))
        summary, structured = await asyncio.gather(summary_task, structured_task)

        await repo.save_summary(
            document_id,
            summary=summary,
            structured_json=structured,
            provider=provider.name,
            model=provider.model,
        )
        await repo.set_status(document_id, "ai_processed", progress=85)
        current = "ai_processed"

    # ── Stage 4 — Chunking + embeddings ────────────────────────────────────────
    set_step("embed")
    chunk_size = int(
        await get_config_value("DOC_CHUNK_SIZE_WORDS", str(settings.DOC_CHUNK_SIZE_WORDS))
        or settings.DOC_CHUNK_SIZE_WORDS
    )
    chunks = chunk_text(full_text, target_words=chunk_size)
    if chunks:
        vectors = await embed_texts([c["content"] for c in chunks])
        embedding_model = (
            await get_config_value("EMBEDDING_MODEL", settings.EMBEDDING_MODEL)
            if any(v is not None for v in vectors)
            else None
        )
        await repo.save_chunks(document_id, chunks, vectors, embedding_model=embedding_model)

    await repo.set_status(document_id, "completed", progress=100, error_message="")
    log.info("doc=%s: pipeline complete (%d pages, %d chunks)",
             document_id, total_pages, len(chunks))
