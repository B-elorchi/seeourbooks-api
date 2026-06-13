"""
Thin DB layer for the documents pipeline — every read/write is wrapped here
so the processor and the route handlers don't reach into supabase URLs.

JSONB columns
─────────────
Both DB backends already handle Python dicts/lists for JSONB columns:
  - _postgres.py registers a JSONB type codec that calls json.dumps() on the
    encoder side.
  - _supabase.py serialises the whole row to JSON via httpx, which PostgREST
    inserts into JSONB columns directly.

So: DO NOT call json.dumps() on values headed for JSONB columns here — passing
raw dicts/lists is correct, and pre-serialising would double-encode the value.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from api.services.db import find, insert, update, upsert
from api.services.documents.errors import PageSaveError

log = logging.getLogger(__name__)

# How many times to retry persisting a single page before giving up.
_PAGE_SAVE_RETRIES = 3


# ── documents ────────────────────────────────────────────────────────────────

async def create_document(row: dict[str, Any]) -> dict[str, Any]:
    """Insert a new row into `documents` and return it."""
    return await insert("documents", row)


async def get_document(document_id: str) -> dict[str, Any] | None:
    rows = await find("documents", filters={"id": document_id}, limit=1)
    return rows[0] if rows else None


async def update_document(document_id: str, patch: dict[str, Any]) -> None:
    """Update one document row. Best-effort — never raises."""
    try:
        await update("documents", filters={"id": document_id}, data=patch)
    except Exception as exc:
        log.warning("update_document(%s) failed: %s", document_id, exc)


async def set_status(document_id: str, status: str, *, progress: int | None = None,
                     error_message: str | None = None) -> None:
    patch: dict[str, Any] = {"status": status}
    if progress is not None:
        patch["progress"] = max(0, min(100, progress))
    if error_message is not None:
        patch["error_message"] = error_message[:1000]
    await update_document(document_id, patch)


# ── document_pages ───────────────────────────────────────────────────────────

async def save_pages(document_id: str, pages: list[dict[str, Any]]) -> None:
    """
    Bulk-write extracted pages — ALL of them, or raise.

    We use upsert(on=(document_id, page_number)) so re-running the processor
    on the same document overwrites old page content instead of duplicating.
    The supabase REST and asyncpg backends both support that conflict target
    on the existing UNIQUE constraint.

    Reliability: a transient DB/network hiccup on a single page must NOT result
    in a document silently missing pages. Each page is retried a few times, and
    if any page still can't be saved we raise PageSaveError so the processor
    marks the document `failed` (with a clear message) instead of advancing with
    incomplete text.
    """
    failed: list[int] = []
    last_exc: Exception | None = None

    for p in pages:
        saved = False
        for attempt in range(_PAGE_SAVE_RETRIES):
            try:
                await upsert(
                    "document_pages",
                    {
                        "document_id": document_id,
                        "page_number": p["page"],
                        "content":     p["content"],
                    },
                    conflict="document_id,page_number",
                )
                saved = True
                break
            except Exception as exc:
                last_exc = exc
                log.warning(
                    "save_pages: page %s for %s failed (attempt %d/%d): %s",
                    p.get("page"), document_id, attempt + 1, _PAGE_SAVE_RETRIES, exc,
                )
                if attempt + 1 < _PAGE_SAVE_RETRIES:
                    await asyncio.sleep(0.5 * (attempt + 1))  # linear backoff
        if not saved:
            failed.append(p["page"])

    if failed:
        raise PageSaveError(
            f"Failed to save {len(failed)} of {len(pages)} page(s) "
            f"(pages: {failed[:20]}{'…' if len(failed) > 20 else ''}). "
            f"Last error: {last_exc}",
            detail={"document_id": document_id, "failed_pages": failed},
        )

    log.info("save_pages: persisted all %d page(s) for %s", len(pages), document_id)


async def get_pages(document_id: str) -> list[dict[str, Any]]:
    rows = await find(
        "document_pages",
        filters={"document_id": document_id},
        select="page_number, content",
        order="page_number ASC",
        limit=10_000,
    )
    return [{"page": r["page_number"], "content": r["content"]} for r in rows]


# ── document_summaries ───────────────────────────────────────────────────────

async def save_summary(
    document_id: str,
    summary: str,
    structured_json: dict[str, Any],
    *,
    provider: str,
    model:    str,
) -> None:
    await upsert(
        "document_summaries",
        {
            "document_id":     document_id,
            "summary":         summary,
            "structured_json": structured_json,   # raw dict — backend handles JSONB
            "provider":        provider,
            "model":           model,
        },
        conflict="document_id",
    )


async def get_summary(document_id: str) -> dict[str, Any] | None:
    rows = await find(
        "document_summaries",
        filters={"document_id": document_id},
        select="summary, structured_json, provider, model",
        limit=1,
    )
    if not rows:
        return None
    row = rows[0]
    sj = row.get("structured_json")
    if isinstance(sj, str):
        try:
            row["structured_json"] = json.loads(sj)
        except json.JSONDecodeError:
            row["structured_json"] = {}
    return row


# ── knowledge_chunks ─────────────────────────────────────────────────────────

async def save_chunks(
    document_id: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float] | None],
    *,
    embedding_model: str | None,
) -> None:
    """
    Bulk-insert chunks + (optional) embeddings.  Existing chunks for this
    document are overwritten on (document_id, chunk_index) conflict.
    """
    for chunk, vec in zip(chunks, embeddings, strict=False):
        try:
            await upsert(
                "knowledge_chunks",
                {
                    "document_id":     document_id,
                    "chunk_index":     chunk["chunk_index"],
                    "content":         chunk["content"],
                    "word_count":      chunk["word_count"],
                    # Raw list of floats (or None) — backend handles JSONB.
                    "embedding":       vec,
                    "embedding_model": embedding_model if vec is not None else None,
                },
                conflict="document_id,chunk_index",
            )
        except Exception as exc:
            log.warning("save_chunks: failed chunk %s for %s: %s",
                        chunk.get("chunk_index"), document_id, exc)
