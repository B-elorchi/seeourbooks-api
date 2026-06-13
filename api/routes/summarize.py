import json
from typing import AsyncIterator, Optional

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.auth import AuthUser, require_user
from api.config.settings import settings
from api.models.requests import SumReq
from api.services.db import find, insert, upsert, update
from api.services.pipeline.orchestrator import (
    _fetch_book_from_catalog, _pick_cached_summary,
)
from api.services.summarizer.chunker import find_book_text, chunk_text
from api.services.summarizer.haiku import run_haiku_pass
from api.services.summarizer.sonnet import run_sonnet_pass
from api.services.summarizer.tashkeel import run_tashkeel_pass

router = APIRouter()


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run(req: SumReq, user_id: Optional[str] = None) -> AsyncIterator[str]:

    # ── 1. Cache check (our internal cache from prior summary runs) ──────────
    try:
        rows = await find(
            "book_summaries",
            filters={
                "book_id":  req.book_id,
                "length":   req.length,
                "style":    req.style,
                "language": req.language,
            },
            select="summary, word_count",
            limit=1,
        )
    except Exception:
        rows = []   # book_summaries table missing → fall through gracefully

    if rows:
        yield sse("cached", {"summary": rows[0]["summary"], "word_count": rows[0]["word_count"]})
        return

    # ── 1b. Production catalog cache check ───────────────────────────────────
    # If the book exists in the client's `books` table with a pre-computed
    # summary (summary_english / summary_en_10min / arabic_summary / etc.),
    # return that immediately.  Saves an entire AI run for books that were
    # already summarised by the client's previous batch job.
    catalog_row = await _fetch_book_from_catalog(req.book_id)
    if catalog_row:
        cached = _pick_cached_summary(catalog_row, req.language)
        if cached:
            wc = len(cached.split())
            yield sse("status", {"msg": "Found cached summary in catalog"})
            yield sse("cached", {"summary": cached, "word_count": wc})
            return

    yield sse("status", {"msg": "Starting…"})

    # ── 2. Ensure book exists in catalog (FK requirement) — skip for numeric  ──
    # ids that already exist in the production catalog (integer PK rejects
    # the upsert anyway, and the row already exists).
    if not (req.book_id or "").isdigit():
        try:
            await upsert(
                "books",
                {"book_id": req.book_id, "title": req.book_id, "status": "pending"},
                "book_id",
            )
        except Exception:
            pass  # production schema may not match; best-effort upsert

    # ── 3. Create or reuse job ────────────────────────────────────────────────
    existing = await find(
        "summary_jobs",
        filters={
            "book_id":  req.book_id,
            "length":   req.length,
            "style":    req.style,
            "language": req.language,
            "status":   ("in", ["queued", "processing"]),
        },
        limit=1,
    )
    if existing:
        job_id = existing[0]["id"]
    else:
        new_job: dict = {
            "book_id":  req.book_id,
            "length":   req.length,
            "style":    req.style,
            "language": req.language,
            "status":   "processing",
        }
        if user_id:
            new_job["user_id"] = user_id
        j = await insert("summary_jobs", new_job)
        job_id = j["id"]

    await update("summary_jobs", {"id": job_id}, {"status": "processing"})

    # ── 4. Load chunks ────────────────────────────────────────────────────────
    # Production `chunks` table has `chunk_id` (UUID) — not `id` — and may have
    # a bigint book_id rather than text.  We try a minimal column selection
    # that exists on both legacy and production schemas; on failure we fall
    # through to the find_book_text path below.
    try:
        chunks = await find(
            "chunks",
            filters={"book_id": req.book_id},
            select="chunk_index, content",
            order="chunk_index ASC",
        )
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning(
            "chunks lookup failed for book_id=%s — falling back to text loader: %s",
            req.book_id, exc,
        )
        chunks = []
    if not chunks:
        yield sse("status", {"msg": "Loading book text…"})
        text = find_book_text(req.book_id)
        if not text:
            yield sse("error", {"msg": f"Book text not found for {req.book_id}"})
            await update("summary_jobs", {"id": job_id}, {"status": "error", "error_msg": "text not found"})
            return
        parts = chunk_text(text)
        yield sse("status", {"msg": f"Chunked into {len(parts)} parts", "total": len(parts)})
        chunks = []
        for i, content in enumerate(parts):
            row = await upsert(
                "chunks",
                {"book_id": req.book_id, "chunk_index": i, "content": content, "token_count": len(content.split())},
                "book_id,chunk_index",
            )
            chunks.append(row)

    total = len(chunks)
    yield sse("status", {"msg": f"Processing {total} chunks…", "total": total})

    # ── 5. Haiku pass — chunk summaries ──────────────────────────────────────
    chunk_sums: list[str] = []

    for chunk in chunks:
        yield ": keepalive\n\n"
        sums = await run_haiku_pass(req.book_id, [chunk], req.language)
        chunk_sums.extend(sums)
        yield sse("chunk_done", {"index": chunk["chunk_index"], "total": total})

    # ── 6. Sonnet pass — final summary (streaming) ────────────────────────────
    yield sse("status", {"msg": "Generating final summary…"})
    full = ""
    async for event, value in run_sonnet_pass(chunk_sums, req.length, req.style, req.language):
        if event == "token":
            full += value
            yield sse("token", {"text": value})
        elif event == "full":
            full = value

    # ── 7. Opus pass — Arabic tashkeel ───────────────────────────────────────
    if req.language == "ar":
        yield sse("status", {"msg": "Applying full tashkeel…"})
        full = await run_tashkeel_pass(full, req.length)

    # ── 8. Cache result ───────────────────────────────────────────────────────
    wc = len(full.split())
    await upsert(
        "book_summaries",
        {
            "book_id":   req.book_id,
            "length":    req.length,
            "style":     req.style,
            "language":  req.language,
            "summary":   full,
            "word_count": wc,
            "model":     settings.MODEL_SONNET,
        },
        "book_id,length,style,language",
    )
    await update("summary_jobs", {"id": job_id}, {"status": "done"})
    yield sse("done", {"summary": full, "word_count": wc, "job_id": job_id})


@router.post("/summarize")
async def summarize(req: SumReq, user: AuthUser = Depends(require_user)):
    return StreamingResponse(
        run(req, user_id=user.id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
