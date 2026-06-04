"""
Legacy read endpoint:
  GET /api/summary/{book_id} → all cached `book_summaries` rows for a book.

Prefer `GET /api/pipeline/output/{book_id}` for new clients — that returns
the full pipeline output (summaries + audio + cover + mindmap).
"""
import logging

import httpx
from fastapi import APIRouter, HTTPException

from api.services.db.supabase import sg

router = APIRouter()
log = logging.getLogger(__name__)


@router.get("/summary/{book_id}")
async def get_summaries(book_id: str):
    """Return all cached summaries for a book. Returns [] when the book has none."""
    # NOTE: removed `audio_url` from select — that column does not exist on
    # `book_summaries` in the current schema (it lived on a different table
    # in an earlier draft). Selecting it caused PostgREST to 400 and the
    # endpoint to 500 for every input.
    select = "length,style,language,summary,word_count,created_at"
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            rows = await sg(
                client,
                f"book_summaries?book_id=eq.{book_id}&select={select}",
            )
        return rows or []
    except httpx.HTTPStatusError as exc:
        log.warning("get_summaries(%s): DB returned %s — %s",
                    book_id, exc.response.status_code, exc.response.text[:300])
        raise HTTPException(
            status_code=502,
            detail=f"Database error while fetching summaries for {book_id!r}",
        ) from exc
    except httpx.RequestError as exc:
        log.error("get_summaries(%s): DB unreachable — %s", book_id, exc)
        raise HTTPException(status_code=503, detail="Database unreachable") from exc
    except Exception as exc:
        log.exception("get_summaries(%s) unexpected failure", book_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
