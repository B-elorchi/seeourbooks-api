"""
Admin routes:
  GET  /api/admin/config              → all current provider settings (flat key:value dict)
  POST /api/admin/config              → { key, value } — update one setting
  GET  /api/admin/metrics             → job counts and timing stats
  GET  /api/admin/jobs                → all pipeline jobs (alias with full detail)
  POST /api/admin/jobs/{job_id}/retry → manually retry a failed/partial job
  GET  /api/admin/costs               → aggregated cost breakdown for the last N days
  GET  /api/admin/openrouter-models   → cached, optionally-filtered live OpenRouter model list
  POST /api/admin/books               → upsert a book row (fetches metadata from Gutenberg if numeric id)
  GET  /api/admin/books/{book_id}     → fetch a single book row from the DB
"""
import io
import logging
import os
import re
import time
import xml.etree.ElementTree as ET
import zipfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from html.parser import HTMLParser

import httpx
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel

from api.auth import require_admin
from api.services.config.runtime import get_all_config, set_config_key, refresh_config_cache, get_config_value
from api.services.db import find, insert, upsert, update
from api.jobs.store import get_job, get_output, reset_for_manual_retry, delete_job, timeout_stuck_jobs, can_retry
from api.models.requests import PipelineReq
from api.services.pipeline.cover import _build_prompt
from api.services.openrouter_keys import openrouter_key_has_credits, reset_openrouter_keys

log = logging.getLogger(__name__)

# Auth: every route in this router requires an admin user when SUPABASE_JWT_SECRET
# is configured.  In dev (no JWT secret) the dependency returns a dummy admin
# so the panel stays usable without a Supabase project.
router = APIRouter(
    prefix="/admin",
    tags=["admin"],
    dependencies=[Depends(require_admin)],
)


class ConfigUpdate(BaseModel):
    key:   str
    value: str


class RerunRequest(BaseModel):
    steps: list[str]  # e.g. ["audio_full", "cover"]


class BookUpsertRequest(BaseModel):
    book_id:     str
    title:       str | None = None   # auto-fetched from Gutenberg when omitted + book_id is numeric
    author:      str | None = None   # same
    language:    str        = "en"
    year:        int | None = None
    pages:       int | None = None
    grade_level: str | None = None
    genres:      list[str]  = []
    status:      str        = "pending"


# ── Gutenberg metadata helper ─────────────────────────────────────────────────

async def _gutenberg_meta(book_id: str) -> dict:
    """
    Fetch title, author, year, pages, and genres from the Gutendex API.
    Returns a dict with whatever fields are available (may be partial).
    """
    result: dict = {}
    try:
        url = f"https://gutendex.com/books?ids={book_id}"
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            data = resp.json()
        books = data.get("results", [])
        if not books:
            return result
        book = books[0]

        result["title"] = book.get("title", "")

        authors = book.get("authors", [])
        if authors:
            name = authors[0].get("name", "")
            # Gutendex format: "Shelley, Mary Wollstonecraft" → "Mary Wollstonecraft Shelley"
            if "," in name:
                last, first = name.split(",", 1)
                result["author"] = f"{first.strip()} {last.strip()}"
            else:
                result["author"] = name

        if book.get("download_count"):
            pass  # no year/pages in Gutendex sadly

        subjects = book.get("subjects", [])
        bookshelves = book.get("bookshelves", [])
        genres_raw = subjects[:3] + bookshelves[:2]
        if genres_raw:
            result["genres"] = [g.split("--")[0].strip() for g in genres_raw]

    except Exception as exc:
        log.debug("Gutenberg meta fetch for %s failed: %s", book_id, exc)
    return result


@router.get("/config")
async def admin_get_config() -> dict:
    """Return all provider settings as a flat key → value dict."""
    return await get_all_config()


@router.post("/config")
async def admin_set_config(body: ConfigUpdate) -> dict:
    """Update a single provider setting. Takes effect on the next pipeline job."""
    if not body.key:
        raise HTTPException(400, "key is required")
    await set_config_key(body.key, body.value)
    return {"ok": True, "key": body.key, "value": body.value}


@router.post("/books", status_code=200)
async def admin_upsert_book(body: BookUpsertRequest) -> dict:
    """
    Insert or update a book row in the database.

    For numeric book_ids (Gutenberg books) where title/author are omitted,
    metadata is automatically fetched from the Gutendex API so you don't
    have to look it up manually.

    After this succeeds, run the pipeline with:
      POST /api/pipeline/run  { "book_id": "<id>", ... }
    """
    title  = body.title
    author = body.author
    genres = body.genres

    # Auto-fetch from Gutenberg when id is numeric and fields are missing
    if body.book_id.isdigit() and (not title or not author):
        meta = await _gutenberg_meta(body.book_id)
        if not title:
            title = meta.get("title", "")
        if not author:
            author = meta.get("author", "")
        if not genres:
            genres = meta.get("genres", [])

    # Last resort: if Gutenberg failed, check whether the book already exists in DB
    if not title:
        try:
            bid_check: object = body.book_id
            try:
                bid_check = int(body.book_id)
            except ValueError:
                pass
            existing = await find("books", filters={"book_id": bid_check}, limit=1)
            if existing:
                title  = existing[0].get("title", "") or ""
                author = author or existing[0].get("author", "") or ""
        except Exception:
            pass

    if not title:
        raise HTTPException(
            422,
            f"Could not determine title for book_id={body.book_id!r}. "
            "Pass `title` explicitly or use a valid Gutenberg numeric id."
        )

    row = {
        "book_id": body.book_id,
        "title":   title,
        "author":  author or "",
        "status":  body.status,
    }
    if body.year:
        row["year"] = body.year
    if body.pages:
        row["pages"] = body.pages
    if body.grade_level:
        row["grade_level"] = body.grade_level
    if genres:
        row["genres"] = genres

    try:
        await upsert("books", row, conflict="book_id")
    except Exception as exc:
        raise HTTPException(502, f"DB upsert failed: {exc}") from exc

    return {
        "ok":      True,
        "book_id": body.book_id,
        "title":   title,
        "author":  author or "",
        "action":  "upserted",
        "next":    f"POST /api/pipeline/run with book_id={body.book_id!r}",
    }


@router.get("/books/{book_id}")
async def admin_get_book(book_id: str) -> dict:
    """Return the book row from the database, or 404 if not found."""
    try:
        bid: object = book_id
        try:
            bid = int(book_id)
        except ValueError:
            pass
        rows = await find("books", filters={"book_id": bid}, limit=1)
    except Exception as exc:
        raise HTTPException(502, f"DB read failed: {exc}") from exc

    if not rows:
        raise HTTPException(404, f"Book {book_id!r} not found in the database")
    return rows[0]


@router.get("/books/{book_id}/cover-prompt")
async def admin_book_cover_prompt(book_id: str) -> dict:
    """
    Build and return the exact cover-image prompt that would be sent to the
    image model for this book.  Uses the book metadata plus the latest available
    summary (from a previous pipeline run).  No image is generated.
    """
    try:
        bid: object = book_id
        try:
            bid = int(book_id)
        except ValueError:
            pass
        rows = await find("books", filters={"book_id": bid}, limit=1)
    except Exception as exc:
        raise HTTPException(502, f"DB read failed: {exc}") from exc

    if not rows:
        raise HTTPException(404, f"Book {book_id!r} not found in the database")

    book = rows[0]

    # Try to reuse the latest generated summary for this book.
    summary: str | None = None
    try:
        output = await get_output(book_id)
        if output:
            result = output.get("result") or output
            sums = result.get("summaries") or {}
            if sums:
                first = next(iter(sums.values()), {})
                summary = first.get("text") or ""
            if not summary:
                summary = result.get("quick_summary") or ""
    except Exception:
        summary = ""

    prompt = await _build_prompt(
        title=book.get("title") or "",
        author=book.get("author") or "",
        summary=summary or None,
        genres=book.get("genres") or None,
        year=book.get("year"),
        language=book.get("language") or "en",
    )

    return {
        "book_id": book_id,
        "prompt":  prompt,
        "summary_chars": len(summary or ""),
    }


# ── In-memory ingest status store (keyed by book_id) ─────────────────────────
# Persists across requests within one server process so the client can poll.
_ingest_status: dict[str, dict] = {}


@router.post("/books/{book_id}/ingest", status_code=202)
async def admin_ingest_book(book_id: str, background_tasks: BackgroundTasks) -> dict:
    """
    Start a background ingest job for this book and return immediately (202).

    Poll GET /api/admin/books/{book_id}/ingest/status to track progress.

    URL resolution order:
      1. epub_download column in the books row
      2. BOOK_FILES_BASE_URL/books/english/{book_id}.epub
      3. txt_download column in the books row
      4. BOOK_FILES_BASE_URL/books/english/{book_id}.txt
    """
    from api.services.config.runtime import get_config_value  # noqa: PLC0415

    # ── Validate book exists ──────────────────────────────────────────────────
    bid: object = book_id
    try:
        bid = int(book_id)
    except ValueError:
        pass

    try:
        rows = await find("books", filters={"book_id": bid}, limit=1)
    except Exception as exc:
        raise HTTPException(502, f"DB read failed: {exc}") from exc

    if not rows:
        raise HTTPException(
            404,
            f"Book {book_id!r} not found. Insert it first via POST /api/admin/books.",
        )

    book = rows[0]

    base = await get_config_value("BOOK_FILES_BASE_URL", "https://files.seeourbook.sa")
    base = base.rstrip("/")

    epub_candidates = [c for c in [book.get("epub_download"), f"{base}/books/english/{book_id}.epub"] if c]
    txt_candidates  = [c for c in [book.get("txt_download"),  f"{base}/books/english/{book_id}.txt"]  if c]

    # Mark as running
    _ingest_status[book_id] = {"status": "running", "book_id": book_id, "title": book.get("title", "")}

    background_tasks.add_task(
        _run_ingest, book_id, bid, book, epub_candidates, txt_candidates
    )

    return {
        "ok":         True,
        "book_id":    book_id,
        "status":     "running",
        "status_url": f"/api/admin/books/{book_id}/ingest/status",
    }


@router.get("/books/{book_id}/ingest/status")
async def admin_ingest_status(book_id: str) -> dict:
    """Poll this endpoint to check whether an ingest job has finished."""
    info = _ingest_status.get(book_id)
    if not info:
        return {"status": "not_started", "book_id": book_id}
    return info


async def _run_ingest(
    book_id: str,
    bid: object,
    book: dict,
    epub_candidates: list[str],
    txt_candidates:  list[str],
) -> None:
    """Background task: download → extract text → chunk → upsert to DB."""
    from api.services.summarizer.chunker import chunk_text  # noqa: PLC0415

    def _set(status: str, **extra: object) -> None:
        _ingest_status[book_id] = {"status": status, "book_id": book_id,
                                    "title": book.get("title", ""), **extra}

    try:
        # ── Download ──────────────────────────────────────────────────────────
        _set("running", step="downloading")
        raw_bytes: bytes = b""
        used_url:  str   = ""
        is_epub:   bool  = False

        _CDN_HEADERS = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "*/*",
        }
        async with httpx.AsyncClient(timeout=300, follow_redirects=True, headers=_CDN_HEADERS) as client:
            for url in epub_candidates:
                try:
                    resp = await client.get(url)
                    if resp.status_code in (200, 206) and len(resp.content) > 1000:
                        raw_bytes, used_url, is_epub = resp.content, url, True
                        log.info("Ingest %s: downloaded EPUB from %s (%d bytes)", book_id, url, len(raw_bytes))
                        break
                    log.warning("Ingest %s: EPUB %s → HTTP %s", book_id, url, resp.status_code)
                except Exception as exc:
                    log.warning("Ingest %s: EPUB %s failed: %s", book_id, url, exc)

            if not raw_bytes:
                for url in txt_candidates:
                    try:
                        resp = await client.get(url)
                        if resp.status_code in (200, 206) and len(resp.content) > 500:
                            raw_bytes, used_url, is_epub = resp.content, url, False
                            log.info("Ingest %s: downloaded TXT from %s (%d bytes)", book_id, url, len(raw_bytes))
                            break
                        log.warning("Ingest %s: TXT %s → HTTP %s", book_id, url, resp.status_code)
                    except Exception as exc:
                        log.warning("Ingest %s: TXT %s failed: %s", book_id, url, exc)

        if not raw_bytes:
            _set("error", error=f"Could not download from any URL. Tried EPUB: {epub_candidates}, TXT: {txt_candidates}")
            return

        # ── Extract text ──────────────────────────────────────────────────────
        _set("running", step="extracting")
        if is_epub:
            text = _extract_epub_text(raw_bytes)
        else:
            text = _strip_gutenberg(raw_bytes.decode("utf-8", errors="replace"))

        text = text.strip()
        if len(text) < 500:
            _set("error", error=f"Extracted text too short ({len(text)} chars). File may be invalid.")
            return

        # ── Chunk ─────────────────────────────────────────────────────────────
        _set("running", step="chunking")
        chunks = chunk_text(text, max_words=1500)
        if not chunks:
            _set("error", error="Chunking produced no segments.")
            return

        # ── Save to DB ────────────────────────────────────────────────────────
        # Production `chunks` schema: book_id (bigint), chunk_index, content,
        # total_chunks, wordcount, book_title, author. No token_count column,
        # no unique (book_id, chunk_index) constraint → plain INSERT.
        _set("running", step=f"saving {len(chunks)} chunks")
        total      = len(chunks)
        book_id_val = bid if isinstance(bid, int) else book_id
        book_title = book.get("title", "") or ""
        author     = book.get("author", "") or ""
        saved = 0
        first_error: Exception | None = None
        for idx, content in enumerate(chunks):
            row = {
                "book_id":      book_id_val,
                "chunk_index":  idx,
                "content":      content,
                "total_chunks": total,
                "wordcount":    len(content.split()),
                "book_title":   book_title,
                "author":       author,
            }
            try:
                await insert("chunks", row)
                saved += 1
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                detail = str(exc).strip() or repr(exc)
                log.warning("chunk[%d] insert failed: %s: %s", idx, type(exc).__name__, detail)

        if saved == 0:
            _set("error", error=f"No chunks could be saved. DB error: {first_error}")
            return

        _set(
            "done",
            source="epub" if is_epub else "txt",
            chunks_saved=saved,
            total_chars=len(text),
            txt_url=used_url,
        )
        log.info("Ingest %s complete: %d chunks saved from %s", book_id, saved, used_url)

    except Exception as exc:
        log.exception("Ingest background task failed for book %s", book_id)
        _set("error", error=str(exc))


# ── HTML text extractor (stdlib only, no extra deps) ─────────────────────────

class _HtmlTextExtractor(HTMLParser):
    """Pull visible text out of an HTML/XHTML document."""
    def __init__(self):
        super().__init__()
        self._skip  = False
        self._parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style", "head"):
            self._skip = True

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _extract_epub_text(epub_bytes: bytes) -> str:
    """
    Extract ordered plain text from an EPUB file (which is a ZIP of HTML/XHTML).

    Reads the OPF spine to get the correct reading order, then extracts
    visible text from each HTML chapter using a stdlib HTML parser.
    """
    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
        names = set(zf.namelist())
        spine_files: list[str] = []

        # ── Parse spine order from OPF ────────────────────────────────────────
        try:
            container_xml = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            container_root = ET.fromstring(container_xml)

            # Find rootfile path (handles namespaced and bare tags)
            opf_path = ""
            for elem in container_root.iter():
                if elem.tag.endswith("rootfile"):
                    opf_path = elem.get("full-path", "")
                    break

            if opf_path and opf_path in names:
                opf_dir = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
                opf_xml = zf.read(opf_path).decode("utf-8", errors="ignore")
                opf_root = ET.fromstring(opf_xml)

                # Build id→href manifest
                manifest: dict[str, str] = {}
                for elem in opf_root.iter():
                    if elem.tag.endswith("item"):
                        item_id   = elem.get("id", "")
                        item_href = elem.get("href", "")
                        media     = elem.get("media-type", "")
                        if "html" in media or item_href.endswith((".html", ".xhtml", ".htm")):
                            # href may be relative to OPF dir
                            full = opf_dir + item_href if not item_href.startswith("/") else item_href.lstrip("/")
                            manifest[item_id] = full

                # Follow spine order
                for elem in opf_root.iter():
                    if elem.tag.endswith("itemref"):
                        idref = elem.get("idref", "")
                        if idref in manifest and manifest[idref] in names:
                            spine_files.append(manifest[idref])
        except Exception as exc:
            log.debug("EPUB OPF parse failed, falling back to sorted filenames: %s", exc)

        # ── Fallback: all HTML files sorted alphabetically ────────────────────
        if not spine_files:
            spine_files = sorted(
                n for n in names
                if n.lower().endswith((".html", ".xhtml", ".htm"))
                and "toc" not in n.lower()
                and "nav" not in n.lower()
            )

        # ── Extract text from each file in spine order ────────────────────────
        parts: list[str] = []
        for fname in spine_files:
            if fname not in names:
                continue
            try:
                html = zf.read(fname).decode("utf-8", errors="ignore")
                parser = _HtmlTextExtractor()
                parser.feed(html)
                chapter_text = re.sub(r"\s{2,}", " ", parser.get_text()).strip()
                if len(chapter_text) > 100:
                    parts.append(chapter_text)
            except Exception as exc:
                log.debug("Failed to read EPUB entry %s: %s", fname, exc)

        return "\n\n".join(parts)


def _strip_gutenberg(text: str) -> str:
    """Remove Project Gutenberg header/footer boilerplate from plain TXT files."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "*END*THE SMALL PRINT",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]
    start_pos = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            start_pos = text.find("\n", idx) + 1
            break
    end_pos = len(text)
    for marker in end_markers:
        idx = text.find(marker, start_pos)
        if idx != -1:
            end_pos = idx
            break
    return text[start_pos:end_pos].strip()


@router.get("/metrics")
async def admin_metrics() -> dict:
    """Return job counts and aggregate stats for the last 500 jobs."""
    try:
        jobs = await find(
            "pipeline_jobs",
            select="status, created_at",
            order="created_at DESC",
            limit=500,
        )
    except Exception as exc:
        log.warning("admin_metrics: DB unreachable — %s", exc)
        # Return empty metrics rather than 500 so the admin panel still loads
        return {"total": 0, "done": 0, "partial": 0, "failed": 0, "running": 0, "queued": 0}

    counts = Counter(j["status"] for j in jobs)
    return {
        "total":   len(jobs),
        "done":    counts.get("done", 0),
        "partial": counts.get("partial", 0),
        "failed":  counts.get("failed", 0),
        "running": counts.get("running", 0),
        "queued":  counts.get("queued", 0),
    }


@router.get("/metrics/queued")
async def admin_queued_metrics(minutes: int = 10) -> dict:
    """
    Return a focused queue snapshot:
      - queued_last_n_minutes: jobs currently queued that were created in the
        last `minutes` minutes (default 10).
      - queued_with_progress: queued jobs that already have at least one step
        completed (done/partial), i.e. waiting for retry after making progress.
      - queued_total: all currently queued jobs.
    """
    try:
        since = datetime.now(timezone.utc) - timedelta(minutes=max(1, minutes))
        since_iso = since.isoformat()
        rows = await find(
            "pipeline_jobs",
            filters={"status": "queued"},
            select="id,created_at,result",
            order="created_at DESC",
            limit=5000,
        )
    except Exception as exc:
        log.warning("admin_queued_metrics: DB unreachable — %s", exc)
        return {
            "minutes": minutes,
            "queued_last_n_minutes": 0,
            "queued_with_progress":  0,
            "queued_total":          0,
        }

    total = len(rows)
    recent = sum(
        1 for r in rows
        if r.get("created_at") and str(r["created_at"]) >= since_iso
    )

    # Jobs that already have at least one finished/partial step in the stored
    # result, or in the granular pipeline_step_results table.
    jobs_with_progress: set[str] = set()
    queued_ids: list[str] = []
    for r in rows:
        job_id = r["id"]
        queued_ids.append(job_id)
        result = r.get("result") or {}
        steps = result.get("steps") or {}
        if any(status in ("done", "partial") for status in steps.values()):
            jobs_with_progress.add(job_id)

    if queued_ids:
        try:
            step_rows = await find(
                "pipeline_step_results",
                filters={
                    "job_id": ("in", queued_ids),
                    "status": ("in", ["done", "partial"]),
                },
                select="job_id",
                limit=10000,
            )
            for sr in step_rows:
                jobs_with_progress.add(sr["job_id"])
        except Exception as exc:
            log.debug("admin_queued_metrics: step_results lookup failed — %s", exc)

    return {
        "minutes": minutes,
        "queued_last_n_minutes": recent,
        "queued_with_progress":  len(jobs_with_progress),
        "queued_total":          total,
    }


@router.get("/jobs")
async def admin_jobs(limit: int = 100) -> list:
    """List all pipeline jobs ordered newest-first."""
    try:
        return await find("pipeline_jobs", order="created_at DESC", limit=limit)
    except Exception as exc:
        log.warning("admin_jobs: DB unreachable — %s", exc)
        return []


# ── Catalog inspector — read-only proxy to a whitelist of tables ──────────────
#
# Powers the "Catalog" tab in the admin UI.  Useful for verifying that:
#   - production books / chunks / covers / etc. are visible to the API
#   - the pipeline is writing into the right tables
#   - per-book data is laid out as expected
#
# Whitelisted to prevent leaking unrelated DB tables through a generic API.
_CATALOG_TABLES: dict[str, dict] = {
    # ── client production tables ──
    "books":                  {"order": "book_id ASC",     "book_id_col": "book_id"},
    "ai_batches":             {"order": "system_created_at DESC", "book_id_col": "book_id"},
    "chunks":                 {"order": "chunk_index ASC", "book_id_col": "book_id"},
    "covers":                 {"order": "created_at DESC", "book_id_col": "bookId"},
    "reviews":                {"order": "updated_at DESC", "book_id_col": "book_id"},
    "audio":                  {"order": "updated_at DESC", "book_id_col": "book_id"},
    # ── seeourbook operational tables ──
    "pipeline_jobs":          {"order": "created_at DESC", "book_id_col": "book_id"},
    "pipeline_step_results":  {"order": "created_at DESC", "book_id_col": None},
    "book_summaries":         {"order": "created_at DESC", "book_id_col": "book_id"},
    "chunk_summaries":        {"order": "created_at DESC", "book_id_col": "book_id"},
    "usage_logs":             {"order": "created_at DESC", "book_id_col": None},
    "provider_config":        {"order": "updated_at DESC", "book_id_col": None},
    "uploaded_documents":     {"order": "created_at DESC", "book_id_col": None},
    "summary_jobs":           {"order": "created_at DESC", "book_id_col": "book_id"},
    # ── new documents pipeline ──
    "documents":              {"order": "created_at DESC", "book_id_col": None},
    "document_pages":         {"order": "page_number ASC", "book_id_col": None},
    "document_summaries":     {"order": "created_at DESC", "book_id_col": None},
    "knowledge_chunks":       {"order": "chunk_index ASC", "book_id_col": None},
}


@router.get("/catalog/tables")
async def admin_catalog_tables() -> dict:
    """List the tables exposed by /catalog/{table} along with their hints."""
    return {
        "tables": [
            {
                "name":            name,
                "default_order":   meta["order"],
                "supports_book_id": meta["book_id_col"] is not None,
            }
            for name, meta in _CATALOG_TABLES.items()
        ],
    }


@router.get("/catalog/{table}")
async def admin_catalog(
    table: str,
    limit:  int = 50,
    offset: int = 0,
    book_id: str | None = None,
) -> dict:
    """
    Return up to `limit` rows from the given table.

    Optional filters:
        book_id — when the table has a book_id-like column.  We map this to the
                  actual column name (some legacy tables use bookId / book_id).

    This endpoint is intentionally read-only and whitelisted.
    """
    if table not in _CATALOG_TABLES:
        raise HTTPException(
            status_code=400,
            detail=f"table {table!r} is not in the catalog whitelist. "
                   f"Allowed: {sorted(_CATALOG_TABLES.keys())}",
        )

    if limit < 1 or limit > 500:
        limit = max(1, min(500, limit))
    if offset < 0:
        offset = 0

    meta    = _CATALOG_TABLES[table]
    filters: dict | None = None

    if book_id and meta["book_id_col"]:
        # The production books / chunks / audio tables use INTEGER book_id.
        # Try numeric coercion first so eq.<int> matches.
        bid: object = book_id
        try:
            bid = int(book_id)
        except ValueError:
            pass
        filters = {meta["book_id_col"]: bid}

    try:
        rows = await find(
            table,
            filters=filters,
            order=meta["order"],
            limit=limit,
            offset=offset,
        )
    except Exception as exc:
        # Surface DB errors with a clean message instead of generic 500
        raise HTTPException(
            status_code=502,
            detail=f"DB read failed for {table!r}: {exc}",
        ) from exc

    return {
        "table":  table,
        "limit":  limit,
        "offset": offset,
        "count":  len(rows),
        "rows":   rows,
    }


@router.get("/costs")
async def admin_costs(days: int = 30) -> dict:
    """
    Aggregated cost breakdown for the last N days.

    Returns totals plus three groupings — by provider, by step, by model —
    each sorted by cost descending.  All amounts are USD estimates derived
    from the rate table in `usage_logger.py`.
    """
    if days < 1:
        days = 1
    if days > 365:
        days = 365

    since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()

    try:
        rows = await find(
            "usage_logs",
            filters={"created_at": ("gte", since_iso)},
            order="created_at DESC",
            limit=20000,
        )
    except Exception:
        # Table absent or query rejected — return an empty (but valid) shape
        rows = []

    by_provider: dict[str, dict] = {}
    by_step:     dict[str, dict] = {}
    by_model:    dict[str, dict] = {}
    total_cost  = 0.0

    for r in rows:
        cost  = float(r.get("cost_usd") or 0)
        units = float(r.get("units")    or 0)
        total_cost += cost

        p = r.get("provider") or "unknown"
        prov = by_provider.setdefault(p, {"provider": p, "calls": 0, "cost_usd": 0.0, "units": 0.0})
        prov["calls"]    += 1
        prov["cost_usd"] += cost
        prov["units"]    += units

        s = r.get("step") or "unknown"
        st = by_step.setdefault(s, {"step": s, "calls": 0, "cost_usd": 0.0})
        st["calls"]    += 1
        st["cost_usd"] += cost

        m = r.get("model") or "unknown"
        md = by_model.setdefault(m, {"model": m, "provider": p, "calls": 0, "cost_usd": 0.0, "units": 0.0,
                                      "unit_type": r.get("unit_type") or ""})
        md["calls"]    += 1
        md["cost_usd"] += cost
        md["units"]    += units

    def _round(rows_: list[dict]) -> list[dict]:
        for x in rows_:
            x["cost_usd"] = round(x["cost_usd"], 4)
            if "units" in x:
                x["units"] = round(x["units"], 2)
        return sorted(rows_, key=lambda x: x["cost_usd"], reverse=True)

    return {
        "days":           days,
        "total_calls":    len(rows),
        "total_cost_usd": round(total_cost, 4),
        "by_provider":    _round(list(by_provider.values())),
        "by_step":        _round(list(by_step.values())),
        "by_model":       _round(list(by_model.values())),
    }


@router.get("/costs/by-book")
async def admin_costs_by_book(days: int = 30, limit: int = 20) -> list:
    """Cost breakdown grouped by book (job), sorted by cost descending."""
    since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        rows = await find("usage_logs", filters={"created_at": ("gte", since_iso)},
                          select="job_id, cost_usd", order="created_at DESC", limit=50000)
    except Exception:
        return []

    # Aggregate by job_id
    by_job: dict[str, float] = {}
    for r in rows:
        jid = r.get("job_id") or "__none__"
        by_job[jid] = by_job.get(jid, 0.0) + float(r.get("cost_usd") or 0)

    job_ids = [j for j in by_job if j != "__none__"]
    if not job_ids:
        return []

    # Fetch book titles from pipeline_jobs
    try:
        jobs = await find("pipeline_jobs",
                          filters={"id": ("in", job_ids[:200])},
                          select="id, book_id, input, user_id")
    except Exception:
        jobs = []

    meta: dict[str, dict] = {j["id"]: j for j in jobs}

    results = []
    for jid, cost in sorted(by_job.items(), key=lambda x: x[1], reverse=True):
        if jid == "__none__" or cost < 0.0001:
            continue
        m = meta.get(jid, {})
        inp = m.get("input") or {}
        if isinstance(inp, str):
            try:
                import json as _json; inp = _json.loads(inp)
            except Exception:
                inp = {}
        results.append({
            "job_id":   jid,
            "book_id":  m.get("book_id") or jid[:8],
            "title":    inp.get("title") or m.get("book_id") or jid[:8],
            "user_id":  m.get("user_id"),
            "cost_usd": round(cost, 4),
        })
        if len(results) >= limit:
            break
    return results


@router.get("/costs/by-user")
async def admin_costs_by_user(days: int = 30) -> list:
    """Cost breakdown grouped by user (via pipeline_jobs.user_id), sorted by cost."""
    since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        rows = await find("usage_logs", filters={"created_at": ("gte", since_iso)},
                          select="job_id, cost_usd", order="created_at DESC", limit=50000)
    except Exception:
        return []

    by_job: dict[str, float] = {}
    for r in rows:
        jid = r.get("job_id") or "__none__"
        by_job[jid] = by_job.get(jid, 0.0) + float(r.get("cost_usd") or 0)

    job_ids = [j for j in by_job if j != "__none__"]
    if not job_ids:
        return []

    try:
        jobs = await find("pipeline_jobs",
                          filters={"id": ("in", job_ids[:200])},
                          select="id, user_id")
    except Exception:
        jobs = []

    # Map job → user
    job_user: dict[str, str] = {j["id"]: (j.get("user_id") or "__anonymous__") for j in jobs}

    # Aggregate by user
    by_user: dict[str, float] = {}
    for jid, cost in by_job.items():
        uid = job_user.get(jid, "__anonymous__")
        by_user[uid] = by_user.get(uid, 0.0) + cost

    # Fetch user emails from app_users
    user_ids = [u for u in by_user if u != "__anonymous__"]
    user_emails: dict[str, str] = {}
    if user_ids:
        try:
            users = await find("app_users",
                               filters={"id": ("in", user_ids[:100])},
                               select="id, email, name")
            user_emails = {u["id"]: (u.get("name") or u.get("email") or u["id"][:8]) for u in users}
        except Exception:
            pass

    results = []
    for uid, cost in sorted(by_user.items(), key=lambda x: x[1], reverse=True):
        if cost < 0.0001:
            continue
        results.append({
            "user_id":  uid,
            "label":    user_emails.get(uid, "Anonymous" if uid == "__anonymous__" else uid[:8]),
            "cost_usd": round(cost, 4),
        })
    return results


@router.get("/costs/daily")
async def admin_costs_daily(days: int = 30) -> list:
    """
    Daily cost totals for the last N days — for a trend line chart.

    Returns a CONTINUOUS series from (today − N + 1) through today: every day
    in the window is present, with cost_usd = 0 for days that had no usage.
    Without this, the chart's X-axis would skip empty days and "7 days" /
    "30 days" would visually compress to however many days happened to have
    cost — which is what the client reported.
    """
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=days - 1)
    since_iso = datetime(start.year, start.month, start.day, tzinfo=timezone.utc).isoformat()
    try:
        rows = await find("usage_logs", filters={"created_at": ("gte", since_iso)},
                          select="created_at, cost_usd", order="created_at DESC", limit=200000)
    except Exception:
        rows = []

    daily: dict[str, float] = {}
    for r in rows:
        day = (r.get("created_at") or "")[:10]
        if day:
            daily[day] = daily.get(day, 0.0) + float(r.get("cost_usd") or 0)

    # Build a zero-filled, gap-free series for the whole window.
    series = []
    for i in range(days):
        d = start + timedelta(days=i)
        key = d.isoformat()
        series.append({"date": key, "cost_usd": round(daily.get(key, 0.0), 4)})
    return series


_MAX_IN_BATCH = 200


async def _book_costs_all_time(limit: int, offset: int) -> list:
    """Read from the book_costs view and enrich with title/author from books."""
    try:
        rows = await find("book_costs", order="total_cost_usd.desc", limit=limit, offset=offset)
    except Exception:
        return []

    book_ids = list({str(r.get("book_id", "")) for r in rows if r.get("book_id")})
    books: dict[str, dict] = {}
    if book_ids:
        for i in range(0, len(book_ids), _MAX_IN_BATCH):
            try:
                batch_rows = await find(
                    "books",
                    filters={"book_id": ("in", book_ids[i:i + _MAX_IN_BATCH])},
                    select="book_id,title,author",
                )
                for b in batch_rows:
                    books[str(b.get("book_id", ""))] = b
            except Exception:
                pass

    results = []
    for r in rows:
        bid = str(r.get("book_id", ""))
        b = books.get(bid, {})
        results.append({
            "book_id":       bid,
            "title":         b.get("title") or bid,
            "author":        b.get("author") or "",
            "user_id":       r.get("user_id"),
            "total_jobs":    r.get("total_jobs", 0),
            "total_calls":   r.get("total_calls", 0),
            "cost_usd":      round(float(r.get("total_cost_usd") or 0), 4),
            "first_call_at": r.get("first_call_at"),
            "last_call_at":  r.get("last_call_at"),
        })
    return results


async def _book_costs_by_days(days: int, limit: int, offset: int) -> list:
    """Aggregate usage_logs over a date window and group by book."""
    since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        logs = await find(
            "usage_logs",
            filters={"created_at": ("gte", since_iso)},
            select="job_id,cost_usd",
            order="created_at DESC",
            limit=50000,
        )
    except Exception:
        return []

    by_job: dict[str, dict] = {}
    for r in logs:
        jid = r.get("job_id") or "__none__"
        entry = by_job.setdefault(jid, {"cost_usd": 0.0, "calls": 0})
        entry["cost_usd"] += float(r.get("cost_usd") or 0)
        entry["calls"] += 1

    job_ids = [j for j in by_job if j != "__none__"]
    if not job_ids:
        return []

    jobs: list[dict] = []
    for i in range(0, len(job_ids), _MAX_IN_BATCH):
        try:
            jobs.extend(await find(
                "pipeline_jobs",
                filters={"id": ("in", job_ids[i:i + _MAX_IN_BATCH])},
                select="id,book_id,user_id",
            ))
        except Exception:
            pass

    by_book: dict[str, dict] = {}
    for j in jobs:
        jid = j.get("id")
        if not jid or jid not in by_job:
            continue
        bid = str(j.get("book_id") or jid[:8])
        entry = by_book.setdefault(bid, {
            "book_id": bid, "title": "", "author": "", "user_id": j.get("user_id"),
            "total_jobs": 0, "total_calls": 0, "cost_usd": 0.0,
        })
        entry["total_jobs"] += 1
        entry["total_calls"] += by_job[jid]["calls"]
        entry["cost_usd"] += by_job[jid]["cost_usd"]

    book_ids = list(by_book.keys())
    books: dict[str, dict] = {}
    if book_ids:
        for i in range(0, len(book_ids), _MAX_IN_BATCH):
            try:
                batch_rows = await find(
                    "books",
                    filters={"book_id": ("in", book_ids[i:i + _MAX_IN_BATCH])},
                    select="book_id,title,author",
                )
                for b in batch_rows:
                    books[str(b.get("book_id", ""))] = b
            except Exception:
                pass

    results = []
    for bid, entry in by_book.items():
        b = books.get(bid, {})
        entry["title"] = b.get("title") or bid
        entry["author"] = b.get("author") or ""
        entry["cost_usd"] = round(entry["cost_usd"], 4)
        results.append(entry)

    results.sort(key=lambda x: x["cost_usd"], reverse=True)
    return results[offset:offset + limit]


async def _user_costs_all_time(limit: int, offset: int) -> list:
    """Read from the user_costs view."""
    try:
        rows = await find("user_costs", order="total_cost_usd.desc", limit=limit, offset=offset)
    except Exception:
        return []

    results = []
    for r in rows:
        results.append({
            "user_id":       r.get("user_id"),
            "label":         r.get("email") or r.get("name") or str(r.get("user_id", ""))[:8],
            "role":          r.get("role"),
            "total_jobs":    r.get("total_jobs", 0),
            "total_calls":   r.get("total_calls", 0),
            "cost_usd":      round(float(r.get("total_cost_usd") or 0), 4),
            "first_call_at": r.get("first_call_at"),
            "last_call_at":  r.get("last_call_at"),
        })
    return results


async def _user_costs_by_days(days: int, limit: int, offset: int) -> list:
    """Aggregate usage_logs over a date window and group by user."""
    since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    try:
        logs = await find(
            "usage_logs",
            filters={"created_at": ("gte", since_iso)},
            select="job_id,cost_usd",
            order="created_at DESC",
            limit=50000,
        )
    except Exception:
        return []

    by_job: dict[str, dict] = {}
    for r in logs:
        jid = r.get("job_id") or "__none__"
        entry = by_job.setdefault(jid, {"cost_usd": 0.0, "calls": 0})
        entry["cost_usd"] += float(r.get("cost_usd") or 0)
        entry["calls"] += 1

    job_ids = [j for j in by_job if j != "__none__"]
    if not job_ids:
        return []

    jobs: list[dict] = []
    for i in range(0, len(job_ids), _MAX_IN_BATCH):
        try:
            jobs.extend(await find(
                "pipeline_jobs",
                filters={"id": ("in", job_ids[i:i + _MAX_IN_BATCH])},
                select="id,user_id",
            ))
        except Exception:
            pass

    by_user: dict[str, dict] = {}
    for j in jobs:
        jid = j.get("id")
        if not jid or jid not in by_job:
            continue
        uid = j.get("user_id") or "__anonymous__"
        entry = by_user.setdefault(uid, {
            "user_id": uid, "label": uid,
            "total_jobs": 0, "total_calls": 0, "cost_usd": 0.0,
        })
        entry["total_jobs"] += 1
        entry["total_calls"] += by_job[jid]["calls"]
        entry["cost_usd"] += by_job[jid]["cost_usd"]

    user_ids = [u for u in by_user if u != "__anonymous__"]
    user_labels: dict[str, str] = {}
    if user_ids:
        for i in range(0, len(user_ids), _MAX_IN_BATCH):
            try:
                rows = await find(
                    "app_users",
                    filters={"id": ("in", user_ids[i:i + _MAX_IN_BATCH])},
                    select="id,email,name",
                )
                for u in rows:
                    user_labels[u["id"]] = u.get("name") or u.get("email") or u["id"][:8]
            except Exception:
                pass

    results = []
    for uid, entry in by_user.items():
        entry["label"] = user_labels.get(uid, "Anonymous" if uid == "__anonymous__" else uid[:8])
        entry["cost_usd"] = round(entry["cost_usd"], 4)
        results.append(entry)

    results.sort(key=lambda x: x["cost_usd"], reverse=True)
    return results[offset:offset + limit]


@router.get("/costs/books")
async def admin_costs_books(days: int = 0, limit: int = 100, offset: int = 0) -> dict:
    """
    Full cost breakdown per book, with pagination.

    - days=0 (default) → all-time totals from the `book_costs` view.
    - days>0           → date-filtered aggregation over the last N days.
    """
    if limit < 1:
        limit = 100
    if limit > 500:
        limit = 500
    if offset < 0:
        offset = 0

    rows = await (_book_costs_by_days(days, limit, offset) if days > 0
                  else _book_costs_all_time(limit, offset))
    return {"days": days, "limit": limit, "offset": offset, "rows": rows}


@router.get("/costs/users")
async def admin_costs_users(days: int = 0, limit: int = 100, offset: int = 0) -> dict:
    """
    Full cost breakdown per user, with pagination.

    - days=0 (default) → all-time totals from the `user_costs` view.
    - days>0           → date-filtered aggregation over the last N days.
    """
    if limit < 1:
        limit = 100
    if limit > 500:
        limit = 500
    if offset < 0:
        offset = 0

    rows = await (_user_costs_by_days(days, limit, offset) if days > 0
                  else _user_costs_all_time(limit, offset))
    return {"days": days, "limit": limit, "offset": offset, "rows": rows}


@router.get("/costs/books/{book_id}")
async def admin_book_cost_details(book_id: str, days: int = 0) -> dict:
    """
    Per-step / per-model cost breakdown for a single book.

    - days=0 (default) → all-time usage for this book.
    - days>0           → usage from the last N days only.
    """
    try:
        jobs = await find(
            "pipeline_jobs",
            filters={"book_id": book_id},
            select="id",
            limit=5000,
        )
    except Exception as exc:
        log.warning("admin_book_cost_details: DB unreachable — %s", exc)
        return {"book_id": book_id, "total_cost_usd": 0.0, "total_calls": 0, "steps": [], "jobs": []}

    job_ids = [j["id"] for j in jobs if j.get("id")]
    if not job_ids:
        return {"book_id": book_id, "total_cost_usd": 0.0, "total_calls": 0, "steps": [], "jobs": []}

    filters: dict = {"job_id": ("in", job_ids)}
    if days > 0:
        since_iso = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        filters["created_at"] = ("gte", since_iso)

    try:
        logs = await find(
            "usage_logs",
            filters=filters,
            select="job_id,step,model,provider,cost_usd",
            order="created_at DESC",
            limit=50000,
        )
    except Exception:
        logs = []

    # Fetch step durations from pipeline_step_results (one row per step per run).
    # Sum duration_sec across all runs so reruns accumulate correctly.
    try:
        step_dur_rows = await find(
            "pipeline_step_results",
            filters={"job_id": ("in", job_ids)},
            select="job_id,step,duration_sec",
            limit=50000,
        )
    except Exception:
        step_dur_rows = []

    # duration per step: sum across all runs; per job: sum of its steps
    step_duration: dict[str, int] = {}
    job_duration: dict[str, int] = {}
    for dr in step_dur_rows:
        sname = dr.get("step") or "unknown"
        jid   = dr.get("job_id") or "__none__"
        dur   = int(dr.get("duration_sec") or 0)
        step_duration[sname] = step_duration.get(sname, 0) + dur
        if jid != "__none__":
            job_duration[jid] = job_duration.get(jid, 0) + dur

    by_step: dict[str, dict] = {}
    by_job: dict[str, dict] = {}
    total_cost = 0.0
    total_calls = 0

    for r in logs:
        step = r.get("step") or "unknown"
        model = r.get("model") or "unknown"
        provider = r.get("provider") or "unknown"
        jid = r.get("job_id") or "__none__"
        cost = float(r.get("cost_usd") or 0)
        total_cost += cost
        total_calls += 1

        sentry = by_step.setdefault(step, {
            "step": step,
            "calls": 0,
            "cost_usd": 0.0,
            "models": {},
        })
        sentry["calls"] += 1
        sentry["cost_usd"] += cost

        mentry = sentry["models"].setdefault(model, {
            "model": model,
            "provider": provider,
            "calls": 0,
            "cost_usd": 0.0,
        })
        mentry["calls"] += 1
        mentry["cost_usd"] += cost

        if jid != "__none__":
            jentry = by_job.setdefault(jid, {"job_id": jid, "calls": 0, "cost_usd": 0.0, "duration_sec": 0})
            jentry["calls"] += 1
            jentry["cost_usd"] += cost

    # Attach durations to jobs
    for jid, jentry in by_job.items():
        jentry["duration_sec"] = job_duration.get(jid, 0)

    # Also include jobs that only appear in step_results (no AI calls logged)
    for jid, dur in job_duration.items():
        if jid not in by_job and jid != "__none__":
            by_job[jid] = {"job_id": jid, "calls": 0, "cost_usd": 0.0, "duration_sec": dur}

    steps = []
    for step_name, data in by_step.items():
        model_list = sorted(
            [{**m, "cost_usd": round(m["cost_usd"], 6)} for m in data["models"].values()],
            key=lambda x: x["cost_usd"],
            reverse=True,
        )
        steps.append({
            "step":         step_name,
            "calls":        data["calls"],
            "cost_usd":     round(data["cost_usd"], 6),
            "duration_sec": step_duration.get(step_name, 0),
            "models":       model_list,
        })
    steps.sort(key=lambda x: x["cost_usd"], reverse=True)

    jobs_out = sorted(
        [{**v, "cost_usd": round(v["cost_usd"], 6)} for v in by_job.values()],
        key=lambda x: x["cost_usd"],
        reverse=True,
    )

    total_duration_sec = sum(step_duration.values())

    return {
        "book_id":            book_id,
        "total_cost_usd":     round(total_cost, 6),
        "total_calls":        total_calls,
        "total_duration_sec": total_duration_sec,
        "steps":              steps,
        "jobs":               jobs_out,
    }


@router.post("/jobs/timeout-stuck", status_code=200)
async def admin_timeout_stuck_jobs(max_age_minutes: int = 60) -> dict:
    """
    Mark any job stuck in 'running' for longer than `max_age_minutes` as failed.
    Default threshold is 60 minutes.
    """
    timed_out = await timeout_stuck_jobs(max_age_minutes)
    return {
        "ok":          True,
        "timed_out":   timed_out,
        "count":       len(timed_out),
        "max_age_minutes": max_age_minutes,
    }


_CREDIT_ERROR_KEYWORDS = (
    "limit", "credits", "insufficient", "exceeded", "quota",
    "low credits", "key limit", "rate limit", "402", "403",
)


def _looks_like_credit_failure(job: dict) -> bool:
    """Return True if a job likely failed because an API key ran out of credits."""
    msg = str(job.get("error_msg") or "").lower()
    return any(k in msg for k in _CREDIT_ERROR_KEYWORDS)


async def auto_retry_credit_failures(background_tasks: BackgroundTasks | None = None) -> list[str]:
    """
    Check whether the active OpenRouter key has credits again.  If it does,
    automatically re-queue jobs that failed because of credit/key-limit errors
    so they can resume without manual admin intervention.

    Returns the list of job IDs that were re-queued.
    """
    from api.routes.pipeline import _run_job  # local import avoids cycle

    if not await openrouter_key_has_credits():
        return []

    # Credits are back — clear any exhaustion state from previous failures.
    reset_openrouter_keys()

    try:
        failed = await find(
            "pipeline_jobs",
            filters={"status": ("in", ["failed", "partial"])},
            select="id,input,result,error_msg,retry_count,max_retries",
            order="created_at DESC",
            limit=500,
        )
    except Exception as exc:
        log.warning("auto_retry_credit_failures: could not query jobs — %s", exc)
        return []

    retried: list[str] = []
    for job in failed:
        if not _looks_like_credit_failure(job):
            continue
        if not can_retry(job):
            continue

        job_id = job["id"]
        try:
            req = PipelineReq.model_validate(job.get("input") or {})
            await reset_for_manual_retry(job_id)
            if background_tasks:
                background_tasks.add_task(_run_job, job_id, req, job.get("result"), False)
            else:
                import asyncio
                asyncio.create_task(_run_job(job_id, req, job.get("result"), False))
            retried.append(job_id)
        except Exception as exc:
            log.warning("auto_retry_credit_failures: could not retry %s — %s", job_id, exc)

    if retried:
        log.info("Auto-retried %d credit-failed job(s) after key regained credits.", len(retried))
    return retried


@router.post("/jobs/auto-retry-credit-failures", status_code=200)
async def admin_auto_retry_credit_failures(background_tasks: BackgroundTasks) -> dict:
    """
    Manually trigger the credit-aware auto-retry loop.  If the OpenRouter key
    has credits, any failed/partial jobs whose error message mentions credit/
    limit/rate-limit keywords are re-queued.
    """
    retried = await auto_retry_credit_failures(background_tasks)
    return {"ok": True, "retried": retried, "count": len(retried)}


# ── Auto-retry stuck failed/partial jobs ─────────────────────────────────────
_SWEEP_ATTEMPTS_KEY = "_sweep_attempts"


async def auto_retry_stuck_jobs() -> list[str]:
    """
    Periodic sweep that re-dispatches jobs stuck in 'failed'/'partial' so their
    incomplete steps get re-run automatically — no manual "Retry" click needed.

    Differs from the two existing retry paths:
      • the inline 8-attempt loop fires only right after a failure, and
      • auto_retry_credit_failures fires only when an exhausted OpenRouter key
        regains credits.
    This catches jobs that exhausted their inline retries and would otherwise
    sit failed forever.

    Guard-rails (so we don't burn money on genuinely-unfixable books):
      • Each job is swept at most AUTO_RETRY_SWEEP_MAX_ATTEMPTS times — the
        counter is persisted in the job's `input` JSON (no schema change).
      • Credit failures are skipped — auto_retry_credit_failures owns those and
        retrying while the key is dry would just burn the sweep budget.
      • Only failed/partial steps re-run (merge-aware), never the done ones.

    Returns the list of job IDs that were re-dispatched.
    """
    import asyncio
    from api.routes.pipeline import _run_job, _failed_steps  # local import avoids cycle

    enabled = (await get_config_value("AUTO_RETRY_SWEEP_ENABLED", "true")).lower() == "true"
    if not enabled:
        return []
    try:
        max_sweeps = int(await get_config_value("AUTO_RETRY_SWEEP_MAX_ATTEMPTS", "3"))
    except (TypeError, ValueError):
        max_sweeps = 3

    try:
        stuck = await find(
            "pipeline_jobs",
            filters={"status": ("in", ["failed", "partial"])},
            select="id,status,input,result,error_msg,retry_count,max_retries",
            order="created_at DESC",
            limit=500,
        )
    except Exception as exc:
        log.warning("auto_retry_stuck_jobs: could not query jobs — %s", exc)
        return []

    retried: list[str] = []
    for job in stuck:
        # Credit failures are owned by the credit-aware loop — don't burn the
        # sweep budget retrying while the key may still be out of credits.
        if _looks_like_credit_failure(job):
            continue

        inp = dict(job.get("input") or {})
        sweeps = int(inp.get(_SWEEP_ATTEMPTS_KEY) or 0)
        if sweeps >= max_sweeps:
            continue  # exhausted — leave for a manual retry

        # A 'partial' job with no incomplete steps has nothing to do — skip it
        # so the sweep doesn't loop on it. A fully 'failed' job with no step map
        # is re-run from scratch (handled inside _run_job).
        if job.get("status") == "partial" and not _failed_steps(job.get("result")):
            continue

        job_id = job["id"]
        try:
            req = PipelineReq.model_validate(inp)
            # Persist the incremented sweep counter so the cap survives restarts.
            inp[_SWEEP_ATTEMPTS_KEY] = sweeps + 1
            await update("pipeline_jobs", {"id": job_id}, {"input": inp})

            await reset_for_manual_retry(job_id)   # fresh inline retries
            asyncio.create_task(_run_job(job_id, req, job.get("result"), False))
            retried.append(job_id)
        except Exception as exc:
            log.warning("auto_retry_stuck_jobs: could not retry %s — %s", job_id, exc)

    if retried:
        log.info("Auto-retry sweep re-dispatched %d stuck job(s).", len(retried))
    return retried


@router.post("/jobs/auto-retry-stuck", status_code=200)
async def admin_auto_retry_stuck_jobs() -> dict:
    """
    Manually trigger the stuck-job sweep. Re-dispatches every 'failed'/'partial'
    job that still has incomplete steps and hasn't exhausted its sweep budget,
    so its remaining steps run again.
    """
    retried = await auto_retry_stuck_jobs()
    return {"ok": True, "retried": retried, "count": len(retried)}


@router.post("/jobs/{job_id}/rerun", status_code=202)
async def admin_rerun_steps(
    job_id: str,
    body: RerunRequest,
    background_tasks: BackgroundTasks,
    force: bool = False,
) -> dict:
    """
    Re-run specific pipeline steps for an existing job.

    By default, steps that already succeeded in the previous run are skipped
    (only failed / running / pending / skipped steps are retried).  Pass
    `?force=true` to override and regenerate even steps that are already done.

    Body: {"steps": ["audio_full", "cover"]}

    - Previous successful results for OTHER steps are preserved in the merged output.
    - Resets retry_count to 0 so the rerun gets 3 fresh auto-retries.
    - Works on any job status (done, failed, partial, queued).
    - Automatically re-runs inject_epub if it was previously done (to include new assets).
    """
    from api.routes.pipeline import _run_job          # noqa: PLC0415
    from api.models.requests import VALID_STEPS       # noqa: PLC0415

    if not body.steps:
        raise HTTPException(422, "steps list must not be empty")

    unknown = [s for s in body.steps if s not in VALID_STEPS]
    if unknown:
        raise HTTPException(
            422,
            f"Unknown step(s): {unknown}. Valid: {sorted(VALID_STEPS)}",
        )

    try:
        job = await get_job(job_id)
    except Exception as exc:
        raise HTTPException(503, f"Database unreachable: {exc}") from exc
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    raw_input = job.get("input")
    if not raw_input:
        raise HTTPException(422, "Job has no stored input — cannot rerun")

    try:
        req = PipelineReq.model_validate(raw_input)
    except Exception as exc:
        raise HTTPException(422, f"Stored input is invalid: {exc}") from exc

    previous_result = job.get("result")
    result_dict = previous_result if isinstance(previous_result, dict) else {}
    step_statuses = result_dict.get("steps", {}) or {}

    # Filter out steps that already succeeded, unless the caller forces them.
    skipped_done: list[str] = []
    if not force:
        steps_to_run = []
        for s in body.steps:
            if step_statuses.get(s) == "done":
                skipped_done.append(s)
            else:
                steps_to_run.append(s)
    else:
        steps_to_run = list(body.steps)

    if not steps_to_run and not skipped_done:
        raise HTTPException(422, "No steps to rerun")
    if not steps_to_run:
        return {
            "ok":           True,
            "job_id":       job_id,
            "status":       "skipped",
            "rerun_steps":  [],
            "skipped_done": skipped_done,
            "message":      "All selected steps already succeeded. Use ?force=true to regenerate them.",
            "status_url":   f"/api/pipeline/status/{job_id}",
        }

    # Auto-add inject_epub if:
    # 1. It's not already in the requested steps
    # 2. It was in the original job input steps
    # 3. It was previously completed (done/partial)
    if "inject_epub" not in steps_to_run:
        original_steps = raw_input.get("steps", []) if isinstance(raw_input, dict) else []
        if "inject_epub" in original_steps and previous_result:
            if step_statuses.get("inject_epub") in ("done", "partial"):
                steps_to_run.append("inject_epub")
                log.info("Auto-adding inject_epub to rerun for job %s (needs to include new assets)", job_id)

    # Override steps to exactly what the admin selected (plus auto-added inject_epub)
    req = req.model_copy(update={"steps": steps_to_run})

    # Debug logging
    log.info("Rerun job %s: force=%s steps_to_run=%s skipped_done=%s", job_id, force, steps_to_run, skipped_done)

    # Make sure the rerun picks up any model/provider changes made in admin.
    await refresh_config_cache()

    await reset_for_manual_retry(job_id)
    # force_steps=True: use exactly the steps we chose, don't override with
    # just-the-failed-steps logic that _run_job normally applies.
    background_tasks.add_task(_run_job, job_id, req, previous_result, True)

    return {
        "ok":           True,
        "job_id":       job_id,
        "status":       "queued",
        "rerun_steps":  steps_to_run,
        "skipped_done": skipped_done,
        "status_url":   f"/api/pipeline/status/{job_id}",
    }


@router.delete("/jobs/{job_id}", status_code=200)
async def admin_delete_job(job_id: str) -> dict:
    """Permanently delete a pipeline job and its step results."""
    job = await get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")
    await delete_job(job_id)
    return {"ok": True, "job_id": job_id, "message": "Job deleted"}


@router.post("/jobs/{job_id}/retry", status_code=202)
async def admin_retry_job(job_id: str, background_tasks: BackgroundTasks) -> dict:
    """
    Manually retry a pipeline job from the admin panel.

    Smart retry — only re-runs steps that failed, merges results with
    the previous partial output so successful steps are not wasted.

    - Resets retry_count to 0 so the job gets a full 3 fresh auto-retries.
    - Works on any status (failed, partial, cancelled).
    """
    # Lazy imports — avoids circular dependency (admin ↔ pipeline)
    from api.routes.pipeline import _run_job, _failed_steps   # noqa: PLC0415

    try:
        job = await get_job(job_id)
    except Exception as exc:
        raise HTTPException(503, f"Database unreachable: {exc}") from exc
    if not job:
        raise HTTPException(404, f"Job {job_id} not found")

    raw_input = job.get("input")
    if not raw_input:
        raise HTTPException(422, "Job has no stored input — cannot retry")

    try:
        req = PipelineReq.model_validate(raw_input)
    except Exception as exc:
        raise HTTPException(422, f"Stored input is invalid: {exc}") from exc

    # Carry the previous (partial) result so _run_job can merge after retry
    previous_result = job.get("result")

    # Tell the user which steps will be re-run
    failed = _failed_steps(previous_result)

    # If the job already succeeded and there is nothing to retry, don't re-run
    # every step from scratch — that would waste credits and overwrite assets.
    if previous_result and not failed:
        return {
            "ok":            True,
            "job_id":        job_id,
            "status":        "skipped",
            "retrying_steps": [],
            "message":       "No failed / running / pending steps to retry. Use /admin/jobs/{job_id}/rerun to regenerate specific steps.",
        }

    # Make sure the retry picks up any model/provider changes made in admin.
    await refresh_config_cache()

    await reset_for_manual_retry(job_id)
    background_tasks.add_task(_run_job, job_id, req, previous_result)

    return {
        "ok":            True,
        "job_id":        job_id,
        "status":        "queued",
        "retrying_steps": failed or "all",   # "all" when there's no prior result
        "status_url":    f"/api/pipeline/status/{job_id}",
    }


# ── TTS voice preview ─────────────────────────────────────────────────────────

class TTSPreviewRequest(BaseModel):
    text:     str = "Hello, this is a voice preview."
    provider: str = "openrouter"   # openrouter | gemini | cartesia | elevenlabs | deepgram
    model:    str = ""             # provider-specific model
    voice:    str = "alloy"        # provider-specific voice
    language: str = "en"


@router.post("/tts-preview")
async def tts_preview(body: TTSPreviewRequest) -> dict:
    """
    Generate a short TTS sample and return the audio as base64.
    Used by the admin panel for live voice preview.
    """
    import tempfile
    import base64
    from api.services.pipeline.tts import synthesize

    # Short preview text — if user sent a long one, truncate
    preview_text = body.text[:500] if body.text else "Hello, this is a voice preview."

    cfg: dict = {}
    lang = body.language.upper()
    cfg[f"TTS_PROVIDER_{lang}"] = body.provider

    # Route provider-specific model/voice config keys so the preview matches
    # how the pipeline actually resolves settings at runtime.
    if body.provider == "gemini":
        if body.model:
            cfg["GEMINI_TTS_MODEL"] = body.model
        if body.voice:
            cfg["GEMINI_TTS_VOICE"] = body.voice
    elif body.provider == "openrouter":
        if body.model:
            cfg["OPENROUTER_TTS_MODEL"] = body.model
        if body.voice:
            # Per-language voice takes priority in the pipeline; also set the shared fallback.
            cfg[f"OPENROUTER_TTS_VOICE_{lang}"] = body.voice
            cfg["OPENROUTER_TTS_VOICE"] = body.voice
    elif body.provider == "elevenlabs":
        if body.voice:
            cfg[f"ELEVENLABS_VOICE_{lang}"] = body.voice
        if body.model:
            cfg["ELEVENLABS_MODEL"] = body.model
    elif body.provider == "cartesia":
        if body.model:
            cfg["CARTESIA_MODEL"] = body.model
        if body.voice:
            cfg[f"CARTESIA_VOICE_{lang}"] = body.voice
    else:
        # deepgram and any other provider that uses the generic TTS_VOICE_* key
        if body.model:
            cfg[f"TTS_VOICE_{lang}"] = body.model
        elif body.voice:
            cfg[f"TTS_VOICE_{lang}"] = body.voice

    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        await synthesize(preview_text, body.language, tmp_path, cfg=cfg)
        with open(tmp_path, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode()
        from api.services.pipeline.tts import _detect_mime_type
        return {
            "ok": True,
            "provider": body.provider,
            "voice": body.voice,
            "language": body.language,
            "audio_base64": audio_b64,
            "mime_type": _detect_mime_type(audio_bytes),
        }
    except Exception as exc:
        log.warning("TTS preview failed: %s", exc)
        raise HTTPException(400, f"TTS preview failed: {exc}") from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# ── OpenRouter live model list (cached proxy) ────────────────────────────────
#
# OpenRouter's public /api/v1/models endpoint requires no auth and lists every
# model currently routable through them — including newly-added image, chat,
# and vision models.  We proxy + cache it for the admin Providers tab so the
# dropdowns always reflect what's actually available without us pushing code.

_OR_MODELS_CACHE: dict[str, tuple[float, list[dict]]] = {}
_OR_MODELS_CACHE_TTL_SEC = 3600   # 1 hour


async def _fetch_openrouter_models_raw() -> list[dict]:
    """Fetch the full OpenRouter model list (cached).  Returns stale cache on failure."""
    now    = time.time()
    cached = _OR_MODELS_CACHE.get("all")
    if cached and (now - cached[0]) < _OR_MODELS_CACHE_TTL_SEC:
        return cached[1]

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get("https://openrouter.ai/api/v1/models")
            r.raise_for_status()
            models = (r.json() or {}).get("data") or []
    except Exception as exc:
        log.warning("OpenRouter /models fetch failed: %s", exc)
        # Return stale cache if any, else empty
        return cached[1] if cached else []

    _OR_MODELS_CACHE["all"] = (now, models)
    return models


_EL_VOICES_CACHE: dict = {}   # {"t": ts, "data": [...]}


@router.get("/elevenlabs-voices")
async def elevenlabs_voices() -> dict:
    """
    Live ElevenLabs voice list for the admin TTS dropdowns.

    Returns each voice's id, name, category, and labels (language / accent /
    gender / description) so the UI can group EN vs AR. Cached for 1 hour.
    """
    import time as _time
    import httpx as _httpx
    from api.config.settings import settings as _settings

    if not _settings.ELEVENLABS_API_KEY:
        raise HTTPException(400, "ELEVENLABS_API_KEY is not set on the server.")

    now = _time.time()
    cached = _EL_VOICES_CACHE.get("data")
    if cached and (now - _EL_VOICES_CACHE.get("t", 0)) < 3600:
        return {"count": len(cached), "voices": cached}

    try:
        async with _httpx.AsyncClient(timeout=20) as client:
            r = await client.get(
                "https://api.elevenlabs.io/v1/voices",
                headers={"xi-api-key": _settings.ELEVENLABS_API_KEY},
            )
            r.raise_for_status()
            data = r.json()
    except Exception as exc:
        raise HTTPException(502, f"ElevenLabs voices fetch failed: {exc}") from exc

    voices = []
    for v in data.get("voices", []):
        labels = v.get("labels") or {}
        voices.append({
            "voice_id":    v.get("voice_id"),
            "name":        v.get("name"),
            "category":    v.get("category"),
            "language":    labels.get("language") or "",
            "accent":      labels.get("accent") or "",
            "gender":      labels.get("gender") or "",
            "description": labels.get("description") or labels.get("use_case") or "",
        })
    voices.sort(key=lambda x: (x.get("name") or "").lower())

    _EL_VOICES_CACHE["t"] = now
    _EL_VOICES_CACHE["data"] = voices
    return {"count": len(voices), "voices": voices}


@router.get("/openrouter-models")
async def openrouter_models(modality: str = "all") -> dict:
    """
    Live OpenRouter model list, optionally filtered by modality.

    `modality` values:
      - "all"     every model OpenRouter routes
      - "image"   models with image OUTPUT (for cover gen)
      - "vision"  models with image INPUT and text OUTPUT (for alt-text)
      - "chat"    models with text OUTPUT (for summarization / mindmap)

    Cached server-side for 1 hour.  Safe to call on every admin tab mount.
    """
    raw = await _fetch_openrouter_models_raw()

    def _out_mods(m: dict) -> list[str]:
        arch = m.get("architecture") or {}
        return arch.get("output_modalities") or []

    def _in_mods(m: dict) -> list[str]:
        arch = m.get("architecture") or {}
        return arch.get("input_modalities") or []

    if modality == "image":
        filtered = [m for m in raw if "image" in _out_mods(m)]
    elif modality == "vision":
        filtered = [m for m in raw if "image" in _in_mods(m) and "text" in _out_mods(m)]
    elif modality == "chat":
        filtered = [m for m in raw if "text" in _out_mods(m) and "image" not in _out_mods(m)]
    else:
        filtered = raw

    # Sort: by name when present, else by id — stable and predictable in the dropdown.
    filtered.sort(key=lambda m: (m.get("name") or m["id"]).lower())

    return {
        "modality": modality,
        "count":    len(filtered),
        "models": [
            {
                "id":      m["id"],
                "name":    m.get("name") or m["id"],
                "context": m.get("context_length"),
            }
            for m in filtered
        ],
    }
