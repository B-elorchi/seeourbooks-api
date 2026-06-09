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
from api.services.config.runtime import get_all_config, set_config_key
from api.services.db import find, insert, upsert
from api.jobs.store import get_job, reset_for_manual_retry
from api.models.requests import PipelineReq

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

        async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
            for url in epub_candidates:
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200 and len(resp.content) > 1000:
                        raw_bytes, used_url, is_epub = resp.content, url, True
                        log.info("Ingest %s: downloaded EPUB from %s (%d bytes)", book_id, url, len(raw_bytes))
                        break
                except Exception as exc:
                    log.debug("EPUB candidate %s failed: %s", url, exc)

            if not raw_bytes:
                for url in txt_candidates:
                    try:
                        resp = await client.get(url)
                        if resp.status_code == 200 and len(resp.content) > 500:
                            raw_bytes, used_url, is_epub = resp.content, url, False
                            log.info("Ingest %s: downloaded TXT from %s (%d bytes)", book_id, url, len(raw_bytes))
                            break
                    except Exception as exc:
                        log.debug("TXT candidate %s failed: %s", url, exc)

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
                log.warning("chunk[%d] insert failed: %s", idx, exc)

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


@router.post("/jobs/{job_id}/rerun", status_code=202)
async def admin_rerun_steps(
    job_id: str,
    body: RerunRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Re-run specific pipeline steps for an existing job — regardless of their
    previous status (done, failed, or skipped).

    Useful when you want to regenerate just one output (e.g. re-generate audio
    or cover) without running the full pipeline again.

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

    # Check if inject_epub was in the original job steps and was completed
    # If so, we need to re-run it to include the new assets
    steps_to_run = list(body.steps)
    previous_result = job.get("result")
    
    # Auto-add inject_epub if:
    # 1. It's not already in the requested steps
    # 2. It was in the original job input steps
    # 3. It was previously completed (done/partial)
    if "inject_epub" not in steps_to_run:
        original_steps = raw_input.get("steps", []) if isinstance(raw_input, dict) else []
        if "inject_epub" in original_steps and previous_result:
            result_dict = previous_result if isinstance(previous_result, dict) else {}
            step_statuses = result_dict.get("steps", {})
            if step_statuses.get("inject_epub") in ("done", "partial"):
                steps_to_run.append("inject_epub")
                log.info("Auto-adding inject_epub to rerun for job %s (needs to include new assets)", job_id)

    # Override steps to exactly what the admin selected (plus auto-added inject_epub)
    req = req.model_copy(update={"steps": steps_to_run})
    
    # Debug logging
    log.info("Rerun job %s: steps_to_run=%s, req.steps=%s", job_id, steps_to_run, req.steps)

    await reset_for_manual_retry(job_id)
    # force_steps=True: use exactly the steps the admin chose, don't override
    # with just-the-failed-steps logic that _run_job normally applies.
    background_tasks.add_task(_run_job, job_id, req, previous_result, True)

    return {
        "ok":          True,
        "job_id":      job_id,
        "status":      "queued",
        "rerun_steps": steps_to_run,
        "status_url":  f"/api/pipeline/status/{job_id}",
    }


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

    await reset_for_manual_retry(job_id)
    background_tasks.add_task(_run_job, job_id, req, previous_result)

    return {
        "ok":            True,
        "job_id":        job_id,
        "status":        "queued",
        "retrying_steps": failed or "all",   # "all" when there's no prior result
        "status_url":    f"/api/pipeline/status/{job_id}",
    }


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
