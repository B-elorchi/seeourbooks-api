"""
POST /api/v2/pipeline/run

Smart pipeline endpoint — does everything in one call:

  1. Look up the book in Supabase.
  2. If it doesn't exist → create the row (auto-fetch metadata from Gutendex).
  3. Check whether chunks already exist.
  4. If no chunks → download EPUB (or TXT fallback) from CDN, extract text,
     chunk it, and save to the `chunks` table.
  5. Start the pipeline job and return 202 immediately.

Request body (all fields except book_id are optional):

  {
    "book_id":  "86",
    "language": "en",          // default "en"
    "source":   "catalog",     // passed through to pipeline metadata
    "steps":    [],            // [] = all steps; or ["cover", "audio_full"] etc.
    "options":  { "length": "10min", "style": "narrative" }
  }

Response (202 Accepted):

  {
    "job_id":     "...",
    "status":     "queued",
    "status_url": "/api/pipeline/status/...",
    "book_id":    "86",
    "title":      "The Scarlet Letter",
    "ingest":     "skipped" | "done",     // whether ingestion was performed
    "chunks":     42                       // total chunks now in DB
  }
"""
import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from html.parser import HTMLParser

import httpx
from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel

from api.jobs.store import create_job, set_failed, set_cancelled, is_cancelled, get_output
from api.models.requests import PipelineReq, PipelineOptions, VALID_STEPS
from api.routes.pipeline import _run_job, JobCancelledError
from api.services.db import find, insert, upsert, update
from api.services.config.runtime import get_config_value
from api.auth.apikey import get_api_key_user

log = logging.getLogger(__name__)

router = APIRouter(prefix="/v2/pipeline")


# ── Request model ─────────────────────────────────────────────────────────────

class V2PipelineReq(BaseModel):
    book_id:  str
    language: str = "en"
    source:   str = "catalog"
    steps:    list[str] = []
    options:  PipelineOptions = PipelineOptions()

    def validate_steps(self) -> None:
        unknown = [s for s in self.steps if s not in VALID_STEPS]
        if unknown:
            raise ValueError(
                f"Unknown step(s): {unknown}. Valid: {sorted(VALID_STEPS)}"
            )


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/run", status_code=202)
async def v2_pipeline_run(req: V2PipelineReq, background_tasks: BackgroundTasks, request: Request) -> dict:
    """
    One-shot smart pipeline. Returns a job_id IMMEDIATELY (202).

    All the slow work — downloading the EPUB, extracting text, chunking,
    saving to DB, and running the pipeline — happens in the background.
    Poll /api/pipeline/status/{job_id} to track progress; if the book file
    can't be found or ingest fails, the job is marked `failed` with the reason.
    """
    try:
        req.validate_steps()
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    book_id = req.book_id.strip()
    if not book_id:
        raise HTTPException(422, "book_id is required")

    bid: object = book_id
    try:
        bid = int(book_id)
    except ValueError:
        pass

    # Quick, non-blocking lookup so we can echo any known title in the response.
    book_row = await _get_book_row(bid)

    # If the stored title is just the id/placeholder, try to resolve it from
    # Gutendex now so the immediate response doesn't show "11" as the title.
    if _is_placeholder_title((book_row or {}).get("title"), book_id) and book_id.isdigit():
        try:
            g = await _gutendex_meta(book_id)
            if g.get("title"):
                patch = {"book_id": bid, "title": g["title"]}
                if g.get("author"):
                    patch["author"] = g["author"]
                await upsert("books", patch, conflict="book_id")
                book_row = {**(book_row or {}), **patch}
                log.info("v2: resolved placeholder title for book %s to %r", book_id, g["title"])
        except Exception as exc:
            log.debug("v2: could not resolve placeholder title for %s: %s", book_id, exc)

    # Build the pipeline request up-front (title/author may be backfilled later).
    pipeline_req = PipelineReq(
        book_id=book_id,
        title=(book_row or {}).get("title") or None,
        author=(book_row or {}).get("author") or None,
        language=req.language,
        steps=req.steps,
        source=req.source,
        options=req.options,
    )

    # Look for a previous completed/partial result so we can reuse existing
    # summaries when the user only asks for a subset of steps (e.g. just cover).
    previous_result = None
    try:
        prev_job = await get_output(book_id)
        if prev_job:
            previous_result = prev_job.get("result")
            log.info("v2: found previous result for book %s — will reuse done steps", book_id)
    except Exception as exc:
        log.debug("v2: could not load previous result for %s: %s", book_id, exc)

    # Attach user_id from API key if present
    api_user = await get_api_key_user(request)
    user_id = api_user.user_id if api_user else None

    # Create the job row now so the client gets an id to poll immediately.
    job_id = await create_job(book_id, pipeline_req.model_dump(), user_id=user_id)

    # Record user→book link (best-effort, non-blocking)
    if user_id:
        try:
            from api.services.db import upsert as db_upsert  # noqa: PLC0415
            await db_upsert("user_books", {"user_id": user_id, "book_id": book_id, "job_id": job_id})
        except Exception:
            pass

    # Do everything heavy in the background: ingest (if needed) → run pipeline.
    background_tasks.add_task(
        _ingest_then_run, job_id, book_id, bid, book_row, req, previous_result,
    )

    return {
        "ok":         True,
        "job_id":     job_id,
        "status":     "queued",
        "status_url": f"/api/pipeline/status/{job_id}",
        "book_id":    book_id,
        "title":      (book_row or {}).get("title", ""),
        "author":     (book_row or {}).get("author", ""),
    }


async def _ingest_then_run(
    job_id: str,
    book_id: str,
    bid: object,
    book_row: dict | None,
    req: V2PipelineReq,
    previous_result: dict | str | None = None,
) -> None:
    """
    Background worker: ensure chunks exist (download + ingest if missing),
    ensure the book row exists, then hand off to the normal pipeline runner.
    Any ingest error is written to the job so the client can see it.
    """
    # Flip the job to "running" right away with an "ingest" marker so the UI
    # doesn't show it stuck in "queued" during the (potentially long) download +
    # text-extraction + chunk-insert phase that happens before _run_job starts.
    try:
        await update(
            "pipeline_jobs",
            {"id": job_id},
            {
                "status": "running",
                "result": {
                    "book_id":      book_id,
                    "status":       "running",
                    "current_step": "ingest",
                    "running_steps": ["ingest"],
                    "steps":        {},
                },
            },
        )
    except Exception as exc:
        log.debug("v2: could not set ingest-running status for %s: %s", job_id, exc)

    try:
        ingest_status, _chunks, _meta, final_row = await _ensure_chunks(
            book_id, bid, book_row, req.language
        )
        log.info("v2: ingest=%s for book %s — handing off to pipeline", ingest_status, book_id)
    except HTTPException as exc:
        # 404 (file not found) / 422 (bad file) / 502 (db) — record on the job.
        await set_failed(job_id, f"ingest: {exc.detail}")
        return
    except Exception as exc:
        log.exception("v2: ingest failed for book %s", book_id)
        await set_failed(job_id, f"ingest: {exc}")
        return

    # Check for cancellation between ingest and pipeline phases.
    if is_cancelled(job_id):
        await set_cancelled(job_id)
        return

    # Hand off to the standard pipeline runner with resolved metadata.
    pipeline_req = PipelineReq(
        book_id=book_id,
        title=final_row.get("title") or None,
        author=final_row.get("author") or None,
        language=req.language,
        steps=req.steps,
        source=req.source,
        options=req.options,
    )
    # force_steps=True: run exactly the steps the user selected (plus auto-added
    # dependencies), don't auto-switch to retrying old failed steps.
    await _run_job(job_id, pipeline_req, previous_result, True)


# ── Helpers ───────────────────────────────────────────────────────────────────

async def _get_book_row(bid: object) -> dict | None:
    """Return the existing book row, or None if not in the DB. Never raises on 'not found'."""
    try:
        rows = await find("books", filters={"book_id": bid}, limit=1)
    except Exception as exc:
        raise HTTPException(502, f"DB read failed: {exc}") from exc
    return rows[0] if rows else None


def _is_placeholder_title(title: str | None, book_id: str) -> bool:
    """Detect titles that are just the id / 'Book <id>' / empty / generic placeholders."""
    if not title:
        return True
    t = str(title).strip()
    if not t:
        return True
    if t == str(book_id).strip():
        return True
    if t.lower() == f"book {book_id}".lower():
        return True
    if t.lower() in {"untitled", "unknown", "n/a", "null", "none"}:
        return True
    return False


async def _ensure_book_row(
    book_id: str,
    bid: object,
    existing: dict | None,
    epub_meta: dict,
) -> dict:
    """
    Make sure a book row exists, creating it if needed.

    Title/author preference order:
      1. existing row values (if the book was already in the DB)
      2. metadata embedded in the EPUB file (dc:title / dc:creator)
      3. Gutendex API (numeric Gutenberg ids only)
      4. fallback "Book <id>" so we never block on a missing title
    """
    if existing:
        # Backfill title/author from the EPUB or Gutendex if the row is missing
        # them or if the title is just a placeholder (e.g. the raw book_id).
        patch: dict = {}
        title_placeholder = _is_placeholder_title(existing.get("title"), book_id)
        if title_placeholder and epub_meta.get("title"):
            patch["title"] = epub_meta["title"]
        if not existing.get("author") and epub_meta.get("author"):
            patch["author"] = epub_meta["author"]

        # For numeric ids, Gutendex is a reliable fallback when EPUB meta is empty.
        if (title_placeholder or not existing.get("author")) and book_id.isdigit():
            g = await _gutendex_meta(book_id)
            if title_placeholder and g.get("title"):
                patch["title"] = g["title"]
            if not existing.get("author") and g.get("author"):
                patch["author"] = g["author"]

        if patch:
            try:
                await upsert("books", {"book_id": bid, **patch}, conflict="book_id")
                existing = {**existing, **patch}
            except Exception as exc:
                log.warning("v2: books backfill failed: %s", exc)
        return existing

    # No row yet — build one from the best metadata we have.
    title  = epub_meta.get("title", "")
    author = epub_meta.get("author", "")

    if (not title or _is_placeholder_title(title, book_id)) and book_id.isdigit():
        g = await _gutendex_meta(book_id)
        title  = title  or g.get("title", "")
        author = author or g.get("author", "")

    if _is_placeholder_title(title, book_id):
        title = f"Book {book_id}"   # last-resort placeholder — pipeline can still run

    row: dict = {
        "book_id": bid,
        "title":   title,
        "author":  author,
    }
    try:
        await upsert("books", row, conflict="book_id")
        log.info("v2: created book row for %s — %r", book_id, title)
    except Exception as exc:
        # chunks.book_id has a FK to books.book_id — if we can't create the
        # parent row, the chunk inserts will fail too. Surface the real reason.
        raise HTTPException(
            502,
            f"Could not create the book row for {book_id!r} in `books`: {exc}. "
            "The `chunks` foreign key requires this row to exist first."
        ) from exc

    return row


async def _ensure_chunks(
    book_id: str,
    bid: object,
    book_row: dict | None,
    language: str,
) -> tuple[str, int, dict, dict]:
    """
    Ensure chunks exist for the book. If not, download EPUB/TXT and ingest.
    Always makes sure the parent `books` row exists (FK requirement) BEFORE
    inserting chunks.

    Returns (ingest_status, total_chunks_in_db, epub_meta, final_book_row).
    Raises HTTP 404 if no source file can be found on the CDN.
    """
    try:
        existing = await find("chunks", filters={"book_id": bid}, limit=1)
    except Exception:
        existing = []

    if existing:
        try:
            all_chunks = await find("chunks", filters={"book_id": bid}, limit=5000)
            count = len(all_chunks)
        except Exception:
            count = 1
        # Book row must already exist (FK), but ensure it anyway for metadata.
        final_row = await _ensure_book_row(book_id, bid, book_row, {})
        log.info("v2: book %s already has %d chunks — skipping ingest", book_id, count)
        return "skipped", count, {}, final_row

    log.info("v2: no chunks for book %s — starting ingest", book_id)
    count, meta, final_row = await _ingest_book(book_id, bid, book_row, language)
    return "done", count, meta, final_row


async def _ingest_book(
    book_id: str,
    bid: object,
    book_row: dict | None,
    language: str,
) -> tuple[int, dict, dict]:
    """
    Download EPUB/TXT from the CDN, extract text + metadata, create the parent
    `books` row (FK requirement), then chunk and save to the `chunks` table.

    Returns (chunks_saved, epub_meta, final_book_row).
    Raises HTTP 404 if the source file can't be found.
    """
    from api.services.summarizer.chunker import chunk_text  # noqa: PLC0415

    base = await get_config_value("BOOK_FILES_BASE_URL", "https://files.seeourbook.sa")
    base = base.rstrip("/")
    book_row = book_row or {}

    # Map the request language to the CDN folder name.
    lang_path = "arabic" if language == "ar" else "english"

    epub_candidates = [
        c for c in [
            book_row.get("epub_download"),
            f"{base}/books/{lang_path}/{book_id}.epub",
        ] if c
    ]
    txt_candidates = [
        c for c in [
            book_row.get("txt_download"),
            f"{base}/books/{lang_path}/{book_id}.txt",
        ] if c
    ]

    raw_bytes: bytes = b""
    used_url:  str   = ""
    is_epub:   bool  = False
    tried:     list[str] = []

    # Mimic a browser User-Agent — some CDNs (including nginx-based file servers)
    # return 403 or a redirect-to-error-page for the default python-httpx agent.
    _HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
    }

    async with httpx.AsyncClient(timeout=300, follow_redirects=True, headers=_HEADERS) as client:
        for url in epub_candidates:
            tried.append(url)
            try:
                r = await client.get(url)
                # Accept 200 OK and 206 Partial Content (range response from some CDNs)
                if r.status_code in (200, 206) and len(r.content) > 1000:
                    raw_bytes, used_url, is_epub = r.content, url, True
                    log.info("v2: downloaded EPUB for %s from %s (%d B)", book_id, url, len(raw_bytes))
                    break
                log.warning("v2: EPUB %s → HTTP %s (skipping)", url, r.status_code)
            except Exception as exc:
                log.warning("v2: EPUB %s failed: %s", url, exc)

        if not raw_bytes:
            for url in txt_candidates:
                tried.append(url)
                try:
                    r = await client.get(url)
                    if r.status_code in (200, 206) and len(r.content) > 500:
                        raw_bytes, used_url, is_epub = r.content, url, False
                        log.info("v2: downloaded TXT for %s from %s (%d B)", book_id, url, len(raw_bytes))
                        break
                    log.warning("v2: TXT %s → HTTP %s (skipping)", url, r.status_code)
                except Exception as exc:
                    log.warning("v2: TXT %s failed: %s", url, exc)

    if not raw_bytes:
        raise HTTPException(
            404,
            f"No source file found for book {book_id!r} (language={language!r}). "
            f"Tried: {tried}. "
            "The file could not be downloaded from the CDN — check the server logs "
            "for the HTTP status code returned for each URL."
        )

    # ── Extract text + metadata ───────────────────────────────────────────────
    meta: dict = {}
    if is_epub:
        text = _extract_epub_text(raw_bytes)
        meta = _extract_epub_meta(raw_bytes)
    else:
        text = _strip_gutenberg(raw_bytes.decode("utf-8", errors="replace"))

    text = text.strip()
    if len(text) < 500:
        raise HTTPException(422, f"Extracted text too short ({len(text)} chars) — file may be corrupt.")

    # Per-language chunk size (admin-configurable: CHUNK_WORDS_EN / CHUNK_WORDS_AR)
    _chunk_key = "CHUNK_WORDS_AR" if language == "ar" else "CHUNK_WORDS_EN"
    try:
        _chunk_words = int(await get_config_value(_chunk_key, "1500") or "1500")
    except (TypeError, ValueError):
        _chunk_words = 1500
    chunks = chunk_text(text, max_words=_chunk_words)
    if not chunks:
        raise HTTPException(422, "Chunking produced no segments.")

    # ── Create the parent `books` row FIRST (chunks.book_id has a FK to it) ────
    final_row = await _ensure_book_row(book_id, bid, book_row or None, meta)

    saved = await _save_chunks(book_id, bid, chunks, final_row, meta)
    log.info("v2: ingested %d chunks for book %s from %s", saved, book_id, used_url)
    return saved, meta, final_row


async def _save_chunks(
    book_id: str,
    bid: object,
    chunks: list[str],
    book_row: dict,
    meta: dict,
) -> int:
    """
    Save chunks to the production `chunks` table.

    Schema (client production):
      chunk_id      uuid    PK, auto (gen_random_uuid) — never set by us
      book_id       bigint
      chunk_index   integer NOT NULL
      content       text    NOT NULL
      total_chunks  integer
      wordcount     bigint
      book_title    text
      author        text

    The table has NO unique (book_id, chunk_index) constraint, so we INSERT
    rather than upsert. book_id is bigint → must be sent as an int.
    """
    total = len(chunks)
    book_title = book_row.get("title") or meta.get("title") or ""
    author     = book_row.get("author") or meta.get("author") or ""

    # book_id is bigint — coerce to int. If the id isn't numeric, leave as-is
    # (a non-numeric id can't match a bigint column, but we surface the error).
    book_id_val: object = bid if isinstance(bid, int) else book_id

    def _row(idx: int, content: str) -> dict:
        return {
            "book_id":      book_id_val,
            "chunk_index":  idx,
            "content":      content,
            "total_chunks": total,
            "wordcount":    len(content.split()),
            "book_title":   book_title,
            "author":       author,
        }

    # ── Probe with the first chunk so we surface the real schema error early ──
    try:
        await insert("chunks", _row(0, chunks[0]))
    except Exception as exc:
        raise HTTPException(
            502,
            f"Could not save chunks for book {book_id!r}. Database rejected the write: {exc}. "
            "Expected `chunks` columns: book_id (bigint), chunk_index, content, "
            "total_chunks, wordcount, book_title, author."
        ) from exc

    # ── Save the remaining chunks ─────────────────────────────────────────────
    saved = 1
    for idx in range(1, total):
        try:
            await insert("chunks", _row(idx, chunks[idx]))
            saved += 1
        except Exception as exc:
            log.warning("v2: chunk[%d] insert failed: %s", idx, exc)

    return saved


# ── Gutendex metadata ─────────────────────────────────────────────────────────

async def _gutendex_meta(book_id: str) -> dict:
    result: dict = {}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.get(f"https://gutendex.com/books?ids={book_id}")
            r.raise_for_status()
            books = r.json().get("results", [])
        if not books:
            return result
        book = books[0]
        result["title"] = book.get("title", "")
        authors = book.get("authors", [])
        if authors:
            name = authors[0].get("name", "")
            if "," in name:
                last, first = name.split(",", 1)
                result["author"] = f"{first.strip()} {last.strip()}"
            else:
                result["author"] = name
    except Exception as exc:
        log.debug("Gutendex fetch for %s failed: %s", book_id, exc)
    return result


# ── EPUB metadata extraction (Dublin Core in the OPF) ────────────────────────

def _extract_epub_meta(epub_bytes: bytes) -> dict:
    """
    Pull title and author from the EPUB's OPF metadata (dc:title / dc:creator).
    Returns {} if nothing usable is found.
    """
    result: dict = {}
    try:
        with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
            names = set(zf.namelist())

            container = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            opf_path = ""
            for elem in ET.fromstring(container).iter():
                if elem.tag.endswith("rootfile"):
                    opf_path = elem.get("full-path", "")
                    break
            if not opf_path or opf_path not in names:
                return result

            opf_root = ET.fromstring(zf.read(opf_path).decode("utf-8", errors="ignore"))
            for elem in opf_root.iter():
                tag = elem.tag.lower()
                text = (elem.text or "").strip()
                if not text:
                    continue
                if tag.endswith("}title") or tag.endswith("title"):
                    result.setdefault("title", text)
                elif tag.endswith("}creator") or tag.endswith("creator"):
                    result.setdefault("author", text)
    except Exception as exc:
        log.debug("EPUB meta extraction failed: %s", exc)
    return result


# ── EPUB text extraction (stdlib only) ───────────────────────────────────────

class _HtmlTextExtractor(HTMLParser):
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
            s = data.strip()
            if s:
                self._parts.append(s)

    def get_text(self) -> str:
        return " ".join(self._parts)


def _extract_epub_text(epub_bytes: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(epub_bytes)) as zf:
        names = set(zf.namelist())
        spine_files: list[str] = []

        try:
            container_xml = zf.read("META-INF/container.xml").decode("utf-8", errors="ignore")
            container_root = ET.fromstring(container_xml)
            opf_path = ""
            for elem in container_root.iter():
                if elem.tag.endswith("rootfile"):
                    opf_path = elem.get("full-path", "")
                    break

            if opf_path and opf_path in names:
                opf_dir = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
                opf_root = ET.fromstring(zf.read(opf_path).decode("utf-8", errors="ignore"))

                manifest: dict[str, str] = {}
                for elem in opf_root.iter():
                    if elem.tag.endswith("item"):
                        href  = elem.get("href", "")
                        media = elem.get("media-type", "")
                        if "html" in media or href.endswith((".html", ".xhtml", ".htm")):
                            full = opf_dir + href if not href.startswith("/") else href.lstrip("/")
                            manifest[elem.get("id", "")] = full

                for elem in opf_root.iter():
                    if elem.tag.endswith("itemref"):
                        idref = elem.get("idref", "")
                        if idref in manifest and manifest[idref] in names:
                            spine_files.append(manifest[idref])
        except Exception as exc:
            log.debug("EPUB OPF parse failed, using sorted fallback: %s", exc)

        if not spine_files:
            spine_files = sorted(
                n for n in names
                if n.lower().endswith((".html", ".xhtml", ".htm"))
                and "toc" not in n.lower()
                and "nav" not in n.lower()
            )

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
