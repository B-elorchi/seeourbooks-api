"""
Document upload route:
  POST /api/document/upload
    multipart form:
      file:     PDF file
      language: en | ar  (default: en)
      steps:    comma-separated e.g. "summarize,audio_full,mindmap"  (default: all)
      length:   3min | 5min | 10min | 15min  (default: 10min)
      style:    narrative | bullets | academic  (default: narrative)

  Returns 202 with job_id immediately.

  Background flow (visible via /api/pipeline/status/{job_id}):
    1. OCR        — if the PDF is scanned (no text layer), run ocrmypdf
                    with Arabic+English tesseract models.
    2. Extract    — PyMuPDF (great Arabic/RTL support) with pdfplumber
                    fallback per page.
    3. Metadata   — an AI model reads the first pages and returns the real
                    book title + author (+ description/category), so the
                    cover is generated with the correct names instead of
                    the filename / "Unknown".
    4. Chapters   — split on Chapter headings, else ~10-page parts.
    5. Pipeline   — same engine as /api/pipeline/run (summary, cover,
                    audio, mindmap, epub …).
"""
import asyncio
import functools
import json as json_module
import logging
import os
import re
import tempfile
import uuid
from pathlib import Path

from pydantic import BaseModel
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File, Form

from api.auth import AuthUser, get_current_user
from api.jobs.store import create_job, set_running, set_done, set_failed, set_partial
from api.models.requests import PipelineReq, PipelineOptions, Chapter
from api.services.db import update as db_update
from api.services.pipeline.orchestrator import run_pipeline

log = logging.getLogger(__name__)

router = APIRouter(prefix="/document", tags=["document"])

from fastapi.responses import StreamingResponse
import httpx

@router.get("/proxy/epub")
async def proxy_epub(url: str):
    async def fetch_stream():
        async with httpx.AsyncClient(follow_redirects=True) as client:
            async with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise HTTPException(status_code=400, detail="Failed to fetch EPUB")
                async for chunk in response.aiter_bytes():
                    yield chunk
    return StreamingResponse(fetch_stream(), media_type="application/epub+zip")

MAX_FILE_SIZE = 80 * 1024 * 1024   # 80 MB


@router.post("/upload", status_code=202)
async def document_upload(
    background_tasks: BackgroundTasks,
    file:     UploadFile  = File(...),
    language: str         = Form("en"),
    steps:    str         = Form(""),           # "summarize,audio_full" or ""
    length:   str         = Form("10min"),
    style:    str         = Form("narrative"),
    user:     AuthUser | None = Depends(get_current_user),
):
    """Upload a PDF and start a pipeline job on its content. Returns 202 immediately."""

    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large — max {MAX_FILE_SIZE // 1024 // 1024} MB")
    if len(raw) < 1000:
        raise HTTPException(422, "File is too small to be a valid PDF")

    # Persist the upload to a temp dir that survives this request — extraction
    # and OCR run in the background so big/scanned PDFs don't block the upload.
    workdir  = Path(tempfile.mkdtemp(prefix="sob_doc_"))
    pdf_path = workdir / "original.pdf"
    pdf_path.write_bytes(raw)

    book_id        = f"pdf_{uuid.uuid4().hex[:8]}"
    fallback_title = os.path.splitext(file.filename)[0].replace("_", " ").title()
    steps_list     = [s.strip() for s in steps.split(",") if s.strip()]

    # Create the job row first so the client gets an id to poll immediately.
    placeholder_req = PipelineReq(
        book_id  = book_id,
        title    = fallback_title,
        language = language,
        steps    = steps_list,
        options  = PipelineOptions(length=length, style=style),
        source   = "pdf_upload",
    )
    job_id = await create_job(book_id, placeholder_req.model_dump(), user_id=user.id if user else None)

    background_tasks.add_task(
        _extract_then_run,
        job_id, book_id, str(pdf_path), fallback_title,
        language, steps_list, length, style,
    )

    return {
        "job_id":      job_id,
        "book_id":     book_id,
        "status":      "queued",
        "status_url":  f"/api/pipeline/status/{job_id}",
    }


# ── Background runner ─────────────────────────────────────────────────────────

async def _extract_then_run(
    job_id: str,
    book_id: str,
    pdf_path: str,
    fallback_title: str,
    language: str,
    steps_list: list[str],
    length: str,
    style: str,
) -> None:
    from api.services.usage_logger import set_job_context, set_step  # noqa: PLC0415
    set_job_context(job_id)

    async def _mark(step_name: str) -> None:
        """Show extraction progress in the job row so the UI isn't stuck on 'queued'."""
        try:
            await db_update("pipeline_jobs", {"id": job_id}, {
                "status": "running",
                "result": {
                    "book_id": book_id, "status": "running",
                    "current_step": step_name, "running_steps": [step_name],
                    "steps": {},
                },
            })
        except Exception:
            pass

    try:
        # ── 1. OCR when the PDF has no text layer (scanned book) ─────────────
        set_step("ocr")
        await _mark("ocr")
        source_path = await _ensure_text_layer(pdf_path)

        # ── 2. Extract text — PyMuPDF (Arabic-aware) + pdfplumber fallback ───
        set_step("extract")
        await _mark("extract")
        import asyncio
        from api.services.documents.extract import extract_pages  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        pages, total_pages = await loop.run_in_executor(None, extract_pages, source_path)
        full_text = "\n\n".join(_sanitize(p["content"]) for p in pages)
        if len(full_text.split()) < 100:
            await set_failed(job_id, "Extracted text too short — the PDF may be empty or image-only and OCR failed.")
            return

        # ── 3. AI metadata: real title + author from the first pages ─────────
        set_step("metadata")
        await _mark("metadata")
        meta = await _extract_metadata(full_text, fallback_title, language)
        title  = meta.get("title")  or fallback_title
        author = meta.get("author") or ""
        log.info("doc job %s: metadata title=%r author=%r", job_id, title, author)

        # ── 4. Split into chapters ────────────────────────────────────────────
        chapters = _split_chapters(full_text)
        if not chapters:
            await set_failed(job_id, "No usable text content found in PDF")
            return

        # ── 5. Run the pipeline with the REAL metadata ────────────────────────
        req = PipelineReq(
            book_id  = book_id,
            title    = title,
            author   = author,
            language = language,
            chapters = chapters,
            steps    = steps_list,
            options  = PipelineOptions(length=length, style=style),
            source   = "pdf_upload",
        )
        # Store the enriched input on the job so retries keep the metadata.
        try:
            await db_update("pipeline_jobs", {"id": job_id}, {"input": req.model_dump()})
        except Exception:
            pass

        await set_running(job_id)
        result = await run_pipeline(req, job_id=job_id)

        # Attach extracted pages so the job detail view can show a text reader.
        # Cap at 300 pages and 3 000 chars/page to stay within JSONB size limits.
        result["extracted_pages"] = [
            {"page": p.get("page", i + 1), "content": _sanitize(p.get("content", ""))[:3000]}
            for i, p in enumerate(pages[:300])
        ]
        result["page_count_extracted"] = total_pages

        if result["status"] == "done":
            await set_done(job_id, result)
        elif result["status"] == "partial":
            await set_partial(job_id, result)
        else:
            await set_failed(job_id, str(result.get("errors", "unknown error")))
    except Exception as e:
        log.exception("document job %s failed", job_id)
        await set_failed(job_id, str(e))
    finally:
        # Clean the temp dir (original + ocr pdf).
        try:
            import shutil
            shutil.rmtree(Path(pdf_path).parent, ignore_errors=True)
        except Exception:
            pass


async def _ensure_text_layer(pdf_path: str) -> str:
    """
    Return a path to a PDF that has extractable text.

    Scanned PDFs (no text layer) are run through ocrmypdf with Arabic+English
    tesseract models. Born-digital PDFs are returned unchanged. If OCR tooling
    is missing on the host, we proceed with the original file and let
    extraction report the problem.
    """
    from api.services.documents.ocr import needs_ocr, run_ocrmypdf  # noqa: PLC0415
    from api.services.config.runtime import get_config_value        # noqa: PLC0415
    from api.config.settings import settings                        # noqa: PLC0415
    import asyncio

    loop = asyncio.get_running_loop()
    try:
        needs = await loop.run_in_executor(None, needs_ocr, pdf_path)
    except Exception as exc:
        log.warning("needs_ocr check failed (%s) — proceeding without OCR", exc)
        return pdf_path

    if not needs:
        return pdf_path

    languages = (
        await get_config_value("DOC_OCR_LANGUAGES", settings.DOC_OCR_LANGUAGES)
        or "ara+eng"
    )
    ocr_path = str(Path(pdf_path).parent / "ocr.pdf")
    log.info("Scanned PDF detected — running OCR (languages=%s)", languages)
    try:
        await run_ocrmypdf(pdf_path, ocr_path, languages=languages)
        return ocr_path
    except Exception as exc:
        log.warning("OCR failed (%s) — falling back to original PDF", exc)
        return pdf_path


async def _extract_metadata(full_text: str, fallback_title: str, language: str) -> dict:
    """
    Ask a chat model to identify the book's real title + author from the
    first pages (cover page, title page, copyright page usually appear there).
    Best-effort: returns {} fields on any failure.
    """
    from api.services.ai_client import chat_complete                # noqa: PLC0415
    from api.services.config.runtime import get_all_config          # noqa: PLC0415
    from api.config.settings import settings                        # noqa: PLC0415

    cfg   = await get_all_config()
    model = cfg.get("MODEL_CHUNK") or cfg.get("MODEL_HAIKU") or settings.MODEL_HAIKU

    sample = full_text[:8000]
    prompt = (
        "Below is the beginning of a book extracted from a PDF. It may include the "
        "cover page, title page, and copyright page — but it may also contain "
        "watermarks or stamps from PDF download websites (e.g. 'Noor-Book.Com', "
        "'pdf-book.net', 'كتب pdf', etc.).\n\n"
        "Your task: identify the book's REAL title and the real human author name. "
        "Also give a one-sentence description and a category.\n\n"
        "Return ONLY valid JSON, no markdown:\n"
        '{"title": "...", "author": "...", "description": "...", "category": "..."}\n\n'
        "Rules:\n"
        "- IGNORE any website names, URLs, or download-site watermarks — these are "
        "NOT part of the title or author.\n"
        "- title must be the actual book title (keep original language — Arabic stays Arabic).\n"
        "- author must be the human author's full name as printed in the book "
        "(Arabic authors stay in Arabic script).\n"
        '- If the author truly cannot be found anywhere in the text, use "" — never invent one.\n'
        f"- If no clear title is found, use: {fallback_title!r}\n\n"
        f"=== BOOK TEXT START ===\n{sample}"
    )

    try:
        raw = await chat_complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
        )
    except Exception as exc:
        log.warning("metadata extraction call failed: %s", exc)
        return {}

    raw = (raw or "").strip()
    if raw.startswith("```"):
        lines = raw.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        data = json_module.loads(raw)
    except json_module.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end <= start:
            return {}
        try:
            data = json_module.loads(raw[start:end + 1])
        except json_module.JSONDecodeError:
            return {}

    return {
        "title":       str(data.get("title") or "").strip(),
        "author":      str(data.get("author") or "").strip(),
        "description": str(data.get("description") or "").strip(),
        "category":    str(data.get("category") or "").strip(),
    }


# ── Text helpers ──────────────────────────────────────────────────────────────

def _sanitize(text: str) -> str:
    """Strip NUL + C0 control chars Postgres rejects (keeps \\t \\n \\r)."""
    if not text:
        return ""
    return "".join(
        ch for ch in text
        if ch in "\t\n\r" or ord(ch) >= 0x20
    )


_HEADING_RE = re.compile(
    r"(?m)^(?:Chapter|CHAPTER|Chap\.?)\s+(\d+|[IVXLC]+)[^\n]*$|"
    r"^(\d+)\.\s+[A-Z][^\n]{3,}$|"
    r"^(?:الفصل|الباب|الجزء)\s+[^\n]{1,40}$"          # Arabic chapter headings
)


def _split_chapters(full_text: str) -> list[Chapter]:
    """
    Split extracted text into chapters.
    1. Chapter-heading patterns (English + Arabic).
    2. Fallback: ~3000-word parts.
    """
    splits = list(_HEADING_RE.finditer(full_text))

    if len(splits) >= 2:
        chapters: list[Chapter] = []
        for i, m in enumerate(splits):
            start   = m.start()
            end     = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
            heading = m.group(0).strip()
            body    = full_text[start + len(heading): end].strip()
            if len(body.split()) > 30:
                chapters.append(Chapter(index=len(chapters) + 1, title=heading, text=body))
        if chapters:
            return chapters

    # Fallback: fixed-size word chunks
    words = full_text.split()
    part_words = 3000
    chapters = []
    for i in range(0, len(words), part_words):
        body = " ".join(words[i: i + part_words]).strip()
        if len(body.split()) > 30:
            idx = len(chapters) + 1
            chapters.append(Chapter(index=idx, title=f"Part {idx}", text=body))
    return chapters


class YouTubeReq(BaseModel):
    url: str
    language: str = "en"
    steps: str = ""


def _json3_events_to_text(data: dict) -> str:
    """Flatten a YouTube json3 caption payload into plain text, one line per event."""
    lines = []
    for event in data.get("events", []):
        text = "".join(seg.get("utf8", "") for seg in (event.get("segs") or [])).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def _plain_subtitle_to_text(raw: str) -> str:
    """Fallback parser for vtt/srv/ttml caption formats — strips markup, timestamps,
    and sequence numbers, keeping only the spoken text lines."""
    text = re.sub(r"<[^>]+>", "", raw)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.upper().startswith("WEBVTT"):
            continue
        if re.match(r"^\d+$", line):          # srv sequence number
            continue
        if "-->" in line or re.match(r"^\d{2}:\d{2}", line):  # timestamp line
            continue
        lines.append(line)
    return "\n".join(lines)


def _extract_youtube_transcript_sync(video_id: str, languages: list[str]) -> tuple[str, str, str]:
    """
    Fetch a YouTube video's captions via yt-dlp (replaces youtube-transcript-api,
    which was getting IP-blocked by YouTube on most requests from this server —
    see api/requirements.txt for details). yt-dlp emulates several official
    YouTube client types internally and is far more resilient to that blocking.

    Runs synchronously — yt-dlp's Python API has no async form — so callers
    must run this in a thread executor. Returns (transcript_text, title, uploader).
    """
    import yt_dlp
    from api.config.settings import settings  # noqa: PLC0415

    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": languages,
        "quiet": True,
        "no_warnings": True,
        "logger": log,  # route yt-dlp's own messages through our logger instead of stderr
        # We only need captions, never a video/audio stream (skip_download=True
        # above), but yt-dlp still resolves a download format as part of
        # building info_dict and raises a hard "Requested format is not
        # available" error if the video has none (e.g. members-only, restricted,
        # or otherwise download-blocked even though captions are readable).
        # This makes that failure non-fatal so caption extraction still proceeds.
        "ignore_no_formats_error": True,
    }
    cookies_file = settings.YOUTUBE_COOKIES_FILE
    if cookies_file and os.path.exists(cookies_file):
        # Authenticated requests are much less likely to hit YouTube's
        # "Sign in to confirm you're not a bot" challenge than anonymous
        # requests from a datacenter IP.
        ydl_opts["cookiefile"] = cookies_file
    elif cookies_file:
        log.warning("YOUTUBE_COOKIES_FILE=%s is set but the file does not exist — "
                    "fetching without cookies", cookies_file)

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    title = info.get("title") or f"YouTube Video {video_id}"
    uploader = info.get("uploader") or "YouTube"

    manual = info.get("subtitles") or {}
    auto = info.get("automatic_captions") or {}

    # Manual (human-uploaded) captions are higher quality than auto-generated
    # ones, so prefer them for each language before falling back to auto.
    track = None
    for lang in languages:
        track = manual.get(lang) or auto.get(lang)
        if track:
            break
    if not track:
        # Nothing in the requested languages — fall back to any manual track
        # rather than failing outright.
        for entries in manual.values():
            track = entries
            break
    if not track:
        raise RuntimeError(f"No captions available for video {video_id} in {languages}")

    entry = next((e for e in track if e.get("ext") == "json3"), track[0])
    resp = httpx.get(entry["url"], headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    resp.raise_for_status()

    if entry.get("ext") == "json3":
        full_text = _json3_events_to_text(resp.json())
    else:
        full_text = _plain_subtitle_to_text(resp.text)

    if not full_text.strip():
        raise RuntimeError(f"Captions for video {video_id} were empty after parsing")

    return full_text, title, uploader


@router.post("/youtube", status_code=202)
async def youtube_upload(
    req: YouTubeReq,
    background_tasks: BackgroundTasks,
    user: AuthUser | None = Depends(get_current_user),
):
    # Extract video ID
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", req.url)
    if not match:
        raise HTTPException(400, "Invalid YouTube URL")
    video_id = match.group(1)

    try:
        loop = asyncio.get_running_loop()
        fn = functools.partial(_extract_youtube_transcript_sync, video_id, ["en", "ar"])
        full_text, video_title, uploader = await loop.run_in_executor(None, fn)
    except Exception as e:
        log.exception("Failed to fetch YouTube transcript")
        raise HTTPException(400, f"Could not extract transcript: {e}")

    book_id = f"yt_{video_id}_{uuid.uuid4().hex[:4]}"
    steps_list = [s.strip() for s in req.steps.split(",") if s.strip()]

    placeholder_req = PipelineReq(
        book_id=book_id,
        title=video_title,
        language=req.language,
        steps=steps_list,
        options=PipelineOptions(length="10min", style="narrative"),
        source="youtube",
    )

    job_id = await create_job(book_id, placeholder_req.model_dump(), user_id=user.id if user else None)

    background_tasks.add_task(
        _run_youtube_pipeline,
        job_id, book_id, video_title, uploader, full_text, req.language, steps_list
    )

    return {
        "job_id": job_id,
        "book_id": book_id,
        "status": "queued",
        "status_url": f"/api/pipeline/status/{job_id}",
    }

async def _run_youtube_pipeline(
    job_id: str,
    book_id: str,
    title: str,
    author: str,
    full_text: str,
    language: str,
    steps_list: list[str],
) -> None:
    from api.services.usage_logger import set_job_context
    set_job_context(job_id)

    try:
        chapters = _split_chapters(full_text)
        if not chapters:
            chapters = [Chapter(index=1, title="Transcript", text=full_text)]

        req = PipelineReq(
            book_id=book_id,
            title=title,
            author=author,
            language=language,
            chapters=chapters,
            steps=steps_list,
            options=PipelineOptions(length="10min", style="narrative"),
            source="youtube",
        )
        
        await db_update("pipeline_jobs", {"id": job_id}, {"input": req.model_dump()})
        await set_running(job_id)
        
        result = await run_pipeline(req, job_id=job_id)
        
        if result["status"] == "done":
            await set_done(job_id, result)
        elif result["status"] == "partial":
            await set_partial(job_id, result)
        else:
            await set_failed(job_id, str(result.get("errors", "unknown error")))
    except Exception as e:
        log.exception("youtube job %s failed", job_id)
        await set_failed(job_id, str(e))
