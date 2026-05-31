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
  Extracts text, splits into chapters, runs the same pipeline as /api/pipeline/run.
"""
import io
import os
import re
import tempfile
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File, Form

from api.jobs.store import create_job, set_running, set_done, set_failed, set_partial
from api.models.requests import PipelineReq, PipelineOptions, Chapter
from api.services.pipeline.orchestrator import run_pipeline

router = APIRouter(prefix="/document", tags=["document"])

MAX_FILE_SIZE = 50 * 1024 * 1024   # 50 MB


@router.post("/upload", status_code=202)
async def document_upload(
    background_tasks: BackgroundTasks,
    file:     UploadFile  = File(...),
    language: str         = Form("en"),
    steps:    str         = Form(""),           # "summarize,audio_full" or ""
    length:   str         = Form("10min"),
    style:    str         = Form("narrative"),
):
    """Upload a PDF and start a pipeline job on its content."""

    # ── Validate ──────────────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are accepted")

    raw = await file.read()
    if len(raw) > MAX_FILE_SIZE:
        raise HTTPException(413, f"File too large — max {MAX_FILE_SIZE // 1024 // 1024} MB")

    # ── Extract text ──────────────────────────────────────────────────────────
    try:
        chapters = _extract_chapters(raw, file.filename)
    except Exception as e:
        raise HTTPException(422, f"Could not extract text from PDF: {e}")

    if not chapters:
        raise HTTPException(422, "No text content found in PDF")

    # ── Build PipelineReq ────────────────────────────────────────────────────
    book_id = f"pdf_{uuid.uuid4().hex[:8]}"
    title   = os.path.splitext(file.filename)[0].replace("_", " ").title()
    steps_list = [s.strip() for s in steps.split(",") if s.strip()]

    req = PipelineReq(
        book_id  = book_id,
        title    = title,
        language = language,
        chapters = chapters,
        steps    = steps_list,
        options  = PipelineOptions(length=length, style=style),
        source   = "pdf_upload",
    )

    # ── Queue job ─────────────────────────────────────────────────────────────
    job_id = await create_job(book_id, req.model_dump())

    background_tasks.add_task(_run_job, job_id, req)

    return {
        "job_id":      job_id,
        "book_id":     book_id,
        "status":      "queued",
        "chapters":    len(chapters),
        "status_url":  f"/api/pipeline/status/{job_id}",
    }


# ── Background runner ─────────────────────────────────────────────────────────

async def _run_job(job_id: str, req: PipelineReq) -> None:
    # Tag every usage log written during this job with the job_id.
    from api.services.usage_logger import set_job_context  # noqa: PLC0415
    set_job_context(job_id)

    await set_running(job_id)
    try:
        result = await run_pipeline(req)
        if result["status"] == "done":
            await set_done(job_id, result)
        elif result["status"] == "partial":
            await set_partial(job_id, result)
        else:
            await set_failed(job_id, str(result.get("errors", "unknown error")))
    except Exception as e:
        await set_failed(job_id, str(e))


# ── PDF text extraction ───────────────────────────────────────────────────────

def _extract_chapters(raw: bytes, filename: str) -> list[Chapter]:
    """
    Extract chapters from a PDF.
    Strategy:
    1. Look for heading-like patterns (Chapter / CHAPTER / numbered headings).
    2. If none found, group pages into ~10-page chunks.
    """
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw))
    pages  = []
    for page in reader.pages:
        txt = page.extract_text() or ""
        pages.append(txt.strip())

    # Merge all text, note page boundaries
    full_text = "\n".join(pages)

    # Detect chapter headings
    heading_re = re.compile(
        r"(?m)^(?:Chapter|CHAPTER|Chap\.?)\s+(\d+|[IVXLC]+)[^\n]*$|"
        r"^(\d+)\.\s+[A-Z][^\n]{3,}$"
    )
    splits = list(heading_re.finditer(full_text))

    if splits:
        chapters: list[Chapter] = []
        for i, m in enumerate(splits):
            start = m.start()
            end   = splits[i + 1].start() if i + 1 < len(splits) else len(full_text)
            heading = m.group(0).strip()
            body    = full_text[start + len(heading): end].strip()
            if len(body.split()) > 30:   # skip near-empty chapters
                chapters.append(Chapter(index=i + 1, title=heading, text=body))
        if chapters:
            return chapters

    # Fallback: 10-page chunks
    chunk_size = 10
    chapters = []
    for i in range(0, len(pages), chunk_size):
        chunk_pages = pages[i: i + chunk_size]
        body = "\n".join(chunk_pages).strip()
        if len(body.split()) > 30:
            idx = len(chapters) + 1
            chapters.append(Chapter(
                index = idx,
                title = f"Part {idx}",
                text  = body,
            ))
    return chapters
