"""
Pipeline orchestrator — the core engine.

Execution phases
─────────────────
  Phase 1  summarize   — 3-pass text summary (must finish first; everything else depends on it)
  Phase 2  (parallel)  — all independent steps run concurrently once summarize is done:
               cover · audio_full · audio_chapters · mindmap · mindmap_chapters
  Phase 3  (parallel)  — steps that depend on Phase-2 outputs:
               alt_text (needs cover) · video (needs audio_full + cover + mindmap)
  Phase 4  inject_epub — runs LAST after all other steps complete (needs ALL assets)

Live config is read from Supabase provider_config via runtime.py at job start.
No restart needed when the admin switches providers.
"""
import asyncio
import json as json_module
import logging
import os
import tempfile
import time
from datetime import datetime, timezone

log = logging.getLogger(__name__)

from api.config.settings import settings
from api.models.requests import PipelineReq, VALID_STEPS
from api.services.config.runtime import get_all_config
from api.services.db import find, upsert, update
from api.services.usage_logger import set_step
from api.services.summarizer.haiku import run_haiku_pass
from api.services.summarizer.sonnet import run_sonnet_pass_sync
from api.services.summarizer.review import run_review_pass
from api.services.pipeline.tts import synthesize
from api.services.pipeline.cover import generate_cover
from api.services.pipeline.mindmap import generate_mermaid_code, render_mermaid_svg, generate_json_mindmap
from api.services.pipeline.alttext import generate_alt_text
from api.services.pipeline.audio import process_audio
from api.services.pipeline.epub import (
    fetch_epub, inject_summary_into_epub,
    EpubError, EpubNotAvailableError,
)
from api.services.pipeline.video import generate_book_video, VideoError
from api.services.pipeline.storage import upload_file, CONTENT_TYPES
from api.jobs.store import is_cancelled
from api.services.db import insert as db_insert

# Single source of truth lives in api/models/requests.py
ALL_STEPS = VALID_STEPS


class JobCancelledError(Exception):
    pass


def _resolve_steps(requested: list[str] | None, previous_result: dict | None = None) -> set[str]:
    """Return the full set of steps to run, including auto-added dependencies."""
    # If requested is None or empty list, run all steps
    # If requested has specific steps, run only those (plus dependencies)
    # Note: empty list [] means "run all", explicit steps like ["cover"] means "run only cover"
    if requested is None or len(requested) == 0:
        return set(ALL_STEPS)
    steps = set(requested)
    
    # Check if we have summary data from previous result
    has_summary = False
    if previous_result:
        _prev = previous_result if isinstance(previous_result, dict) else {}
        _psums = _prev.get("summaries") or {}
        if _psums:
            _first_sum = next(iter(_psums.values()), {})
            has_summary = bool(_first_sum.get("text"))
    
    # Enforce dependencies (skip if data already available from previous result)
    if "audio_full" in steps or "audio_chapters" in steps:
        steps.add("summarize")
    if "mindmap" in steps or "mindmap_chapters" in steps:
        steps.add("summarize")
    if "alt_text" in steps:
        steps.add("cover")
    if "inject_epub" in steps and not has_summary:
        # Only require summarize if we don't have summary data from previous result
        steps.add("summarize")
    if "video" in steps:
        steps.add("summarize")
        steps.add("audio_full")
    return steps


def _fmt_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s" if m else f"{s}s"


def _fmt_audio_duration(seconds: int | None) -> str | None:
    if seconds is None:
        return None
    m = seconds // 60
    s = seconds % 60
    return f"{m}:{s:02d}"


async def _compute_job_cost(job_id: str | None) -> dict:
    """Sum cost from usage_logs for this job. Returns zeros if unreachable."""
    empty = {"total_usd": 0.0, "calls": 0, "by_step": {}, "by_provider": {}}
    if not job_id:
        return empty
    try:
        rows = await find("usage_logs", filters={"job_id": job_id}, limit=20000)
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).debug("cost rollup failed for job %s: %s", job_id, exc)
        return empty

    total = 0.0
    by_step: dict[str, float] = {}
    by_provider: dict[str, float] = {}
    for r in rows:
        c = float(r.get("cost_usd") or 0)
        total += c
        step = r.get("step") or "unknown"
        prov = r.get("provider") or "unknown"
        by_step[step]     = round(by_step.get(step, 0.0) + c, 6)
        by_provider[prov] = round(by_provider.get(prov, 0.0) + c, 6)

    return {
        "total_usd":   round(total, 6),
        "calls":       len(rows),
        "by_step":     by_step,
        "by_provider": by_provider,
    }


# ── Per-step DB persistence helpers ──────────────────────────────────────────
# All helpers are best-effort: they log on failure but never raise so a DB
# hiccup never breaks the pipeline result.

async def _persist_step_result(
    job_id: str | None,
    step: str,
    status: str,
    output_url: str | None = None,
    error_msg: str | None = None,
    duration_sec: int | None = None,
) -> None:
    """Insert a row into pipeline_step_results."""
    if not job_id:
        return
    try:
        await db_insert("pipeline_step_results", {
            "job_id":       job_id,
            "step":         step,
            "status":       status,
            "output_url":   output_url,
            "error_msg":    error_msg,
            "duration_sec": duration_sec,
        })
    except Exception as exc:
        log.debug("pipeline_step_results insert failed (%s/%s): %s", step, status, exc)


async def _persist_book_summary(
    book_id: str,
    language: str,
    length: str,
    style: str,
    full_summary: str,
    model_sonnet: str,
) -> None:
    """Save the full summary to books table columns and book_summaries cache."""
    if not full_summary:
        return

    # ── books table ───────────────────────────────────────────────────────────
    if book_id.isdigit():
        try:
            # Choose the right column based on language + length
            if language == "ar":
                col = "arabic_summary_v2"
            elif length == "10min":
                col = "summary_en_10min"
            else:
                col = "summary_english"
            await update(
                "books",
                filters={"book_id": int(book_id)},
                data={col: full_summary, "status": "summarized"},
            )
        except Exception as exc:
            log.debug("books summary update failed for %s: %s", book_id, exc)

    # ── book_summaries cache ──────────────────────────────────────────────────
    try:
        await db_insert("book_summaries", {
            "book_id":    book_id,
            "length":     length,
            "style":      style,
            "language":   language,
            "summary":    full_summary,
            "word_count": len(full_summary.split()),
            "model":      model_sonnet,
        })
    except Exception as exc:
        log.debug("book_summaries insert failed for %s: %s", book_id, exc)


async def _persist_chunk_summaries(
    book_id: str,
    chapter_results: list[dict],
) -> None:
    """Write each chapter's summary back to chunks.summary by chunk_index."""
    if not book_id.isdigit() or not chapter_results:
        return
    for ch in chapter_results:
        if not ch.get("summary"):
            continue
        try:
            await update(
                "chunks",
                filters={"book_id": int(book_id), "chunk_index": ch["index"]},
                data={"summary": ch["summary"]},
            )
        except Exception as exc:
            log.debug("chunks.summary update failed for chunk %s: %s", ch["index"], exc)


async def _persist_cover(
    book_id: str,
    title: str,
    cover_url: str,
) -> None:
    """Insert into covers table and set books.cover_status."""
    try:
        await db_insert("covers", {
            "bookid":   int(book_id) if book_id.isdigit() else None,
            "title":    title,
            "coverurl": cover_url,
        })
    except Exception as exc:
        log.debug("covers insert failed for %s: %s", book_id, exc)

    if book_id.isdigit():
        try:
            await update(
                "books",
                filters={"book_id": int(book_id)},
                data={"cover_status": "done"},
            )
        except Exception as exc:
            log.debug("books.cover_status update failed for %s: %s", book_id, exc)


async def _persist_audio(
    book_id: str,
    language: str,
    audio_url: str,
) -> None:
    """Upsert into the audio table."""
    if not book_id.isdigit():
        return
    try:
        url_col    = "ar_url"    if language == "ar" else "en_url"
        status_col = "ar_status" if language == "ar" else "en_status"
        await upsert(
            "audio",
            {
                "book_id":   int(book_id),
                url_col:     audio_url,
                status_col:  "done",
            },
            conflict="book_id",
        )
    except Exception as exc:
        log.debug("audio upsert failed for %s: %s", book_id, exc)


async def _ensure_book_row(req: "PipelineReq") -> None:
    if req.book_id and req.book_id.isdigit():
        return
    payload: dict = {
        "book_id": req.book_id,
        "title":   req.title or req.book_id,
        "author":  req.author or "",
    }
    try:
        await upsert("books", payload, conflict="book_id")
    except Exception as e:
        import logging
        logging.getLogger(__name__).debug(
            "Could not upsert books row for %s: %s", req.book_id, e,
        )


async def _fetch_catalog_chapters(book_id: str) -> list[dict]:
    try:
        int(book_id)
    except (TypeError, ValueError):
        return []
    try:
        rows = await find(
            "chunks",
            filters={"book_id": book_id},
            select="chunk_index, content",
            order="chunk_index ASC",
        )
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning(
            "Could not fetch chunks for book_id=%s: %s", book_id, exc,
        )
        return []
    return [
        {"index": r["chunk_index"], "title": f"Chapter {r['chunk_index']}", "text": r["content"]}
        for r in rows
    ]


async def _fetch_book_from_catalog(book_id: str) -> dict | None:
    try:
        bid = int(book_id)
    except (TypeError, ValueError):
        return None
    try:
        rows = await find(
            "books",
            filters={"book_id": bid},
            select=(
                "book_id, title, author, pages, category, description, "
                "summary_english, summary_en_10min, "
                "arabic_summary, arabic_summary_v2"
            ),
            limit=1,
        )
    except Exception as exc:
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "catalog lookup for book_id=%s failed: %s", book_id, exc,
        )
        return None
    return rows[0] if rows else None


async def _fetch_gutenberg_metadata(book_id: str) -> dict:
    import logging as _log
    import httpx
    result = {"title": "", "author": ""}
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
            if "," in name:
                parts = name.split(",", 1)
                result["author"] = f"{parts[1].strip()} {parts[0].strip()}"
            else:
                result["author"] = name
    except Exception as exc:
        _log.getLogger(__name__).debug(
            "Gutendex metadata fetch for book_id=%s failed: %s", book_id, exc,
        )
    return result


def _pick_cached_summary(book_row: dict, language: str) -> str | None:
    _NULL_SENTINELS = {"", "null", "none", "nil", "n/a", "na", "undefined"}

    def _nonempty(value) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if not stripped or stripped.lower() in _NULL_SENTINELS:
            return None
        return stripped

    if (language or "en").lower() == "ar":
        return (
            _nonempty(book_row.get("arabic_summary_v2"))
            or _nonempty(book_row.get("arabic_summary"))
        )
    return (
        _nonempty(book_row.get("summary_en_10min"))
        or _nonempty(book_row.get("summary_english"))
    )


async def run_pipeline(
    req: PipelineReq,
    *,
    job_id: str | None = None,
    previous_result: dict | str | None = None,
) -> dict:
    """
    Execute the pipeline and return the full result dict.

    Phase 1  — summarize (sequential)
    Phase 2  — cover · audio_full · audio_chapters · mindmap · mindmap_chapters  (parallel)
    Phase 3  — alt_text · video  (parallel, depend on Phase-2 outputs)
    Phase 4  — inject_epub (runs LAST, after all other steps complete)
    """

    started = time.time()
    cfg     = await get_all_config()
    
    # Parse previous_result for dependency resolution
    _prev_parsed = None
    if previous_result:
        _prev_parsed = previous_result if isinstance(previous_result, dict) else {}
        if isinstance(previous_result, str):
            try:
                _prev_parsed = json_module.loads(previous_result)
            except Exception:
                _prev_parsed = {}
    
    steps   = _resolve_steps(req.steps, _prev_parsed)
    log.info("Job %s: req.steps=%s, resolved_steps=%s", job_id, req.steps, steps)
    errors: dict[str, str]   = {}
    step_status: dict[str, str] = {s: "skipped" for s in ALL_STEPS}

    # ── Shared result variables (updated by each step, read by checkpoint) ───
    chapter_results: list[dict] = []
    full_summary:    str        = ""
    quick_summary:   str        = ""
    full_audio:      dict | None = None
    full_audio_path: str | None = None
    chapter_audio:   dict[int, str] = {}
    cover_url:       str | None = None
    alt_text:        str | None = None
    mindmap_url:     str | None = None
    mindmap_data:    dict | None = None
    mindmap_path_saved: str | None = None
    chapter_mindmap: dict[int, dict] = {}
    epub_url:        str | None = None
    video_url:       str | None = None
    video_meta:      dict | None = None

    # ── Pre-load previous result so live checkpoints always carry forward
    # assets from steps that aren't being re-run in this pass.
    if previous_result:
        if isinstance(previous_result, str):
            try:
                previous_result = json_module.loads(previous_result)
            except Exception:
                previous_result = {}
        _prev: dict = previous_result or {}
        _pfiles = _prev.get("files") or {}
        _pmeta  = _prev.get("metadata") or {}
        _paudio = _prev.get("audio") or {}
        _psums  = _prev.get("summaries") or {}

        cover_url   = cover_url   or _pfiles.get("cover")
        mindmap_url = mindmap_url or _pfiles.get("mindmap")
        epub_url    = epub_url    or _pfiles.get("epub")
        video_url   = video_url   or _pfiles.get("video")
        alt_text    = alt_text    or _pmeta.get("cover_alt_text")

        _lang_key = f"full_{req.language}"
        if _lang_key in _paudio and not full_audio:
            full_audio = _paudio[_lang_key]

        # Quick / full summary (needed if summarize is skipped this run)
        if not full_summary and _psums:
            _first_sum = next(iter(_psums.values()), {})
            full_summary  = _first_sum.get("text", "")
            quick_summary = _prev.get("quick_summary", "")
            log.info("Loaded summary from previous result: %s chars", len(full_summary) if full_summary else 0)

        # Chapter audio & mindmaps
        for _fc in _pfiles.get("chapters") or []:
            _idx = _fc.get("index")
            if _idx is None:
                continue
            if _fc.get("audio_url") and _idx not in chapter_audio:
                chapter_audio[_idx] = _fc["audio_url"]
            if _fc.get("mindmap_url") and _idx not in chapter_mindmap:
                chapter_mindmap[_idx] = {
                    "url":    _fc["mindmap_url"],
                    "data":   None,
                    "format": "mermaid",
                }

        # Chapter results (summaries)
        if not chapter_results and _prev.get("chapters"):
            chapter_results = [dict(ch) for ch in _prev["chapters"]]
            log.info("Loaded %d chapters from previous result", len(chapter_results))

    # ── Live checkpoint ───────────────────────────────────────────────────────
    async def _checkpoint() -> None:
        if not job_id:
            return
        if is_cancelled(job_id):
            raise JobCancelledError(f"Job {job_id} was cancelled")
        try:
            lang = req.language
            current_step = next(
                (s for s, st in step_status.items() if st == "running"), None
            )
            running_steps = [s for s, st in step_status.items() if st == "running"]
            partial = {
                "book_id":         req.book_id,
                "status":          "running",
                "current_step":    current_step,
                "running_steps":   running_steps,
                "generated_at":    datetime.now(timezone.utc).isoformat(),
                "processing_time": _fmt_duration(round(time.time() - started, 1)),
                "steps":           dict(step_status),
                "metadata": {
                    "title":          req.title,
                    "author":         req.author,
                    "year":           req.year,
                    "pages":          req.pages,
                    "grade_level":    req.grade_level,
                    "genres":         req.genres,
                    "cover_url":      cover_url,
                    "cover_alt_text": alt_text,
                },
                "quick_summary": quick_summary,
                "summaries": (
                    {
                        f"{req.options.length}_{lang}": {
                            "text":       full_summary,
                            "word_count": len(full_summary.split()) if full_summary else 0,
                            "style":      req.options.style,
                            "language":   lang,
                        }
                    } if full_summary else {}
                ),
                "audio":   ({f"full_{lang}": full_audio} if full_audio else {}),
                "mindmap": (
                    {"url": mindmap_url, "data": mindmap_data} if mindmap_data else
                    {"url": mindmap_url} if mindmap_url else
                    None
                ),
                "epub":  ({f"enriched_{lang}": {"url": epub_url}} if epub_url else None),
                "video": (
                    {
                        f"summary_{lang}": {
                            "url":              video_url,
                            "duration_seconds": (video_meta or {}).get("duration_seconds"),
                            "size_mb":          (video_meta or {}).get("size_mb"),
                            "width":            (video_meta or {}).get("width"),
                            "height":           (video_meta or {}).get("height"),
                            "provider":         (video_meta or {}).get("provider"),
                            "silent":           (video_meta or {}).get("silent", False),
                        },
                    } if video_url else None
                ),
                "chapters": chapter_results,
                "errors":   dict(errors),
            }
            await update(
                "pipeline_jobs",
                filters={"id": job_id},
                data={"result": partial, "status": "running"},
            )
        except JobCancelledError:
            raise
        except Exception as exc:
            import logging as _log
            _log.getLogger(__name__).warning("checkpoint write failed: %s", exc)

    # ── Production catalog enrichment ─────────────────────────────────────────
    if not (req.title and req.author and req.summary):
        book_row = await _fetch_book_from_catalog(req.book_id)
        if book_row:
            updates: dict = {}
            catalog_title = book_row.get("title", "")
            if not req.title and catalog_title and catalog_title != req.book_id:
                updates["title"] = catalog_title
            if not req.author and book_row.get("author"):
                updates["author"] = book_row["author"]
            if not req.summary:
                cached = _pick_cached_summary(book_row, req.language)
                if cached:
                    updates["summary"] = cached
            if not req.pages and book_row.get("pages"):
                updates["pages"] = book_row["pages"]
            if updates:
                req = req.model_copy(update=updates)

    if req.book_id and req.book_id.isdigit() and not req.title:
        gutenberg_meta = await _fetch_gutenberg_metadata(req.book_id)
        meta_updates: dict = {}
        if gutenberg_meta.get("title"):
            meta_updates["title"] = gutenberg_meta["title"]
        if not req.author and gutenberg_meta.get("author"):
            meta_updates["author"] = gutenberg_meta["author"]
        if meta_updates:
            req = req.model_copy(update=meta_updates)

    await _ensure_book_row(req)

    # ── Resolve chapters ──────────────────────────────────────────────────────
    if req.chapters:
        chapters = [{"index": c.index, "title": c.title, "text": c.text} for c in req.chapters]
    else:
        chapters = await _fetch_catalog_chapters(req.book_id)

    if not chapters and not (req.summary and req.summary.strip()):
        return {
            "book_id":         req.book_id,
            "status":          "failed",
            "generated_at":    datetime.now(timezone.utc).isoformat(),
            "processing_time": "0s",
            "steps":           step_status,
            "errors": {"input": (
                f"No input data found for book_id={req.book_id!r}. "
                "Supply either `chapters` or a pre-computed `summary` in the "
                "request, OR use a numeric book_id that exists in the catalog "
                "`chunks` table."
            )},
        }

    # Mark every planned step as "pending" + write immediate checkpoint
    for s in steps:
        step_status[s] = "pending"
    await _checkpoint()

    # ── Resolved models from live config ─────────────────────────────────────
    model_haiku      = cfg.get("MODEL_HAIKU",      settings.MODEL_HAIKU)
    model_chunk      = cfg.get("MODEL_CHUNK") or cfg.get("MODEL_HAIKU", settings.MODEL_HAIKU)
    model_sonnet     = cfg.get("MODEL_SONNET",     settings.MODEL_SONNET)
    model_mindmap    = cfg.get("MODEL_MINDMAP",    settings.MODEL_MINDMAP)
    mindmap_format   = cfg.get("MINDMAP_FORMAT",   settings.MINDMAP_FORMAT)
    tts_enabled      = cfg.get("PIPELINE_STEP_TTS",        "true") == "true"
    cover_enabled    = cfg.get("PIPELINE_STEP_COVER",       "true") == "true"
    alttext_enabled  = cfg.get("PIPELINE_STEP_ALTTEXT",     "true") == "true"
    mindmap_enabled  = cfg.get("PIPELINE_STEP_MINDMAP",     "true") == "true"
    epub_enabled     = cfg.get("PIPELINE_STEP_INJECT_EPUB", "true") == "true"
    video_enabled    = cfg.get("PIPELINE_STEP_VIDEO",       "true") == "true"
    audio_proc_enabled = cfg.get("PIPELINE_STEP_AUDIO_PROCESSING", "true") == "true"
    base_url         = cfg.get("BOOK_FILES_BASE_URL") or settings.BOOK_FILES_BASE_URL
    video_provider   = cfg.get("VIDEO_PROVIDER") or settings.VIDEO_PROVIDER

    with tempfile.TemporaryDirectory() as tmp:
        cover_path_saved = os.path.join(tmp, "cover.jpg")

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 1 — SUMMARIZE
        # Everything downstream needs the full_summary + chapter_results.
        # ═════════════════════════════════════════════════════════════════════
        if "summarize" in steps:
            step_status["summarize"] = "running"
            set_step("summarize")
            await _checkpoint()
            try:
                if req.summary:
                    full_summary = req.summary
                    sentences = [s.strip() for s in full_summary.replace(".\n", ". ").split(". ") if s.strip()]
                    quick_summary = ". ".join(sentences[:2]) + "." if sentences else full_summary[:200]
                    # Load per-chapter summaries from DB so audio_chapters/mindmap_chapters can run
                    if chapters and req.book_id and req.book_id.isdigit():
                        try:
                            chunk_rows = await find(
                                "chunks",
                                filters={"book_id": req.book_id},
                                select="chunk_index, summary",
                                order="chunk_index ASC",
                            )
                            chunk_sum_map = {r["chunk_index"]: r.get("summary") or "" for r in chunk_rows}
                            chapter_results = [
                                {
                                    "index":         ch["index"],
                                    "title":         ch["title"],
                                    "summary":       chunk_sum_map.get(ch["index"], ""),
                                    "read_time_min": max(1, len((chunk_sum_map.get(ch["index"]) or "").split()) // 200),
                                }
                                for ch in chapters
                                if chunk_sum_map.get(ch["index"])
                            ]
                            log.info("Loaded %d chapter summaries from chunks for cached-summary job", len(chapter_results))
                        except Exception as exc:
                            log.debug("Could not load chunk summaries for cached summary: %s", exc)
                            chapter_results = []
                    else:
                        chapter_results = []
                    step_status["summarize"] = "done"
                else:
                    haiku_conc = max(1, int(cfg.get("HAIKU_CONCURRENCY", "6")))
                    sem = asyncio.Semaphore(haiku_conc)

                    async def _summarize_chunk(ch: dict) -> dict:
                        async with sem:
                            chunk = {
                                "id":          f"{req.book_id}_ch{ch['index']}",
                                "chunk_index": ch["index"],
                                "content":     ch["text"],
                            }
                            try:
                                sums = await run_haiku_pass(
                                    req.book_id, [chunk], req.language, model=model_chunk,
                                )
                            except Exception as exc:
                                import logging as _log
                                _log.getLogger(__name__).warning(
                                    "Haiku pass failed for chunk %s: %s", ch["index"], exc,
                                )
                                sums = []
                            summary = sums[0] if sums else ""
                            return {
                                "index":         ch["index"],
                                "title":         ch["title"],
                                "summary":       summary,
                                "read_time_min": max(1, len(summary.split()) // 200) if summary else 1,
                            }

                    chapter_results = list(await asyncio.gather(
                        *[_summarize_chunk(ch) for ch in chapters]
                    ))
                    chapter_results.sort(key=lambda c: c["index"])

                    chunk_summaries = [c["summary"] for c in chapter_results if c.get("summary")]

                    if chunk_summaries:
                        full_summary = await run_sonnet_pass_sync(
                            chunk_summaries,
                            req.options.length,
                            req.options.style,
                            req.language,
                            model_override=model_sonnet,
                        )
                        full_summary = await run_review_pass(
                            full_summary,
                            req.options.length,
                            req.options.style,
                            req.language,
                            model=model_haiku,
                        )
                        sentences = [s.strip() for s in full_summary.replace(".\n", ". ").split(". ") if s.strip()]
                        quick_summary = ". ".join(sentences[:2]) + "." if sentences else full_summary[:200]

                    step_status["summarize"] = "done"
                    # Save chunk summaries back to chunks.summary
                    await _persist_chunk_summaries(req.book_id, chapter_results)

            except JobCancelledError:
                raise
            except Exception as e:
                errors["summarize"] = str(e)
                step_status["summarize"] = "failed"

            _sum_status = step_status["summarize"]
            _sum_t = round(time.time() - started)
            await _persist_step_result(job_id, "summarize", _sum_status, duration_sec=_sum_t)
            if _sum_status == "done":
                await _persist_book_summary(
                    req.book_id, req.language, req.options.length, req.options.style,
                    full_summary, model_sonnet,
                )
            await _checkpoint()

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 2 — PARALLEL: cover · audio_full · audio_chapters ·
        #                      mindmap · mindmap_chapters
        # All of these are independent of each other (only need summarize done).
        # ═════════════════════════════════════════════════════════════════════

        # ── cover ─────────────────────────────────────────────────────────────
        async def _do_cover() -> None:
            nonlocal cover_url
            if "cover" not in steps or not cover_enabled:
                return
            step_status["cover"] = "running"
            set_step("cover")
            await _checkpoint()
            try:
                await generate_cover(
                    req.title or req.book_id,
                    req.author or "",
                    cover_path_saved,
                    cfg,
                    summary  = full_summary or quick_summary or None,
                    genres   = req.genres,
                    year     = req.year,
                    language = req.language,
                )
                key = f"books/{req.book_id}/cover.jpg"
                cover_url = upload_file(cover_path_saved, key, CONTENT_TYPES[".jpg"])
                step_status["cover"] = "done"
            except JobCancelledError:
                raise
            except Exception as e:
                errors["cover"] = str(e)
                step_status["cover"] = "failed"
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "cover", step_status["cover"], output_url=cover_url, duration_sec=_t)
            if step_status["cover"] == "done":
                await _persist_cover(req.book_id, req.title or req.book_id, cover_url)
            await _checkpoint()

        # ── audio_full ────────────────────────────────────────────────────────
        async def _do_audio_full() -> None:
            nonlocal full_audio, full_audio_path
            if "audio_full" not in steps or not tts_enabled or not full_summary:
                return
            step_status["audio_full"] = "running"
            set_step("audio_full")
            await _checkpoint()
            try:
                raw  = os.path.join(tmp, "audio_raw.mp3")
                proc = os.path.join(tmp, "audio.mp3")
                await synthesize(full_summary, req.language, raw, cfg)
                if audio_proc_enabled:
                    loop = asyncio.get_event_loop()
                    meta = await loop.run_in_executor(
                        None, process_audio, raw, proc,
                        req.title or req.book_id, req.author or "",
                    )
                    src = proc
                else:
                    meta = {}
                    src = raw
                key = f"books/{req.book_id}/audio_{req.language}_{req.options.length}.mp3"
                url = upload_file(src, key, CONTENT_TYPES[".mp3"])
                full_audio = {
                    "url":      url,
                    "duration": _fmt_audio_duration(meta.get("duration_seconds")),
                    "size_mb":  meta.get("size_mb"),
                }
                full_audio_path = src
                step_status["audio_full"] = "done"
            except JobCancelledError:
                raise
            except Exception as e:
                errors["audio_full"] = str(e)
                step_status["audio_full"] = "failed"
            _audio_url = (full_audio or {}).get("url")
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "audio_full", step_status["audio_full"], output_url=_audio_url, duration_sec=_t)
            if step_status["audio_full"] == "done" and _audio_url:
                await _persist_audio(req.book_id, req.language, _audio_url)
            await _checkpoint()

        # ── audio_chapters ────────────────────────────────────────────────────
        async def _do_audio_chapters() -> None:
            nonlocal chapter_audio, chapter_results
            if "audio_chapters" not in steps or not tts_enabled or not chapter_results:
                return
            step_status["audio_chapters"] = "running"
            set_step("audio_chapters")
            await _checkpoint()
            ch_errors = 0
            ch_processed = 0
            ch_skipped_no_summary = 0
            
            # Create a map to update chapter_results by index
            audio_key = f"audio_{req.language}"
            
            for ch in chapter_results:
                if not ch.get("summary"):
                    ch_skipped_no_summary += 1
                    log.warning("Chapter %d has no summary, skipping audio generation", ch["index"])
                    continue
                try:
                    ch_raw  = os.path.join(tmp, f"ch{ch['index']}_raw.mp3")
                    ch_proc = os.path.join(tmp, f"ch{ch['index']}.mp3")
                    await synthesize(ch["summary"], req.language, ch_raw, cfg)
                    if audio_proc_enabled:
                        loop = asyncio.get_event_loop()
                        await loop.run_in_executor(
                            None, process_audio, ch_raw, ch_proc,
                            ch["title"], req.author or "",
                        )
                        src = ch_proc
                    else:
                        src = ch_raw
                    idx = ch["index"]
                    k   = f"books/{req.book_id}/chapters/ch_{idx:02d}_{req.language}.mp3"
                    audio_url = upload_file(src, k, CONTENT_TYPES[".mp3"])
                    chapter_audio[idx] = audio_url
                    
                    # Update chapter_results with audio URL for dashboard display
                    ch[audio_key] = audio_url
                    
                    ch_processed += 1
                    await _checkpoint()   # checkpoint per chapter so progress is visible
                except JobCancelledError:
                    raise
                except Exception as e:
                    ch_errors += 1
                    errors[f"audio_chapter_{ch['index']}"] = str(e)

            # Calculate status based on actual processing, not just chapter count
            total_attempted = ch_processed + ch_errors
            if ch_errors > 0 and ch_processed == 0:
                step_status["audio_chapters"] = "failed"
            elif ch_errors > 0:
                step_status["audio_chapters"] = "partial"
            elif ch_processed > 0:
                step_status["audio_chapters"] = "done"
            elif ch_skipped_no_summary > 0:
                # All chapters skipped due to missing summaries
                step_status["audio_chapters"] = "failed"
                errors["audio_chapters"] = f"All {ch_skipped_no_summary} chapters skipped: no summaries available"
            else:
                step_status["audio_chapters"] = "skipped"
            
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "audio_chapters", step_status["audio_chapters"], duration_sec=_t)
            await _checkpoint()

        # ── mindmap ───────────────────────────────────────────────────────────
        async def _do_mindmap() -> None:
            nonlocal mindmap_url, mindmap_data, mindmap_path_saved
            if "mindmap" not in steps or not mindmap_enabled or not full_summary:
                return
            step_status["mindmap"] = "running"
            set_step("mindmap")
            await _checkpoint()
            try:
                if mindmap_format == "json":
                    mindmap_data = await generate_json_mindmap(
                        req.title or req.book_id, full_summary, req.language, model=model_mindmap
                    )
                    json_path = os.path.join(tmp, "mindmap.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json_module.dump(mindmap_data, f, ensure_ascii=False)
                    key = f"books/{req.book_id}/mindmap.json"
                    mindmap_url = upload_file(json_path, key, CONTENT_TYPES[".json"])
                else:
                    mermaid = await generate_mermaid_code(
                        req.title or req.book_id, full_summary, req.language, model=model_mindmap
                    )
                    svg_path = os.path.join(tmp, "mindmap.svg")
                    await render_mermaid_svg(mermaid, svg_path)
                    key = f"books/{req.book_id}/mindmap.svg"
                    mindmap_url = upload_file(svg_path, key, CONTENT_TYPES[".svg"])
                    mindmap_path_saved = svg_path
                step_status["mindmap"] = "done"
            except JobCancelledError:
                raise
            except Exception as e:
                errors["mindmap"] = str(e)
                step_status["mindmap"] = "failed"
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "mindmap", step_status["mindmap"], output_url=mindmap_url, duration_sec=_t)
            await _checkpoint()

        # ── mindmap_chapters ──────────────────────────────────────────────────
        async def _do_mindmap_chapters() -> None:
            nonlocal chapter_mindmap, chapter_results
            if "mindmap_chapters" not in steps or not mindmap_enabled or not chapter_results:
                return
            step_status["mindmap_chapters"] = "running"
            set_step("mindmap_chapters")
            await _checkpoint()
            ch_errors = 0
            ch_skipped_no_summary = 0

            mm_conc = max(1, int(cfg.get("MINDMAP_CONCURRENCY", "4")))
            mm_sem  = asyncio.Semaphore(mm_conc)

            async def _chapter_mindmap(ch: dict) -> tuple[int, dict | None, str | None]:
                idx      = ch["index"]
                ch_title = ch.get("title") or f"Chapter {idx}"
                ch_text  = ch["summary"]
                async with mm_sem:
                    try:
                        if mindmap_format == "json":
                            data = await generate_json_mindmap(
                                ch_title, ch_text, req.language, model=model_mindmap
                            )
                            json_path = os.path.join(tmp, f"ch{idx}_mindmap.json")
                            with open(json_path, "w", encoding="utf-8") as f:
                                json_module.dump(data, f, ensure_ascii=False)
                            key = f"books/{req.book_id}/chapters/ch_{idx:02d}_mindmap.json"
                            url = upload_file(json_path, key, CONTENT_TYPES[".json"])
                            return idx, {"url": url, "data": data, "format": "json"}, None
                        else:
                            mermaid = await generate_mermaid_code(
                                ch_title, ch_text, req.language, model=model_mindmap
                            )
                            svg_path = os.path.join(tmp, f"ch{idx}_mindmap.svg")
                            await render_mermaid_svg(mermaid, svg_path)
                            key = f"books/{req.book_id}/chapters/ch_{idx:02d}_mindmap.svg"
                            url = upload_file(svg_path, key, CONTENT_TYPES[".svg"])
                            return idx, {"url": url, "data": None, "format": "mermaid"}, None
                    except Exception as e:
                        return idx, None, str(e)

            # Count chapters that will be skipped
            chapters_with_summary = [ch for ch in chapter_results if ch.get("summary")]
            ch_skipped_no_summary = len(chapter_results) - len(chapters_with_summary)
            
            if ch_skipped_no_summary > 0:
                log.warning("%d chapters skipped for mindmap: no summaries available", ch_skipped_no_summary)

            mm_results = await asyncio.gather(
                *[_chapter_mindmap(ch) for ch in chapters_with_summary]
            )
            
            # Update chapter_results with mindmap URLs for dashboard display
            ch_processed = 0
            for idx, result, err in mm_results:
                if result is not None:
                    chapter_mindmap[idx] = result
                    ch_processed += 1
                    # Find the chapter and add mindmap URL
                    for ch in chapter_results:
                        if ch["index"] == idx:
                            ch["mindmap_url"] = result.get("url")
                            ch["mindmap_format"] = result.get("format")
                            if result.get("data"):
                                ch["mindmap_data"] = result.get("data")
                            break
                if err is not None:
                    ch_errors += 1
                    errors[f"mindmap_chapter_{idx}"] = err

            # Calculate status based on actual processing
            if ch_errors > 0 and ch_processed == 0:
                step_status["mindmap_chapters"] = "failed"
            elif ch_errors > 0:
                step_status["mindmap_chapters"] = "partial"
            elif ch_processed > 0:
                step_status["mindmap_chapters"] = "done"
            elif ch_skipped_no_summary > 0:
                step_status["mindmap_chapters"] = "failed"
                errors["mindmap_chapters"] = f"All {ch_skipped_no_summary} chapters skipped: no summaries available"
            else:
                step_status["mindmap_chapters"] = "skipped"
            
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "mindmap_chapters", step_status["mindmap_chapters"], duration_sec=_t)
            await _checkpoint()

        # ── Run Phase 2 in parallel ───────────────────────────────────────────
        phase2_coros = [
            _do_cover(),
            _do_audio_full(),
            _do_audio_chapters(),
            _do_mindmap(),
            _do_mindmap_chapters(),
        ]
        phase2_results = await asyncio.gather(*phase2_coros, return_exceptions=True)
        # Re-raise cancellation; other exceptions are already recorded per-step
        for r in phase2_results:
            if isinstance(r, JobCancelledError):
                raise r
            if isinstance(r, Exception):
                import logging as _log
                _log.getLogger(__name__).warning("Phase-2 step raised unexpected: %s", r)

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 3 — PARALLEL: alt_text · video
        # alt_text needs cover; video needs audio_full + summary + cover + mindmap
        # ═════════════════════════════════════════════════════════════════════

        # ── alt_text ──────────────────────────────────────────────────────────
        async def _do_alt_text() -> None:
            nonlocal alt_text
            if (
                "alt_text" not in steps or not alttext_enabled
                or not cover_url or not os.path.exists(cover_path_saved)
            ):
                return
            step_status["alt_text"] = "running"
            set_step("alt_text")
            await _checkpoint()
            try:
                alt_text = await generate_alt_text(cover_path_saved, req.title or req.book_id, req.language)
                step_status["alt_text"] = "done"
            except JobCancelledError:
                raise
            except Exception as e:
                errors["alt_text"] = str(e)
                step_status["alt_text"] = "failed"
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "alt_text", step_status["alt_text"], duration_sec=_t)
            await _checkpoint()

        # ── video ─────────────────────────────────────────────────────────────
        async def _do_video() -> None:
            nonlocal video_url, video_meta
            if "video" not in steps or not video_enabled or not full_summary:
                return
            step_status["video"] = "running"
            set_step("video")
            await _checkpoint()
            try:
                video_out   = os.path.join(tmp, "video.mp4")
                use_cover   = cover_path_saved if cover_url and os.path.exists(cover_path_saved) else None
                use_mindmap = mindmap_path_saved if mindmap_path_saved and os.path.exists(mindmap_path_saved) else None

                video_meta = await generate_book_video(
                    title         = req.title or req.book_id,
                    author        = req.author or "",
                    summary_text  = full_summary,
                    language      = req.language,
                    audio_path    = full_audio_path,
                    cover_path    = use_cover,
                    mindmap_path  = use_mindmap,
                    chapters      = chapter_results,
                    output_path   = video_out,
                    provider_name = video_provider,
                )
                key = f"books/{req.book_id}/video_{req.language}.mp4"
                video_url = upload_file(video_out, key, CONTENT_TYPES[".mp4"])
                step_status["video"] = "done"
            except VideoError as e:
                errors["video"] = str(e)
                step_status["video"] = "failed"
            except JobCancelledError:
                raise
            except Exception as e:
                errors["video"] = str(e)
                step_status["video"] = "failed"
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "video", step_status["video"], output_url=video_url, duration_sec=_t)
            await _checkpoint()

        # ── inject_epub ── PHASE 4: runs LAST so it has ALL assets ────────────
        async def _do_inject_epub() -> None:
            nonlocal epub_url
            log.info("_do_inject_epub called: inject_epub in steps=%s, epub_enabled=%s, has_full_summary=%s", 
                     "inject_epub" in steps, epub_enabled, bool(full_summary))
            if "inject_epub" not in steps or not epub_enabled or not full_summary:
                log.info("_do_inject_epub skipped: steps=%s, epub_enabled=%s, full_summary=%s", 
                         steps, epub_enabled, bool(full_summary))
                return
            step_status["inject_epub"] = "running"
            set_step("inject_epub")
            await _checkpoint()
            try:
                # Try to fetch source EPUB from CDN; fall back to creating from scratch
                src_path: str | None = None
                if base_url:
                    try:
                        _src = os.path.join(tmp, "source.epub")
                        await fetch_epub(req.book_id, req.language, _src)
                        src_path = _src
                        log.info("inject_epub: using source EPUB from CDN")
                    except EpubNotAvailableError as e:
                        log.info("inject_epub: source EPUB not available (%s) — creating from scratch", e)
                        src_path = None
                else:
                    log.info("inject_epub: no BOOK_FILES_BASE_URL — creating EPUB from scratch")

                out_path = os.path.join(tmp, f"{req.book_id}_{req.language}.epub")
                
                # Download cover from CDN if we have a URL but no local file
                _cover_for_epub = None
                if cover_url and not os.path.exists(cover_path_saved):
                    try:
                        import httpx
                        async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
                            r = await client.get(cover_url)
                            if r.status_code == 200 and r.content:
                                Path(cover_path_saved).write_bytes(r.content)
                                log.info("Downloaded cover from CDN for EPUB injection")
                    except Exception as e:
                        log.warning("Could not download cover for EPUB: %s", e)
                
                if cover_url and os.path.exists(cover_path_saved):
                    _cover_for_epub = cover_path_saved
                
                # Debug logging for EPUB injection
                log.info("Injecting EPUB for book %s", req.book_id)
                log.info("  - chapter_results count: %d", len(chapter_results))
                log.info("  - chapter_results with summaries: %d", sum(1 for ch in chapter_results if ch.get("summary")))
                log.info("  - chapter_audio keys: %s", list(chapter_audio.keys()))
                log.info("  - chapter_mindmap keys: %s", list(chapter_mindmap.keys()))
                log.info("  - full_audio: %s", (full_audio or {}).get("url", "None"))
                log.info("  - mindmap_url: %s", mindmap_url or "None")
                log.info("  - cover_path: %s", _cover_for_epub or "None")
                
                await inject_summary_into_epub(
                    src_path,
                    out_path,
                    title            = req.title or req.book_id,
                    author           = req.author or "",
                    summary_text     = full_summary,
                    language         = req.language,
                    cover_path       = _cover_for_epub,
                    chapters         = chapter_results,
                    chapter_audio    = chapter_audio,
                    chapter_mindmap  = chapter_mindmap,
                    audio_url        = (full_audio or {}).get("url"),
                    mindmap_url      = mindmap_url,
                )
                # Storage key uses the clean filename: books/{id}/{id}_{lang}.epub
                key = f"books/{req.book_id}/{req.book_id}_{req.language}.epub"
                epub_url = upload_file(out_path, key, CONTENT_TYPES[".epub"])
                step_status["inject_epub"] = "done"
            except JobCancelledError:
                raise
            except (EpubError, Exception) as e:
                errors["inject_epub"] = str(e)
                step_status["inject_epub"] = "failed"
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "inject_epub", step_status["inject_epub"], output_url=epub_url, duration_sec=_t)
            await _checkpoint()

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 3 — PARALLEL: alt_text · video
        # (inject_epub runs AFTER Phase 3 completes, so it has all assets)
        # ═════════════════════════════════════════════════════════════════════
        phase3_results = await asyncio.gather(
            _do_alt_text(),
            _do_video(),
            return_exceptions=True,
        )
        for r in phase3_results:
            if isinstance(r, JobCancelledError):
                raise r
        
        # ═════════════════════════════════════════════════════════════════════
        # PHASE 4 — inject_epub (runs LAST, after all other steps complete)
        # This ensures the EPUB includes all generated assets:
        # - cover image
        # - full audio
        # - chapter audio
        # - chapter mindmaps
        # - alt_text
        # ═════════════════════════════════════════════════════════════════════
        await _do_inject_epub()

    # ── Attach audio + mindmap URLs to chapter results ────────────────────────
    lang = req.language
    for ch in chapter_results:
        ch[f"audio_{lang}"] = chapter_audio.get(ch["index"])
        cm = chapter_mindmap.get(ch["index"])
        if cm:
            ch["mindmap_url"]    = cm["url"]
            ch["mindmap_format"] = cm["format"]
            if cm["data"] is not None:
                ch["mindmap_data"] = cm["data"]

    # ── Overall status ────────────────────────────────────────────────────────
    active = {s for s in steps if s in ALL_STEPS}
    failed = sum(1 for s in active if step_status.get(s) == "failed")
    status = "done" if failed == 0 else ("failed" if failed == len(active) else "partial")

    elapsed = round(time.time() - started, 1)
    cost    = await _compute_job_cost(job_id)

    summary_key = f"{req.options.length}_{req.language}"

    files = {
        "cover":      cover_url,
        "audio_full": (full_audio or {}).get("url"),
        "mindmap":    mindmap_url,
        "epub":       epub_url,
        "video":      video_url,
        "chapters": [
            {
                "index":       ch["index"],
                "title":       ch.get("title"),
                "audio_url":   chapter_audio.get(ch["index"]),
                "mindmap_url": (chapter_mindmap.get(ch["index"]) or {}).get("url"),
            }
            for ch in chapter_results
        ],
    }

    return {
        "book_id":         req.book_id,
        "status":          status,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "processing_time": _fmt_duration(elapsed),
        "cost":            cost,
        "files":           files,
        "steps":           step_status,
        "metadata": {
            "title":          req.title,
            "author":         req.author,
            "year":           req.year,
            "pages":          req.pages,
            "grade_level":    req.grade_level,
            "genres":         req.genres,
            "cover_url":      cover_url,
            "cover_alt_text": alt_text,
        },
        "quick_summary": quick_summary,
        "summaries": {
            summary_key: {
                "text":       full_summary,
                "word_count": len(full_summary.split()) if full_summary else 0,
                "style":      req.options.style,
                "language":   req.language,
            }
        } if full_summary else {},
        "audio": {f"full_{lang}": full_audio} if full_audio else {},
        "mindmap": (
            {"url": mindmap_url, "data": mindmap_data} if mindmap_data else
            {"url": mindmap_url} if mindmap_url else
            None
        ),
        "epub": ({f"enriched_{lang}": {"url": epub_url}} if epub_url else None),
        "video": (
            {
                f"summary_{lang}": {
                    "url":              video_url,
                    "duration_seconds": (video_meta or {}).get("duration_seconds"),
                    "size_mb":          (video_meta or {}).get("size_mb"),
                    "width":            (video_meta or {}).get("width"),
                    "height":           (video_meta or {}).get("height"),
                    "provider":         (video_meta or {}).get("provider"),
                    "silent":           (video_meta or {}).get("silent", False),
                },
            } if video_url else None
        ),
        "chapters": chapter_results,
        "errors":   errors,
    }
