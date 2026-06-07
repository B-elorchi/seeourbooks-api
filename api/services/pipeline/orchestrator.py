"""
Pipeline orchestrator — the core engine.

Steps (each independent, continues even if one fails):
  summarize       Pass 1 (Haiku chapter summaries) + Pass 2 (Sonnet full) + Pass 3 (Haiku review)
  audio_full      TTS of full summary → single MP3
  audio_chapters  TTS of each chapter → one MP3 per chapter
  cover           AI cover image
  alt_text        Alt text for the cover (depends on cover)
  mindmap         AI mind map SVG

Live config is read from Supabase provider_config via runtime.py at job start.
No restart needed when the admin switches providers.
"""
import asyncio
import json as json_module
import os
import tempfile
import time
from datetime import datetime, timezone

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

# Single source of truth lives in api/models/requests.py
ALL_STEPS = VALID_STEPS


def _resolve_steps(requested: list[str]) -> set[str]:
    """Return the full set of steps to run, including auto-added dependencies."""
    steps = set(requested) if requested else set(ALL_STEPS)
    # Enforce dependencies
    if "audio_full" in steps or "audio_chapters" in steps:
        steps.add("summarize")
    if "mindmap" in steps or "mindmap_chapters" in steps:
        steps.add("summarize")
    if "alt_text" in steps:
        steps.add("cover")
    if "inject_epub" in steps:
        # Needs the summary text. Cover is OPTIONAL — if the cover step
        # also runs we use its output; otherwise we keep the original EPUB cover.
        steps.add("summarize")
    if "video" in steps:
        # Video uses the full TTS audio as its narration track and the summary
        # text for sentence-aligned subtitles.  Cover + mindmap are visual
        # bonuses — auto-add them when they aren't explicitly excluded.
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


async def _ensure_book_row(req: "PipelineReq") -> None:
    """
    Make sure a row exists in the `books` table for this book_id.

    Two cases:

    1. Production catalog book — numeric `book_id`, integer PK.  The row
       already exists; do nothing.  Trying to upsert a string `book_id`
       into an INTEGER column fails immediately.

    2. Custom JSON / PDF upload — string `book_id`.  We try to upsert a
       MINIMAL row (just book_id + title + author).  Many production
       deployments don't have the extra columns (`language`, `grade_level`,
       `genres`, `status`) so we don't send them.  If the books table is
       schema-locked or integer-keyed (most production setups), the upsert
       silently fails and we just log it — the rest of the pipeline doesn't
       depend on this row in any modern table (no FKs in our migrations).
    """
    # Skip for production catalog ids — the row is already there
    if req.book_id and req.book_id.isdigit():
        return

    # Minimal upsert that won't blow up on schemas missing optional columns.
    # `title` and `author` are present in every books schema we've seen.
    payload: dict = {
        "book_id": req.book_id,
        "title":   req.title or req.book_id,
        "author":  req.author or "",
    }

    try:
        await upsert("books", payload, conflict="book_id")
    except Exception as e:
        # Log at DEBUG — this is best-effort and the rest of the pipeline
        # does NOT depend on the books row existing.  Surfacing it as a
        # WARNING in every job cluttered the logs.
        import logging
        logging.getLogger(__name__).debug(
            "Could not upsert books row for %s (likely schema mismatch — safe to ignore): %s",
            req.book_id, e,
        )


async def _fetch_catalog_chapters(book_id: str) -> list[dict]:
    """
    Load chapters from the chunks table when no chapters are supplied in the request.

    Production `chunks` tables typically key `book_id` as BIGINT.  Querying
    with a non-numeric string (e.g. "book_0127", "pdf_8f8d3285") causes
    PostgREST to fail with a 400 cast error.  So we only attempt the
    lookup when the id is actually numeric; otherwise we return an empty
    list and let the caller decide what to do.
    """
    # Skip for non-numeric ids (custom_json / PDF uploads) — saves a guaranteed-400
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


# ── Production catalog enrichment ─────────────────────────────────────────────
#
# The production `books` table (integer PK book_id) carries pre-computed
# summaries in dedicated columns:
#     summary_english       — long-form English summary
#     summary_en_10min      — ~10-minute English summary (preferred for audio)
#     arabic_summary        — long-form Arabic summary
#     arabic_summary_v2     — newer Arabic summary (preferred for audio)
#
# When the client posts just  {"book_id": "12345"}  we look the row up,
# fill in missing title / author, and reuse the cached summary so audio +
# cover + mindmap can run without re-summarizing the whole book.

async def _fetch_book_from_catalog(book_id: str) -> dict | None:
    """
    Look up a row in the production `books` table by integer book_id.

    Returns None when:
      - book_id is non-numeric (custom_json upload — not a catalog book)
      - the books table is unreachable / different schema
      - no row matches
    """
    try:
        bid = int(book_id)
    except (TypeError, ValueError):
        return None  # custom string id — not a catalog book

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
        # Books table missing / schema mismatch / network — fall through silently
        import logging as _logging
        _logging.getLogger(__name__).debug(
            "catalog lookup for book_id=%s failed: %s", book_id, exc,
        )
        return None

    return rows[0] if rows else None


async def _fetch_gutenberg_metadata(book_id: str) -> dict:
    """
    Fetch title and author from the Gutendex API for a numeric Gutenberg book_id.

    Returns a dict with keys 'title' and 'author' (both may be empty strings on failure).
    """
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
            # Gutendex format: "Barrie, J. M. (James Matthew)" — reverse to "J. M. Barrie"
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
    """
    Return the best pre-computed summary on the books row for the language.

    Tries the 10-minute summary first (preferred — sized for our audio target),
    then falls back to the long-form summary.

    NUL-string handling
    ───────────────────
    Production data sometimes stores the LITERAL STRING "null" (or "None",
    "NULL", etc.) where a real database NULL was intended.  Treating those
    strings as a real summary would cause downstream TTS to literally narrate
    the word "null".  We strip those sentinel values out below.
    """
    _NULL_SENTINELS = {"", "null", "none", "nil", "n/a", "na", "undefined"}

    def _nonempty(value) -> str | None:
        if not isinstance(value, str):
            return None
        stripped = value.strip()
        if not stripped:
            return None
        # Reject sentinel strings that production data has been observed using
        # in place of actual NULL (e.g. "null", "None", etc.).
        if stripped.lower() in _NULL_SENTINELS:
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
) -> dict:
    """
    Execute the pipeline and return the full result dict.

    `job_id`, when supplied, enables live progress checkpointing — after every
    step the orchestrator writes a partial result to `pipeline_jobs.result`
    so the frontend's 3s poll sees progress immediately instead of waiting
    until the whole pipeline finishes.
    """

    started = time.time()
    cfg     = await get_all_config()          # live admin config — Supabase + settings fallback
    steps   = _resolve_steps(req.steps)
    errors: dict[str, str]   = {}
    step_status: dict[str, str] = {s: "skipped" for s in ALL_STEPS}

    # ── Hoisted result-variables ─────────────────────────────────────────────
    # Declared up front (instead of inside each step block) so the live
    # checkpoint closure below can read whatever has been produced so far.
    # All start as None / [] and get populated by the step that owns them.
    chapter_results: list[dict] = []
    full_summary:    str        = ""
    quick_summary:   str        = ""
    full_audio:      dict | None = None
    chapter_audio:   dict[int, str] = {}
    cover_url:       str | None = None
    alt_text:        str | None = None
    mindmap_url:     str | None = None
    mindmap_data:    dict | None = None
    chapter_mindmap: dict[int, dict] = {}
    epub_url:        str | None = None
    video_url:       str | None = None
    video_meta:      dict | None = None

    # ── Live progress checkpointing ──────────────────────────────────────────
    # After every step, write a snapshot of step_status + every result field
    # produced so far to `pipeline_jobs.result`.  The frontend's 3-second poll
    # picks it up — chips turn green AND output panels (cover, audio, mindmap,
    # epub, video) populate one at a time as each step completes.
    async def _checkpoint() -> None:
        if not job_id:
            return
        try:
            lang = req.language
            partial = {
                "book_id":         req.book_id,
                "status":          "running",
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
        except Exception as exc:
            # Best-effort — a failed checkpoint must never break the pipeline.
            import logging as _log
            _log.getLogger(__name__).warning("checkpoint write failed: %s", exc)

    # ── Production catalog enrichment ─────────────────────────────────────────
    # When the client posts only {"book_id": "12345"} we look up the production
    # `books` row and fill in any missing title / author / summary from it.
    # If a pre-computed summary exists on the books row (summary_en_10min for
    # English or arabic_summary_v2 for Arabic), we use it as `req.summary` so
    # the Pass 1/2 re-summarization is skipped and downstream steps (audio,
    # cover, mindmap) run directly on the cached text.
    if not (req.title and req.author and req.summary):
        book_row = await _fetch_book_from_catalog(req.book_id)
        if book_row:
            updates: dict = {}
            # Treat title == book_id as a missing title (placeholder inserted during initial import)
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

    # For numeric book_ids still missing a real title (e.g. catalog row had title == book_id),
    # fall back to Gutendex API to retrieve the actual title and author.
    if req.book_id and req.book_id.isdigit() and not req.title:
        gutenberg_meta = await _fetch_gutenberg_metadata(req.book_id)
        meta_updates: dict = {}
        if gutenberg_meta.get("title"):
            meta_updates["title"] = gutenberg_meta["title"]
        if not req.author and gutenberg_meta.get("author"):
            meta_updates["author"] = gutenberg_meta["author"]
        if meta_updates:
            req = req.model_copy(update=meta_updates)

    # Ensure the FK target exists — chunk_summaries / book_summaries reference books(book_id).
    # custom_json requests for a book not in the catalog would otherwise 409.
    await _ensure_book_row(req)

    # ── Resolve chapters ──────────────────────────────────────────────────────
    # Three valid input sources, checked in order:
    #   1. req.chapters       — chapter list supplied directly in the request
    #   2. catalog `chunks`   — only when book_id is numeric and the row exists
    #   3. req.summary        — pre-computed summary text supplied directly
    # The pipeline can still run with #3 alone (audio + cover + mindmap all
    # consume the summary text), even when there are zero chapters.
    if req.chapters:
        chapters = [{"index": c.index, "title": c.title, "text": c.text} for c in req.chapters]
    else:
        # _fetch_catalog_chapters returns [] on bad book_id / 400 — never raises
        chapters = await _fetch_catalog_chapters(req.book_id)

    # Hard-fail when there's literally NO input the pipeline can work with
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

    # ── Resolved models from live config ────────────────────────────────────
    model_haiku      = cfg.get("MODEL_HAIKU",      settings.MODEL_HAIKU)
    model_sonnet     = cfg.get("MODEL_SONNET",     settings.MODEL_SONNET)
    model_mindmap    = cfg.get("MODEL_MINDMAP",    settings.MODEL_MINDMAP)
    mindmap_format   = cfg.get("MINDMAP_FORMAT",   settings.MINDMAP_FORMAT)

    with tempfile.TemporaryDirectory() as tmp:

        # ─────────────────────────────────────────────────────────────────────
        # STEP: summarize  (Pass 1 Haiku + Pass 2 Sonnet + Pass 3 Haiku review)
        # ─────────────────────────────────────────────────────────────────────
        chapter_results: list[dict] = []
        full_summary  = ""
        quick_summary = ""

        if "summarize" in steps:
            step_status["summarize"] = "running"
            set_step("summarize")
            try:
                # ── Pre-computed summary shortcut ──────────────────────────────
                # If the caller sent a summary field, use it directly and skip
                # Pass 1 (Haiku) and Pass 2 (Sonnet).  Still runs Pass 3 (review)
                # if the language is Arabic (tashkeel correction).
                if req.summary:
                    full_summary = req.summary
                    sentences = [s.strip() for s in full_summary.replace(".\n", ". ").split(". ") if s.strip()]
                    quick_summary = ". ".join(sentences[:2]) + "." if sentences else full_summary[:200]
                    step_status["summarize"] = "done"
                    # Build placeholder chapter_results (no per-chapter data)
                    chapter_results = []

                else:
                    for ch in chapters:
                        chunk = {
                            "id":          f"{req.book_id}_ch{ch['index']}",
                            "chunk_index": ch["index"],
                            "content":     ch["text"],
                        }
                        sums = await run_haiku_pass(
                            req.book_id, [chunk], req.language, model=model_haiku,
                        )
                        chapter_results.append({
                            "index":         ch["index"],
                            "title":         ch["title"],
                            "summary":       sums[0] if sums else "",
                            "read_time_min": max(1, len(sums[0].split()) // 200) if sums else 1,
                        })

                    chunk_summaries = [c["summary"] for c in chapter_results if c.get("summary")]

                    # Pass 2 — Sonnet full summary
                    if chunk_summaries:
                        full_summary = await run_sonnet_pass_sync(
                            chunk_summaries,
                            req.options.length,
                            req.options.style,
                            req.language,
                            model_override=model_sonnet,
                        )

                        # Pass 3 — Haiku review (quality check + light correction)
                        full_summary = await run_review_pass(
                            full_summary,
                            req.options.length,
                            req.options.style,
                            req.language,
                            model=model_haiku,
                        )

                        # Quick summary — first 2 sentences of the full summary
                        sentences = [s.strip() for s in full_summary.replace(".\n", ". ").split(". ") if s.strip()]
                        quick_summary = ". ".join(sentences[:2]) + "." if sentences else full_summary[:200]

                    step_status["summarize"] = "done"
            except Exception as e:
                errors["summarize"] = str(e)
                step_status["summarize"] = "failed"
            await _checkpoint()

        # Summary key e.g. "10min_en"
        summary_key = f"{req.options.length}_{req.language}"

        # ─────────────────────────────────────────────────────────────────────
        # STEP: audio_full
        # ─────────────────────────────────────────────────────────────────────
        full_audio: dict | None = None
        # Local on-disk path to the finished audio file — kept around so the
        # video step can re-use it as the narration track without re-downloading.
        full_audio_path: str | None = None
        tts_enabled = cfg.get("PIPELINE_STEP_TTS", "true") == "true"

        if "audio_full" in steps and tts_enabled and full_summary:
            step_status["audio_full"] = "running"
            set_step("audio_full")
            try:
                raw = os.path.join(tmp, "audio_raw.mp3")
                proc = os.path.join(tmp, "audio.mp3")
                await synthesize(full_summary, req.language, raw, cfg)

                audio_proc_enabled = cfg.get("PIPELINE_STEP_AUDIO_PROCESSING", "true") == "true"
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
            except Exception as e:
                errors["audio_full"] = str(e)
                step_status["audio_full"] = "failed"
            await _checkpoint()

        # ─────────────────────────────────────────────────────────────────────
        # STEP: audio_chapters
        # ─────────────────────────────────────────────────────────────────────
        chapter_audio: dict[int, str] = {}   # index → URL

        if "audio_chapters" in steps and tts_enabled and chapter_results:
            step_status["audio_chapters"] = "running"
            set_step("audio_chapters")
            ch_errors = 0
            for ch in chapter_results:
                if not ch.get("summary"):
                    continue
                try:
                    ch_raw  = os.path.join(tmp, f"ch{ch['index']}_raw.mp3")
                    ch_proc = os.path.join(tmp, f"ch{ch['index']}.mp3")
                    await synthesize(ch["summary"], req.language, ch_raw, cfg)

                    audio_proc_enabled = cfg.get("PIPELINE_STEP_AUDIO_PROCESSING", "true") == "true"
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
                    chapter_audio[idx] = upload_file(src, k, CONTENT_TYPES[".mp3"])
                except Exception as e:
                    ch_errors += 1
                    errors[f"audio_chapter_{ch['index']}"] = str(e)

            step_status["audio_chapters"] = "failed" if ch_errors == len(chapter_results) else (
                "partial" if ch_errors else "done"
            )
            await _checkpoint()

        # ─────────────────────────────────────────────────────────────────────
        # STEP: cover
        # ─────────────────────────────────────────────────────────────────────
        cover_url: str | None = None
        cover_path_saved = os.path.join(tmp, "cover.jpg")

        cover_enabled = cfg.get("PIPELINE_STEP_COVER", "true") == "true"
        if "cover" in steps and cover_enabled:
            step_status["cover"] = "running"
            set_step("cover")
            try:
                # Feed the actual book content into the prompt so the cover
                # visually reflects the story — not just the title.
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
            except Exception as e:
                errors["cover"] = str(e)
                step_status["cover"] = "failed"
            await _checkpoint()

        # ─────────────────────────────────────────────────────────────────────
        # STEP: alt_text  (requires cover)
        # ─────────────────────────────────────────────────────────────────────
        alt_text: str | None = None
        alttext_enabled = cfg.get("PIPELINE_STEP_ALTTEXT", "true") == "true"

        if "alt_text" in steps and alttext_enabled and cover_url and os.path.exists(cover_path_saved):
            step_status["alt_text"] = "running"
            set_step("alt_text")
            try:
                alt_text = await generate_alt_text(cover_path_saved, req.title or req.book_id, req.language)
                step_status["alt_text"] = "done"
            except Exception as e:
                errors["alt_text"] = str(e)
                step_status["alt_text"] = "failed"
            await _checkpoint()

        # ─────────────────────────────────────────────────────────────────────
        # STEP: mindmap
        # ─────────────────────────────────────────────────────────────────────
        mindmap_url: str | None = None
        mindmap_data: dict | None = None
        # Local on-disk path to the mindmap (SVG for mermaid, n/a for JSON)
        # so the video step can render a mindmap reveal stage.
        mindmap_path_saved: str | None = None
        mindmap_enabled = cfg.get("PIPELINE_STEP_MINDMAP", "true") == "true"

        if "mindmap" in steps and mindmap_enabled and full_summary:
            step_status["mindmap"] = "running"
            set_step("mindmap")
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
            except Exception as e:
                errors["mindmap"] = str(e)
                step_status["mindmap"] = "failed"
            await _checkpoint()

        # ─────────────────────────────────────────────────────────────────────
        # STEP: mindmap_chapters  (one mindmap per chapter)
        # ─────────────────────────────────────────────────────────────────────
        # index → {"url": str, "data": dict | None, "format": "mermaid" | "json"}
        chapter_mindmap: dict[int, dict] = {}

        if "mindmap_chapters" in steps and mindmap_enabled and chapter_results:
            step_status["mindmap_chapters"] = "running"
            set_step("mindmap_chapters")
            ch_errors = 0
            for ch in chapter_results:
                if not ch.get("summary"):
                    continue
                try:
                    idx       = ch["index"]
                    ch_title  = ch.get("title") or f"Chapter {idx}"
                    ch_text   = ch["summary"]

                    if mindmap_format == "json":
                        data = await generate_json_mindmap(
                            ch_title, ch_text, req.language, model=model_mindmap
                        )
                        json_path = os.path.join(tmp, f"ch{idx}_mindmap.json")
                        with open(json_path, "w", encoding="utf-8") as f:
                            json_module.dump(data, f, ensure_ascii=False)
                        key = f"books/{req.book_id}/chapters/ch_{idx:02d}_mindmap.json"
                        url = upload_file(json_path, key, CONTENT_TYPES[".json"])
                        chapter_mindmap[idx] = {"url": url, "data": data, "format": "json"}
                    else:
                        mermaid = await generate_mermaid_code(
                            ch_title, ch_text, req.language, model=model_mindmap
                        )
                        svg_path = os.path.join(tmp, f"ch{idx}_mindmap.svg")
                        await render_mermaid_svg(mermaid, svg_path)
                        key = f"books/{req.book_id}/chapters/ch_{idx:02d}_mindmap.svg"
                        url = upload_file(svg_path, key, CONTENT_TYPES[".svg"])
                        chapter_mindmap[idx] = {"url": url, "data": None, "format": "mermaid"}
                except Exception as e:
                    ch_errors += 1
                    errors[f"mindmap_chapter_{ch['index']}"] = str(e)

            # Status reflects how many chapter mindmaps actually got generated
            attempted = sum(1 for c in chapter_results if c.get("summary"))
            step_status["mindmap_chapters"] = (
                "failed"  if ch_errors == attempted and attempted > 0 else
                "partial" if ch_errors > 0 else
                "done"    if attempted > 0 else
                "skipped"
            )
            await _checkpoint()

        # ─────────────────────────────────────────────────────────────────────
        # STEP: inject_epub  (fetch source EPUB, inject summary + cover, upload)
        # ─────────────────────────────────────────────────────────────────────
        epub_url: str | None = None
        epub_enabled = cfg.get("PIPELINE_STEP_INJECT_EPUB", "true") == "true"
        base_url     = cfg.get("BOOK_FILES_BASE_URL") or settings.BOOK_FILES_BASE_URL

        if "inject_epub" in steps and epub_enabled and full_summary:
            step_status["inject_epub"] = "running"
            set_step("inject_epub")
            if not base_url:
                errors["inject_epub"] = (
                    "BOOK_FILES_BASE_URL is not configured — set it in Admin → "
                    "Providers → EPUB Source to enable inject_epub."
                )
                step_status["inject_epub"] = "skipped"
            else:
                try:
                    src_path = os.path.join(tmp, "source.epub")
                    out_path = os.path.join(tmp, "enriched.epub")

                    await fetch_epub(req.book_id, req.language, src_path)
                    await inject_summary_into_epub(
                        src_path, out_path,
                        title        = req.title or req.book_id,
                        author       = req.author or "",
                        summary_text = full_summary,
                        language     = req.language,
                        cover_path   = (
                            cover_path_saved
                            if cover_url and os.path.exists(cover_path_saved)
                            else None
                        ),
                    )
                    key = f"books/{req.book_id}/enriched_{req.language}.epub"
                    epub_url = upload_file(out_path, key, CONTENT_TYPES[".epub"])
                    step_status["inject_epub"] = "done"
                except EpubNotAvailableError as e:
                    # 404 on the source EPUB — degrade to skipped, not failed.
                    # This avoids painting the whole job red just because the
                    # client's catalog hasn't got an EPUB for this book_id yet.
                    errors["inject_epub"] = str(e)
                    step_status["inject_epub"] = "skipped"
                except EpubError as e:
                    errors["inject_epub"] = str(e)
                    step_status["inject_epub"] = "failed"
                except Exception as e:
                    errors["inject_epub"] = str(e)
                    step_status["inject_epub"] = "failed"
            await _checkpoint()

        # ─────────────────────────────────────────────────────────────────────
        # STEP: video  (slideshow video with optional TTS narration + subtitles)
        # ─────────────────────────────────────────────────────────────────────
        # Soft requirement: a full_summary so we have chapter cards + subtitles.
        # If audio_full succeeded → video uses it as the narration track and the
        # video duration matches the audio length.
        # If audio_full was skipped/disabled → video runs as a SILENT slideshow
        # at a fixed duration (useful for local testing without TTS keys).
        video_url:  str | None = None
        video_meta: dict | None = None
        video_enabled  = cfg.get("PIPELINE_STEP_VIDEO", "true") == "true"
        video_provider = cfg.get("VIDEO_PROVIDER") or settings.VIDEO_PROVIDER

        if "video" in steps and video_enabled and full_summary:
            step_status["video"] = "running"
            set_step("video")
            try:
                video_out = os.path.join(tmp, "video.mp4")

                # Cover & mindmap are visual bonuses — use them when present.
                use_cover  = (
                    cover_path_saved
                    if cover_url and os.path.exists(cover_path_saved)
                    else None
                )
                use_mindmap = (
                    mindmap_path_saved
                    if mindmap_path_saved and os.path.exists(mindmap_path_saved)
                    else None
                )

                video_meta = await generate_book_video(
                    title         = req.title or req.book_id,
                    author        = req.author or "",
                    summary_text  = full_summary,
                    language      = req.language,
                    audio_path    = full_audio_path,   # None → silent slideshow
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
            except Exception as e:
                errors["video"] = str(e)
                step_status["video"] = "failed"
            await _checkpoint()

    # ── Attach audio URLs to chapter results ─────────────────────────────────
    lang = req.language
    for ch in chapter_results:
        ch[f"audio_{lang}"] = chapter_audio.get(ch["index"])
        # Attach per-chapter mindmap if generated
        cm = chapter_mindmap.get(ch["index"])
        if cm:
            ch["mindmap_url"]    = cm["url"]
            ch["mindmap_format"] = cm["format"]
            if cm["data"] is not None:
                ch["mindmap_data"] = cm["data"]

    # ── Determine overall status ──────────────────────────────────────────────
    active = {s for s in steps if s in ALL_STEPS}
    failed = sum(1 for s in active if step_status.get(s) == "failed")
    status = "done" if failed == 0 else ("failed" if failed == len(active) else "partial")

    elapsed = round(time.time() - started, 1)

    return {
        "book_id":         req.book_id,
        "status":          status,
        "generated_at":    datetime.now(timezone.utc).isoformat(),
        "processing_time": _fmt_duration(elapsed),
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
        "audio": {
            f"full_{lang}": full_audio,
        } if full_audio else {},
        "mindmap": (
            {"url": mindmap_url, "data": mindmap_data} if mindmap_data else
            {"url": mindmap_url} if mindmap_url else
            None
        ),
        "epub": (
            {f"enriched_{lang}": {"url": epub_url}} if epub_url else None
        ),
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
