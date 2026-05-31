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
from api.services.db import find, upsert
from api.services.usage_logger import set_step
from api.services.summarizer.haiku import run_haiku_pass
from api.services.summarizer.sonnet import run_sonnet_pass_sync
from api.services.summarizer.review import run_review_pass
from api.services.pipeline.tts import synthesize
from api.services.pipeline.cover import generate_cover
from api.services.pipeline.mindmap import generate_mermaid_code, render_mermaid_svg, generate_json_mindmap
from api.services.pipeline.alttext import generate_alt_text
from api.services.pipeline.audio import process_audio
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

    chunk_summaries / book_summaries / chunks all have FK references to books,
    so a custom_json request for a book that was never ingested into the catalog
    would fail with a foreign-key violation on the first cache write.

    Upsert is idempotent: a real catalog book is unchanged; a custom_json book
    gets a row created from whatever metadata the request supplied.
    """
    try:
        await upsert(
            "books",
            {
                "book_id":     req.book_id,
                "title":       req.title       or req.book_id,
                "author":      req.author      or "",
                "language":    req.language    or "en",
                "year":        req.year,
                "pages":       req.pages,
                "grade_level": req.grade_level,
                "genres":      req.genres or [],
                "status":      "ready",
            },
            conflict="book_id",
        )
    except Exception as e:
        # Log but don't crash — if the row already exists with stricter rules
        # the FK will still resolve. Surfacing the original step error is more useful.
        import logging
        logging.getLogger(__name__).warning("Could not upsert books row for %s: %s", req.book_id, e)


async def _fetch_catalog_chapters(book_id: str) -> list[dict]:
    """Load chapters from the chunks table when no chapters are supplied in the request."""
    rows = await find(
        "chunks",
        filters={"book_id": book_id},
        select="chunk_index, content",
        order="chunk_index ASC",
    )
    return [
        {"index": r["chunk_index"], "title": f"Chapter {r['chunk_index']}", "text": r["content"]}
        for r in rows
    ]


async def run_pipeline(req: PipelineReq) -> dict:
    """Execute the pipeline and return the full result dict."""

    started = time.time()
    cfg     = await get_all_config()          # live admin config — Supabase + settings fallback
    steps   = _resolve_steps(req.steps)
    errors: dict[str, str]   = {}
    step_status: dict[str, str] = {s: "skipped" for s in ALL_STEPS}

    # Ensure the FK target exists — chunk_summaries / book_summaries reference books(book_id).
    # custom_json requests for a book not in the catalog would otherwise 409.
    await _ensure_book_row(req)

    # ── Resolve chapters ──────────────────────────────────────────────────────
    if req.chapters:
        chapters = [{"index": c.index, "title": c.title, "text": c.text} for c in req.chapters]
    else:
        try:
            chapters = await _fetch_catalog_chapters(req.book_id)
        except Exception as e:
            return {
                "book_id":         req.book_id,
                "status":          "failed",
                "generated_at":    datetime.now(timezone.utc).isoformat(),
                "processing_time": "0s",
                "steps":           step_status,
                "errors":          {"catalog_fetch": str(e)},
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

        # Summary key e.g. "10min_en"
        summary_key = f"{req.options.length}_{req.language}"

        # ─────────────────────────────────────────────────────────────────────
        # STEP: audio_full
        # ─────────────────────────────────────────────────────────────────────
        full_audio: dict | None = None
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
                step_status["audio_full"] = "done"
            except Exception as e:
                errors["audio_full"] = str(e)
                step_status["audio_full"] = "failed"

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

        # ─────────────────────────────────────────────────────────────────────
        # STEP: mindmap
        # ─────────────────────────────────────────────────────────────────────
        mindmap_url: str | None = None
        mindmap_data: dict | None = None
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
                step_status["mindmap"] = "done"
            except Exception as e:
                errors["mindmap"] = str(e)
                step_status["mindmap"] = "failed"

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
        "chapters": chapter_results,
        "errors":   errors,
    }
