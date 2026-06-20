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
from pathlib import Path

log = logging.getLogger(__name__)

from api.config.settings import settings
from api.models.requests import PipelineReq, VALID_STEPS
from api.services.config.runtime import get_all_config
from api.services.db import find, upsert, update
from api.services.usage_logger import set_step
from api.services.summarizer.haiku import run_haiku_pass
from api.services.summarizer.sonnet import run_sonnet_pass_sync
from api.services.summarizer.review import run_review_pass
from api.services.summarizer.quality import score_summary_coverage
from api.services.summarizer.translate import translate_summary
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
from api.jobs.store import is_cancelled, get_step_results
from api.services.db import insert as db_insert

import re as _re

# ── Summary preamble stripper ─────────────────────────────────────────────────
# AI models sometimes add conversational openers ("Of course. Here is…") and
# script markers ("(Start of Audio Script)") before the actual summary text.
# This function removes them so the stored/displayed summary is clean.
_PREAMBLE_RE = _re.compile(
    r"^\s*("
    r"of course[.,\s]|certainly[.,\s]|sure[.,\s]|"
    r"here is |here's |below is |the following is |"
    r"i('ve| have) (created|written|prepared)|"
    r"as requested[.,\s]|"
    r"welcome[.,\s].*?presentation"
    r")",
    _re.IGNORECASE,
)
_SCRIPT_MARKER_RE = _re.compile(
    r"^\s*\(?\s*(start|end)\s+of\s+(audio\s+)?script\s*\)?\s*$",
    _re.IGNORECASE,
)


def _clean_summary(text: str) -> str:
    """Strip AI conversational preamble and script markers from a summary."""
    lines = text.splitlines()
    # Drop leading lines that are preamble, horizontal rules, or script markers
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if stripped in ("***", "---", "___", "* * *"):
            lines.pop(0)
            continue
        if _SCRIPT_MARKER_RE.match(stripped):
            lines.pop(0)
            continue
        if _PREAMBLE_RE.match(stripped):
            lines.pop(0)
            continue
        break
    # Drop trailing script markers / horizontal rules
    while lines:
        stripped = lines[-1].strip()
        if not stripped or stripped in ("***", "---", "___", "* * *"):
            lines.pop()
            continue
        if _SCRIPT_MARKER_RE.match(stripped):
            lines.pop()
            continue
        break
    return "\n".join(lines).strip()


# Single source of truth lives in api/models/requests.py
ALL_STEPS = VALID_STEPS


def _cfg_int(cfg: dict[str, str], key: str, default: int = 0) -> int:
    """Parse an integer config value; return `default` on missing/invalid."""
    try:
        return int(cfg.get(key) or default)
    except (TypeError, ValueError):
        return default


def _resolve_target_words(options: "PipelineOptions", cfg: dict[str, str], language: str) -> int | None:
    """
    Convert user length preset / custom char count to a target word count.
    Falls back to None when no preset is active, so callers can use legacy
    SUMMARY_MAX_WORDS_* or time-based length presets.
    """
    lang = (language or "en").lower()
    preset = (options.length_preset or "").strip().lower()

    if preset == "custom":
        if options.max_chars and options.max_chars > 0:
            return max(1, options.max_chars // 5)
        return None

    if preset in {"small", "medium", "large"}:
        key = f"SUMMARY_LENGTH_{preset.upper()}_{lang.upper()}"
        chars = _cfg_int(cfg, key)
        if chars > 0:
            return max(1, chars // 5)
        return None

    return None


class JobCancelledError(Exception):
    pass


def _has_translated_summary(previous_result: dict | None, source_lang: str) -> bool:
    """Return True if the previous result already contains a translated full summary."""
    if not previous_result or not isinstance(previous_result, dict):
        return False
    target_lang = "en" if (source_lang or "en") == "ar" else "ar"
    for asset in (previous_result.get("summaries") or {}).values():
        if asset.get("language") == target_lang and asset.get("text"):
            return True
    return False


def _has_translated_chapters(previous_result: dict | None, source_lang: str) -> bool:
    """Return True if the previous result already has translated chapter assets."""
    if not previous_result or not isinstance(previous_result, dict):
        return False
    target_lang = "en" if (source_lang or "en") == "ar" else "ar"
    chapters = previous_result.get("chapters") or []
    if not chapters:
        return False
    found = 0
    for ch in chapters:
        if ch.get("translated_summary") or ch.get(f"audio_{target_lang}") or ch.get(f"mindmap_{target_lang}_url"):
            found += 1
    # Consider it available if at least half the chapters have a translated asset.
    return found >= len(chapters) / 2


def _resolve_steps(
    requested: list[str] | None,
    previous_result: dict | None = None,
    source_lang: str = "en",
) -> set[str]:
    """Return the full set of steps to run, including auto-added dependencies."""
    # If requested is None or empty list, run all steps
    # If requested has specific steps, run only those (plus dependencies)
    # Note: empty list [] means "run all", explicit steps like ["cover"] means "run only cover"
    if requested is None or len(requested) == 0:
        return set(ALL_STEPS)
    steps = set(requested)

    # Steps that produce translated outputs.
    translate_steps = {
        "translate",
        "audio_full_translate", "audio_chapters_translate",
        "mindmap_translate", "mindmap_chapters_translate",
    }

    # Check if we already have summary data from a previous run. If so, the
    # 'summarize' dependency is satisfied and we must NOT re-add (and thus
    # regenerate) it — the existing summary is reused from previous_result.
    has_summary = False
    if previous_result:
        _prev = previous_result if isinstance(previous_result, dict) else {}
        _psums = _prev.get("summaries") or {}
        if _psums:
            _first_sum = next(iter(_psums.values()), {})
            has_summary = bool(_first_sum.get("text"))

    # Enforce dependencies. Only auto-add 'summarize' when we don't already
    # have a usable summary from the previous result.
    if not has_summary:
        needs_summary = ({"audio_full", "audio_chapters", "mindmap", "mindmap_chapters",
                          "inject_epub", "video"} | translate_steps) & steps
        if needs_summary:
            steps.add("summarize")

    # 'translate' is required for any translated-output step unless the previous
    # result already has the needed translated content.
    if ({"audio_full_translate", "mindmap_translate"} & steps) and not _has_translated_summary(previous_result, source_lang):
        steps.add("translate")
    if ({"audio_chapters_translate", "mindmap_chapters_translate"} & steps) and not _has_translated_chapters(previous_result, source_lang):
        steps.add("translate")

    if "alt_text" in steps:
        steps.add("cover")
    if "video" in steps:
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


async def _persist_book_details(book_id: str, fields: dict) -> None:
    """
    Best-effort write of computed book details to the `books` table.

    Some columns may not exist in every deployment's schema, and PostgREST
    rejects the WHOLE patch if any single column is unknown. So we try the full
    batch first, then fall back to writing each field on its own — this way the
    known columns still get saved even if one column name is wrong.
    """
    if not book_id.isdigit() or not fields:
        return
    clean = {k: v for k, v in fields.items() if v is not None}
    if not clean:
        return
    bid = int(book_id)
    try:
        await update("books", filters={"book_id": bid}, data=clean)
        return
    except Exception:
        pass  # fall back to per-field writes
    for k, v in clean.items():
        try:
            await update("books", filters={"book_id": bid}, data={k: v})
        except Exception as exc:
            log.debug("books.%s update skipped for %s: %s", k, book_id, exc)


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
    chunks: int | None = None,
) -> None:
    """Upsert into the audio table (en_url/ar_url + status + chapter chunk count)."""
    if not book_id.isdigit():
        return
    try:
        url_col    = "ar_url"    if language == "ar" else "en_url"
        status_col = "ar_status" if language == "ar" else "en_status"
        chunks_col = "ar_chunks" if language == "ar" else "en_chunks"
        payload: dict = {"book_id": int(book_id)}
        if audio_url:                       # don't overwrite an existing url with ""
            payload[url_col]    = audio_url
            payload[status_col] = "done"
        if chunks is not None:
            payload[chunks_col] = chunks
        if len(payload) > 1:
            await upsert("audio", payload, conflict="book_id")
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
        if r.get("content") and str(r["content"]).strip()
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


async def _resolve_book_title(book_id: str) -> str | None:
    """
    Try to resolve a human-readable title for a book_id.

    1. Look up the production catalog (`books` table).
    2. If the book_id is numeric, try Gutendex.
    3. Return None if nothing useful is found.
    """
    if not book_id:
        return None

    # 1. Catalog lookup
    try:
        bid = int(book_id)
        rows = await find(
            "books",
            filters={"book_id": bid},
            select="title",
            limit=1,
        )
        if rows:
            title = (rows[0].get("title") or "").strip()
            log.debug("_resolve_book_title catalog lookup for %s: %r", book_id, title)
            if title and title.lower() != book_id.lower():
                return title
    except (TypeError, ValueError):
        pass
    except Exception as exc:
        log.debug("title resolution catalog lookup failed: %s", exc)

    # 2. Gutenberg fallback (numeric IDs only)
    if book_id.isdigit():
        try:
            meta = await _fetch_gutenberg_metadata(book_id)
            title = (meta.get("title") or "").strip()
            log.debug("_resolve_book_title gutendex lookup for %s: %r", book_id, title)
            if title:
                return title
        except Exception as exc:
            log.debug("title resolution Gutenberg lookup failed: %s", exc)

    return None


def _pick_cached_summary(book_row: dict, language: str) -> str | None:
    """
    Return the best cached summary for the given language.

    IMPORTANT — order matters: we prefer the FULL summary over the short
    "10-minute / quick" version. This value becomes the book's `full_summary`,
    which is what the audio (TTS) and downstream steps narrate. Picking the
    10-min/quick text here would make the audio read the Quick Summary instead
    of the full Summary.

    Columns (catalog `books` table):
      English — summary_english   = full Summary        (preferred)
                summary_en_10min  = Quick / 10-min read (fallback)
      Arabic  — arabic_summary_v2 = full Summary v2     (preferred)
                arabic_summary    = full Summary v1      (fallback)
    """
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
        _nonempty(book_row.get("summary_english"))     # full Summary first
        or _nonempty(book_row.get("summary_en_10min"))  # quick/10-min fallback
    )


async def run_pipeline(
    req: PipelineReq,
    *,
    job_id: str | None = None,
    previous_result: dict | str | None = None,
    force_regenerate_summary: bool = False,
    forced_steps: set[str] | None = None,
) -> dict:
    """
    Execute the pipeline and return the full result dict.

    Phase 1  — summarize (sequential)
    Phase 2  — cover · audio_full · audio_chapters · mindmap · mindmap_chapters  (parallel)
    Phase 3  — alt_text · video  (parallel, depend on Phase-2 outputs)
    Phase 4  — inject_epub (runs LAST, after all other steps complete)

    forced_steps — steps the user explicitly asked to (re)generate. A step that
    already succeeded in `previous_result` is reused as-is UNLESS it is listed
    here, so completed work (e.g. a good cover) is never silently regenerated.
    """

    started = time.time()
    cfg     = await get_all_config()
    forced_steps = forced_steps or set()

    # Arabic tashkeel toggle: admin config default, overridable per request body.
    _tashkeel_cfg  = cfg.get("ARABIC_TASHKEEL_ENABLED", "true").lower() != "false"
    _tashkeel_body = req.options.arabic_tashkeel   # bool | None
    tashkeel_enabled = _tashkeel_body if _tashkeel_body is not None else _tashkeel_cfg
    
    # Parse previous_result for dependency resolution
    _prev_parsed = None
    if previous_result:
        _prev_parsed = previous_result if isinstance(previous_result, dict) else {}
        if isinstance(previous_result, str):
            try:
                _prev_parsed = json_module.loads(previous_result)
            except Exception:
                _prev_parsed = {}
    
    steps   = _resolve_steps(req.steps, _prev_parsed, source_lang=req.language)
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
    summary_qa:      dict | None = None   # {score, passed, threshold, missing, …}
    translated_summary: str        = ""   # summary translated into the other language
    translated_lang:    str        = ""   # "en" or "ar"
    translated_audio:   dict | None = None  # audio of the translated summary
    translated_mindmap_url:  str | None = None
    translated_mindmap_data: dict | None = None
    translated_mindmap_path_saved: str | None = None
    chapter_audio_translated: dict[int, str] = {}
    chapter_mindmap_translated: dict[int, dict] = {}

    # Flags populated from the previous result (if any).
    _prev_qa_failed: bool = False
    _prev_missing_topics: list[str] | None = None

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

        # Detect whether the previous run's summary QA failed so we can force
        # a fresh regeneration instead of reusing the same low-coverage summary.
        _prev_qa = _prev.get("summary_qa") or {}
        _prev_qa_failed = _prev_qa.get("passed") is False
        _prev_missing_topics = _prev_qa.get("missing") if _prev_qa_failed else None
        if _prev_qa_failed:
            log.info(
                "Previous summary QA failed (score=%s) — will force regeneration. "
                "Missing topics: %s",
                _prev_qa.get("score"), _prev_missing_topics,
            )

        cover_url   = cover_url   or _pfiles.get("cover")
        mindmap_url = mindmap_url or _pfiles.get("mindmap")
        epub_url    = epub_url    or _pfiles.get("epub")
        video_url   = video_url   or _pfiles.get("video")
        alt_text    = alt_text    or _pmeta.get("cover_alt_text")

        _lang_key = f"full_{req.language}"
        if _lang_key in _paudio and not full_audio:
            full_audio = _paudio[_lang_key]

        # Quick / full summary (needed if summarize is skipped this run).
        # Don't preload when QA failed or when summarize is force-requested.
        if not full_summary and _psums and not _prev_qa_failed and not force_regenerate_summary:
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

        # Translated full summary / audio / mindmap (so translate-* steps can reuse)
        target_lang = "en" if (req.language or "en") == "ar" else "ar"
        if not translated_summary:
            for asset in (_prev.get("summaries") or {}).values():
                if asset.get("language") == target_lang and asset.get("text"):
                    translated_summary = asset["text"]
                    translated_lang = target_lang
                    break
        if not translated_audio and f"full_{target_lang}" in _paudio:
            translated_audio = _paudio[f"full_{target_lang}"]
            translated_lang = target_lang
        _pmindmap_t = _prev.get(f"mindmap_{target_lang}") or {}
        if not translated_mindmap_url and _pmindmap_t.get("url"):
            translated_mindmap_url = _pmindmap_t["url"]
            translated_mindmap_data = _pmindmap_t.get("data")
            translated_lang = target_lang

        # Translated chapter audio & mindmaps
        for _fc in _pfiles.get("chapters") or []:
            _idx = _fc.get("index")
            if _idx is None:
                continue
            if _fc.get(f"audio_{target_lang}_url") and _idx not in chapter_audio_translated:
                chapter_audio_translated[_idx] = _fc[f"audio_{target_lang}_url"]
            if _fc.get(f"mindmap_{target_lang}_url") and _idx not in chapter_mindmap_translated:
                chapter_mindmap_translated[_idx] = {
                    "url":    _fc[f"mindmap_{target_lang}_url"],
                    "data":   None,
                    "format": "mermaid",
                }

    # ── Also load from pipeline_step_results table (more reliable than JSONB result) ──
    if job_id:
        try:
            _step_rows = await get_step_results(job_id)
            if _step_rows:
                log.info("Loaded %d step results from pipeline_step_results table", len(_step_rows))
                for row in _step_rows:
                    _step_name = row.get("step")
                    _step_status = row.get("status")
                    _output_url = row.get("output_url")
                    if not _step_name:
                        continue
                    # Only trust 'done' status from persisted step results
                    if _step_status == "done":
                        step_status[_step_name] = "done"
                        # Restore output URLs for each step type
                        if _step_name == "cover" and _output_url and not cover_url:
                            cover_url = _output_url
                        elif _step_name == "audio_full" and _output_url and not full_audio:
                            full_audio = {"url": _output_url}
                        elif _step_name == "audio_full_translate" and _output_url and not translated_audio:
                            translated_audio = {"url": _output_url}
                        elif _step_name == "mindmap" and _output_url and not mindmap_url:
                            mindmap_url = _output_url
                        elif _step_name == "mindmap_translate" and _output_url and not translated_mindmap_url:
                            translated_mindmap_url = _output_url
                        elif _step_name == "inject_epub" and _output_url and not epub_url:
                            epub_url = _output_url
                        elif _step_name == "video" and _output_url and not video_url:
                            video_url = _output_url
                        elif _step_name == "alt_text" and _output_url and not alt_text:
                            alt_text = _output_url
                log.info("Restored step statuses from DB: %s", 
                         {s: st for s, st in step_status.items() if st == "done"})
        except Exception as exc:
            log.debug("Could not load step results from DB: %s", exc)

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
                "summary_qa": summary_qa,
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

    async def _skip_step(name: str) -> None:
        """
        Mark a requested step as 'skipped' (disabled in config or missing input)
        and clear any stale error/result from a previous run, so a step turned
        off in the admin panel never lingers as 'failed' in the UI.
        """
        step_status[name] = "skipped"
        errors.pop(name, None)
        await _persist_step_result(job_id, name, "skipped")

    _DONE_SENTINEL = object()

    def _already_done(name: str, asset=_DONE_SENTINEL) -> bool:
        """
        True when `name` already succeeded in a previous run and the user did
        NOT explicitly force it this time — so we keep the existing output
        instead of regenerating it. When an `asset` handle is passed, it must
        also be truthy (guards against a 'done' status with a missing URL).
        """
        if name in forced_steps:
            return False
        if step_status.get(name) != "done":
            return False
        if asset is not _DONE_SENTINEL and not asset:
            return False
        return True

    # ── Production catalog enrichment ─────────────────────────────────────────
    # Treat a title that is just the book_id as missing — some callers send the
    # id in the title field, which would otherwise get stamped onto the cover.
    _title_is_missing = not req.title or str(req.title).strip() == str(req.book_id).strip()
    if _title_is_missing or not req.author or not req.summary:
        book_row = await _fetch_book_from_catalog(req.book_id)
        if book_row:
            updates: dict = {}
            catalog_title = book_row.get("title", "")
            if _title_is_missing and catalog_title and catalog_title != req.book_id:
                updates["title"] = catalog_title
            if not req.author and book_row.get("author"):
                updates["author"] = book_row["author"]
            if not req.summary and not _prev_qa_failed and not force_regenerate_summary:
                # Skip loading cached summary when QA previously failed OR when
                # summarize was explicitly force-requested — regenerate fresh.
                cached = _pick_cached_summary(book_row, req.language)
                if cached:
                    updates["summary"] = cached
            if not req.pages and book_row.get("pages"):
                updates["pages"] = book_row["pages"]
            if updates:
                req = req.model_copy(update=updates)

    _title_is_missing = not req.title or str(req.title).strip() == str(req.book_id).strip()
    if req.book_id and req.book_id.isdigit() and _title_is_missing:
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

    # Per-step on/off toggles from Admin → Settings → Pipeline Steps.
    # A step disabled here is removed from the run set and marked skipped.
    step_enabled: dict[str, bool] = {
        "summarize":                  cfg.get("PIPELINE_STEP_SUMMARIZE", "true") == "true",
        "translate":                  cfg.get("PIPELINE_STEP_TRANSLATE", "true") == "true",
        "audio_full":                 (cfg.get("PIPELINE_STEP_AUDIO_FULL", "true") == "true") and tts_enabled,
        "audio_chapters":             (cfg.get("PIPELINE_STEP_AUDIO_CHAPTERS", "true") == "true") and tts_enabled,
        "audio_full_translate":       (cfg.get("PIPELINE_STEP_AUDIO_FULL_TRANSLATE", "true") == "true") and tts_enabled,
        "audio_chapters_translate":   (cfg.get("PIPELINE_STEP_AUDIO_CHAPTERS_TRANSLATE", "true") == "true") and tts_enabled,
        "cover":                      (cfg.get("PIPELINE_STEP_COVER", "true") == "true") and cover_enabled,
        "alt_text":                   (cfg.get("PIPELINE_STEP_ALTTEXT", "true") == "true") and alttext_enabled,
        "mindmap":                    (cfg.get("PIPELINE_STEP_MINDMAP", "true") == "true") and mindmap_enabled,
        "mindmap_chapters":           (cfg.get("PIPELINE_STEP_MINDMAP", "true") == "true") and mindmap_enabled,
        "mindmap_translate":          (cfg.get("PIPELINE_STEP_MINDMAP_TRANSLATE", "true") == "true") and mindmap_enabled,
        "mindmap_chapters_translate": (cfg.get("PIPELINE_STEP_MINDMAP_CHAPTERS_TRANSLATE", "true") == "true") and mindmap_enabled,
        "inject_epub":                (cfg.get("PIPELINE_STEP_INJECT_EPUB", "true") == "true") and epub_enabled,
        "video":                      (cfg.get("PIPELINE_STEP_VIDEO", "true") == "true") and video_enabled,
    }

    # Apply admin-level step toggles: disabled steps are removed from the run
    # set and explicitly marked skipped (this also clears stale errors on rerun).
    for _s in list(steps):
        if not step_enabled.get(_s, True):
            await _skip_step(_s)
            steps.discard(_s)
    if steps:
        log.info("Job %s: enabled steps after admin toggles: %s", job_id, sorted(steps))

    # Per-language summary length + chapter-summary length overrides (0 = use preset)
    _lang_up         = (req.language or "en").upper()
    _target_words    = _resolve_target_words(req.options, cfg, req.language)
    summary_max_words = _target_words or _cfg_int(cfg, f"SUMMARY_MAX_WORDS_{_lang_up}")
    chapter_max_words = _target_words if _target_words else _cfg_int(cfg, "CHAPTER_SUMMARY_MAX_WORDS")
    # Summary QA / coverage gating
    qa_enabled       = cfg.get("SUMMARY_QA_ENABLED", "true") == "true"
    qa_model         = cfg.get("SUMMARY_QA_MODEL") or "deepseek/deepseek-chat"
    qa_threshold     = _cfg_int(cfg, "SUMMARY_QA_THRESHOLD") or 70
    # Cross-language translation + optional target-language audio.
    # Default to a cheap, capable model — translation is a low-complexity task
    # and Sonnet here cost ~$0.11/call (≈$16 on a 144-chapter book). gpt-4.1-mini
    # via OpenRouter does it for a fraction of a cent. Admins can override with
    # the TRANSLATE_MODEL config key.
    translate_enabled   = cfg.get("TRANSLATE_SUMMARY_ENABLED", "true") == "true"
    translate_model     = cfg.get("TRANSLATE_MODEL") or "openai/gpt-4.1-mini"
    # Per-chapter translation is OFF by default — it costs O(chapters) model
    # calls (144 on a big book) to produce translated chapter audio/mindmaps
    # that are rarely consumed. Only the FINAL summary is translated unless an
    # admin explicitly opts in via TRANSLATE_CHAPTERS_ENABLED=true.
    translate_chapters_enabled = cfg.get("TRANSLATE_CHAPTERS_ENABLED", "false") == "true"
    target_lang         = "en" if (req.language or "en") == "ar" else "ar"

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
                    
                    # Try to load per-chapter summaries from DB first
                    _db_loaded = False
                    if chapters and req.book_id and req.book_id.isdigit():
                        try:
                            chunk_rows = await find(
                                "chunks",
                                filters={"book_id": req.book_id},
                                select="chunk_index, summary",
                                order="chunk_index ASC",
                            )
                            chunk_sum_map = {
                                r["chunk_index"]: r.get("summary") or ""
                                for r in chunk_rows
                                if r.get("summary") and str(r["summary"]).strip()
                            }
                            if chunk_sum_map:
                                chapter_results = [
                                    {
                                        "index":         ch["index"],
                                        "title":         ch["title"],
                                        "summary":       chunk_sum_map.get(ch["index"], ""),
                                        "read_time_min": max(1, len((chunk_sum_map.get(ch["index"]) or "").split()) // 200),
                                    }
                                    for ch in chapters
                                    if ch["index"] in chunk_sum_map
                                ]
                                log.info("Loaded %d chapter summaries from chunks for cached-summary job", len(chapter_results))
                                _db_loaded = True
                        except Exception as exc:
                            log.debug("Could not load chunk summaries for cached summary: %s", exc)
                    
                    # If no DB summaries, generate them from chapter text
                    if not _db_loaded and chapters:
                        log.info("No DB chapter summaries found — generating from chapter text")
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
                                        max_words=chapter_max_words or None,
                                        tashkeel_enabled=tashkeel_enabled,
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
                        log.info("Generated %d chapter summaries from text", len(chapter_results))
                        # Save back to DB for future use
                        await _persist_chunk_summaries(req.book_id, chapter_results)

                    # Determine actual summarize status based on valid summaries
                    _valid = [c for c in chapter_results if c.get("summary") and str(c["summary"]).strip()]
                    if not _valid and chapters:
                        step_status["summarize"] = "failed"
                        errors["summarize"] = (
                            "All chapter summaries are empty. "
                            "The source text may be missing, unextractable, or the AI model failed."
                        )
                    elif len(_valid) < len(chapters):
                        step_status["summarize"] = "partial"
                        errors["summarize"] = (
                            f"{len(chapters) - len(_valid)} of {len(chapters)} chapters failed to summarize. "
                            f"Audio is blocked until all chapters complete. "
                            f"Retry — only the failed chapters will be re-processed."
                        )
                    else:
                        step_status["summarize"] = "done"
                else:
                    haiku_conc = max(1, int(cfg.get("HAIKU_CONCURRENCY", "6")))
                    sem = asyncio.Semaphore(haiku_conc)

                    # ── Smart resume: reload already-summarized chapters ───────
                    # On retry, we skip chapters that already have a summary so
                    # only the chapters that actually failed are re-processed.
                    _existing: dict[int, str] = {}

                    # 1) From previous pipeline result (any book_id type)
                    for _pc in (_prev_parsed or {}).get("chapters", []):
                        _pi  = _pc.get("index")
                        _ps  = (_pc.get("summary") or "").strip()
                        if _pi is not None and _ps:
                            _existing[_pi] = _ps

                    # 2) From DB chunks table (numeric book_ids only)
                    if req.book_id.isdigit() and not _existing:
                        try:
                            _dbrows = await find(
                                "chunks",
                                filters={"book_id": req.book_id},
                                select="chunk_index,summary",
                                order="chunk_index ASC",
                            )
                            for _r in (_dbrows or []):
                                _pi = _r.get("chunk_index")
                                _ps = (_r.get("summary") or "").strip()
                                if _pi is not None and _ps:
                                    _existing[_pi] = _ps
                            if _existing:
                                log.info(
                                    "Smart resume: loaded %d chapter summaries from DB",
                                    len(_existing),
                                )
                        except Exception as _exc:
                            log.debug("Could not preload chunk summaries from DB: %s", _exc)

                    _chapters_to_run = [ch for ch in chapters if ch["index"] not in _existing]
                    if _existing:
                        log.info(
                            "Smart resume: %d/%d chapters already done, running %d",
                            len(_existing), len(chapters), len(_chapters_to_run),
                        )

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
                                    max_words=chapter_max_words or None,
                                    tashkeel_enabled=tashkeel_enabled,
                                )
                            except Exception as exc:
                                log.warning("Haiku pass failed for chunk %s: %s", ch["index"], exc)
                                sums = []
                            summary = sums[0] if sums else ""
                            return {
                                "index":         ch["index"],
                                "title":         ch["title"],
                                "summary":       summary,
                                "read_time_min": max(1, len(summary.split()) // 200) if summary else 1,
                            }

                    # Pre-fill from cache, then run remaining chapters
                    chapter_results = [
                        {
                            "index":         _pi,
                            "title":         next(
                                (c["title"] for c in chapters if c["index"] == _pi),
                                f"Chapter {_pi + 1}",
                            ),
                            "summary":       _ps,
                            "read_time_min": max(1, len(_ps.split()) // 200),
                        }
                        for _pi, _ps in _existing.items()
                    ]
                    if _chapters_to_run:
                        _new = list(await asyncio.gather(
                            *[_summarize_chunk(ch) for ch in _chapters_to_run]
                        ))
                        chapter_results.extend(_new)
                    chapter_results.sort(key=lambda c: c["index"])

                    chunk_summaries = [c["summary"] for c in chapter_results if c.get("summary")]

                    if chunk_summaries:
                        full_summary = await run_sonnet_pass_sync(
                            chunk_summaries,
                            req.options.length,
                            req.options.style,
                            req.language,
                            model_override=model_sonnet,
                            max_words=summary_max_words or None,
                            missing_topics=_prev_missing_topics,
                            tashkeel_enabled=tashkeel_enabled,
                        )
                        full_summary = await run_review_pass(
                            full_summary,
                            req.options.length,
                            req.options.style,
                            req.language,
                            model=model_haiku,
                            max_words=summary_max_words or None,
                        )
                        full_summary = _clean_summary(full_summary)
                        sentences = [s.strip() for s in full_summary.replace(".\n", ". ").split(". ") if s.strip()]
                        quick_summary = ". ".join(sentences[:2]) + "." if sentences else full_summary[:200]

                    # Determine actual summarize status based on valid summaries
                    _valid = [c for c in chapter_results if c.get("summary") and str(c["summary"]).strip()]
                    if not _valid and chapters:
                        step_status["summarize"] = "failed"
                        errors["summarize"] = (
                            "All chapter summaries are empty. "
                            "The source text may be missing, unextractable, or the AI model failed."
                        )
                    elif len(_valid) < len(chapters):
                        step_status["summarize"] = "partial"
                        errors["summarize"] = (
                            f"{len(chapters) - len(_valid)} of {len(chapters)} chapters failed to summarize. "
                            f"Audio is blocked until all chapters complete. "
                            f"Retry — only the failed chapters will be re-processed."
                        )
                    else:
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
                # Write real `books` columns: status + the book's total word count
                # (sum of chapter source text). Only columns that exist in the
                # schema are written.
                _total_words = sum(len((c.get("text") or "").split()) for c in chapters)
                await _persist_book_details(req.book_id, {
                    "status":         "summarized",
                    "totalwordcount": _total_words or None,
                })
            await _checkpoint()

        # ═════════════════════════════════════════════════════════════════════
        # SUMMARY QA — score how well the full summary covers the whole book.
        # Gates audio generation: audio steps only run when score ≥ threshold.
        # ═════════════════════════════════════════════════════════════════════
        if qa_enabled and full_summary and chapter_results:
            try:
                _chap_notes = [c.get("summary", "") for c in chapter_results if c.get("summary")]
                summary_qa = await score_summary_coverage(
                    full_summary, _chap_notes, req.language,
                    model=qa_model, threshold=qa_threshold,
                )
                summary_qa["threshold"] = qa_threshold
                log.info(
                    "Summary QA: score=%s passed=%s (threshold=%s) model=%s",
                    summary_qa.get("score"), summary_qa.get("passed"),
                    qa_threshold, qa_model,
                )
            except Exception as exc:
                log.warning("Summary QA failed (%s) — allowing audio to proceed", exc)
                summary_qa = {"score": -1, "passed": True, "threshold": qa_threshold,
                              "reason": f"QA error: {exc}", "missing": []}
            await _checkpoint()

        # Audio is blocked when QA explicitly failed OR when summarize is partial
        # (some chapters are still missing — audio would contain incomplete stories).
        _summarize_partial = step_status.get("summarize") == "partial"
        audio_blocked = (
            bool(summary_qa and summary_qa.get("passed") is False)
            or _summarize_partial
        )
        if _summarize_partial and not errors.get("audio_blocked"):
            errors["audio_blocked"] = (
                "Audio blocked: some chapters failed to summarize. "
                "Retry the job — only failed chapters will be re-processed."
            )

        # ═════════════════════════════════════════════════════════════════════
        # TRANSLATION — produce the summary in the OTHER language too.
        # Always runs when the legacy TRANSLATE_SUMMARY_ENABLED flag is true.
        # When the explicit "translate" step is requested, we also translate
        # every chapter summary so downstream target-language audio/mindmaps
        # can be generated.
        # ═════════════════════════════════════════════════════════════════════
        translate_step_requested = "translate" in steps
        # Reuse an existing translation instead of regenerating it, unless the
        # user explicitly forced translate OR the summary was (re)generated this
        # pass (in which case the old translation is stale).
        _translate_already_done = (
            "translate" not in forced_steps
            and "summarize" not in steps
            and bool(translated_summary)
        )
        if _translate_already_done and translate_step_requested:
            step_status["translate"] = "done"
            log.info("translate already done — reusing existing translation (not forced)")
        if (translate_enabled or translate_step_requested) and full_summary and not audio_blocked and not _translate_already_done:
            if translate_step_requested:
                step_status["translate"] = "running"
                set_step("translate")
                await _checkpoint()
            try:
                translated_summary = await translate_summary(
                    full_summary, req.language, target_lang, model=translate_model,
                    tashkeel_enabled=tashkeel_enabled,
                )
                if translated_summary:
                    translated_lang = target_lang
                    log.info("Translated summary %s→%s (%d words)",
                             req.language, target_lang, len(translated_summary.split()))
                    # Persist the translated summary to the books table.
                    await _persist_book_summary(
                        req.book_id, target_lang, req.options.length, req.options.style,
                        translated_summary, translate_model,
                    )
                    if translate_step_requested:
                        step_status["translate"] = "done"
            except Exception as exc:
                log.warning("translation step failed: %s", exc)
                if translate_step_requested:
                    step_status["translate"] = "failed"
                    errors["translate"] = str(exc)
            if translate_step_requested:
                _t = round(time.time() - started)
                await _persist_step_result(job_id, "translate", step_status["translate"],
                                           error_msg=errors.get("translate"), duration_sec=_t)
            await _checkpoint()

        # Translate per-chapter summaries only when (a) an admin has explicitly
        # opted into chapter translation AND (b) a step that actually uses them is
        # requested (audio_chapters_translate or mindmap_chapters_translate).
        # Translating chapters costs O(N) model calls on large books (144 on a
        # 144-chapter book) while producing output that is rarely consumed — so
        # this is OFF by default and must be turned on deliberately.
        _needs_chapter_translations = translate_chapters_enabled and bool(
            steps & {"audio_chapters_translate", "mindmap_chapters_translate"}
        )
        if translate_step_requested and translated_summary and chapter_results and _needs_chapter_translations:
            try:
                # Bound concurrency — firing one translate call per chapter all at
                # once (200+ on a big book) hammers the provider's rate limit and
                # the resulting 429 retry-storm can stall the step for many minutes.
                tr_conc = max(1, int(cfg.get("TRANSLATE_CONCURRENCY", "8")))
                tr_sem  = asyncio.Semaphore(tr_conc)

                async def _translate_chapter(ch: dict) -> None:
                    if not ch.get("summary"):
                        return
                    async with tr_sem:
                        tr = await translate_summary(
                            ch["summary"], req.language, target_lang, model=translate_model,
                            tashkeel_enabled=tashkeel_enabled,
                        )
                    if tr:
                        ch["translated_summary"] = tr

                await asyncio.gather(*[_translate_chapter(ch) for ch in chapter_results])
                _tr_count = sum(1 for ch in chapter_results if ch.get("translated_summary"))
                log.info("Translated %d/%d chapter summaries %s→%s",
                         _tr_count, len(chapter_results), req.language, target_lang)
                if step_status.get("translate") == "done" and _tr_count < len(chapter_results):
                    step_status["translate"] = "partial"
            except Exception as exc:
                log.warning("chapter translation failed: %s", exc)
                if step_status.get("translate") == "done":
                    step_status["translate"] = "partial"
                errors["translate_chapters"] = str(exc)
            await _checkpoint()

        # ═════════════════════════════════════════════════════════════════════
        # PHASE 2 — PARALLEL: cover · audio_full · audio_chapters ·
        #                      mindmap · mindmap_chapters
        # All of these are independent of each other (only need summarize done).
        # ═════════════════════════════════════════════════════════════════════

        # ── cover ─────────────────────────────────────────────────────────────
        async def _do_cover() -> None:
            nonlocal cover_url
            if "cover" not in steps:
                return
            if _already_done("cover", cover_url):
                log.info("cover already done — reusing existing image (not forced)")
                return
            if not cover_enabled:
                await _skip_step("cover")
                return
            step_status["cover"] = "running"
            set_step("cover")
            await _checkpoint()
            try:
                # Never stamp the raw book_id onto a cover. If the request didn't
                # include a real title, try to fetch one from the catalog/Gutenberg.
                cover_title = req.title
                if not cover_title or cover_title.strip() == str(req.book_id).strip():
                    cover_title = await _resolve_book_title(req.book_id)
                if not cover_title:
                    cover_title = "Untitled"

                log.info("Generating cover for book_id=%s with title=%r", req.book_id, cover_title)
                await generate_cover(
                    cover_title,
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
                await _persist_cover(req.book_id, cover_title, cover_url)
            await _checkpoint()

        # ── audio_full ────────────────────────────────────────────────────────
        async def _do_audio_full() -> None:
            nonlocal full_audio, full_audio_path, translated_audio
            if "audio_full" not in steps:
                return
            if _already_done("audio_full", full_audio):
                log.info("audio_full already done — reusing existing audio (not forced)")
                return
            if not tts_enabled or not full_summary:
                await _skip_step("audio_full")
                return
            if audio_blocked:
                step_status["audio_full"] = "failed"
                errors["audio_full"] = (
                    f"Blocked: summary coverage {summary_qa.get('score')}% is below the "
                    f"required {qa_threshold}%. Improve the summary, then retry."
                )
                await _persist_step_result(job_id, "audio_full", "failed",
                                           error_msg=errors["audio_full"])
                await _checkpoint()
                return
            step_status["audio_full"] = "running"
            set_step("audio_full")
            await _checkpoint()
            try:
                raw  = os.path.join(tmp, "audio_raw.mp3")
                proc = os.path.join(tmp, "audio.mp3")
                await synthesize(full_summary, req.language, raw, cfg, audio_style=req.options.audio_style)
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
                from api.services.pipeline.watermark import stamp_audio  # noqa: PLC0415
                stamp_audio(src, cfg)
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
            nonlocal chapter_audio, chapter_results, chapter_audio_translated
            if "audio_chapters" not in steps:
                return
            if _already_done("audio_chapters"):
                log.info("audio_chapters already done — reusing existing audio (not forced)")
                return
            if not tts_enabled or not chapter_results:
                await _skip_step("audio_chapters")
                return
            if audio_blocked:
                step_status["audio_chapters"] = "failed"
                errors["audio_chapters"] = (
                    f"Blocked: summary coverage {summary_qa.get('score')}% is below the "
                    f"required {qa_threshold}%. Improve the summary, then retry."
                )
                await _persist_step_result(job_id, "audio_chapters", "failed",
                                           error_msg=errors["audio_chapters"])
                await _checkpoint()
                return
            step_status["audio_chapters"] = "running"
            set_step("audio_chapters")
            await _checkpoint()
            ch_errors = 0
            ch_processed = 0

            # Create a map to update chapter_results by index
            audio_key = f"audio_{req.language}"

            chapters_with_summary = [ch for ch in chapter_results if ch.get("summary")]
            ch_skipped_no_summary = len(chapter_results) - len(chapters_with_summary)
            if ch_skipped_no_summary > 0:
                log.warning("%d chapters skipped for audio: no summaries available", ch_skipped_no_summary)

            # Run chapter TTS in parallel (bounded) — a 200-chapter book was
            # taking 30+ min when this loop was sequential.
            audio_conc = max(1, int(cfg.get("AUDIO_CONCURRENCY", "4")))
            audio_sem  = asyncio.Semaphore(audio_conc)

            async def _one_chapter_audio(ch: dict) -> tuple[int, str | None, str | None]:
                idx = ch["index"]
                async with audio_sem:
                    try:
                        ch_raw  = os.path.join(tmp, f"ch{idx}_raw.mp3")
                        ch_proc = os.path.join(tmp, f"ch{idx}.mp3")
                        await synthesize(ch["summary"], req.language, ch_raw, cfg, audio_style=req.options.audio_style)
                        if audio_proc_enabled:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None, process_audio, ch_raw, ch_proc,
                                ch["title"], req.author or "",
                            )
                            src = ch_proc
                        else:
                            src = ch_raw
                        from api.services.pipeline.watermark import stamp_audio as _sa  # noqa: PLC0415
                        _sa(src, cfg)
                        k = f"books/{req.book_id}/chapters/ch_{idx:02d}_{req.language}.mp3"
                        audio_url = upload_file(src, k, CONTENT_TYPES[".mp3"])
                        chapter_audio[idx] = audio_url
                        ch[audio_key] = audio_url   # for dashboard display
                        await _checkpoint()         # live progress (also detects cancel)
                        return idx, audio_url, None
                    except JobCancelledError:
                        raise
                    except Exception as e:
                        return idx, None, str(e)

            _audio_results = await asyncio.gather(
                *[_one_chapter_audio(ch) for ch in chapters_with_summary]
            )
            for idx, audio_url, err in _audio_results:
                if audio_url:
                    ch_processed += 1
                if err is not None:
                    ch_errors += 1
                    errors[f"audio_chapter_{idx}"] = err

            # Calculate status based on actual processing, not just chapter count
            if ch_errors > 0 and ch_processed == 0:
                step_status["audio_chapters"] = "failed"
                # Collect first error message as the step-level error for display
                first_err = next(
                    (v for k, v in errors.items() if k.startswith("audio_chapter_")), None
                )
                errors["audio_chapters"] = (
                    f"All {ch_errors} chapter(s) failed. "
                    + (f"First error: {first_err}" if first_err else "Check logs for details.")
                )
            elif ch_errors > 0:
                step_status["audio_chapters"] = "partial"
                errors["audio_chapters"] = f"{ch_errors} of {ch_errors + ch_processed} chapter(s) failed"
            elif ch_processed > 0:
                step_status["audio_chapters"] = "done"
            elif ch_skipped_no_summary > 0:
                step_status["audio_chapters"] = "failed"
                errors["audio_chapters"] = f"All {ch_skipped_no_summary} chapters skipped: no summaries available"
            else:
                step_status["audio_chapters"] = "skipped"
            
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "audio_chapters", step_status["audio_chapters"], duration_sec=_t)
            # Record the chapter-audio count on the audio table (en_chunks/ar_chunks).
            if chapter_audio:
                await _persist_audio(req.book_id, req.language,
                                     (full_audio or {}).get("url") or "",
                                     chunks=len(chapter_audio))
            await _checkpoint()

        # ── audio_full_translate ──────────────────────────────────────────────
        async def _do_audio_full_translate() -> None:
            nonlocal translated_audio
            if "audio_full_translate" not in steps:
                return
            if _already_done("audio_full_translate", translated_audio):
                log.info("audio_full_translate already done — reusing existing audio (not forced)")
                return
            if not tts_enabled or not translated_summary or not translated_lang:
                await _skip_step("audio_full_translate")
                return
            if audio_blocked:
                step_status["audio_full_translate"] = "failed"
                errors["audio_full_translate"] = (
                    f"Blocked: summary coverage {summary_qa.get('score')}% is below the "
                    f"required {qa_threshold}%."
                )
                await _persist_step_result(job_id, "audio_full_translate", "failed",
                                           error_msg=errors["audio_full_translate"])
                await _checkpoint()
                return
            step_status["audio_full_translate"] = "running"
            set_step("audio_full_translate")
            await _checkpoint()
            try:
                t_raw  = os.path.join(tmp, "audio_t_raw.mp3")
                t_proc = os.path.join(tmp, "audio_t.mp3")
                await synthesize(translated_summary, translated_lang, t_raw, cfg, audio_style=req.options.audio_style)
                if audio_proc_enabled:
                    loop = asyncio.get_event_loop()
                    t_meta = await loop.run_in_executor(
                        None, process_audio, t_raw, t_proc,
                        req.title or req.book_id, req.author or "",
                    )
                    t_src = t_proc
                else:
                    t_meta = {}
                    t_src = t_raw
                from api.services.pipeline.watermark import stamp_audio as _stamp_audio  # noqa: PLC0415
                _stamp_audio(t_src, cfg)
                t_key = f"books/{req.book_id}/audio_{translated_lang}_{req.options.length}.mp3"
                t_url = upload_file(t_src, t_key, CONTENT_TYPES[".mp3"])
                translated_audio = {
                    "url":      t_url,
                    "duration": _fmt_audio_duration(t_meta.get("duration_seconds")),
                    "size_mb":  t_meta.get("size_mb"),
                }
                await _persist_audio(req.book_id, translated_lang, t_url)
                step_status["audio_full_translate"] = "done"
                log.info("Target-language audio generated (%s)", translated_lang)
            except JobCancelledError:
                raise
            except Exception as e:
                errors["audio_full_translate"] = str(e)
                step_status["audio_full_translate"] = "failed"
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "audio_full_translate", step_status["audio_full_translate"],
                                       output_url=(translated_audio or {}).get("url"), duration_sec=_t)
            await _checkpoint()

        # ── audio_chapters_translate ──────────────────────────────────────────
        async def _do_audio_chapters_translate() -> None:
            nonlocal chapter_audio_translated, chapter_results
            if "audio_chapters_translate" not in steps:
                return
            if _already_done("audio_chapters_translate"):
                log.info("audio_chapters_translate already done — reusing existing audio (not forced)")
                return
            if not tts_enabled or not chapter_results or not translated_summary or not translated_lang:
                await _skip_step("audio_chapters_translate")
                return
            if audio_blocked:
                step_status["audio_chapters_translate"] = "failed"
                errors["audio_chapters_translate"] = (
                    f"Blocked: summary coverage {summary_qa.get('score')}% is below the "
                    f"required {qa_threshold}%."
                )
                await _persist_step_result(job_id, "audio_chapters_translate", "failed",
                                           error_msg=errors["audio_chapters_translate"])
                await _checkpoint()
                return
            step_status["audio_chapters_translate"] = "running"
            set_step("audio_chapters_translate")
            await _checkpoint()
            ch_errors = 0
            ch_processed = 0
            target_audio_key = f"audio_{translated_lang}"

            chapters_with_translated = [ch for ch in chapter_results if ch.get("translated_summary")]

            audio_conc = max(1, int(cfg.get("AUDIO_CONCURRENCY", "4")))
            audio_sem  = asyncio.Semaphore(audio_conc)

            async def _one_chapter_audio_t(ch: dict) -> tuple[int, str | None, str | None]:
                idx = ch["index"]
                async with audio_sem:
                    try:
                        ch_t_raw  = os.path.join(tmp, f"ch{idx}_{translated_lang}_raw.mp3")
                        ch_t_proc = os.path.join(tmp, f"ch{idx}_{translated_lang}.mp3")
                        await synthesize(ch["translated_summary"], translated_lang, ch_t_raw, cfg, audio_style=req.options.audio_style)
                        if audio_proc_enabled:
                            loop = asyncio.get_event_loop()
                            await loop.run_in_executor(
                                None, process_audio, ch_t_raw, ch_t_proc,
                                ch["title"], req.author or "",
                            )
                            src = ch_t_proc
                        else:
                            src = ch_t_raw
                        from api.services.pipeline.watermark import stamp_audio as _sa_t  # noqa: PLC0415
                        _sa_t(src, cfg)
                        k = f"books/{req.book_id}/chapters/ch_{idx:02d}_{translated_lang}.mp3"
                        audio_url = upload_file(src, k, CONTENT_TYPES[".mp3"])
                        chapter_audio_translated[idx] = audio_url
                        ch[target_audio_key] = audio_url
                        await _checkpoint()
                        return idx, audio_url, None
                    except JobCancelledError:
                        raise
                    except Exception as e:
                        return idx, None, str(e)

            _t_results = await asyncio.gather(
                *[_one_chapter_audio_t(ch) for ch in chapters_with_translated]
            )
            for idx, audio_url, err in _t_results:
                if audio_url:
                    ch_processed += 1
                if err is not None:
                    ch_errors += 1
                    errors[f"audio_chapter_{idx}_{translated_lang}"] = err

            if ch_errors > 0 and ch_processed == 0:
                step_status["audio_chapters_translate"] = "failed"
            elif ch_errors > 0:
                step_status["audio_chapters_translate"] = "partial"
                errors["audio_chapters_translate"] = f"{ch_errors} chapter(s) failed"
            elif ch_processed > 0:
                step_status["audio_chapters_translate"] = "done"
            else:
                step_status["audio_chapters_translate"] = "skipped"

            _t = round(time.time() - started)
            await _persist_step_result(job_id, "audio_chapters_translate",
                                       step_status["audio_chapters_translate"], duration_sec=_t)
            await _checkpoint()

        # ── mindmap ───────────────────────────────────────────────────────────
        async def _do_mindmap() -> None:
            nonlocal mindmap_url, mindmap_data, mindmap_path_saved
            nonlocal translated_mindmap_url, translated_mindmap_data, translated_mindmap_path_saved
            if "mindmap" not in steps:
                return
            if _already_done("mindmap", mindmap_url):
                log.info("mindmap already done — reusing existing mindmap (not forced)")
                return
            if not mindmap_enabled or not full_summary:
                await _skip_step("mindmap")
                return
            step_status["mindmap"] = "running"
            set_step("mindmap")
            await _checkpoint()
            try:
                from api.services.pipeline.watermark import stamp_mindmap_json, stamp_mindmap_mermaid  # noqa: PLC0415
                if mindmap_format == "json":
                    mindmap_data = await generate_json_mindmap(
                        req.title or req.book_id, full_summary, req.language, model=model_mindmap
                    )
                    mindmap_data = stamp_mindmap_json(mindmap_data, cfg)
                    json_path = os.path.join(tmp, "mindmap.json")
                    with open(json_path, "w", encoding="utf-8") as f:
                        json_module.dump(mindmap_data, f, ensure_ascii=False)
                    key = f"books/{req.book_id}/mindmap.json"
                    mindmap_url = upload_file(json_path, key, CONTENT_TYPES[".json"])
                else:
                    mermaid = await generate_mermaid_code(
                        req.title or req.book_id, full_summary, req.language, model=model_mindmap
                    )
                    mermaid = stamp_mindmap_mermaid(mermaid, cfg)
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
            nonlocal chapter_mindmap, chapter_results, chapter_mindmap_translated
            if "mindmap_chapters" not in steps:
                return
            if _already_done("mindmap_chapters"):
                log.info("mindmap_chapters already done — reusing existing mindmaps (not forced)")
                return
            if not mindmap_enabled or not chapter_results:
                await _skip_step("mindmap_chapters")
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
                        from api.services.pipeline.watermark import stamp_mindmap_json, stamp_mindmap_mermaid  # noqa: PLC0415
                        if mindmap_format == "json":
                            data = await generate_json_mindmap(
                                ch_title, ch_text, req.language, model=model_mindmap
                            )
                            data = stamp_mindmap_json(data, cfg)
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
                            mermaid = stamp_mindmap_mermaid(mermaid, cfg)
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
                first_err = next(
                    (v for k, v in errors.items() if k.startswith("mindmap_chapter_")), None
                )
                errors["mindmap_chapters"] = (
                    f"All {ch_errors} chapter(s) failed. "
                    + (f"First error: {first_err}" if first_err else "Check logs for details.")
                )
            elif ch_errors > 0:
                step_status["mindmap_chapters"] = "partial"
                errors["mindmap_chapters"] = f"{ch_errors} of {ch_errors + ch_processed} chapter(s) failed"
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

        # ── mindmap_translate ─────────────────────────────────────────────────
        async def _do_mindmap_translate() -> None:
            nonlocal translated_mindmap_url, translated_mindmap_data, translated_mindmap_path_saved
            if "mindmap_translate" not in steps:
                return
            if _already_done("mindmap_translate", translated_mindmap_url):
                log.info("mindmap_translate already done — reusing existing mindmap (not forced)")
                return
            if not mindmap_enabled or not translated_summary or not translated_lang:
                await _skip_step("mindmap_translate")
                return
            step_status["mindmap_translate"] = "running"
            set_step("mindmap_translate")
            await _checkpoint()
            try:
                from api.services.pipeline.watermark import stamp_mindmap_json, stamp_mindmap_mermaid  # noqa: PLC0415
                if mindmap_format == "json":
                    t_data = await generate_json_mindmap(
                        req.title or req.book_id, translated_summary, translated_lang, model=model_mindmap
                    )
                    t_data = stamp_mindmap_json(t_data, cfg)
                    t_json_path = os.path.join(tmp, f"mindmap_{translated_lang}.json")
                    with open(t_json_path, "w", encoding="utf-8") as f:
                        json_module.dump(t_data, f, ensure_ascii=False)
                    t_key = f"books/{req.book_id}/mindmap_{translated_lang}.json"
                    translated_mindmap_url = upload_file(t_json_path, t_key, CONTENT_TYPES[".json"])
                    translated_mindmap_data = t_data
                else:
                    t_mermaid = await generate_mermaid_code(
                        req.title or req.book_id, translated_summary, translated_lang, model=model_mindmap
                    )
                    t_mermaid = stamp_mindmap_mermaid(t_mermaid, cfg)
                    t_svg_path = os.path.join(tmp, f"mindmap_{translated_lang}.svg")
                    await render_mermaid_svg(t_mermaid, t_svg_path)
                    t_key = f"books/{req.book_id}/mindmap_{translated_lang}.svg"
                    translated_mindmap_url = upload_file(t_svg_path, t_key, CONTENT_TYPES[".svg"])
                    translated_mindmap_path_saved = t_svg_path
                step_status["mindmap_translate"] = "done"
            except JobCancelledError:
                raise
            except Exception as e:
                errors["mindmap_translate"] = str(e)
                step_status["mindmap_translate"] = "failed"
            _t = round(time.time() - started)
            await _persist_step_result(job_id, "mindmap_translate", step_status["mindmap_translate"],
                                       output_url=translated_mindmap_url, duration_sec=_t)
            await _checkpoint()

        # ── mindmap_chapters_translate ────────────────────────────────────────
        async def _do_mindmap_chapters_translate() -> None:
            nonlocal chapter_mindmap_translated, chapter_results
            if "mindmap_chapters_translate" not in steps:
                return
            if _already_done("mindmap_chapters_translate"):
                log.info("mindmap_chapters_translate already done — reusing existing mindmaps (not forced)")
                return
            if not mindmap_enabled or not chapter_results or not translated_summary or not translated_lang:
                await _skip_step("mindmap_chapters_translate")
                return
            step_status["mindmap_chapters_translate"] = "running"
            set_step("mindmap_chapters_translate")
            await _checkpoint()
            ch_errors = 0
            ch_processed = 0

            mm_conc = max(1, int(cfg.get("MINDMAP_CONCURRENCY", "4")))
            mm_sem  = asyncio.Semaphore(mm_conc)

            async def _chapter_mindmap_target(ch: dict) -> tuple[int, dict | None, str | None]:
                idx      = ch["index"]
                ch_title = ch.get("title") or f"Chapter {idx}"
                ch_text  = ch["translated_summary"]
                async with mm_sem:
                    try:
                        from api.services.pipeline.watermark import stamp_mindmap_json, stamp_mindmap_mermaid  # noqa: PLC0415
                        if mindmap_format == "json":
                            data = await generate_json_mindmap(
                                ch_title, ch_text, translated_lang, model=model_mindmap
                            )
                            data = stamp_mindmap_json(data, cfg)
                            json_path = os.path.join(tmp, f"ch{idx}_mindmap_{translated_lang}.json")
                            with open(json_path, "w", encoding="utf-8") as f:
                                json_module.dump(data, f, ensure_ascii=False)
                            key = f"books/{req.book_id}/chapters/ch_{idx:02d}_mindmap_{translated_lang}.json"
                            url = upload_file(json_path, key, CONTENT_TYPES[".json"])
                            return idx, {"url": url, "data": data, "format": "json"}, None
                        else:
                            mermaid = await generate_mermaid_code(
                                ch_title, ch_text, translated_lang, model=model_mindmap
                            )
                            mermaid = stamp_mindmap_mermaid(mermaid, cfg)
                            svg_path = os.path.join(tmp, f"ch{idx}_mindmap_{translated_lang}.svg")
                            await render_mermaid_svg(mermaid, svg_path)
                            key = f"books/{req.book_id}/chapters/ch_{idx:02d}_mindmap_{translated_lang}.svg"
                            url = upload_file(svg_path, key, CONTENT_TYPES[".svg"])
                            return idx, {"url": url, "data": None, "format": "mermaid"}, None
                    except Exception as e:
                        return idx, None, str(e)

            chapters_with_translated = [ch for ch in chapter_results if ch.get("translated_summary")]
            if chapters_with_translated:
                mm_target_results = await asyncio.gather(
                    *[_chapter_mindmap_target(ch) for ch in chapters_with_translated]
                )
                for idx, result, err in mm_target_results:
                    if result is not None:
                        chapter_mindmap_translated[idx] = result
                        ch_processed += 1
                        for ch in chapter_results:
                            if ch["index"] == idx:
                                ch[f"mindmap_{translated_lang}_url"] = result.get("url")
                                ch["mindmap_format"] = result.get("format")
                                if result.get("data"):
                                    ch[f"mindmap_{translated_lang}_data"] = result.get("data")
                                break
                    if err is not None:
                        ch_errors += 1
                        log.warning("target-language chapter mindmap %s failed: %s", idx, err)
                        errors[f"mindmap_chapter_{idx}_{translated_lang}"] = err

            if ch_errors > 0 and ch_processed == 0:
                step_status["mindmap_chapters_translate"] = "failed"
                errors["mindmap_chapters_translate"] = f"All {ch_errors} chapter(s) failed"
            elif ch_errors > 0:
                step_status["mindmap_chapters_translate"] = "partial"
                errors["mindmap_chapters_translate"] = f"{ch_errors} chapter(s) failed"
            elif ch_processed > 0:
                step_status["mindmap_chapters_translate"] = "done"
            else:
                step_status["mindmap_chapters_translate"] = "skipped"

            _t = round(time.time() - started)
            await _persist_step_result(job_id, "mindmap_chapters_translate",
                                       step_status["mindmap_chapters_translate"], duration_sec=_t)
            await _checkpoint()

        # ── Run Phase 2 in parallel ───────────────────────────────────────────
        phase2_coros = [
            _do_cover(),
            _do_audio_full(),
            _do_audio_chapters(),
            _do_audio_full_translate(),
            _do_audio_chapters_translate(),
            _do_mindmap(),
            _do_mindmap_chapters(),
            _do_mindmap_translate(),
            _do_mindmap_chapters_translate(),
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
            if "alt_text" not in steps:
                return
            if _already_done("alt_text", alt_text):
                log.info("alt_text already done — reusing existing (not forced)")
                return
            if not alttext_enabled or not cover_url or not os.path.exists(cover_path_saved):
                await _skip_step("alt_text")
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
            if "video" not in steps:
                return
            if _already_done("video", video_url):
                log.info("video already done — reusing existing video (not forced)")
                return
            if not video_enabled or not full_summary:
                await _skip_step("video")
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
            if "inject_epub" not in steps:
                return
            if _already_done("inject_epub", epub_url):
                log.info("inject_epub already done — reusing existing EPUB (not forced)")
                return
            if not epub_enabled or not full_summary:
                log.info("_do_inject_epub skipped: steps=%s, epub_enabled=%s, full_summary=%s",
                         steps, epub_enabled, bool(full_summary))
                await _skip_step("inject_epub")
                return
            # Don't bake an incomplete/low-quality book into the downloadable EPUB.
            # Same gate as audio: skip when the summary failed QA or some chapters
            # are still missing (summarize = partial). The EPUB will be built on a
            # later retry once the summary is complete and passes quality.
            if audio_blocked:
                log.info(
                    "_do_inject_epub skipped: audio_blocked (summary failed QA or "
                    "chapters incomplete) — EPUB not built from incomplete content",
                )
                await _skip_step("inject_epub")
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
        if translated_lang:
            ch[f"audio_{translated_lang}"] = chapter_audio_translated.get(ch["index"])
        cm = chapter_mindmap.get(ch["index"])
        if cm:
            ch["mindmap_url"]    = cm["url"]
            ch["mindmap_format"] = cm["format"]
            if cm["data"] is not None:
                ch["mindmap_data"] = cm["data"]
        cmt = chapter_mindmap_translated.get(ch["index"])
        if cmt and translated_lang:
            ch[f"mindmap_{translated_lang}_url"] = cmt["url"]
            if cmt["data"] is not None:
                ch[f"mindmap_{translated_lang}_data"] = cmt["data"]

    # ── Overall status ────────────────────────────────────────────────────────
    active = {s for s in steps if s in ALL_STEPS}
    failed = sum(1 for s in active if step_status.get(s) == "failed")
    status = "done" if failed == 0 else ("failed" if failed == len(active) else "partial")

    elapsed = round(time.time() - started, 1)
    cost    = await _compute_job_cost(job_id)

    # ── Final status write-back to the books table ────────────────────────────
    # Asset URLs live in their own tables (audio.en_url/ar_url, covers.coverurl);
    # the books table only carries status flags. cover_status is set by
    # _persist_cover; here we set the overall pipeline status.
    await _persist_book_details(req.book_id, {
        "status": "complete" if status == "done" else status,
    })

    summary_key = f"{req.options.length}_{req.language}"

    files = {
        "cover":      cover_url,
        "audio_full": (full_audio or {}).get("url"),
        "mindmap":    mindmap_url,
        "epub":       epub_url,
        "video":      video_url,
        "chapters": [
            {
                "index":        ch["index"],
                "title":        ch.get("title"),
                "audio_url":    chapter_audio.get(ch["index"]),
                "audio_en_url": chapter_audio.get(ch["index"]) if lang == "en" else chapter_audio_translated.get(ch["index"]),
                "audio_ar_url": chapter_audio_translated.get(ch["index"]) if lang == "en" else chapter_audio.get(ch["index"]),
                "mindmap_url":    (chapter_mindmap.get(ch["index"]) or {}).get("url"),
                "mindmap_en_url": ((chapter_mindmap.get(ch["index"]) if lang == "en" else chapter_mindmap_translated.get(ch["index"])) or {}).get("url"),
                "mindmap_ar_url": ((chapter_mindmap_translated.get(ch["index"]) if lang == "en" else chapter_mindmap.get(ch["index"])) or {}).get("url"),
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
            **({
                summary_key: {
                    "text":       full_summary,
                    "word_count": len(full_summary.split()) if full_summary else 0,
                    "style":      req.options.style,
                    "language":   req.language,
                }
            } if full_summary else {}),
            **({
                f"{req.options.length}_{translated_lang}": {
                    "text":       translated_summary,
                    "word_count": len(translated_summary.split()),
                    "style":      req.options.style,
                    "language":   translated_lang,
                    "translated": True,
                }
            } if translated_summary and translated_lang else {}),
        },
        "audio": {
            **({f"full_{lang}": full_audio} if full_audio else {}),
            **({f"full_{translated_lang}": translated_audio} if translated_audio and translated_lang else {}),
        },
        "mindmap": (
            {"url": mindmap_url, "data": mindmap_data} if mindmap_data else
            {"url": mindmap_url} if mindmap_url else
            None
        ),
        "mindmap_en": (
            {"url": mindmap_url, "data": mindmap_data} if lang == "en" and mindmap_url else
            {"url": translated_mindmap_url, "data": translated_mindmap_data} if translated_lang == "en" and translated_mindmap_url else
            None
        ),
        "mindmap_ar": (
            {"url": mindmap_url, "data": mindmap_data} if lang == "ar" and mindmap_url else
            {"url": translated_mindmap_url, "data": translated_mindmap_data} if translated_lang == "ar" and translated_mindmap_url else
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
        "summary_qa": summary_qa,
        "errors":   errors,
    }
