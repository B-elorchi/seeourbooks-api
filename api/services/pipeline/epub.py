"""
EPUB enrichment — fetches an existing book EPUB and injects the AI-generated
summary as a new front-matter chapter (and optionally replaces the cover).

URL pattern for the source EPUB (configured via BOOK_FILES_BASE_URL):
    {base}/books/english/{book_id}.epub      ← language == "en"
    {base}/books/arabic/{book_id}.epub       ← language == "ar"

Public surface
──────────────
    fetch_epub(book_id, language, output_path)
    inject_summary_into_epub(epub_path, output_path, *, title, author,
                              summary_text, language, cover_path=None)
    EpubError                — raised on any unrecoverable EPUB issue.
    EpubNotAvailableError    — source EPUB URL not configured or 404.

The HTML templates are ported from the reference `epub_auto_injector.py`
script — RTL-aware Arabic layout with Amiri/Traditional Arabic fallback, LTR
English layout with Georgia.

Design notes
────────────
- ebooklib is synchronous.  Heavy work runs in `loop.run_in_executor` so the
  pipeline event loop stays responsive.
- HTML body content is escaped via `html.escape` and split on blank-line
  paragraph boundaries — no raw HTML from the summarizer ever reaches the
  rendered chapter.
- The summary chapter is prepended to the spine + ToC so the reader sees it
  before chapter 1.
"""
from __future__ import annotations

import asyncio
import functools
import logging
from html import escape
from pathlib import Path
from urllib.parse import quote

import httpx

from api.config.settings import settings

log = logging.getLogger(__name__)


# ── Errors ───────────────────────────────────────────────────────────────────

class EpubError(Exception):
    """Generic EPUB read/write/inject failure."""


class EpubNotAvailableError(EpubError):
    """The source EPUB cannot be located (no base URL or 404)."""


# ── Templates ────────────────────────────────────────────────────────────────

_AR_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="ar" lang="ar" dir="rtl">
<head>
  <meta charset="utf-8"/>
  <title>ملخص الكتاب</title>
  <style>
    body {{
      font-family: 'Amiri', 'Traditional Arabic', Arial, sans-serif;
      direction: rtl; text-align: right;
      line-height: 2.0; margin: 2em;
      color: #1a1a1a; background: #fafaf8;
    }}
    h1 {{ font-size: 1.6em; color: #2c3e50;
          border-bottom: 2px solid #e67e22;
          padding-bottom: 0.4em; margin-bottom: 0.6em; }}
    .meta  {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 1.4em; }}
    .text  {{ font-size: 1.05em; text-align: justify; }}
    .badge {{ background: #e67e22; color: white;
              padding: 0.2em 0.8em; border-radius: 4px;
              font-size: 0.8em; margin-bottom: 1em;
              display: inline-block; }}
    footer {{ text-align: center; color: #aaa;
              font-size: 0.8em; margin-top: 2em; }}
  </style>
</head>
<body>
  <span class="badge">ملخص</span>
  <h1>{title}</h1>
  <p class="meta">المؤلف: {author}</p>
  <div class="text">{body}</div>
  <footer>SeeOurBook.com — مكتبتك الرقمية</footer>
</body>
</html>"""

_EN_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en" lang="en" dir="ltr">
<head>
  <meta charset="utf-8"/>
  <title>Book Summary</title>
  <style>
    body {{
      font-family: Georgia, 'Times New Roman', serif;
      direction: ltr; text-align: left;
      line-height: 1.8; margin: 2em;
      color: #1a1a1a; background: #fafaf8;
    }}
    h1 {{ font-size: 1.6em; color: #2c3e50;
          border-bottom: 2px solid #2980b9;
          padding-bottom: 0.4em; margin-bottom: 0.6em; }}
    .meta  {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 1.4em; }}
    .text  {{ font-size: 1.05em; text-align: justify; }}
    .badge {{ background: #2980b9; color: white;
              padding: 0.2em 0.8em; border-radius: 4px;
              font-size: 0.8em; margin-bottom: 1em;
              display: inline-block; }}
    footer {{ text-align: center; color: #aaa;
              font-size: 0.8em; margin-top: 2em; }}
  </style>
</head>
<body>
  <span class="badge">Summary</span>
  <h1>{title}</h1>
  <p class="meta">Author: {author}</p>
  <div class="text">{body}</div>
  <footer>SeeOurBook.com — Your Digital Library</footer>
</body>
</html>"""


def _paragraphs_html(text: str) -> str:
    """Convert plain text → safely-escaped <p>...</p> blocks."""
    if not text:
        return "  <p></p>"
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text.strip()]
    out: list[str] = []
    for p in paras:
        # Single-line breaks inside a paragraph become <br/>.
        # Escape first to prevent HTML injection from the summarizer.
        safe = escape(p).replace("\n", "<br/>")
        out.append(f"  <p>{safe}</p>")
    return "\n".join(out)


# ── URL resolution ───────────────────────────────────────────────────────────

def _epub_source_url(book_id: str, language: str) -> str:
    base_cfg = (settings.BOOK_FILES_BASE_URL or "").strip().rstrip("/")
    if not base_cfg:
        raise EpubNotAvailableError(
            "BOOK_FILES_BASE_URL is not configured — set it in admin or .env "
            "to enable the inject_epub step."
        )
    folder = "arabic" if (language or "en").lower() == "ar" else "english"
    return f"{base_cfg}/books/{folder}/{quote(book_id, safe='')}.epub"


# ── Public: fetch source EPUB ────────────────────────────────────────────────

async def fetch_epub(book_id: str, language: str, output_path: str) -> str:
    """
    Download the source EPUB for (book_id, language) to `output_path`.

    Raises:
        EpubNotAvailableError — base URL unset or remote returns 404.
        EpubError             — any other download failure.
    """
    url = _epub_source_url(book_id, language)
    log.info("Fetching EPUB from %s", url)

    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            r = await client.get(url)
    except httpx.ConnectError as exc:
        # DNS resolution failure, refused connection, etc. — treat the same
        # way as a 404: the EPUB simply isn't available at the configured
        # URL.  Pipeline degrades to "skipped" instead of "failed" so the
        # rest of the job still succeeds.
        raise EpubNotAvailableError(
            f"EPUB host unreachable for {url} ({exc}). "
            "Either BOOK_FILES_BASE_URL is misconfigured or the host is down. "
            "Set BOOK_FILES_BASE_URL='' in Admin → Providers → EPUB Source to "
            "disable this step entirely."
        ) from exc
    except httpx.TimeoutException as exc:
        # Slow upstream — also degradable; user can retry later.
        raise EpubNotAvailableError(
            f"EPUB host timed out fetching {url}: {exc}"
        ) from exc
    except httpx.RequestError as exc:
        raise EpubNotAvailableError(
            f"EPUB request error for {url}: {exc}"
        ) from exc

    if r.status_code == 404:
        raise EpubNotAvailableError(
            f"source EPUB not found at {url} (404)"
        )
    if r.status_code != 200:
        # 5xx / 403 etc. — treat as transient unavailability, don't paint the
        # whole job red.  Admins can re-run later or update BOOK_FILES_BASE_URL.
        raise EpubNotAvailableError(
            f"EPUB host returned {r.status_code} for {url}"
        )

    if not r.content or len(r.content) < 100:
        raise EpubNotAvailableError(f"empty response body from {url}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(r.content)
    return output_path


# ── Synchronous ebooklib worker (runs in executor) ───────────────────────────

def _detect_image_media_type(data: bytes, ext_hint: str) -> str:
    """Detect actual image media type from magic bytes, fallback to ext hint."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    ext = ext_hint.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    if ext in ("png", "webp", "gif"):
        return f"image/{ext}"
    return "image/jpeg"


def _inject_sync(
    epub_path: str,
    output_path: str,
    *,
    title: str,
    author: str,
    summary_text: str,
    language: str,
    cover_path: str | None,
) -> None:
    try:
        from ebooklib import epub  # local import keeps cold start fast
    except ImportError as exc:
        raise EpubError(
            "EbookLib is not installed — run `pip install EbookLib`"
        ) from exc

    try:
        book = epub.read_epub(epub_path)
    except Exception as exc:
        raise EpubError(f"could not parse source EPUB: {exc}") from exc

    # Fall back to EPUB metadata when caller didn't supply title/author
    if not title:
        meta = book.get_metadata("DC", "title")
        title = meta[0][0] if meta else "Untitled"
    if not author:
        meta = book.get_metadata("DC", "creator")
        author = meta[0][0] if meta else "Unknown"

    # ── Cover replacement (optional) ─────────────────────────────────────────
    if cover_path and Path(cover_path).is_file():
        try:
            data = Path(cover_path).read_bytes()
            media = _detect_image_media_type(data, Path(cover_path).suffix)
            ext = "jpg" if media == "image/jpeg" else media.split("/", 1)[1]
            book.set_cover(f"cover.{ext}", data)
            log.info("EPUB cover replaced (%s, %d bytes)", media, len(data))
        except Exception as exc:
            # Non-fatal — keep going with the original cover.
            log.warning("cover replacement failed (%s) — keeping original", exc)

    # ── Summary chapter ──────────────────────────────────────────────────────
    is_arabic = (language or "en").lower() == "ar"
    template  = _AR_TEMPLATE if is_arabic else _EN_TEMPLATE
    chapter_title = "ملخص الكتاب" if is_arabic else "Book Summary"

    html = template.format(
        title  = escape(title),
        author = escape(author),
        body   = _paragraphs_html(summary_text),
    )

    slug = Path(epub_path).stem
    file_name = f"summary_{(language or 'en').lower()}.xhtml"
    item = epub.EpubHtml(
        uid       = f"{slug}_summary_{(language or 'en').lower()}",
        title     = chapter_title,
        file_name = file_name,
        lang      = (language or "en").lower(),
    )
    item.content = html.encode("utf-8")
    book.add_item(item)

    # Replace any earlier injection of the same filename, then prepend
    existing_spine = [
        sp for sp in book.spine
        if not (isinstance(sp, tuple) and getattr(sp[0], "file_name", None) == file_name)
        and not (hasattr(sp, "file_name") and sp.file_name == file_name)
    ]
    book.spine = [item] + existing_spine

    # Prepend to the ToC
    new_link = epub.Link(file_name, chapter_title, item.id)
    old_toc  = list(book.toc) if book.toc else []
    book.toc = (new_link, *old_toc)

    # ── Write the enriched EPUB ──────────────────────────────────────────────
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(output_path, book)
    except Exception as exc:
        raise EpubError(f"could not write enriched EPUB: {exc}") from exc


# ── Public: async injection ──────────────────────────────────────────────────

async def inject_summary_into_epub(
    epub_path:    str,
    output_path:  str,
    *,
    title:        str,
    author:       str,
    summary_text: str,
    language:     str,
    cover_path:   str | None = None,
) -> str:
    """
    Inject the summary (+ optional cover) into `epub_path`, write to
    `output_path`.  Returns the output path.

    Heavy work runs in the default executor so the pipeline event loop
    stays responsive — ebooklib has no async API.
    """
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        _inject_sync,
        epub_path, output_path,
        title=title, author=author, summary_text=summary_text,
        language=language, cover_path=cover_path,
    )
    await loop.run_in_executor(None, fn)
    return output_path
