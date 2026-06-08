"""
EPUB enrichment — injects AI-generated content into an existing book EPUB,
OR creates a fresh EPUB from scratch when no source is available.

New structure (in reading order):
  1. Cover image
  2. Front-matter page  — full summary text + full audio + mind-map link
  3. Original book chapters (from source EPUB spine), each followed by:
       └── Chapter-insights page — chapter summary + chapter audio + chapter mindmap

When building from scratch (no source EPUB):
  1. Cover page
  2. Front-matter page
  3. One chapter-insights page per chapter (no original text available)

Output filename is always  {book_id}_{language}.epub
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


# ── Errors ────────────────────────────────────────────────────────────────────

class EpubError(Exception):
    """Generic EPUB read/write/inject failure."""


class EpubNotAvailableError(EpubError):
    """The source EPUB cannot be located (no base URL or 404)."""


# ── HTML helpers ──────────────────────────────────────────────────────────────

def _paragraphs_html(text: str) -> str:
    if not text:
        return "<p>—</p>"
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paras:
        paras = [text.strip()]
    return "\n".join(
        f'<p>{escape(p).replace(chr(10), "<br/>")}</p>'
        for p in paras
    )


def _base_styles(accent: str, font: str, is_arabic: bool) -> str:
    align = "right" if is_arabic else "left"
    return f"""
    body {{
      font-family: {font};
      direction: {'rtl' if is_arabic else 'ltr'};
      text-align: {align};
      line-height: 1.9;
      margin: 2em 2.5em;
      color: #1a1a1a;
      background: #fafaf8;
    }}
    h1 {{
      font-size: 1.55em; color: #2c3e50;
      border-bottom: 3px solid {accent};
      padding-bottom: 0.3em; margin-bottom: 0.6em;
    }}
    h2 {{
      font-size: 1.15em; color: #34495e;
      margin-top: 1.6em; margin-bottom: 0.4em;
    }}
    .meta {{ color: #7f8c8d; font-size: 0.88em; margin-bottom: 1.5em; }}
    .badge {{
      display: inline-block;
      background: {accent}; color: white;
      padding: 0.2em 0.9em; border-radius: 5px;
      font-size: 0.8em; margin-bottom: 1em;
    }}
    .summary-text {{ font-size: 1.02em; text-align: justify; }}
    .asset-box {{
      border: 1px solid #ddd; border-radius: 8px;
      padding: 1em 1.2em; margin: 1.2em 0;
      background: #f5f5f0;
    }}
    .asset-box h3 {{
      font-size: 0.95em; color: #555;
      margin: 0 0 0.5em 0; text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .asset-link {{
      display: block;
      color: {accent};
      text-decoration: underline;
      word-break: break-all;
      font-size: 0.93em;
      margin: 0.3em 0;
    }}
    .chapter-insights {{
      border-top: 2px solid {accent};
      margin-top: 2em; padding-top: 1em;
    }}
    .insight-label {{
      font-size: 0.75em; text-transform: uppercase;
      letter-spacing: 0.08em; color: #999;
      margin-bottom: 0.8em;
    }}
    footer {{
      text-align: center; color: #bbb;
      font-size: 0.78em; margin-top: 2.5em;
      border-top: 1px solid #eee; padding-top: 0.8em;
    }}
    """


def _xhtml_page(
    uid: str,
    title: str,
    body_html: str,
    language: str,
    author: str = "",
    accent: str = "",
    font: str = "",
    show_header: bool = True,
) -> str:
    """Render a complete XHTML page ready for ebooklib."""
    is_arabic = (language or "en").lower() == "ar"
    lang_attr = "ar" if is_arabic else "en"
    dir_attr  = 'dir="rtl"' if is_arabic else 'dir="ltr"'
    _accent   = accent or ("#e67e22" if is_arabic else "#2980b9")
    _font     = font   or (
        "Amiri, 'Traditional Arabic', Arial, sans-serif"
        if is_arabic else
        "Georgia, 'Times New Roman', serif"
    )
    author_label = "المؤلف" if is_arabic else "Author"
    footer_text  = "مكتبتك الرقمية" if is_arabic else "Your Digital Library"

    header_html = (
        f'<h1>{escape(title)}</h1>'
        f'<p class="meta">{author_label}: {escape(author)}</p>'
        if show_header and title else ""
    )

    return f"""<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"
      xml:lang="{lang_attr}" lang="{lang_attr}" {dir_attr}>
<head>
  <meta charset="utf-8"/>
  <title>{escape(title)}</title>
  <style>{_base_styles(_accent, _font, is_arabic)}</style>
</head>
<body>
  {header_html}
  {body_html}
  <footer>SeeOurBook — {footer_text}</footer>
</body>
</html>"""


# ── URL resolution ────────────────────────────────────────────────────────────

def _epub_source_url(book_id: str, language: str) -> str:
    base_cfg = (settings.BOOK_FILES_BASE_URL or "").strip().rstrip("/")
    if not base_cfg:
        raise EpubNotAvailableError(
            "BOOK_FILES_BASE_URL is not configured — set it in admin or .env "
            "to enable the inject_epub step."
        )
    folder = "arabic" if (language or "en").lower() == "ar" else "english"
    return f"{base_cfg}/books/{folder}/{quote(book_id, safe='')}.epub"


# ── Public: fetch source EPUB ─────────────────────────────────────────────────

async def fetch_epub(book_id: str, language: str, output_path: str) -> str:
    url = _epub_source_url(book_id, language)
    log.info("Fetching EPUB from %s", url)
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            r = await client.get(url)
    except (httpx.ConnectError, httpx.TimeoutException, httpx.RequestError) as exc:
        raise EpubNotAvailableError(f"EPUB host unreachable for {url}: {exc}") from exc

    if r.status_code == 404:
        raise EpubNotAvailableError(f"source EPUB not found at {url} (404)")
    if r.status_code != 200:
        raise EpubNotAvailableError(f"EPUB host returned {r.status_code} for {url}")
    if not r.content or len(r.content) < 100:
        raise EpubNotAvailableError(f"empty response body from {url}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_bytes(r.content)
    return output_path


# ── Image media-type detection ────────────────────────────────────────────────

def _detect_image_media_type(data: bytes, ext_hint: str) -> str:
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


# ── Page builders ─────────────────────────────────────────────────────────────

def _build_front_matter(
    epub_mod,
    *,
    slug: str,
    title: str,
    author: str,
    summary_text: str,
    language: str,
    audio_url: str | None,
    mindmap_url: str | None,
) -> object:
    """
    Front-matter page: full summary + full audio link + mindmap link.
    This appears immediately after the cover in the reading order.
    """
    is_arabic   = (language or "en").lower() == "ar"
    page_title  = "ملخص الكتاب" if is_arabic else "Book Summary"
    badge_text  = "ملخص" if is_arabic else "Summary"

    # ── Summary section ───────────────────────────────────────────────────────
    body_parts = [
        f'<p class="badge">{escape(badge_text)}</p>',
        f'<div class="summary-text">{_paragraphs_html(summary_text)}</div>',
    ]

    # ── Full audio ────────────────────────────────────────────────────────────
    if audio_url:
        audio_label  = "الصوت الكامل" if is_arabic else "Full Audio"
        dl_label     = "تحميل" if is_arabic else "Download MP3"
        body_parts.append(
            f'<div class="asset-box">'
            f'<h3>🔊 {audio_label}</h3>'
            f'<a class="asset-link" href="{escape(audio_url)}">{escape(audio_url)}</a>'
            f'<p style="font-size:0.85em;color:#777;margin-top:0.4em">'
            f'<a href="{escape(audio_url)}">{dl_label} ↗</a></p>'
            f'</div>'
        )

    # ── Mind map ──────────────────────────────────────────────────────────────
    if mindmap_url:
        mm_label = "خريطة الذهن" if is_arabic else "Mind Map"
        body_parts.append(
            f'<div class="asset-box">'
            f'<h3>🗺 {mm_label}</h3>'
            f'<a class="asset-link" href="{escape(mindmap_url)}">{escape(mindmap_url)}</a>'
            f'</div>'
        )

    html = _xhtml_page(
        uid=f"{slug}_front",
        title=title,
        author=author,
        body_html="\n".join(body_parts),
        language=language,
    )
    item = epub_mod.EpubHtml(
        uid       = f"{slug}_front_{language}",
        title     = page_title,
        file_name = f"seeourbook_front_{language}.xhtml",
        lang      = language,
    )
    item.content = html.encode("utf-8")
    return item


def _build_chapter_insights(
    epub_mod,
    *,
    slug: str,
    ch: dict,
    language: str,
    chapter_audio: dict[int, str],
    chapter_mindmap: dict[int, dict],
) -> object | None:
    """
    Per-chapter insights page: chapter summary + audio + mindmap.
    Inserted into the spine AFTER the original chapter page.
    Returns None if the chapter has no summary.
    """
    ch_summary = (ch.get("summary") or "").strip()
    if not ch_summary:
        return None

    idx       = ch.get("index", 0)
    ch_title  = ch.get("title") or f"Chapter {idx}"
    is_arabic = (language or "en").lower() == "ar"

    insights_label = "تحليل الفصل" if is_arabic else "Chapter Insights"
    sum_label      = "ملخص" if is_arabic else "Summary"
    audio_label    = "صوت الفصل" if is_arabic else "Chapter Audio"
    mm_label       = "خريطة ذهنية" if is_arabic else "Chapter Mind Map"

    body_parts = [
        f'<div class="chapter-insights">',
        f'<p class="insight-label">✦ {insights_label} — {escape(ch_title)}</p>',

        # Summary
        f'<div class="asset-box">',
        f'<h3>📝 {sum_label}</h3>',
        f'<div class="summary-text">{_paragraphs_html(ch_summary)}</div>',
        f'</div>',
    ]

    # Chapter audio
    audio_url = chapter_audio.get(idx)
    if audio_url:
        body_parts += [
            f'<div class="asset-box">',
            f'<h3>🎧 {audio_label}</h3>',
            f'<a class="asset-link" href="{escape(audio_url)}">{escape(audio_url)}</a>',
            f'</div>',
        ]

    # Chapter mindmap
    cm = chapter_mindmap.get(idx)
    mm_url = cm.get("url") if cm else None
    if mm_url:
        body_parts += [
            f'<div class="asset-box">',
            f'<h3>🗺 {mm_label}</h3>',
            f'<a class="asset-link" href="{escape(mm_url)}">{escape(mm_url)}</a>',
            f'</div>',
        ]

    body_parts.append('</div>')  # close .chapter-insights

    safe_slug = f"{slug}_ch{idx}"
    html = _xhtml_page(
        uid      = safe_slug,
        title    = f"{insights_label}: {ch_title}",
        author   = "",
        body_html= "\n".join(body_parts),
        language = language,
        show_header = False,
    )
    item = epub_mod.EpubHtml(
        uid       = f"{safe_slug}_{language}",
        title     = f"{insights_label}: {ch_title}",
        file_name = f"seeourbook_ch{idx:03d}_{language}.xhtml",
        lang      = language,
    )
    item.content = html.encode("utf-8")
    return item


# ── Synchronous ebooklib worker ───────────────────────────────────────────────

def _inject_sync(
    epub_path: str | None,
    output_path: str,
    *,
    title: str,
    author: str,
    summary_text: str,
    language: str,
    cover_path: str | None,
    chapters: list[dict] | None,
    chapter_audio: dict[int, str] | None,
    chapter_mindmap: dict[int, dict] | None,
    audio_url: str | None,
    mindmap_url: str | None,
) -> None:
    try:
        from ebooklib import epub
    except ImportError as exc:
        raise EpubError("EbookLib is not installed — run `pip install EbookLib`") from exc

    chapters        = chapters        or []
    chapter_audio   = chapter_audio   or {}
    chapter_mindmap = chapter_mindmap or {}

    # ── Load or create the EPUB book ─────────────────────────────────────────
    from_scratch = True
    if epub_path and Path(epub_path).is_file() and Path(epub_path).stat().st_size > 100:
        try:
            book = epub.read_epub(epub_path)
            from_scratch = False
            log.info("Loaded source EPUB: %s", epub_path)
        except Exception as exc:
            log.warning("Could not parse source EPUB (%s) — creating from scratch", exc)
            book = epub.EpubBook()
    else:
        book = epub.EpubBook()

    if from_scratch:
        log.info("Building EPUB from scratch for %r / %s", title, language)
        book.set_identifier(f"seeourbook_{title}_{language}")
        book.set_language(language or "en")
        book.add_item(epub.EpubNcx())
        book.add_item(epub.EpubNav())
        book.spine = ["nav"]

    # Fill in metadata
    if not title:
        meta  = book.get_metadata("DC", "title")
        title = meta[0][0] if meta else "Untitled"
    if not author:
        meta   = book.get_metadata("DC", "creator")
        author = meta[0][0] if meta else "Unknown"
    book.set_title(title)
    book.add_author(author)

    slug = f"{title}_{language}".replace(" ", "_")[:40]

    # ── 1. Cover ──────────────────────────────────────────────────────────────
    if cover_path and Path(cover_path).is_file():
        try:
            data  = Path(cover_path).read_bytes()
            media = _detect_image_media_type(data, Path(cover_path).suffix)
            ext   = "jpg" if media == "image/jpeg" else media.split("/", 1)[1]
            book.set_cover(f"cover.{ext}", data)
            log.info("Cover set (%s, %d bytes)", media, len(data))
        except Exception as exc:
            log.warning("Cover set failed (%s) — continuing", exc)

    # ── 2. Front-matter page ──────────────────────────────────────────────────
    front_item = _build_front_matter(
        epub,
        slug=slug, title=title, author=author,
        summary_text=summary_text, language=language,
        audio_url=audio_url, mindmap_url=mindmap_url,
    )
    book.add_item(front_item)

    # ── 3. Per-chapter insights pages ─────────────────────────────────────────
    # Build a map: chapter index → insights item
    insight_items: dict[int, object] = {}
    for ch in chapters:
        item = _build_chapter_insights(
            epub,
            slug=slug, ch=ch, language=language,
            chapter_audio=chapter_audio,
            chapter_mindmap=chapter_mindmap,
        )
        if item:
            book.add_item(item)
            insight_items[ch.get("index", 0)] = item

    # ── Rebuild spine ─────────────────────────────────────────────────────────
    # Strategy:
    #   • Remove any previously injected seeourbook pages from the spine
    #   • Put front-matter first (after "nav")
    #   • For existing EPUBs: interleave insight pages after each original chapter
    #   • For from-scratch EPUBs: list front + insights in chapter order

    injected_fnames = {front_item.file_name} | {
        it.file_name for it in insight_items.values()
    }

    def _spine_fname(sp) -> str | None:
        """Extract the filename from a spine entry regardless of its type."""
        if isinstance(sp, str):
            return sp
        if isinstance(sp, tuple):
            sp = sp[0]
        return getattr(sp, "file_name", None)

    # Clean old spine of injected pages
    clean_spine = [
        sp for sp in book.spine
        if (_spine_fname(sp) or "") not in injected_fnames
        and sp not in ("nav", "cover")
    ]

    if from_scratch or not clean_spine:
        # From scratch: front-matter then chapters in order
        new_spine: list = ["nav", front_item]
        for ch in sorted(chapters, key=lambda c: c.get("index", 0)):
            item = insight_items.get(ch.get("index", 0))
            if item:
                new_spine.append(item)
        book.spine = new_spine
    else:
        # Existing EPUB: insert insights after every N-th original chapter
        # We split the clean_spine into chapter-sized groups and interleave.
        # Heuristic: treat each existing spine item as one book chapter,
        # and pair it with insight_items by position.
        new_spine = ["nav", front_item]
        sorted_insights = [
            insight_items[ch.get("index", 0)]
            for ch in sorted(chapters, key=lambda c: c.get("index", 0))
            if ch.get("index", 0) in insight_items
        ]
        n_orig = len(clean_spine)
        n_ins  = len(sorted_insights)

        if n_orig == 0:
            new_spine += sorted_insights
        elif n_ins == 0:
            new_spine += clean_spine
        else:
            # Distribute insights evenly across original chapters
            # If counts match exactly, pair 1:1.
            # Otherwise insert an insight page after every k original pages.
            k = max(1, n_orig // max(n_ins, 1))
            ins_idx = 0
            for i, sp in enumerate(clean_spine):
                new_spine.append(sp)
                # After every k-th original chapter, append the next insight
                if ins_idx < n_ins and (i + 1) % k == 0:
                    new_spine.append(sorted_insights[ins_idx])
                    ins_idx += 1
            # Append any remaining insights at the end
            while ins_idx < n_ins:
                new_spine.append(sorted_insights[ins_idx])
                ins_idx += 1

        book.spine = new_spine

    # ── Rebuild ToC ───────────────────────────────────────────────────────────
    is_arabic  = (language or "en").lower() == "ar"
    toc_items: list = []

    # Front-matter first
    toc_items.append(epub.Link(front_item.file_name, front_item.title, front_item.id))

    # Old ToC entries (minus any previously injected ones)
    old_toc = [
        entry for entry in (list(book.toc) if book.toc else [])
        if hasattr(entry, "href") and not any(
            entry.href.startswith(fn) for fn in injected_fnames
        )
    ]
    toc_items += old_toc

    # Chapter insights
    insights_section_title = "تحليل الفصول" if is_arabic else "Chapter Insights"
    if insight_items:
        toc_items.append(epub.Section(insights_section_title, [
            epub.Link(it.file_name, it.title, it.id)
            for it in sorted(insight_items.values(), key=lambda x: x.file_name)
        ]))

    book.toc = tuple(toc_items)

    # ── Write ─────────────────────────────────────────────────────────────────
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        epub.write_epub(output_path, book)
        size = Path(output_path).stat().st_size
        log.info("Enriched EPUB written: %s (%d bytes)", output_path, size)
    except Exception as exc:
        raise EpubError(f"could not write enriched EPUB: {exc}") from exc


# ── Public async wrapper ──────────────────────────────────────────────────────

async def inject_summary_into_epub(
    epub_path:       str | None,
    output_path:     str,
    *,
    title:           str,
    author:          str,
    summary_text:    str,
    language:        str,
    cover_path:      str | None = None,
    chapters:        list[dict] | None = None,
    chapter_audio:   dict[int, str] | None = None,
    chapter_mindmap: dict[int, dict] | None = None,
    audio_url:       str | None = None,
    mindmap_url:     str | None = None,
) -> str:
    """
    Inject AI content into `epub_path` (or create from scratch if None/missing).
    Writes the enriched EPUB to `output_path` and returns the path.
    Heavy ebooklib work runs in the executor to keep the event loop free.
    """
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        _inject_sync,
        epub_path, output_path,
        title=title, author=author, summary_text=summary_text,
        language=language, cover_path=cover_path,
        chapters=chapters, chapter_audio=chapter_audio,
        chapter_mindmap=chapter_mindmap,
        audio_url=audio_url, mindmap_url=mindmap_url,
    )
    await loop.run_in_executor(None, fn)
    return output_path
