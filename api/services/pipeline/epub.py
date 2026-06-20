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
import io
import logging
import re
import warnings
import zipfile
from html import escape
from pathlib import Path
from urllib.parse import quote

import httpx

# Silence ebooklib's own deprecation noise (ignore_ncx default + rootfile xpath).
# These are harmless library warnings emitted on every read_epub() and only
# clutter the server logs, hiding real errors. Scoped to the ebooklib module so
# we don't suppress warnings from anywhere else.
warnings.filterwarnings("ignore", category=UserWarning,   module="ebooklib.epub")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib.epub")

from api.config.settings import settings


# ── Chapter-number parsing (for interleaving insights after the right chapter) ─

_ROMAN_VALUES = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}


def _roman_to_int(s: str) -> int | None:
    s = s.lower().strip()
    if not s or any(ch not in _ROMAN_VALUES for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        val = _ROMAN_VALUES[ch]
        total += -val if val < prev else val
        prev = max(prev, val)
    return total or None


def _parse_chapter_num(title: str | None) -> int | None:
    """Extract a chapter number from a title like 'CHAPTER XII. ...' or 'Chapter 7'."""
    if not title:
        return None
    t = title.strip()
    # "Chapter 12" / "Chapter XII"
    m = re.search(r"chapter\s+([ivxlcdm]+|\d+)", t, re.IGNORECASE)
    if m:
        tok = m.group(1)
        return int(tok) if tok.isdigit() else _roman_to_int(tok)
    # Leading "12." or "12 "
    m = re.match(r"\s*(\d+)[\.\s]", t)
    if m:
        return int(m.group(1))
    return None

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


# ── EPUB loading helpers ──────────────────────────────────────────────────────

def _read_epub_patched(epub_path: str, original_exc: Exception):
    """
    Fallback reader: when ebooklib fails because the zip has manifest entries
    pointing to files that don't exist (e.g. a font that was deleted from the
    archive), rebuild the zip in-memory with those items removed from
    content.opf, then re-read with ebooklib.

    Returns an EpubBook on success, or None if the patched zip also fails.
    """
    try:
        with zipfile.ZipFile(epub_path, "r") as zin:
            names_in_zip = set(zin.namelist())

            # Find and patch the OPF manifest to remove missing items
            opf_name: str | None = None
            for name in names_in_zip:
                if name.endswith(".opf"):
                    opf_name = name
                    break

            if opf_name is None:
                return None

            opf_bytes = zin.read(opf_name)
            # Remove <item> elements whose href points to a missing file.
            # The href may be relative to the OPF directory.
            opf_dir = opf_name.rsplit("/", 1)[0] + "/" if "/" in opf_name else ""
            def _item_missing(m: re.Match) -> str:
                tag = m.group(0)
                href_m = re.search(r'href=["\']([^"\']+)["\']', tag)
                if not href_m:
                    return tag
                href = href_m.group(1)
                full = opf_dir + href if not href.startswith("/") else href.lstrip("/")
                return "" if full not in names_in_zip else tag

            patched_opf = re.sub(
                r'<item\b[^>]*/?>',
                _item_missing,
                opf_bytes.decode("utf-8", errors="replace"),
            ).encode("utf-8")

            # Rebuild zip in-memory with patched OPF
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
                for name in names_in_zip:
                    if name == opf_name:
                        zout.writestr(name, patched_opf)
                    else:
                        zout.writestr(name, zin.read(name))

        buf.seek(0)
        # Write patched bytes back to a temp path, then read with ebooklib
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".epub", delete=False) as tmp:
            tmp.write(buf.read())
            tmp_path = tmp.name
        try:
            from ebooklib import epub as _epub
            book = _epub.read_epub(tmp_path)
        finally:
            os.unlink(tmp_path)
        return book

    except Exception as patch_exc:
        log.warning("EPUB manifest patch also failed (%s) — will build from scratch", patch_exc)
        return None


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
    
    # Debug logging
    log.info("EPUB injection starting for %r", title)
    log.info("  - summary length: %d chars", len(summary_text or ""))
    log.info("  - cover: %s", cover_path if cover_path and Path(cover_path).is_file() else "None")
    log.info("  - audio_url: %s", audio_url or "None")
    log.info("  - mindmap_url: %s", mindmap_url or "None")
    log.info("  - chapters: %d", len(chapters))
    log.info("  - chapter_audio keys: %s", list(chapter_audio.keys()))
    log.info("  - chapter_mindmap keys: %s", list(chapter_mindmap.keys()))

    # ── Load or create the EPUB book ─────────────────────────────────────────
    from_scratch = True
    if epub_path and Path(epub_path).is_file() and Path(epub_path).stat().st_size > 100:
        try:
            book = epub.read_epub(epub_path)
            from_scratch = False
            log.info("Loaded source EPUB: %s", epub_path)
        except Exception as exc:
            # Some EPUBs have manifest entries for files not present in the zip
            # (e.g. a font referenced in content.opf but missing from the archive).
            # ebooklib hard-fails on these. Patch: rebuild the zip without the
            # missing items and re-read, so we keep all real content.
            book = _read_epub_patched(epub_path, exc)
            if book is not None:
                from_scratch = False
                log.info("Loaded source EPUB after patching missing manifest items: %s", epub_path)
            else:
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
    # IMPORTANT: use create_page=False. ebooklib's default create_page=True adds
    # an empty cover XHTML whose body can't be parsed during nav generation,
    # which crashes write_epub with "Document is empty". We embed the image +
    # cover metadata (for the reader thumbnail) and build our own cover PAGE
    # below so the cover is also visible as the first page.
    cover_page_item = None
    if cover_path and Path(cover_path).is_file():
        try:
            data       = Path(cover_path).read_bytes()
            media      = _detect_image_media_type(data, Path(cover_path).suffix)
            ext        = "jpg" if media == "image/jpeg" else media.split("/", 1)[1]
            cover_name = f"cover.{ext}"
            book.set_cover(cover_name, data, create_page=False)

            # A valid cover page that displays the image full-bleed.
            cover_html = (
                "<?xml version='1.0' encoding='utf-8'?>\n"
                "<!DOCTYPE html>\n"
                '<html xmlns="http://www.w3.org/1999/xhtml"><head>'
                '<meta charset="utf-8"/><title>Cover</title>'
                "<style>body{margin:0;padding:0;text-align:center;}"
                "img{max-width:100%;max-height:100vh;}</style></head>"
                f'<body><img src="{cover_name}" alt="Cover"/></body></html>'
            )
            cover_page_item = epub.EpubHtml(
                uid       = "seeourbook_cover",
                title     = "Cover",
                file_name = "seeourbook_cover.xhtml",
                lang      = language or "en",
            )
            cover_page_item.content = cover_html.encode("utf-8")
            book.add_item(cover_page_item)
            log.info("Cover set (%s, %d bytes) + cover page added", media, len(data))
        except Exception as exc:
            log.warning("Cover set failed (%s) — continuing", exc)

    # ── 2. Front-matter page ──────────────────────────────────────────────────
    log.info("Building front matter page...")
    front_item = _build_front_matter(
        epub,
        slug=slug, title=title, author=author,
        summary_text=summary_text, language=language,
        audio_url=audio_url, mindmap_url=mindmap_url,
    )
    book.add_item(front_item)
    log.info("Front matter page added: %s", front_item.file_name)

    # ── 3. Per-chapter insights pages ─────────────────────────────────────────
    # Build a map: chapter index → insights item
    log.info("Building chapter insight pages...")
    insight_items: dict[int, object] = {}
    for ch in chapters:
        ch_index = ch.get("index", 0)
        ch_summary = (ch.get("summary") or "").strip()
        log.info("  Chapter %d: title=%r, has_summary=%s", ch_index, ch.get("title"), bool(ch_summary))
        item = _build_chapter_insights(
            epub,
            slug=slug, ch=ch, language=language,
            chapter_audio=chapter_audio,
            chapter_mindmap=chapter_mindmap,
        )
        if item:
            book.add_item(item)
            insight_items[ch_index] = item
            log.info("    -> insight page added: %s", item.file_name)
        else:
            log.info("    -> no insight page (no summary)")
    log.info("Total insight pages: %d", len(insight_items))

    # ── Rebuild spine ─────────────────────────────────────────────────────────
    # Goal:
    #   [cover] → nav → front-matter → [orig chapter, its insight, orig chapter, its insight…]
    #
    # Matching strategy:
    #   1. Build a map from original spine item → chapter number using ToC titles
    #   2. Walk the original spine in order; after each item whose chapter number
    #      matches an insight index, insert that insight
    #   3. Any unmatched insights are appended at the end
    is_arabic       = (language or "en").lower() == "ar"
    injected_fnames = {front_item.file_name} | {
        it.file_name for it in insight_items.values()
    }
    if cover_page_item is not None:
        injected_fnames.add(cover_page_item.file_name)

    def _resolve_spine_entry(sp):
        """Return (item_or_idref, fname). Resolves idref strings to item objects."""
        idref = sp[0] if isinstance(sp, tuple) else sp
        if isinstance(idref, str):
            item = book.get_item_with_id(idref)
            if item is not None:
                return item, getattr(item, "file_name", None)
            return idref, idref          # unresolved special (e.g. "nav")
        # already an item object
        return idref, getattr(idref, "file_name", None)

    # Normalise original spine → ordered list of resolved content items
    try:
        from ebooklib.epub import EpubNav, EpubNcx
    except Exception:  # pragma: no cover
        EpubNav = EpubNcx = ()  # type: ignore

    orig_items: list = []
    for sp in book.spine:
        idref = sp[0] if isinstance(sp, tuple) else sp
        # Skip nav/cover by their special idref before resolving
        if isinstance(idref, str) and idref.lower() in ("nav", "cover"):
            continue
        item, fname = _resolve_spine_entry(sp)
        # Skip nav/cover documents and anything we injected
        if fname in (None, "nav", "cover", "nav.xhtml"):
            continue
        if EpubNav and isinstance(item, (EpubNav, EpubNcx)):
            continue
        if (fname or "") in injected_fnames:
            continue
        orig_items.append((item, fname))

    log.info("Original spine items (after filtering): %d", len(orig_items))

    # insights sorted by chapter index
    sorted_chs = sorted(chapters, key=lambda c: c.get("index", 0))
    insights_in_order = [
        (ch.get("index", 0), insight_items[ch.get("index", 0)])
        for ch in sorted_chs
        if ch.get("index", 0) in insight_items
    ]
    insight_by_index = {idx: it for idx, it in insights_in_order}

    # Build a map: file_name → chapter number, from the original ToC titles.
    # We store BOTH the raw fname and fname+.xhtml to handle ebooklib's
    # idref vs file_name mismatch.
    fname_to_num: dict[str, int] = {}
    def _walk_toc(entries):
        for e in entries or []:
            if isinstance(e, tuple):                 # (Section, [children])
                _walk_toc(e[1])
            else:
                href  = getattr(e, "href", "") or ""
                fname = href.split("#", 1)[0]
                num   = _parse_chapter_num(getattr(e, "title", ""))
                if fname and num is not None:
                    fname_to_num.setdefault(fname, num)
                    # Also store without .xhtml extension for idref matching
                    if fname.endswith('.xhtml'):
                        fname_to_num.setdefault(fname[:-6], num)
                    elif fname.endswith('.html'):
                        fname_to_num.setdefault(fname[:-5], num)
    _walk_toc(list(book.toc) if book.toc else [])
    log.info("ToC chapter number map: %s", fname_to_num)

    # Identify which spine items are "chapter-like" (have a chapter number in ToC).
    # We match insights to chapters by RELATIVE ORDER: the 1st chapter-like item
    # gets the 1st insight, the 2nd chapter-like item gets the 2nd insight, etc.
    # This works regardless of whether chapter numbers are 0-based or 1-based.
    chapter_like_items: list[tuple[object, str, int]] = []   # (item, fname, position)
    non_chapter_items: list[tuple[object, str]] = []         # (item, fname)
    for pos, (item, fname) in enumerate(orig_items):
        if fname_to_num.get(fname or "") is not None:
            chapter_like_items.append((item, fname, pos))
        else:
            non_chapter_items.append((item, fname))
    
    log.info("Chapter-like spine items: %d, non-chapter items: %d", 
             len(chapter_like_items), len(non_chapter_items))

    # Map: relative chapter position (0,1,2…) → insight index
    # insights_in_order is already sorted by chapter index
    insight_for_ch_pos: dict[int, tuple[int, object]] = {}
    for ch_pos, (idx, it) in enumerate(insights_in_order):
        insight_for_ch_pos[ch_pos] = (idx, it)

    # Cover page first (if we built one), then nav, then the summary front matter.
    new_spine: list = []
    if cover_page_item is not None:
        new_spine.append(cover_page_item)
    new_spine += ["nav", front_item]
    used_indices: set[int] = set()

    if orig_items:
        # Rebuild spine: non-chapter items first (in order), then interleave
        # chapter items with their insights.
        # Actually, we need to preserve original order, so we walk orig_items
        # and insert insights right after chapter-like items.
        ch_pos = 0
        for item, fname in orig_items:
            new_spine.append(item)
            is_chapter = fname_to_num.get(fname or "") is not None
            if is_chapter:
                if ch_pos in insight_for_ch_pos:
                    idx, it = insight_for_ch_pos[ch_pos]
                    if idx not in used_indices:
                        new_spine.append(it)
                        used_indices.add(idx)
                        log.info("  Interleaved insight for chapter pos %d (index %d) after %s", 
                                 ch_pos, idx, fname)
                ch_pos += 1
        
        # Any insights we couldn't match → append at the end in order
        for idx, it in insights_in_order:
            if idx not in used_indices:
                new_spine.append(it)
                log.info("  Appended unmatched insight for chapter index %d at end", idx)
    else:
        # From scratch (no original content): front then all insights in order
        for idx, it in insights_in_order:
            new_spine.append(it)

    book.spine = new_spine
    log.info("EPUB spine rebuilt with %d items", len(new_spine))

    # ── Rebuild ToC ───────────────────────────────────────────────────────────
    toc_items: list = [
        epub.Link(front_item.file_name, front_item.title, front_item.id)
    ]

    # Keep the original ToC structure (minus any previously-injected pages)
    def _clean_toc(entries):
        out = []
        for e in entries or []:
            if isinstance(e, tuple):
                sec, kids = e
                out.append((sec, _clean_toc(kids)))
            elif hasattr(e, "href") and e.href.split("#", 1)[0] not in injected_fnames:
                out.append(e)
        return out
    toc_items += _clean_toc(list(book.toc) if book.toc else [])

    # Chapter insights section (also navigable as a group).
    # Nested ToC sections must be a (Section, [children]) tuple — NOT a bare
    # Section (whose 2nd ctor arg is an href, not a child list).
    if insight_items:
        insights_section_title = "تحليل الفصول" if is_arabic else "Chapter Insights"
        toc_items.append((
            epub.Section(insights_section_title),
            [
                epub.Link(it.file_name, it.title, it.id)
                for _, it in insights_in_order
            ],
        ))

    book.toc = tuple(toc_items)
    log.info("EPUB ToC rebuilt with %d items", len(toc_items))

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
