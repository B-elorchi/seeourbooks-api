"""
Text extraction from a (possibly OCR'd) PDF.

We use PyMuPDF as the primary extractor — it's fast, handles Arabic + RTL
layouts correctly, and preserves line breaks better than pdfplumber for the
kinds of documents we ingest.  If PyMuPDF produces empty text for a page we
fall back to pdfplumber for that page.
"""
from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import TypedDict

from api.services.documents.errors import (
    EmptyExtractionError,
    InvalidPDFError,
)

log = logging.getLogger(__name__)


class PageText(TypedDict):
    page:    int     # 1-based page number
    content: str


# Remove form-feed, soft-hyphen, BOM, and other invisible characters that
# routinely sneak into OCR output and mess with downstream models.
_INVISIBLES_RE = re.compile(r"[­​-‏‪-‮⁠﻿\x0c]")

# Join hyphenated line breaks ("inter-\nnational" → "international").
# Common in OCR output and in justified text PDFs.
_HYPHEN_BREAK_RE = re.compile(r"-\n(\w)")

# Collapse 3+ blank lines into a single double-newline so paragraph boundaries
# are preserved without runaway whitespace.
_TRIPLE_NEWLINE_RE = re.compile(r"\n{3,}")


def _normalize(text: str) -> str:
    """Light whitespace + invisible-character cleanup. Preserves paragraph breaks."""
    if not text:
        return ""
    text = _INVISIBLES_RE.sub("", text)
    text = _HYPHEN_BREAK_RE.sub(r"\1", text)
    text = _TRIPLE_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def _extract_page_pymupdf(doc, idx: int) -> str:
    """Extract one page via PyMuPDF."""
    return doc[idx].get_text("text") or ""


def _extract_page_pdfplumber_fallback(pdf_path: str, idx: int) -> str:
    """Last-ditch fallback when PyMuPDF returns nothing for a page."""
    try:
        import pdfplumber
    except ImportError:
        return ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if idx >= len(pdf.pages):
                return ""
            return pdf.pages[idx].extract_text() or ""
    except Exception as exc:
        log.debug("pdfplumber fallback failed on page %d: %s", idx, exc)
        return ""


def extract_pages(pdf_path: str | Path) -> tuple[list[PageText], int]:
    """
    Extract text from every page of `pdf_path`.

    Returns (pages, total_pages):
        pages       — list[{page, content}] for pages with non-empty content.
                      Pages that were truly blank are omitted.
        total_pages — the actual page count of the PDF (NOT len(pages)).
                      Used by the processor to compute progress %.

    Raises:
        InvalidPDFError      — file can't be opened.
        EmptyExtractionError — every page returned empty text (likely a
                               scanned PDF that failed to OCR).
    """
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise InvalidPDFError("PyMuPDF is not installed") from exc

    pdf_path = str(pdf_path)

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise InvalidPDFError(f"Could not open PDF: {exc}") from exc

    total_pages = doc.page_count
    pages: list[PageText] = []

    try:
        for i in range(total_pages):
            text = _extract_page_pymupdf(doc, i)
            if not text.strip():
                # Fallback for tough pages — pdfplumber sometimes finds text
                # that PyMuPDF misses, especially in scanned-then-OCR'd PDFs.
                text = _extract_page_pdfplumber_fallback(pdf_path, i)

            text = _normalize(text)
            if text:
                pages.append({"page": i + 1, "content": text})
    finally:
        doc.close()

    if not pages:
        raise EmptyExtractionError(
            "No text could be extracted from any page — PDF may be a scanned "
            "image and need OCR, or it may be image-only with no text layer.",
            detail={"total_pages": total_pages},
        )

    log.info("Extracted %d/%d non-empty pages from %s", len(pages), total_pages, pdf_path)
    return pages, total_pages


# ── Language detection (very rough, fast) ────────────────────────────────────

# Unicode ranges
_AR_RANGE = (0x0600, 0x06FF)        # Arabic
_LATIN_RANGE = (0x0041, 0x007A)     # ASCII letters


def detect_language(pages: list[PageText]) -> str:
    """
    Return the primary language code for the document.

    Returns one of:
        "ara"   — majority Arabic characters
        "eng"   — majority Latin characters
        "mixed" — neither side is >70%
    """
    counter: Counter[str] = Counter()
    sample = " ".join(p["content"] for p in pages[:5])
    for ch in sample[:20_000]:
        code = ord(ch)
        if _AR_RANGE[0] <= code <= _AR_RANGE[1]:
            counter["ara"] += 1
        elif _LATIN_RANGE[0] <= code <= _LATIN_RANGE[1] or 0x0061 <= code <= 0x007A:
            counter["eng"] += 1

    total = counter["ara"] + counter["eng"]
    if total == 0:
        return "mixed"
    ar_ratio = counter["ara"] / total
    if ar_ratio >= 0.7:
        return "ara"
    if ar_ratio <= 0.3:
        return "eng"
    return "mixed"
