"""
OCR service for the documents pipeline.

Two responsibilities:
  1. needs_ocr(pdf_path)       — detect whether a PDF already has a text layer.
                                  Scanned-image PDFs need OCR; born-digital PDFs
                                  do not.
  2. run_ocrmypdf(in, out, ...) — shell out to ocrmypdf to produce a searchable
                                  PDF with embedded text from tesseract.

We intentionally shell out to the system `ocrmypdf` binary instead of importing
the `ocrmypdf` Python module directly — ocrmypdf's internal API is not stable
between minor versions, while the CLI is.  This also lets the host install a
single OS package and not depend on the python module being importable.

Host requirements (Debian/Ubuntu):
    sudo apt-get install -y tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng \
                            ghostscript ocrmypdf
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import subprocess
from pathlib import Path

from api.services.documents.errors import (
    InvalidPDFError,
    OCRFailure,
    OCRMissingError,
)

log = logging.getLogger(__name__)

# How long ocrmypdf is allowed to run on a single PDF before we abort.
# Large scanned books take a while — tesseract is single-threaded per page by
# default, and ocrmypdf parallelises across pages.  30 minutes is generous
# without being unbounded.
_OCR_TIMEOUT_SEC = 30 * 60


# ── Detection: does this PDF already have a text layer? ──────────────────────

def needs_ocr(pdf_path: str | os.PathLike) -> bool:
    """
    Return True when the PDF should be run through OCR.

    Heuristic: open the PDF with PyMuPDF, pull text from the first 5 pages.
    If the total extractable text is less than ~50 characters, we assume the
    PDF is a scanned image (or a malformed text-less PDF) and needs OCR.
    Born-digital PDFs always have plenty of text on page 1.

    Lazy import of fitz so the import error from a broken install surfaces
    at the boundary instead of at module load.
    """
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        # Catch BaseException-ish failures too — on Windows, PyMuPDF can fail
        # at import with an OSError / FileNotFoundError when the bundled DLLs
        # can't be loaded (architecture mismatch, missing VC++ runtime, etc.).
        # We want to surface the actual cause to the admin, not a misleading
        # "not installed" message.
        import sys as _sys
        raise OCRMissingError(
            f"PyMuPDF (fitz) failed to import: {type(exc).__name__}: {exc}. "
            f"Python interpreter is {_sys.executable!r} (version {_sys.version.split()[0]}). "
            f"Install PyMuPDF in THIS interpreter with: "
            f"\"{_sys.executable}\" -m pip install PyMuPDF==1.24.10"
        ) from exc

    try:
        with fitz.open(str(pdf_path)) as doc:
            if doc.is_encrypted:
                raise InvalidPDFError("PDF is encrypted — cannot be processed")
            sample_pages = min(5, doc.page_count)
            chars = 0
            for i in range(sample_pages):
                chars += len((doc[i].get_text("text") or "").strip())
                if chars > 100:
                    return False  # already has text — fast path
            return chars < 50
    except InvalidPDFError:
        raise
    except Exception as exc:
        raise InvalidPDFError(f"Could not open PDF: {exc}") from exc


# ── OCR run ──────────────────────────────────────────────────────────────────

def _ocrmypdf_available() -> bool:
    """Cached check that the ocrmypdf binary is on PATH."""
    return shutil.which("ocrmypdf") is not None


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


async def run_ocrmypdf(
    input_path:  str | os.PathLike,
    output_path: str | os.PathLike,
    *,
    languages: str = "ara+eng",
) -> Path:
    """
    Run ocrmypdf on `input_path`, write a searchable PDF to `output_path`.

    Idempotent: if `output_path` already exists with non-zero size, we assume
    a previous OCR succeeded and skip re-running.

    Raises:
        OCRMissingError — ocrmypdf or tesseract not installed.
        OCRFailure      — ocrmypdf ran but failed (non-zero exit, timeout).
        InvalidPDFError — the input PDF is malformed.
    """
    input_path  = Path(input_path)
    output_path = Path(output_path)

    # Idempotent skip
    if output_path.exists() and output_path.stat().st_size > 0:
        log.info("OCR output already exists at %s — skipping", output_path)
        return output_path

    if not _ocrmypdf_available():
        raise OCRMissingError(
            "ocrmypdf is not installed on this host. "
            "Install with: sudo apt-get install -y ocrmypdf tesseract-ocr "
            "tesseract-ocr-ara tesseract-ocr-eng ghostscript"
        )
    if not _tesseract_available():
        raise OCRMissingError(
            "tesseract is not installed on this host. "
            "Install with: sudo apt-get install -y tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ocrmypdf flags:
    #   --skip-text       Don't re-OCR pages that already have text — safer than --force-ocr
    #                     for mixed documents (some scanned, some digital).
    #   --jobs            Parallel pages.  os.cpu_count() / 2 stays polite.
    #   --output-type pdf Standard PDF (not PDF/A) — smaller files, faster.
    #   --quiet           Silences progress bars (we capture stderr ourselves).
    jobs = max(1, (os.cpu_count() or 2) // 2)
    cmd = [
        "ocrmypdf",
        "-l", languages,
        "--skip-text",
        "--jobs", str(jobs),
        "--output-type", "pdf",
        "--quiet",
        str(input_path),
        str(output_path),
    ]

    log.info("Running OCR: %s", " ".join(cmd))

    # Run subprocess on a thread so we don't block the event loop.
    loop = asyncio.get_running_loop()

    def _run() -> subprocess.CompletedProcess:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            timeout=_OCR_TIMEOUT_SEC,
        )

    try:
        proc = await loop.run_in_executor(None, _run)
    except subprocess.TimeoutExpired as exc:
        raise OCRFailure(
            f"ocrmypdf timed out after {_OCR_TIMEOUT_SEC // 60} minutes",
            detail={"input": str(input_path)},
        ) from exc
    except FileNotFoundError as exc:
        # ocrmypdf binary vanished mid-run (container hot-swap, PATH change)
        raise OCRMissingError("ocrmypdf binary disappeared mid-run") from exc

    # ocrmypdf returns exit codes documented at
    # https://ocrmypdf.readthedocs.io/en/latest/advanced.html#return-codes
    #   0 = success
    #   6 = already has text + --skip-text → input copied to output (still success)
    #   anything else = failure
    if proc.returncode in (0, 6):
        if not output_path.exists() or output_path.stat().st_size == 0:
            raise OCRFailure(
                "ocrmypdf reported success but produced no output file",
                detail={"stderr": proc.stderr.decode(errors="replace")[-500:]},
            )
        return output_path

    # Specific exit codes worth surfacing
    stderr = proc.stderr.decode(errors="replace")[-800:]
    if proc.returncode == 2:
        raise InvalidPDFError(
            f"ocrmypdf rejected input PDF: {stderr.strip()[:400]}",
        )
    raise OCRFailure(
        f"ocrmypdf exited with code {proc.returncode}",
        detail={"stderr": stderr},
    )
