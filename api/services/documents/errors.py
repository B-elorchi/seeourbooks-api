"""
Typed exceptions for the documents pipeline.

Each exception carries a `code` (machine-readable, stored on
`documents.error_message` for clients) and a `http_status` hint used by the
routes layer to map errors to HTTP responses.

Catch the base `DocumentError` to handle any pipeline-related failure.
"""
from __future__ import annotations


class DocumentError(Exception):
    """Base class for every documents-pipeline error."""
    code: str = "document_error"
    http_status: int = 500

    def __init__(self, message: str = "", *, detail: dict | None = None) -> None:
        super().__init__(message or self.code)
        self.detail = detail or {}


class UnsupportedFileType(DocumentError):
    """Uploaded file is not a PDF (wrong extension or bad magic bytes)."""
    code = "unsupported_file_type"
    http_status = 415


class InvalidPDFError(DocumentError):
    """The file is a PDF but malformed / unreadable / encrypted without password."""
    code = "invalid_pdf"
    http_status = 422


class OCRMissingError(DocumentError):
    """ocrmypdf / tesseract is not installed on the host."""
    code = "ocr_unavailable"
    http_status = 503


class OCRFailure(DocumentError):
    """ocrmypdf ran but exited non-zero, or timed out."""
    code = "ocr_failed"
    http_status = 500


class EmptyExtractionError(DocumentError):
    """After OCR + extraction, no usable text was produced from the PDF."""
    code = "empty_extraction"
    http_status = 422


class AIFailureError(DocumentError):
    """AI provider call failed (auth, billing, timeout, malformed JSON, etc.)."""
    code = "ai_failure"
    http_status = 502


class DocumentNotFound(DocumentError):
    """No document row with the given id."""
    code = "document_not_found"
    http_status = 404


class PageSaveError(DocumentError):
    """One or more extracted pages could not be persisted to the database."""
    code = "page_save_failed"
    http_status = 500
