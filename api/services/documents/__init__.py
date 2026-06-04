"""
Documents pipeline — OCR + text extraction + AI structured analysis + chunking.

Public surface (used by `api.routes.documents` and `api.services.documents.processor`):

    DocumentError, InvalidPDFError, OCRMissingError, OCRFailure,
    EmptyExtractionError, AIFailureError, UnsupportedFileType, DocumentNotFound
"""
from api.services.documents.errors import (
    DocumentError,
    InvalidPDFError,
    OCRMissingError,
    OCRFailure,
    EmptyExtractionError,
    AIFailureError,
    UnsupportedFileType,
    DocumentNotFound,
)

__all__ = [
    "DocumentError",
    "InvalidPDFError",
    "OCRMissingError",
    "OCRFailure",
    "EmptyExtractionError",
    "AIFailureError",
    "UnsupportedFileType",
    "DocumentNotFound",
]
