from pathlib import Path
from api.config.settings import settings


def find_book_text(book_id: str) -> str | None:
    """Search for book text file in english/, arabic/, or root TEXT_DIR."""
    for subdir in ["english", "arabic", ""]:
        for ext in [".txt", ".md"]:
            p = (
                settings.TEXT_DIR / subdir / f"{book_id}{ext}"
                if subdir
                else settings.TEXT_DIR / f"{book_id}{ext}"
            )
            if p.exists():
                return p.read_text(encoding="utf-8")
    return None


def chunk_text(text: str, max_words: int | None = None) -> list[str]:
    """Split text into ~max_words chunks, breaking at sentence boundaries."""
    if max_words is None:
        max_words = settings.CHUNK_SIZE_WORDS

    words, chunks = text.split(), []
    while words:
        part = " ".join(words[:max_words])
        words = words[max_words:]
        if words:
            cut = part.rfind(". ")
            if cut > len(part) * 0.7:
                words = part[cut + 2:].split() + words
                part = part[:cut + 1]
        chunks.append(part.strip())
    return [c for c in chunks if c]
