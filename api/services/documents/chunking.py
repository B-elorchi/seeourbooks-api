"""
Word-count based chunking for the knowledge base.

We target ~750 words per chunk by default — small enough for most embedding
models' context (≈1000 tokens), large enough to carry semantic context.

Strategy:
  1. Split the full document into paragraphs (separated by blank lines).
  2. Greedily concatenate paragraphs into chunks until we hit the word target.
  3. If a single paragraph is bigger than the target, split it on sentence
     boundaries.  If even a single sentence is bigger (rare), split on words.

Each chunk preserves natural boundaries — important for embedding quality
because mid-sentence cuts destroy semantic coherence.
"""
from __future__ import annotations

import re
from typing import TypedDict


class Chunk(TypedDict):
    chunk_index: int
    content:     str
    word_count:  int


# Paragraph = one or more blank lines between blocks of text
_PARAGRAPH_RE = re.compile(r"\n\s*\n+")

# Sentence boundary: ., !, ?, Arabic question mark, ellipsis followed by space
_SENTENCE_RE  = re.compile(r"(?<=[.!?؟…])\s+")


def _word_count(s: str) -> int:
    return len(s.split())


def _split_long_paragraph(para: str, target: int) -> list[str]:
    """Break an over-sized paragraph on sentence (then word) boundaries."""
    pieces: list[str] = []
    buf = ""
    for sent in _SENTENCE_RE.split(para):
        sent = sent.strip()
        if not sent:
            continue
        candidate = (buf + " " + sent).strip() if buf else sent
        if _word_count(candidate) > target:
            if buf:
                pieces.append(buf.strip())
            # If even one sentence is bigger than the target, hard-split on words
            if _word_count(sent) > target:
                words = sent.split()
                for i in range(0, len(words), target):
                    pieces.append(" ".join(words[i: i + target]))
                buf = ""
            else:
                buf = sent
        else:
            buf = candidate
    if buf.strip():
        pieces.append(buf.strip())
    return pieces


def chunk_text(text: str, target_words: int = 750) -> list[Chunk]:
    """
    Split `text` into ~target_words chunks for the knowledge base.

    `text` should be the concatenated text of every extracted page (or any
    other corpus the caller wants to chunk).  Empty / whitespace-only input
    returns [].
    """
    if not text or not text.strip():
        return []

    target = max(50, target_words)
    paragraphs = [p.strip() for p in _PARAGRAPH_RE.split(text) if p.strip()]

    chunks: list[str] = []
    buf = ""
    buf_words = 0

    def flush() -> None:
        nonlocal buf, buf_words
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""
        buf_words = 0

    for para in paragraphs:
        words = _word_count(para)

        # Outlier paragraph — split it independently
        if words > target * 1.5:
            flush()
            for piece in _split_long_paragraph(para, target):
                chunks.append(piece)
            continue

        # Adding this paragraph would push us past the target — flush first
        if buf_words and buf_words + words > target:
            flush()

        buf = (buf + "\n\n" + para).strip() if buf else para
        buf_words += words

    flush()

    return [
        {"chunk_index": i, "content": c, "word_count": _word_count(c)}
        for i, c in enumerate(chunks)
    ]
