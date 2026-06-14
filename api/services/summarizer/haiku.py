import logging

from api.config.settings import settings
from api.services.ai_client import chat_complete
from api.services.db import find, upsert

log = logging.getLogger(__name__)

# A chunk below this many words is unlikely to yield a useful summary — skip
# rather than waste a model call. (TOC fragments, blank pages, headers, etc.)
_MIN_CHUNK_WORDS = 20


AR_TASHKEEL_INSTRUCTION = (
    "\n\nمهم جداً: اكتب النص العربي بالتشكيل الكامل (الفتحة والضمة والكسرة والسكون والشدة والتنوين) "
    "على كل كلمة، وبشكل خاص على أواخر الكلمات (الإعراب) وعلى الكلمات التي تحتمل أكثر من قراءة. "
    "النص سيُستخدم مباشرةً في تحويل النص إلى صوت (TTS) لذا الدقة في النطق ضرورية."
)


async def run_haiku_pass(
    book_id: str,
    chunks: list[dict],
    language: str,
    model: str | None = None,
    max_words: int | None = None,
) -> list[str]:
    """
    Summarise each chunk — Pass 1 of the pipeline.
    Cached per (chunk_id, language). Skips any chunk already in DB.
    Supports any model via ai_client routing (Anthropic, OpenAI, OpenRouter).

    max_words — optional cap on each chapter summary's length (admin-configurable
    via CHAPTER_SUMMARY_MAX_WORDS). When None/0, uses the default "3-5 sentences".

    Resilience: if the primary model returns an empty completion (rate-limit
    swallowed by SDK, content filter, reasoning budget burn, etc.), we retry
    once with an OpenRouter Claude Haiku fallback. Only after both attempts
    return empty do we raise — so a single dud model call no longer fails the
    whole job. All fallbacks route through OpenRouter so we never depend on a
    separate native Anthropic / OpenAI key for resilience.
    """
    primary = model or settings.MODEL_HAIKU
    # Always route the local "empty completion" fallback through OpenRouter so
    # this path works even when no native Anthropic key is configured. If the
    # primary IS this OpenRouter Haiku, drop the fallback to avoid a pointless
    # retry against the same endpoint.
    _OR_HAIKU = "anthropic/claude-haiku-4-5"
    fallback = None if primary == _OR_HAIKU else _OR_HAIKU
    lang_name = "Arabic" if language == "ar" else "English"
    tashkeel  = AR_TASHKEEL_INSTRUCTION if language == "ar" else ""
    summaries: list[str] = []

    if max_words and max_words > 0:
        length_instr = f"in about {max_words} words"
        max_tokens   = int(max_words * 2.5) + 100
    else:
        length_instr = "in 3-5 sentences"
        max_tokens   = 512

    prompt_template = (
        f"Summarise this book excerpt in {lang_name} {length_instr}. "
        f"Focus on key ideas.{tashkeel}\n\n{{content}}"
    )

    for chunk in chunks:
        cid, idx = chunk["id"], chunk["chunk_index"]
        content = (chunk.get("content") or "").strip()
        word_count = len(content.split())

        existing = await find(
            "chunk_summaries",
            filters={"chunk_id": cid, "language": language},
            select="summary",
            limit=1,
        )
        if existing:
            summaries.append(existing[0]["summary"])
            continue

        # Skip unsalvageable chunks BEFORE calling the model — saves cost and
        # surfaces the real problem (no extractable text) in the logs.
        if word_count < _MIN_CHUNK_WORDS:
            log.warning(
                "Skipping chunk %s: only %d words of content — likely an "
                "extraction failure or blank section.",
                cid, word_count,
            )
            raise ValueError(
                f"Chunk {cid} has only {word_count} words — source text appears "
                f"empty or unextractable."
            )

        messages = [{"role": "user", "content": prompt_template.format(content=content)}]

        summary = await chat_complete(model=primary, messages=messages, max_tokens=max_tokens)

        if (not summary or not summary.strip()) and fallback:
            log.warning(
                "Model %r returned empty summary for chunk %s (%d words) — "
                "retrying with fallback %r",
                primary, cid, word_count, fallback,
            )
            summary = await chat_complete(model=fallback, messages=messages, max_tokens=max_tokens)

        if not summary or not summary.strip():
            raise ValueError(
                f"Model returned empty summary for chunk {cid} "
                f"(primary={primary}, fallback={fallback}, words={word_count})"
            )

        await upsert(
            "chunk_summaries",
            {
                "book_id":     book_id,
                "chunk_id":    cid,
                "chunk_index": idx,
                "language":    language,
                "summary":     summary,
                "model":       model,
            },
            "chunk_id,language",
        )
        summaries.append(summary)

    return summaries
