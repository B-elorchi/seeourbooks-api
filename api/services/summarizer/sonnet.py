import asyncio
import logging
import re
from typing import AsyncIterator

from api.config.settings import settings, SUMMARY_LENGTHS
from api.services.ai_client import chat_complete, chat_stream
from api.services.summarizer.haiku import AR_TASHKEEL_INSTRUCTION

log = logging.getLogger(__name__)

def _strip_tashkeel(text: str) -> str:
    return re.sub(r'[\u0617-\u061A\u064B-\u0652]', '', text)

STYLE_MAP = {
    "narrative": "flowing narrative prose",
    "bullets":   "clear bullet points",
    "academic":  "formal academic style",
}

# Approx. safe size (characters) for the combined section summaries fed into the
# final summary prompt. Most models allow far more, but staying conservative
# leaves room for the instructions + the generated output and avoids a
# context_length_exceeded error on very large books (hundreds of chapters).
# ~120k chars ≈ ~30k tokens — comfortably inside a 32k-context model with output.
_MAX_COMBINED_CHARS = 120_000

# Substrings (lower-cased) that indicate the model rejected the request because
# the input was too large — used to trigger a reduce-and-retry fallback.
_CONTEXT_ERROR_HINTS = (
    "context length", "context_length", "maximum context", "context window",
    "too long", "prompt is too long", "input is too long", "too many tokens",
    "reduce the length", "string too long", "max_tokens",
)


def _looks_like_context_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return any(h in msg for h in _CONTEXT_ERROR_HINTS)


def _combined_size(items: list[str]) -> int:
    """Approx. size of the joined section summaries, incl. separators."""
    return sum(len(s) for s in items) + 6 * len(items)


async def _reduce_chunk_summaries(
    chunk_summaries: list[str],
    language: str,
    model: str,
    max_chars: int = _MAX_COMBINED_CHARS,
) -> list[str]:
    """
    Collapse a very large set of section summaries into a smaller set that fits
    the model's context (map-reduce). Groups summaries into batches under the
    budget, condenses each batch into one intermediate summary, and repeats
    until the whole set fits. Returns the (possibly unchanged) reduced list.
    """
    if _combined_size(chunk_summaries) <= max_chars or len(chunk_summaries) <= 1:
        return chunk_summaries

    lang_name   = "Arabic" if language == "ar" else "English"
    batch_budget = max(max_chars // 3, 20_000)
    sem = asyncio.Semaphore(4)   # bound fan-out so we don't trip rate limits

    async def _reduce_one(batch: list[str]) -> str:
        combined = "\n\n---\n\n".join(batch)
        prompt = (
            f"Condense the following section summaries into ONE cohesive "
            f"{lang_name} summary that preserves all key topics, names, events "
            f"and details. Output ONLY the summary — no preamble.\n\n{combined}"
        )
        async with sem:
            return await chat_complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2000,
            )

    current = chunk_summaries
    # Bounded passes — each level shrinks the set; 4 is plenty even for 1000+ chapters.
    for _ in range(4):
        if _combined_size(current) <= max_chars:
            break
        batches: list[list[str]] = []
        buf: list[str] = []
        size = 0
        for s in current:
            s_len = len(s) + 6
            if buf and size + s_len > batch_budget:
                batches.append(buf)
                buf, size = [], 0
            buf.append(s)
            size += s_len
        if buf:
            batches.append(buf)
        log.info(
            "Sonnet pass: reducing %d section summaries (%d chars) → %d batch(es)",
            len(current), _combined_size(current), len(batches),
        )
        current = list(await asyncio.gather(*[_reduce_one(b) for b in batches]))

    return current


def _build_prompt(chunk_summaries: list[str], length: str, style: str, language: str,
                  target_words: int | None = None,
                  missing_topics: list[str] | None = None,
                  tashkeel_enabled: bool = True) -> str:
    lang_name = "Arabic" if language == "ar" else "English"
    target    = target_words if (target_words and target_words > 0) else SUMMARY_LENGTHS.get(length, 750)
    tashkeel  = AR_TASHKEEL_INSTRUCTION if (language == "ar" and tashkeel_enabled) else ""
    combined  = "\n\n---\n\n".join(
        f"Section {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )
    missing_note = ""
    if missing_topics:
        missing_list = "\n".join(f"- {t}" for t in missing_topics)
        missing_note = (
            f"\n\nCRITICAL — a previous version of this summary scored below the "
            f"required coverage threshold because it missed these topics. You MUST "
            f"explicitly cover ALL of them in your summary:\n{missing_list}"
        )
    return (
        f"Create a book summary in {lang_name} for an audio presentation "
        f"of ~{length} (~{target} words).\n"
        f"Style: {STYLE_MAP.get(style, 'narrative prose')}.\n"
        f"Based on these section summaries:\n\n{combined}\n\n"
        f"Write the complete summary in {lang_name}. Target: ~{target} words.\n"
        f"IMPORTANT: Output ONLY the summary itself — no preamble, no introduction, "
        f"no 'Of course' or 'Here is', no script markers, no horizontal rules, "
        f"no closing remarks. Start directly with the first sentence of the summary."
        f"{missing_note}"
        f"{tashkeel}"
    )


async def run_sonnet_pass(
    chunk_summaries: list[str],
    length: str,
    style: str,
    language: str,
    model: str | None = None,
    missing_topics: list[str] | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """
    Streaming version — yields ("token", text) and finally ("full", full_text).
    Used by the /api/summarize SSE endpoint.
    Supports any model via ai_client routing (Anthropic, OpenAI, OpenRouter).
    """
    model  = model or settings.MODEL_SONNET
    target = SUMMARY_LENGTHS.get(length, 750)
    chunk_summaries = await _reduce_chunk_summaries(chunk_summaries, language, model)
    prompt = _build_prompt(chunk_summaries, length, style, language, missing_topics=missing_topics)
    full   = ""

    async for token in chat_stream(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=min(4096, target * 2),
    ):
        full += token
        yield "token", token

    yield "full", full


async def run_sonnet_pass_sync(
    chunk_summaries: list[str],
    length: str,
    style: str,
    language: str,
    model_override: str | None = None,
    max_words: int | None = None,
    missing_topics: list[str] | None = None,
    tashkeel_enabled: bool = True,
) -> str:
    """
    Non-streaming version — returns the complete summary text.
    Used by the pipeline orchestrator.
    Supports any model via ai_client routing.

    max_words — admin override (SUMMARY_MAX_WORDS_*). When None/0, the length
    preset is used (3min=450, 5min=750, 10min=1500, 15min=2250).
    missing_topics — topics the previous summary missed (from QA); included in
    the prompt so the model knows what to fix on retry.
    """
    model  = model_override or settings.MODEL_SONNET
    target = max_words if (max_words and max_words > 0) else SUMMARY_LENGTHS.get(length, 750)

    # Proactively collapse a very large set of section summaries so the final
    # prompt fits the model context (big books = hundreds of chapters).
    chunk_summaries = await _reduce_chunk_summaries(chunk_summaries, language, model)
    prompt = _build_prompt(chunk_summaries, length, style, language,
                           target_words=target, missing_topics=missing_topics,
                           tashkeel_enabled=tashkeel_enabled)

    try:
        result = await chat_complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(4096, target * 2),
        )
        if language == "ar":
            clean_len = len(_strip_tashkeel(result))
            if clean_len < 3000:
                log.info("Arabic summary too short (%d chars without tashkeel). Expanding...", clean_len)
                expansion_prompt = "النص السابق قصير جداً. الرجاء توسيع وتفصيل الملخص بشكل كبير ليصبح طويلاً ومفصلاً مع الحفاظ على الأسلوب المطلوب."
                result = await chat_complete(
                    model=model,
                    messages=[
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": result},
                        {"role": "user", "content": expansion_prompt}
                    ],
                    max_tokens=min(4096, target * 2),
                )
        return result
    except Exception as exc:
        if not _looks_like_context_error(exc):
            raise
        # Reactive fallback: the model still rejected the size — reduce harder
        # (half the budget) and retry once.
        log.warning(
            "Sonnet pass hit a context-length error — reducing input and retrying: %s",
            str(exc)[:200],
        )
        reduced = await _reduce_chunk_summaries(
            chunk_summaries, language, model, max_chars=_MAX_COMBINED_CHARS // 2,
        )
        prompt = _build_prompt(reduced, length, style, language,
                               target_words=target, missing_topics=missing_topics,
                               tashkeel_enabled=tashkeel_enabled)
        return await chat_complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=min(4096, target * 2),
        )
