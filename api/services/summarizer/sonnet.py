from typing import AsyncIterator

from api.config.settings import settings, SUMMARY_LENGTHS
from api.services.ai_client import chat_complete, chat_stream
from api.services.summarizer.haiku import AR_TASHKEEL_INSTRUCTION

STYLE_MAP = {
    "narrative": "flowing narrative prose",
    "bullets":   "clear bullet points",
    "academic":  "formal academic style",
}


def _build_prompt(chunk_summaries: list[str], length: str, style: str, language: str,
                  target_words: int | None = None) -> str:
    lang_name = "Arabic" if language == "ar" else "English"
    target    = target_words if (target_words and target_words > 0) else SUMMARY_LENGTHS.get(length, 750)
    tashkeel  = AR_TASHKEEL_INSTRUCTION if language == "ar" else ""
    combined  = "\n\n---\n\n".join(
        f"Section {i + 1}:\n{s}" for i, s in enumerate(chunk_summaries)
    )
    return (
        f"Create a book summary in {lang_name} for an audio presentation "
        f"of ~{length} (~{target} words).\n"
        f"Style: {STYLE_MAP.get(style, 'narrative prose')}.\n"
        f"Based on these section summaries:\n\n{combined}\n\n"
        f"Write the complete summary in {lang_name}. Target: ~{target} words."
        f"{tashkeel}"
    )


async def run_sonnet_pass(
    chunk_summaries: list[str],
    length: str,
    style: str,
    language: str,
    model: str | None = None,
) -> AsyncIterator[tuple[str, str]]:
    """
    Streaming version — yields ("token", text) and finally ("full", full_text).
    Used by the /api/summarize SSE endpoint.
    Supports any model via ai_client routing (Anthropic, OpenAI, OpenRouter).
    """
    model  = model or settings.MODEL_SONNET
    target = SUMMARY_LENGTHS.get(length, 750)
    prompt = _build_prompt(chunk_summaries, length, style, language)
    full   = ""

    async for token in chat_stream(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=target * 2,
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
) -> str:
    """
    Non-streaming version — returns the complete summary text.
    Used by the pipeline orchestrator.
    Supports any model via ai_client routing.

    max_words — admin override (SUMMARY_MAX_WORDS_*). When None/0, the length
    preset is used (3min=450, 5min=750, 10min=1500, 15min=2250).
    """
    model  = model_override or settings.MODEL_SONNET
    target = max_words if (max_words and max_words > 0) else SUMMARY_LENGTHS.get(length, 750)
    prompt = _build_prompt(chunk_summaries, length, style, language, target_words=target)

    return await chat_complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=target * 2,
    )
