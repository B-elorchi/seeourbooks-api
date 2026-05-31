"""
Pass 3 — Review pass.
Checks word count, language consistency, style, and repetition.
Returns a corrected summary only if needed; otherwise returns the original unchanged.
Supports any model via ai_client routing (Anthropic, OpenAI, OpenRouter).
"""
from api.config.settings import settings, SUMMARY_LENGTHS
from api.services.ai_client import chat_complete


async def run_review_pass(
    summary: str,
    length:   str,
    style:    str,
    language: str,
    model:    str | None = None,
) -> str:
    """
    Quality-check the final summary and correct it if needed.
    Returns the (possibly corrected) summary text.
    """
    model     = model or settings.MODEL_HAIKU
    target    = SUMMARY_LENGTHS.get(length, 750)
    lang_name = "Arabic" if language == "ar" else "English"
    word_count = len(summary.split())
    low, high  = int(target * 0.85), int(target * 1.15)

    prompt = (
        f"You are a book summary quality reviewer. Check this {lang_name} summary:\n\n"
        f"---\n{summary}\n---\n\n"
        f"Criteria:\n"
        f"1. Word count: {word_count} — target is {target} words ({low}–{high} is acceptable)\n"
        f"2. Language must be entirely {lang_name} — no mixing\n"
        f"3. Style: {style} — must match throughout\n"
        f"4. No repeated ideas or sentences\n\n"
        f"If ALL criteria pass, return the summary UNCHANGED.\n"
        f"If ANY criterion fails, return a corrected version.\n"
        f"Return ONLY the summary text — no preamble, no labels, no explanations."
    )

    return await chat_complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=target * 3,
    )
