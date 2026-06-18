"""
Pass 3 — Review pass.
Checks word count, language consistency, style, and repetition.
Returns a corrected summary only if needed; otherwise returns the original unchanged.
Supports any model via ai_client routing (Anthropic, OpenAI, OpenRouter).
"""
import logging

from api.config.settings import settings, SUMMARY_LENGTHS
from api.services.ai_client import chat_complete

log = logging.getLogger(__name__)

# Phrases that indicate the model returned an assessment/rejection instead of
# summary text.  When detected we discard the output and fall back to the
# original so that garbage text is never stored as the book summary.
_ASSESSMENT_INDICATORS = (
    # Arabic assessment phrases
    "لا يستوفي", "معايير القبول", "يتطلب توسيع", "طول النص", "أقل بكثير",
    "غير مقبول", "لا يلبي", "يحتاج إلى توسيع",
    # English assessment phrases
    "does not meet", "doesn't meet", "acceptance criteria",
    "needs expansion", "word count is", "significantly shorter",
    "below the target", "not acceptable", "requires expansion",
)


async def run_review_pass(
    summary: str,
    length:   str,
    style:    str,
    language: str,
    model:    str | None = None,
    max_words: int | None = None,
) -> str:
    """
    Quality-check the final summary and correct it if needed.
    Returns the (possibly corrected) summary text.

    max_words — admin override (SUMMARY_MAX_WORDS_*). When None/0, uses preset.
    """
    model     = model or settings.MODEL_HAIKU
    target    = max_words if (max_words and max_words > 0) else SUMMARY_LENGTHS.get(length, 750)
    lang_name = "Arabic" if language == "ar" else "English"
    word_count = len(summary.split())
    low, high  = int(target * 0.85), int(target * 1.15)

    prompt = (
        f"You are a book summary editor. Polish this {lang_name} summary:\n\n"
        f"---\n{summary}\n---\n\n"
        f"Editing criteria:\n"
        f"1. Word count: {word_count} words — target is {target} words ({low}–{high} is acceptable). "
        f"If the text is too short, expand it with relevant detail. "
        f"If too long, condense it.\n"
        f"2. Language must be entirely {lang_name} — fix any mixing.\n"
        f"3. Style: {style} — correct any deviations.\n"
        f"4. Remove repeated ideas or sentences.\n\n"
        f"CRITICAL RULES — you MUST follow these exactly:\n"
        f"• Output ONLY the final summary text itself.\n"
        f"• NEVER write explanations, assessments, critiques, or rejection messages.\n"
        f"• NEVER write phrases like 'this summary does not meet' or 'يتطلب توسيع'.\n"
        f"• If you cannot fully expand the text to the target length, output the best "
        f"version you can — do NOT explain why.\n"
        f"• Start immediately with the first word of the summary content."
    )

    result = await chat_complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=target * 3,
    )

    # Safety net: detect if the model returned assessment/rejection text instead
    # of actual summary content, and fall back to the original to prevent garbage
    # from being stored as the book summary.
    result_lower = result.lower()
    if any(ind in result_lower or ind in result for ind in _ASSESSMENT_INDICATORS):
        log.warning(
            "Review pass returned assessment/rejection text — falling back to original. "
            "Snippet: %r", result[:200],
        )
        return summary

    # Also fall back if the result is drastically shorter than the original
    # (model truncated instead of improving).
    if len(result.split()) < len(summary.split()) * 0.4:
        log.warning(
            "Review pass returned text much shorter than original (%d vs %d words) "
            "— falling back to original.",
            len(result.split()), len(summary.split()),
        )
        return summary

    return result
