"""
Summary translation.

Produces a clean translation of a finished summary into the other language so
every book ends up with BOTH an English and an Arabic summary. Translation is
done with a configurable chat model (TRANSLATE_MODEL).
"""
import logging

from api.config.settings import settings
from api.services.ai_client import chat_complete
from api.services.summarizer.haiku import AR_TASHKEEL_INSTRUCTION

log = logging.getLogger(__name__)


async def translate_summary(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str | None = None,
) -> str:
    """
    Translate `text` from source_lang to target_lang as a clean, natural,
    publication-quality summary (not a literal word-for-word translation).
    Returns the translated text, or "" on failure.
    """
    if not text or source_lang == target_lang:
        return ""

    model = model or settings.MODEL_SONNET
    src_name = "Arabic" if source_lang == "ar" else "English"
    tgt_name = "Arabic" if target_lang == "ar" else "English"
    tashkeel = AR_TASHKEEL_INSTRUCTION if target_lang == "ar" else ""

    prompt = (
        f"Translate the following book summary from {src_name} into {tgt_name}.\n"
        f"Requirements:\n"
        f"- Produce natural, fluent, publication-quality {tgt_name} — NOT a literal "
        f"word-for-word translation.\n"
        f"- Preserve ALL the content, structure, paragraphs, and meaning.\n"
        f"- Do not add, remove, or summarise — translate the whole text faithfully.\n"
        f"- Return ONLY the translated text, no preamble or notes.{tashkeel}\n\n"
        f"=== TEXT ({src_name}) ===\n{text}"
    )

    # Allow generous output — translation length ≈ source length.
    approx_tokens = min(8000, int(len(text.split()) * 3) + 500)
    try:
        out = await chat_complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=approx_tokens,
        )
        out = (out or "").strip()
        # Strip common preamble phrases the model adds despite the instruction.
        # E.g. "هَذِهِ تَرْجَمَةُ النَّصِّ بِالتَّشْكِيلِ الكَامِلِ:\n\n"
        # or   "Here is the translation:\n\n"
        _preamble_re = (
            r"^(هَذِهِ تَرْجَمَةُ|هذه ترجمة|إليكم الترجمة|الترجمة:|"
            r"here is the translation|here's the translation|"
            r"below is the translation|translation:)\s*[:\n]+"
        )
        import re as _re
        out = _re.sub(_preamble_re, "", out, count=1, flags=_re.IGNORECASE).strip()
        return out
    except Exception as exc:
        log.warning("translate_summary failed (%s→%s): %s", source_lang, target_lang, exc)
        return ""
