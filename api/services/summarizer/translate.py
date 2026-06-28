"""
Summary translation.

Produces a clean translation of a finished summary into the other language so
every book ends up with BOTH an English and an Arabic summary. Translation is
done with a configurable chat model (TRANSLATE_MODEL).
"""
import logging
import asyncio
import re

from api.config.settings import settings
from api.services.ai_client import chat_complete
from api.services.summarizer.haiku import AR_TASHKEEL_INSTRUCTION

log = logging.getLogger(__name__)


async def translate_summary(
    text: str,
    source_lang: str,
    target_lang: str,
    model: str | None = None,
    tashkeel_enabled: bool = True,
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
    tashkeel = AR_TASHKEEL_INSTRUCTION if (target_lang == "ar" and tashkeel_enabled) else ""

    async def _translate_chunk(chunk_text: str) -> str:
        prompt = (
            f"Translate the following book summary from {src_name} into {tgt_name}.\n"
            f"Requirements:\n"
            f"- Produce natural, fluent, publication-quality {tgt_name} — NOT a literal "
            f"word-for-word translation.\n"
            f"- Preserve ALL the content, structure, paragraphs, and meaning.\n"
            f"- Do not add, remove, or summarise — translate the whole text faithfully.\n"
            f"- Return ONLY the translated text, no preamble or notes.\n"
            f"- DO NOT repeat or include any of the original {src_name} text in your output.{tashkeel}\n\n"
            f"=== TEXT ({src_name}) ===\n{chunk_text}"
        )

        words = len(chunk_text.split())
        approx_tokens = min(8192, words * 4 + 1000)
            
        try:
            out = await chat_complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=approx_tokens,
            )
            out = (out or "").strip()
            # Strip common preamble phrases the model adds despite the instruction.
            _preamble_re = (
                r"^(هَذِهِ تَرْجَمَةُ|هذه ترجمة|إليكم الترجمة|الترجمة:|"
                r"here is the translation|here's the translation|"
                r"below is the translation|translation:)\s*[:\n]+"
            )
            out = re.sub(_preamble_re, "", out, count=1, flags=re.IGNORECASE).strip()
            return out
        except Exception as exc:
            log.warning("translate_chunk failed (%s→%s): %s", source_lang, target_lang, exc)
            return ""

    # Split text into chunks to bypass max_tokens limits (especially for Arabic tashkeel)
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_words = 0
    
    for p in paragraphs:
        w = len(p.split())
        if current_words + w > 800 and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [p]
            current_words = w
        else:
            current_chunk.append(p)
            current_words += w
            
    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    # Translate all chunks in parallel
    results = await asyncio.gather(*[_translate_chunk(c) for c in chunks])
    
    # Filter out empty results and join
    final_text = "\n\n".join(r for r in results if r)
    return final_text
