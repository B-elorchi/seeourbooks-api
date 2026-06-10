from api.config.settings import settings
from api.services.ai_client import chat_complete
from api.services.db import find, upsert


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
    """
    model     = model or settings.MODEL_HAIKU
    lang_name = "Arabic" if language == "ar" else "English"
    tashkeel  = AR_TASHKEEL_INSTRUCTION if language == "ar" else ""
    summaries: list[str] = []

    if max_words and max_words > 0:
        length_instr = f"in about {max_words} words"
        max_tokens   = int(max_words * 2.5) + 100
    else:
        length_instr = "in 3-5 sentences"
        max_tokens   = 512

    for chunk in chunks:
        cid, idx = chunk["id"], chunk["chunk_index"]

        existing = await find(
            "chunk_summaries",
            filters={"chunk_id": cid, "language": language},
            select="summary",
            limit=1,
        )
        if existing:
            summaries.append(existing[0]["summary"])
            continue

        summary = await chat_complete(
            model=model,
            messages=[{
                "role":    "user",
                "content": (
                    f"Summarise this book excerpt in {lang_name} {length_instr}. "
                    f"Focus on key ideas.{tashkeel}\n\n{chunk['content']}"
                ),
            }],
            max_tokens=max_tokens,
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
