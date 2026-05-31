from api.config.settings import settings, SUMMARY_LENGTHS
from api.services.ai_client import chat_complete


async def run_tashkeel_pass(text: str, length: str, model: str | None = None) -> str:
    """
    Arabic only — rewrite text with full diacritics (tashkeel) on every letter.
    Required for accurate TTS pronunciation.
    Supports any model via ai_client routing (Anthropic, OpenAI, OpenRouter).
    """
    model  = model or settings.MODEL_OPUS
    target = SUMMARY_LENGTHS.get(length, 750)

    prompt = (
        "أنت متخصص في اللغة العربية الفصحى وعلم التشكيل. مهمتك إضافة التشكيل الكامل على النص التالي.\n\n"
        "القواعد الصارمة:\n"
        "١. ضَعْ حَرَكَةً على كُلِّ حَرْفٍ في كُلِّ كَلِمَة.\n"
        "٢. أضِفِ الشَّدَّة على كل حرف مُشَدَّد.\n"
        "٣. أضِفِ التَّنْوِين على نهايات الكلمات المُنَوَّنَة.\n"
        "٤. لا تُغَيِّر أي كلمة — فقط أضِفِ التشكيل.\n\n"
        f"النص:\n\n{text}"
    )

    return await chat_complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=int(target * 2.5),
    )
