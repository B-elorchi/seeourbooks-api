"""
Alt text generation for cover images.
Supports Claude and OpenAI, configurable per language.
"""
import base64
import anthropic
from openai import AsyncOpenAI
from api.config.settings import settings
from api.services.usage_logger import log_text_usage


def _sniff_media_type(data: bytes) -> str:
    """
    Detect the real image media type from magic bytes.
    Needed because Gemini returns PNG even when we save as .jpg — Anthropic
    rejects mismatched media_type headers.
    """
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    # Default to jpeg — Anthropic will surface a clearer error if it's truly wrong
    return "image/jpeg"


def _read_image(path: str) -> tuple[str, str]:
    """Return (base64-encoded data, detected media type) for the image at path."""
    with open(path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode(), _sniff_media_type(raw)


async def generate_alt_text(image_path: str, title: str, language: str) -> str:
    """Generate a 1-2 sentence description of the cover image."""
    provider = settings.ALTTEXT_PROVIDER_EN if language == "en" else settings.ALTTEXT_PROVIDER_AR
    model    = settings.ALTTEXT_MODEL_EN    if language == "en" else settings.ALTTEXT_MODEL_AR

    lang_name = "Arabic" if language == "ar" else "English"
    prompt = (
        f"Describe this book cover image for '{title}' in 1-2 sentences in {lang_name}. "
        "Focus on the visual style, colors, and mood. Keep it concise."
    )

    if provider == "claude":
        return await _claude_alt_text(image_path, prompt, model)
    elif provider == "openai":
        return await _openai_alt_text(image_path, prompt, model)
    else:
        raise ValueError(f"Unknown alt text provider: {provider}")


async def _claude_alt_text(image_path: str, prompt: str, model: str) -> str:
    b64, media_type = _read_image(image_path)
    ai = anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    msg = await ai.messages.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text",  "text": prompt},
            ],
        }],
    )
    usage = getattr(msg, "usage", None)
    await log_text_usage(
        provider     = "anthropic",
        model        = model,
        input_tokens = getattr(usage, "input_tokens",  0) if usage else 0,
        output_tokens= getattr(usage, "output_tokens", 0) if usage else 0,
        step         = "alt_text",
    )
    return msg.content[0].text


async def _openai_alt_text(image_path: str, prompt: str, model: str) -> str:
    b64, media_type = _read_image(image_path)
    client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text",      "text": prompt},
            ],
        }],
    )
    usage = getattr(response, "usage", None)
    await log_text_usage(
        provider     = "openai",
        model        = model,
        input_tokens = getattr(usage, "prompt_tokens",     0) if usage else 0,
        output_tokens= getattr(usage, "completion_tokens", 0) if usage else 0,
        step         = "alt_text",
    )
    return response.choices[0].message.content
