"""
Alt text generation for cover images.

Routes by **model name** (not just a provider field) — matches the routing
used in ai_client.py so admins can pick any vision-capable model and it
goes to the right API:

    Model contains "/"           → OpenRouter (OpenAI-compatible vision format)
    Model starts with "claude-"  → native Anthropic SDK
    Anything else                → native OpenAI SDK

Examples that all "just work" after restart:
    google/gemini-2.5-pro          (OpenRouter → Google)
    anthropic/claude-sonnet-4-6    (OpenRouter → Anthropic)
    openai/gpt-4.1-mini            (OpenRouter → OpenAI)
    claude-sonnet-4-6              (native Anthropic)
    gpt-4.1-mini                   (native OpenAI)
"""
import base64
import logging

import anthropic
from openai import AsyncOpenAI

from api.config.settings import settings
from api.services.config.runtime import get_all_config
from api.services.usage_logger import log_text_usage

log = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


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
    return "image/jpeg"


def _read_image(path: str) -> tuple[str, str]:
    """Return (base64-encoded data, detected media type) for the image at path."""
    with open(path, "rb") as f:
        raw = f.read()
    return base64.b64encode(raw).decode(), _sniff_media_type(raw)


# ── Routing helpers (mirror ai_client.py) ────────────────────────────────────

def _is_openrouter(model: str) -> bool:
    return "/" in model


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude-") and "/" not in model


# ── Public entry point ───────────────────────────────────────────────────────

async def generate_alt_text(image_path: str, title: str, language: str) -> str:
    """
    Generate a 1-2 sentence description of the cover image.

    Reads ALTTEXT_MODEL_EN / ALTTEXT_MODEL_AR from the LIVE admin config
    (not from settings.py) so admin changes take effect on the next call.
    """
    cfg = await get_all_config()

    if (language or "en").lower() == "ar":
        model = cfg.get("ALTTEXT_MODEL_AR") or settings.ALTTEXT_MODEL_AR
    else:
        model = cfg.get("ALTTEXT_MODEL_EN") or settings.ALTTEXT_MODEL_EN

    lang_name = "Arabic" if (language or "en").lower() == "ar" else "English"
    prompt = (
        f"Describe this book cover image for '{title}' in 1-2 sentences in {lang_name}. "
        "Focus on the visual style, colors, and mood. Keep it concise."
    )

    if _is_openrouter(model):
        return await _openrouter_alt_text(image_path, prompt, model)
    if _is_anthropic(model):
        return await _claude_alt_text(image_path, prompt, model)
    # Native OpenAI
    return await _openai_alt_text(image_path, prompt, model)


# ── Per-route implementations ────────────────────────────────────────────────

async def _claude_alt_text(image_path: str, prompt: str, model: str) -> str:
    if not settings.ANTHROPIC_API_KEY:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. Add it to .env or switch ALTTEXT_MODEL_* "
            "to an OpenRouter / OpenAI model in Admin → Providers → Alt Text."
        )
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
    if not settings.OPENAI_API_KEY:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to .env or switch ALTTEXT_MODEL_* "
            "to an OpenRouter model (vendor/model format) in Admin → Providers → Alt Text."
        )
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
    return response.choices[0].message.content or ""


async def _openrouter_alt_text(image_path: str, prompt: str, model: str) -> str:
    """
    Route any model with a `vendor/` prefix through OpenRouter.
    OpenRouter uses the OpenAI Chat Completions API, including the
    `image_url` content type for vision.
    """
    if not settings.OPENROUTER_API_KEY:
        raise ValueError(
            f"OPENROUTER_API_KEY is not set — required to use {model!r} via OpenRouter. "
            "Add it to .env or pick a native Claude / OpenAI model in Admin → Providers → Alt Text."
        )
    b64, media_type = _read_image(image_path)
    client = AsyncOpenAI(
        base_url       = _OPENROUTER_BASE,
        api_key        = settings.OPENROUTER_API_KEY,
        default_headers= {
            "HTTP-Referer": "https://seeourbook.sa",
            "X-Title":      "SeeOurBook",
        },
    )
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
        provider     = "openrouter",
        model        = model,
        input_tokens = getattr(usage, "prompt_tokens",     0) if usage else 0,
        output_tokens= getattr(usage, "completion_tokens", 0) if usage else 0,
        step         = "alt_text",
    )
    return response.choices[0].message.content or ""
