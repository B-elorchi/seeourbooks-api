"""
AI cover image generation.

Routing logic
─────────────
  Native OpenAI  (OPENAI_API_KEY  → api.openai.com)
    dall-e-3, dall-e-2, gpt-image-1
    openai/dall-e-3, openai/dall-e-2, openai/gpt-image-1   ← openai/ prefix stripped

  OpenRouter — Gemini image models  (OPENROUTER_API_KEY → openrouter.ai/api/v1)
    google/gemini-2.5-flash-image
    google/gemini-2.0-flash-exp:image
    Uses chat completions with image_url in response content (not images.generate).

  OpenRouter — FLUX / SD  (OPENROUTER_API_KEY → openrouter.ai/api/v1)
    black-forest-labs/flux-1.1-pro
    black-forest-labs/flux-schnell
    stability-ai/stable-diffusion-3.5-large
    … any other  vendor/model  with a non-openai, non-google vendor

  NOTE: openai/* image models are NOT available on OpenRouter — they are
  always sent to api.openai.com regardless of the slash prefix.
"""
import base64
import logging
import httpx
from openai import AsyncOpenAI

from api.config.settings import settings
from api.services.usage_logger import log_image_usage
from api.services.config.runtime import get_config_value, PROMPT_COVER_DEFAULT

log = logging.getLogger(__name__)


async def _build_prompt(title: str, author: str, summary: str | None,
                        genres: list[str] | None, year: int | None,
                        language: str | None) -> str:
    """Compose a content-aware cover prompt from the book's metadata + summary."""

    # ── Details block (only include what we actually have) ──────────────────
    parts: list[str] = []
    if year:
        parts.append(f"- Published: {year}")
    if genres:
        parts.append(f"- Genres / categories: {', '.join(genres)}")
    if language:
        parts.append(f"- Original language: {language}")
    details = "\n".join(parts) if parts else "- (no additional metadata)"

    # ── Summary block — trim to keep prompt under model limits ──────────────
    if summary:
        snippet = summary.strip().replace("\n", " ")
        if len(snippet) > 1200:
            snippet = snippet[:1200].rsplit(" ", 1)[0] + "…"
    else:
        snippet = "(no summary provided — invent a visual that matches the title and genre)"

    # ── Genre hint for visual style ──────────────────────────────────────────
    genre_hint = (", ".join(genres) if genres else "general").lower()

    template = await get_config_value("PROMPT_COVER", PROMPT_COVER_DEFAULT)

    # Treat blank / placeholder author values as "no author known". We must
    # NEVER stamp the literal word "Unknown" (or a placeholder) onto a cover.
    clean_author = (author or "").strip()
    if clean_author.lower() in ("", "unknown", "n/a", "anonymous", "غير معروف", "مجهول"):
        clean_author = ""

    prompt = template.format(
        title      = title or "Untitled",
        author     = clean_author,
        details    = details,
        summary    = snippet,
        genre_hint = genre_hint,
    )

    # When the author is unknown, override any author-rendering instructions in
    # the template: produce a title-only cover rather than printing "Unknown".
    if not clean_author:
        prompt += (
            "\n\nIMPORTANT — AUTHOR IS UNKNOWN:\n"
            "- Do NOT render any author name, byline, placeholder, or the word "
            "\"Unknown\" anywhere on the cover.\n"
            "- Render ONLY the title as text. Leave the lower author area as "
            "clean negative space / artwork with no text."
        )
    return prompt

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ── Size maps ─────────────────────────────────────────────────────────────────

# gpt-image-1 valid: 1024x1024 | 1024x1536 | 1536x1024 | auto
_GPT_IMAGE_1_SIZE_MAP = {
    "1024x1792": "1024x1536",
    "1792x1024": "1536x1024",
    "512x512":   "1024x1024",
    "256x256":   "1024x1024",
}

# dall-e-3 valid: 1024x1024 | 1024x1792 | 1792x1024
_DALLE3_SIZE_MAP = {
    "1024x1536": "1024x1792",
    "1536x1024": "1792x1024",
    "512x512":   "1024x1024",
    "256x256":   "1024x1024",
    "auto":      "1024x1024",
}

# dall-e-2 valid: square only
_DALLE2_SIZE_MAP = {
    "1024x1792": "1024x1024",
    "1024x1536": "1024x1024",
    "1792x1024": "1024x1024",
    "1536x1024": "1024x1024",
    "1024x1024": "1024x1024",
    "512x512":   "512x512",
    "256x256":   "256x256",
    "auto":      "1024x1024",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vendor(model: str) -> str:
    """Return vendor prefix, e.g. 'black-forest-labs' from 'black-forest-labs/flux-1.1-pro'."""
    return model.split("/", 1)[0] if "/" in model else ""


def _bare(model: str) -> str:
    """Strip vendor prefix."""
    return model.split("/", 1)[1] if "/" in model else model


def _uses_openrouter(model: str) -> bool:
    """True when the model should be sent to OpenRouter instead of native OpenAI."""
    v = _vendor(model)
    return bool(v) and v != "openai"


def _is_gemini_image(model: str) -> bool:
    """True for Google Gemini image-generation models on OpenRouter."""
    return model.startswith("google/")


async def _download_url(url: str, output_path: str) -> None:
    async with httpx.AsyncClient(timeout=60) as http:
        r = await http.get(url)
        r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)


# ── OpenRouter — Gemini image generation ─────────────────────────────────────

def _extract_image_url_from_message(message: dict) -> str | None:
    """
    Scan an OpenRouter chat-completions message for a generated image URL.
    Returns the URL (http(s):// or data:image/...;base64,...) or None.

    Gemini image models on OpenRouter return the image in one of:
      message["images"][i]["image_url"]["url"]    ← most common
      message["images"][i]["image_url"]           ← alt shape (raw string)
      message["content"] = [{"type":"image_url", "image_url":{"url":...}}]
      message["content"] = "data:image/png;base64,..."
    """
    # Path 1: non-standard `images` field (OpenRouter's Gemini extension)
    images = message.get("images")
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, dict):
            url = first.get("image_url")
            if isinstance(url, dict):
                url = url.get("url")
            if isinstance(url, str) and url:
                return url
        if isinstance(first, str) and first:
            return first

    # Path 2: multimodal content parts
    content = message.get("content")
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                iu = part.get("image_url")
                if isinstance(iu, dict):
                    return iu.get("url")
                if isinstance(iu, str):
                    return iu

    # Path 3: plain string data URL
    if isinstance(content, str) and content.startswith("data:"):
        return content

    return None


async def _write_image_url(image_url: str, output_path: str) -> None:
    """Persist either a data: URL or an http(s):// URL to disk."""
    if image_url.startswith("data:"):
        b64 = image_url.split(",", 1)[1]
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(b64))
    else:
        await _download_url(image_url, output_path)


async def _generate_gemini_openrouter(model: str, prompt: str, output_path: str) -> None:
    """
    Generate image via a Google Gemini model on OpenRouter.

    Uses raw httpx to access the full chat-completions response — the OpenAI
    SDK strips OpenRouter's non-standard `message.images` field where Gemini
    actually returns the generated image, leaving `message.content` as None.
    """
    if not settings.OPENROUTER_API_KEY:
        raise ValueError(
            f"OPENROUTER_API_KEY is not set. Add it to your .env file to use {model}. "
            "Get a key at https://openrouter.ai/keys"
        )

    headers = {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
    }
    payload = {
        "model":      model,
        "messages":   [{"role": "user", "content": prompt}],
        "modalities": ["image", "text"],   # ← tells OpenRouter we want image output
        "max_tokens": 4096,
    }

    async with httpx.AsyncClient(timeout=120) as http:
        r = await http.post(
            f"{_OPENROUTER_BASE}/chat/completions",
            headers=headers,
            json=payload,
        )
        r.raise_for_status()
        body = r.json()

    try:
        message = body["choices"][0]["message"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Gemini response had no choices[0].message. Body keys: {list(body.keys()) if isinstance(body, dict) else type(body)}"
        ) from exc

    image_url = _extract_image_url_from_message(message)
    if not image_url:
        # Surface enough of the message for the admin to diagnose
        preview = {k: type(v).__name__ for k, v in message.items()}
        raise RuntimeError(
            f"No image found in Gemini response from {model}. "
            f"Message fields: {preview}. Raw content preview: {str(message)[:300]}"
        )

    await _write_image_url(image_url, output_path)


# ── OpenRouter — FLUX / SD image generation (images.generate) ─────────────────

async def _generate_openrouter(model: str, prompt: str, size: str, output_path: str) -> None:
    """Generate image via OpenRouter (FLUX / Stable Diffusion / etc.)."""
    if not settings.OPENROUTER_API_KEY:
        raise ValueError(
            f"OPENROUTER_API_KEY is not set. Add it to your .env file to use {model}. "
            "Get a key at https://openrouter.ai/keys"
        )

    client = AsyncOpenAI(base_url=_OPENROUTER_BASE, api_key=settings.OPENROUTER_API_KEY)

    # OpenRouter image models generally accept standard sizes; pass through as-is.
    # They do NOT support the 'quality' parameter — omit it.
    resp = await client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        n=1,
    )
    item = resp.data[0]
    if getattr(item, "b64_json", None):
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(item.b64_json))
    else:
        await _download_url(item.url, output_path)


# ── Native OpenAI image generation ───────────────────────────────────────────

async def _generate_openai(bare_model: str, prompt: str, quality: str, size: str,
                            output_path: str) -> None:
    """
    Generate image via native OpenAI API using EXACTLY the model the admin picked.
    No silent fallback — if the chosen model fails, the real error is raised so
    the admin knows what to change in the panel.
    """
    api_key = settings.OPENAI_API_KEY
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set. Add it to your .env file, or switch IMAGE_MODEL "
            "in Admin → Providers → Cover Image to a FLUX model (uses OPENROUTER_API_KEY)."
        )
    if api_key.startswith("sk-or-"):
        raise ValueError(
            "Your OPENAI_API_KEY looks like an OpenRouter key (starts with 'sk-or-'). "
            "OpenAI image models need a native OpenAI key from platform.openai.com. "
            "Either fix OPENAI_API_KEY in .env, or switch IMAGE_MODEL in the admin panel "
            "to a FLUX / Gemini / Stable Diffusion model (those use OPENROUTER_API_KEY)."
        )

    client = AsyncOpenAI(api_key=api_key)

    # Resolve size per model — admin's IMAGE_SIZE choice is respected if valid,
    # remapped to the nearest valid size for that model if not.
    if bare_model == "dall-e-2":
        safe_size = _DALLE2_SIZE_MAP.get(size, "1024x1024")
        resp = await client.images.generate(
            model="dall-e-2", prompt=prompt[:1000], size=safe_size, n=1)
        await _download_url(resp.data[0].url, output_path)

    elif bare_model == "gpt-image-1":
        safe_size = _GPT_IMAGE_1_SIZE_MAP.get(size, size)
        resp = await client.images.generate(
            model="gpt-image-1", prompt=prompt, size=safe_size, quality=quality, n=1)
        item = resp.data[0]
        if getattr(item, "b64_json", None):
            with open(output_path, "wb") as f:
                f.write(base64.b64decode(item.b64_json))
        else:
            await _download_url(item.url, output_path)

    elif bare_model == "dall-e-3":
        safe_size = _DALLE3_SIZE_MAP.get(size, size)
        resp = await client.images.generate(
            model="dall-e-3", prompt=prompt, size=safe_size, quality=quality, n=1)
        await _download_url(resp.data[0].url, output_path)

    else:
        # Unknown native model name — pass through, let OpenAI's error message speak
        resp = await client.images.generate(
            model=bare_model, prompt=prompt, size=size, quality=quality, n=1)
        await _download_url(resp.data[0].url, output_path)


# ── Deprecated / renamed model aliases ───────────────────────────────────────
# Maps stale DB values to their current replacements so old saved configs keep
# working after an OpenRouter model is retired or renamed.
_MODEL_ALIASES: dict[str, str] = {
    "google/gemini-2.0-flash-exp:image": "google/gemini-2.5-flash-image",
}


# ── Public entry point ────────────────────────────────────────────────────────

async def generate_cover(
    title:       str,
    author:      str,
    output_path: str,
    cfg:         dict | None = None,
    *,
    summary:     str | None = None,
    genres:      list[str] | None = None,
    year:        int | None = None,
    language:    str | None = None,
) -> str:
    """
    Generate a cover image and save to output_path. Returns output_path.

    The prompt is built from the book's actual content (summary, genres, year)
    so the generated image reflects the story — not just the title.
    """
    cfg = cfg or {}

    # ── Resolve image model per language ────────────────────────────────────
    # Fallback chain:
    #   1. IMAGE_MODEL_{LANG}  in admin config (admin picked a language-specific model)
    #   2. IMAGE_MODEL         in admin config (legacy single-language setting)
    #   3. IMAGE_MODEL_{LANG}  hard default from settings.py
    #   4. IMAGE_MODEL         hard default from settings.py
    lang_key = "IMAGE_MODEL_AR" if (language or "").lower() == "ar" else "IMAGE_MODEL_EN"
    lang_default = settings.IMAGE_MODEL_AR if (language or "").lower() == "ar" else settings.IMAGE_MODEL_EN
    model = (
        cfg.get(lang_key)
        or cfg.get("IMAGE_MODEL")
        or lang_default
        or settings.IMAGE_MODEL
    )

    quality = cfg.get("IMAGE_QUALITY", settings.IMAGE_QUALITY)
    size    = cfg.get("IMAGE_SIZE",    settings.IMAGE_SIZE)
    prompt  = await _build_prompt(title, author, summary, genres, year, language)

    # Silently upgrade any stale/deprecated model name saved in the DB
    if model in _MODEL_ALIASES:
        log.warning("IMAGE_MODEL %r is deprecated — using %r instead", model, _MODEL_ALIASES[model])
        model = _MODEL_ALIASES[model]

    if _uses_openrouter(model) and _is_gemini_image(model):
        # Gemini image models — chat completions endpoint, no size param
        await _generate_gemini_openrouter(model, prompt, output_path)
        provider_tag = "openrouter"
    elif _uses_openrouter(model):
        # FLUX, SD, and other non-OpenAI OpenRouter image models
        await _generate_openrouter(model, prompt, size, output_path)
        provider_tag = "openrouter"
    else:
        # Native OpenAI models (dall-e-3, gpt-image-1, dall-e-2, openai/dall-e-3…)
        await _generate_openai(_bare(model), prompt, quality, size, output_path)
        provider_tag = "openai"

    await log_image_usage(provider=provider_tag, model=model, count=1)

    # Watermark the generated image
    from api.services.pipeline.watermark import stamp_image  # noqa: PLC0415
    stamp_image(output_path, cfg)

    return output_path
