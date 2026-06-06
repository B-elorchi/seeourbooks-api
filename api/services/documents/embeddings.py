"""
Embedding generation for knowledge chunks.

When EMBEDDING_PROVIDER is unset, this is a graceful no-op — chunks are
still saved (without vectors), and a later batch job can backfill them.

Supported providers
───────────────────
    openai      — native OpenAI API.   Uses OPENAI_API_KEY.
                  Models: text-embedding-3-small / -large
    deepseek    — native DeepSeek API.  Uses DEEPSEEK_API_KEY.
    openrouter  — any vendor/model identifier.  Uses OPENROUTER_API_KEY.
                  Examples:
                      openai/text-embedding-3-small
                      voyage/voyage-3
                      mistralai/mistral-embed
                      cohere/embed-multilingual-v3.0

All routes use the OpenAI-compatible SDK shape — DeepSeek and OpenRouter
both expose `embeddings.create` with the same request/response schema.

Smart auto-routing
──────────────────
If `EMBEDDING_MODEL` contains a "/" (e.g. "openai/text-embedding-3-small")
the request automatically goes through OpenRouter regardless of what
EMBEDDING_PROVIDER is set to.  Saves admins from having to keep two
settings in sync.
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from api.config.settings import settings
from api.services.config.runtime import get_config_value
from api.services.usage_logger import log_text_usage

log = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


async def _resolve_provider() -> tuple[str, str]:
    """Return (provider_name, model_name) from runtime config + settings fallback."""
    provider = (await get_config_value("EMBEDDING_PROVIDER", settings.EMBEDDING_PROVIDER)).lower()
    model    = await get_config_value("EMBEDDING_MODEL",    settings.EMBEDDING_MODEL)
    return provider, model


def _client_for(provider: str) -> AsyncOpenAI | None:
    """Build a client for the chosen provider.  Returns None when the key is missing."""
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            return None
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    if provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            return None
        return AsyncOpenAI(
            api_key  = settings.DEEPSEEK_API_KEY,
            base_url = settings.DEEPSEEK_BASE_URL,
        )
    if provider == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            return None
        return AsyncOpenAI(
            api_key         = settings.OPENROUTER_API_KEY,
            base_url        = _OPENROUTER_BASE,
            default_headers = {
                "HTTP-Referer": "https://seeourbook.sa",
                "X-Title":      "SeeOurBook",
            },
        )
    return None


async def embed_texts(texts: list[str]) -> list[list[float] | None]:
    """
    Embed each text into a vector.

    Returns one entry per input:
      - list[float]  when embedding succeeded.
      - None         when embeddings are disabled or that single text errored.

    Failures are isolated to individual entries — one bad text won't fail
    the whole batch.
    """
    if not texts:
        return []

    provider, model = await _resolve_provider()

    # ── Smart auto-routing: any vendor/model identifier → OpenRouter ────────
    # Saves admins from having to set EMBEDDING_PROVIDER=openrouter AND
    # EMBEDDING_MODEL=openai/text-embedding-3-small.  The "/" in the model
    # is unambiguous — only OpenRouter uses that convention.
    if model and "/" in model and provider != "openrouter":
        log.info(
            "EMBEDDING_MODEL %r looks like an OpenRouter identifier — "
            "auto-routing via openrouter (ignoring EMBEDDING_PROVIDER=%s).",
            model, provider or "(unset)",
        )
        provider = "openrouter"

    if not provider:
        log.info("EMBEDDING_PROVIDER not set — skipping vector generation")
        return [None] * len(texts)

    client = _client_for(provider)
    if client is None:
        log.warning(
            "Embedding provider %r not usable (missing API key) — skipping. "
            "Add the corresponding *_API_KEY to .env or pick a different provider.",
            provider,
        )
        return [None] * len(texts)

    # Batch the request — every supported provider accepts a list of strings.
    try:
        resp = await client.embeddings.create(model=model, input=texts)
    except Exception as exc:
        log.warning(
            "Embedding batch failed (provider=%s model=%s): %s — chunks saved without vectors",
            provider, model, exc,
        )
        return [None] * len(texts)

    # Cost logging — embeddings are priced per input token.  We approximate
    # with character count / 4 when usage is missing (rough token estimate).
    usage = getattr(resp, "usage", None)
    input_tokens = (
        getattr(usage, "prompt_tokens", 0) if usage
        else sum(len(t) // 4 for t in texts)
    )
    await log_text_usage(
        provider     = provider,
        model        = model,
        input_tokens = input_tokens,
        output_tokens= 0,
        step         = "embed",
    )

    out: list[list[float] | None] = []
    for item in resp.data:
        vec = getattr(item, "embedding", None)
        out.append(list(vec) if vec is not None else None)

    # Defensive: pad if the provider returned fewer vectors than we asked for
    while len(out) < len(texts):
        out.append(None)
    return out
