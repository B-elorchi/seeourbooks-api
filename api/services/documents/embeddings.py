"""
Embedding generation for knowledge chunks.

When EMBEDDING_PROVIDER is unset, this is a graceful no-op — chunks are
still saved (without vectors), and a later batch job can backfill them.

Supported providers:
    openai    — text-embedding-3-small / -large
    deepseek  — deepseek embeddings

Both use the OpenAI SDK shape (DeepSeek is OpenAI-compatible).
"""
from __future__ import annotations

import logging

from openai import AsyncOpenAI

from api.config.settings import settings
from api.services.config.runtime import get_config_value
from api.services.usage_logger import log_text_usage

log = logging.getLogger(__name__)


async def _resolve_provider() -> tuple[str, str]:
    """Return (provider_name, model_name) from runtime config + settings fallback."""
    provider = (await get_config_value("EMBEDDING_PROVIDER", settings.EMBEDDING_PROVIDER)).lower()
    model    = await get_config_value("EMBEDDING_MODEL",    settings.EMBEDDING_MODEL)
    return provider, model


def _client_for(provider: str) -> AsyncOpenAI | None:
    if provider == "openai":
        if not settings.OPENAI_API_KEY:
            return None
        return AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
    if provider == "deepseek":
        if not settings.DEEPSEEK_API_KEY:
            return None
        return AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
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
    if not provider:
        log.info("EMBEDDING_PROVIDER not set — skipping vector generation")
        return [None] * len(texts)

    client = _client_for(provider)
    if client is None:
        log.warning("Embedding provider %r not usable (missing API key) — skipping", provider)
        return [None] * len(texts)

    # Batch the request — both OpenAI and DeepSeek accept a list of strings.
    try:
        resp = await client.embeddings.create(model=model, input=texts)
    except Exception as exc:
        log.warning("Embedding batch failed (%s) — chunks saved without vectors", exc)
        return [None] * len(texts)

    # Cost logging — embeddings are priced per input token.  We approximate
    # with character count when usage is missing.
    usage = getattr(resp, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) if usage else sum(len(t) // 4 for t in texts)
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
