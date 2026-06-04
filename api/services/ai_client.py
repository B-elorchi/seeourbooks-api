"""
Unified AI client — routes text completions to the right provider by model name.

Routing rules (checked in order):
  model contains "/"            → OpenRouter  (openai-compat, e.g. "anthropic/claude-haiku-4-5")
  model starts with "claude-"   → Native Anthropic SDK
  anything else                 → Native OpenAI SDK   (gpt-*, o1-*, o3-*, …)

Usage:
    text = await chat_complete("claude-haiku-4-5", messages=[...], max_tokens=512)
    text = await chat_complete("anthropic/claude-haiku-4-5", ...)   # same model via OpenRouter
    text = await chat_complete("gpt-4.1-mini", ...)
    text = await chat_complete("openai/gpt-4.1-mini", ...)          # same model via OpenRouter

    async for token in chat_stream("claude-sonnet-4-6", messages=[...], max_tokens=2000):
        print(token, end="", flush=True)
"""
from __future__ import annotations

import logging
from typing import AsyncIterator

import anthropic
import openai

from api.config.settings import settings
from api.services.usage_logger import log_text_usage

log = logging.getLogger(__name__)

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _provider_label(model: str) -> str:
    """Return a short provider tag for cost logging."""
    if "/" in model:
        return "openrouter"
    if model.startswith("claude-"):
        return "anthropic"
    return "openai"


# ── Fallback chains ──────────────────────────────────────────────────────────
# If a primary model fails with a recoverable error (credit exhausted, rate-limit,
# provider 5xx, network timeout), the request is retried with the next model in
# this chain.  Each entry is the list of fallbacks tried AFTER the primary.
#
# Design notes:
#   - Claude → equivalent OpenAI tier (Haiku → 4.1-mini, Sonnet/Opus → 4.1)
#   - OpenAI → equivalent Claude tier
#   - OpenRouter wrappers fall back to native — if OpenRouter is down, native
#     Anthropic/OpenAI is often still up.
#
# Admins can override per-model via the provider_config table: set
# FALLBACK_<model> = "<m1>,<m2>,<m3>" to use a custom chain.
_DEFAULT_FALLBACK_CHAINS: dict[str, list[str]] = {
    # Native Anthropic → OpenRouter Anthropic → Native OpenAI tier-equivalent
    "claude-haiku-4-5":          ["anthropic/claude-haiku-4-5",  "openai/gpt-4.1-mini", "gpt-4.1-mini"],
    "claude-haiku-4-5-20251001": ["anthropic/claude-haiku-4-5",  "openai/gpt-4.1-mini", "gpt-4.1-mini"],
    "claude-sonnet-4-6":         ["anthropic/claude-sonnet-4-6", "openai/gpt-4.1",      "gpt-4.1"],
    "claude-opus-4-7":           ["anthropic/claude-opus-4-7",   "openai/gpt-4.1",      "gpt-4.1"],
    # Native OpenAI → Claude tier-equivalent via OpenRouter then native
    "gpt-4.1-mini":              ["openai/gpt-4.1-mini",  "anthropic/claude-haiku-4-5",  "claude-haiku-4-5"],
    "gpt-4.1":                   ["openai/gpt-4.1",       "anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"],
    # OpenRouter wrappers → native counterpart
    "anthropic/claude-haiku-4-5":  ["claude-haiku-4-5",  "openai/gpt-4.1-mini", "gpt-4.1-mini"],
    "anthropic/claude-sonnet-4-6": ["claude-sonnet-4-6", "openai/gpt-4.1",      "gpt-4.1"],
    "anthropic/claude-opus-4-7":   ["claude-opus-4-7",   "openai/gpt-4.1",      "gpt-4.1"],
    "openai/gpt-4.1-mini":         ["gpt-4.1-mini",      "anthropic/claude-haiku-4-5",  "claude-haiku-4-5"],
    "openai/gpt-4.1":              ["gpt-4.1",           "anthropic/claude-sonnet-4-6", "claude-sonnet-4-6"],
}


# Substrings that flag a known-recoverable error.  Lower-cased message check —
# keeps us decoupled from anthropic/openai SDK exception class names.
_RECOVERABLE_KEYWORDS: tuple[str, ...] = (
    # Billing / quota
    "credit balance", "credit_balance", "out of credits",
    "insufficient_quota", "insufficient quota",
    "billing", "payment required",
    # Rate limit
    "rate limit", "rate_limit", "rate_limit_error", "too many requests",
    # Network / outage
    "timed out", "timeout",
    "connection refused", "connection reset", "remote disconnected",
    "service unavailable", "temporarily unavailable",
    "bad gateway", "gateway timeout",
    "name or service not known",
)

# HTTP status codes considered recoverable when we can extract them.
_RECOVERABLE_STATUS: frozenset[int] = frozenset({402, 408, 425, 429, 500, 502, 503, 504})


def _exception_status(exc: BaseException) -> int | None:
    """Extract an HTTP status code from an SDK exception if available."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    resp = getattr(exc, "response", None)
    if resp is not None:
        status = getattr(resp, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def _is_recoverable(exc: BaseException) -> bool:
    """Should this exception trigger a fallback to the next model in the chain?"""
    status = _exception_status(exc)
    if status in _RECOVERABLE_STATUS:
        return True
    msg = str(exc).lower()
    return any(k in msg for k in _RECOVERABLE_KEYWORDS)


async def _resolve_fallback_chain(model: str) -> list[str]:
    """
    Return the ordered list of models to try.  The first element is always the
    caller-supplied model; the rest come from the admin override (if any) or
    the hardcoded default chain.
    """
    # Admin override: provider_config key  FALLBACK_<model> = "m1,m2,m3"
    chain_override: list[str] = []
    try:
        from api.services.config.runtime import get_config_value
        raw = await get_config_value(f"FALLBACK_{model}", "")
        if raw:
            chain_override = [m.strip() for m in raw.split(",") if m.strip()]
    except Exception:
        pass  # config lookup is best-effort

    extras = chain_override or _DEFAULT_FALLBACK_CHAINS.get(model, [])
    # De-duplicate while keeping order; ensure primary is at index 0
    seen: set[str] = set()
    ordered: list[str] = []
    for m in [model, *extras]:
        if m and m not in seen:
            ordered.append(m)
            seen.add(m)
    return ordered


async def _fallback_enabled() -> bool:
    try:
        from api.services.config.runtime import get_config_value
        raw = (await get_config_value("ENABLE_MODEL_FALLBACK",
                                       str(settings.ENABLE_MODEL_FALLBACK))).lower()
        return raw in ("1", "true", "yes", "on")
    except Exception:
        return settings.ENABLE_MODEL_FALLBACK


# ── Client factories ──────────────────────────────────────────────────────────

def _is_openrouter(model: str) -> bool:
    return "/" in model


def _is_anthropic(model: str) -> bool:
    return model.startswith("claude-") and "/" not in model


def _anthropic_client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)


def _openai_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(api_key=settings.OPENAI_API_KEY)


def _openrouter_client() -> openai.AsyncOpenAI:
    if not settings.OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env to use OpenRouter models."
        )
    return openai.AsyncOpenAI(
        base_url=_OPENROUTER_BASE,
        api_key=settings.OPENROUTER_API_KEY,
        default_headers={
            "HTTP-Referer": "https://seeourbook.sa",
            "X-Title":      "SeeOurBook",
        },
    )


# ── Helpers for message normalisation ─────────────────────────────────────────

def _prepend_system(system: str | None, messages: list[dict]) -> list[dict]:
    """Insert a system message at position 0 for OpenAI-style APIs."""
    if system:
        return [{"role": "system", "content": system}] + list(messages)
    return list(messages)


# ── Public interface ──────────────────────────────────────────────────────────

async def chat_complete(
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    system: str | None = None,
) -> str:
    """
    Single-turn completion. Returns the full response text.
    Works with any model — Anthropic native, OpenAI native, or OpenRouter.

    If ENABLE_MODEL_FALLBACK is on (default) and the call fails with a
    recoverable error — credit exhausted, rate-limit, provider 5xx, network
    timeout — the request is automatically retried with the next model in the
    fallback chain.  Non-recoverable errors (bad input, auth, unknown model)
    surface immediately so they are not silently masked.
    """
    if not await _fallback_enabled():
        return await _chat_complete_single(model, messages, max_tokens, system)

    chain = await _resolve_fallback_chain(model)
    last_exc: BaseException | None = None

    for attempt_model in chain:
        try:
            result = await _chat_complete_single(attempt_model, messages, max_tokens, system)
            if attempt_model != model:
                log.warning(
                    "Model fallback succeeded: primary=%r → used=%r (after %s)",
                    model, attempt_model, last_exc and type(last_exc).__name__,
                )
            return result
        except Exception as exc:
            last_exc = exc
            if not _is_recoverable(exc):
                # Bad input, auth failure, unknown model — don't mask.
                raise
            log.warning(
                "Model %r failed with %s — %s; trying next in fallback chain",
                attempt_model, type(exc).__name__, str(exc)[:240],
            )

    # Every model in the chain failed with a recoverable error.
    assert last_exc is not None
    raise last_exc


async def _chat_complete_single(
    model: str,
    messages: list[dict],
    max_tokens: int,
    system: str | None,
) -> str:
    """One attempt at a chat completion, no fallback logic."""
    if _is_openrouter(model):
        client = _openrouter_client()
        resp = await client.chat.completions.create(
            model=model,
            messages=_prepend_system(system, messages),
            max_tokens=max_tokens,
        )
        usage = getattr(resp, "usage", None)
        await log_text_usage(
            provider     = "openrouter",
            model        = model,
            input_tokens = getattr(usage, "prompt_tokens",     0) if usage else 0,
            output_tokens= getattr(usage, "completion_tokens", 0) if usage else 0,
        )
        return resp.choices[0].message.content or ""

    if _is_anthropic(model):
        client = _anthropic_client()
        kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        msg = await client.messages.create(**kwargs)
        usage = getattr(msg, "usage", None)
        await log_text_usage(
            provider     = "anthropic",
            model        = model,
            input_tokens = getattr(usage, "input_tokens",  0) if usage else 0,
            output_tokens= getattr(usage, "output_tokens", 0) if usage else 0,
        )
        return msg.content[0].text

    # Native OpenAI
    client = _openai_client()
    resp = await client.chat.completions.create(
        model=model,
        messages=_prepend_system(system, messages),
        max_tokens=max_tokens,
    )
    usage = getattr(resp, "usage", None)
    await log_text_usage(
        provider     = "openai",
        model        = model,
        input_tokens = getattr(usage, "prompt_tokens",     0) if usage else 0,
        output_tokens= getattr(usage, "completion_tokens", 0) if usage else 0,
    )
    return resp.choices[0].message.content or ""


async def chat_stream(
    model: str,
    messages: list[dict],
    max_tokens: int = 1024,
    system: str | None = None,
) -> AsyncIterator[str]:
    """
    Streaming completion. Yields text tokens as they arrive.
    Works with any model — Anthropic native, OpenAI native, or OpenRouter.
    """
    if _is_openrouter(model):
        client = _openrouter_client()
        stream = await client.chat.completions.create(
            model=model,
            messages=_prepend_system(system, messages),
            max_tokens=max_tokens,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta
        return

    if _is_anthropic(model):
        client = _anthropic_client()
        kwargs: dict = dict(model=model, max_tokens=max_tokens, messages=messages)
        if system:
            kwargs["system"] = system
        async with client.messages.stream(**kwargs) as stream:
            async for token in stream.text_stream:
                yield token
        return

    # Native OpenAI
    client = _openai_client()
    stream = await client.chat.completions.create(
        model=model,
        messages=_prepend_system(system, messages),
        max_tokens=max_tokens,
        stream=True,
    )
    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
