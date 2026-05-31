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

from typing import AsyncIterator

import anthropic
import openai

from api.config.settings import settings
from api.services.usage_logger import log_text_usage

_OPENROUTER_BASE = "https://openrouter.ai/api/v1"


def _provider_label(model: str) -> str:
    """Return a short provider tag for cost logging."""
    if "/" in model:
        return "openrouter"
    if model.startswith("claude-"):
        return "anthropic"
    return "openai"


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
    """
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
