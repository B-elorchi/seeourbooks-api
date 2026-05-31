"""
Usage / cost logger.

Every external paid API call (chat completion, TTS, image generation, vision)
records one row in the `usage_logs` table.  The admin Costs tab aggregates
those rows for per-provider / per-step / per-model breakdowns.

How the calling code uses it
────────────────────────────
    from api.services.usage_logger import (
        set_job_context, step_context,
        log_text_usage, log_tts_usage, log_image_usage,
    )

    # Once per pipeline job (in pipeline.py _run_job)
    set_job_context(job_id)

    # Around each pipeline step (in orchestrator.py)
    with step_context("summarize"):
        ...
        await log_text_usage(provider="anthropic", model="claude-haiku-4-5",
                             input_tokens=1234, output_tokens=567)

The context vars mean the producers (ai_client, tts, cover, alttext) don't
need job_id / step plumbed through their function signatures.

Failure isolation
─────────────────
Every insert is wrapped in try/except — a logging failure NEVER bubbles up
into the pipeline.  Worst case we lose a cost row.
"""
from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager

from api.services.db import insert

log = logging.getLogger(__name__)


# ── Context vars — populated at the top of the pipeline ──────────────────────

_job_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_job_id", default=None
)
_step_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "usage_step", default=None
)


def set_job_context(job_id: str | None) -> None:
    """Tag every subsequent usage log with this job_id."""
    _job_id_var.set(job_id)


@contextmanager
def step_context(step: str):
    """`with step_context("summarize"): ...` — scope usage logs to a pipeline step."""
    token = _step_var.set(step)
    try:
        yield
    finally:
        _step_var.reset(token)


def set_step(step: str) -> None:
    """
    Set the current pipeline step without scoping.  Use this in the orchestrator
    right after `step_status[step] = "running"` — simpler than wrapping every
    try-block in a `with` statement, and good enough because each step runs
    sequentially within the same task.
    """
    _step_var.set(step)


# ── Rate table ───────────────────────────────────────────────────────────────
# Approximate USD prices used to estimate cost per call.  Real billing should
# come from each provider's dashboard — these numbers are for in-app guidance.
#
# Text models: (input_per_1m_tokens, output_per_1m_tokens) in USD
_TEXT_RATES: dict[str, tuple[float, float]] = {
    # Anthropic
    "claude-haiku-4-5":          (1.00,  5.00),
    "claude-haiku-4-5-20251001": (1.00,  5.00),
    "claude-sonnet-4-6":         (3.00, 15.00),
    "claude-opus-4-7":           (15.00, 75.00),
    # OpenAI
    "gpt-4.1-mini": (0.40,  1.60),
    "gpt-4.1":      (2.00,  8.00),
    "o3-mini":      (1.10,  4.40),
    "o1-mini":      (1.10,  4.40),
}

# TTS: USD per 1000 characters
_TTS_RATES_PER_1K_CHARS: dict[str, float] = {
    "deepgram":   0.015,
    "elevenlabs": 0.30,
    "cartesia":   0.065,
    "gemini":     0.05,
}

# Image generation: USD per image
_IMAGE_RATES_PER_CALL: dict[str, float] = {
    "dall-e-3":                                0.04,
    "gpt-image-1":                             0.04,
    "dall-e-2":                                0.02,
    "google/gemini-2.5-flash-image":           0.04,
    "google/gemini-2.0-flash-exp:image":       0.04,
    "black-forest-labs/flux-1.1-pro":          0.04,
    "black-forest-labs/flux-schnell":          0.003,
    "stability-ai/stable-diffusion-3.5-large": 0.04,
}


def _bare_model(model: str) -> str:
    """'openai/gpt-4.1-mini' → 'gpt-4.1-mini'."""
    return model.split("/", 1)[1] if "/" in model else model


def _text_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = _TEXT_RATES.get(model) or _TEXT_RATES.get(_bare_model(model))
    if not rates:
        return 0.0
    in_rate, out_rate = rates
    return (input_tokens / 1_000_000) * in_rate + (output_tokens / 1_000_000) * out_rate


def _tts_cost_usd(provider: str, characters: int) -> float:
    rate = _TTS_RATES_PER_1K_CHARS.get(provider, 0.0)
    return (characters / 1000.0) * rate


def _image_cost_usd(model: str, count: int) -> float:
    rate = _IMAGE_RATES_PER_CALL.get(model) or _IMAGE_RATES_PER_CALL.get(_bare_model(model), 0.0)
    return count * rate


# ── Public logging entry points ──────────────────────────────────────────────

async def _safe_insert(row: dict) -> None:
    """Best-effort insert. Logs but never raises on failure."""
    try:
        await insert("usage_logs", row)
    except Exception as exc:
        log.warning("usage_logs insert failed: %s", exc)


async def log_text_usage(
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    job_id: str | None = None,
    step: str | None = None,
) -> None:
    """Log a chat completion or vision call (token-priced)."""
    if input_tokens <= 0 and output_tokens <= 0:
        return
    await _safe_insert({
        "job_id":    job_id or _job_id_var.get(),
        "step":      step or _step_var.get() or "unknown",
        "provider":  provider,
        "model":     model,
        "units":     float(input_tokens + output_tokens),
        "unit_type": "tokens",
        "cost_usd":  round(_text_cost_usd(model, input_tokens, output_tokens), 6),
    })


async def log_tts_usage(
    *,
    provider: str,
    model: str,
    characters: int,
    job_id: str | None = None,
    step: str | None = None,
) -> None:
    """Log a TTS call (per-character priced)."""
    if characters <= 0:
        return
    await _safe_insert({
        "job_id":    job_id or _job_id_var.get(),
        "step":      step or _step_var.get() or "audio_full",
        "provider":  provider,
        "model":     model,
        "units":     float(characters),
        "unit_type": "characters",
        "cost_usd":  round(_tts_cost_usd(provider, characters), 6),
    })


async def log_image_usage(
    *,
    provider: str,
    model: str,
    count: int = 1,
    job_id: str | None = None,
    step: str | None = None,
) -> None:
    """Log an image generation call (per-image priced)."""
    if count <= 0:
        return
    await _safe_insert({
        "job_id":    job_id or _job_id_var.get(),
        "step":      step or _step_var.get() or "cover",
        "provider":  provider,
        "model":     model,
        "units":     float(count),
        "unit_type": "images",
        "cost_usd":  round(_image_cost_usd(model, count), 6),
    })
