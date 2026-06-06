"""
AI provider abstraction for the documents pipeline.

Each provider implements the `TextAnalysisProvider` Protocol:

    async def generate_summary(text: str, language: str) -> str
    async def generate_structured_json(text: str, language: str) -> dict

Three concrete implementations are bundled:

    DeepSeekProvider   — calls api.deepseek.com (OpenAI-compatible).  Default.
    OpenAIProvider     — calls api.openai.com via the openai SDK.
    ClaudeProvider     — routes through `ai_client.chat_complete`, which gets
                         automatic ENABLE_MODEL_FALLBACK + cost logging for free.

The factory `get_provider(name)` reads runtime config and returns the right
implementation.  When the chosen provider isn't configured (e.g. DEEPSEEK_API_KEY
empty), it falls back to ClaudeProvider so the pipeline never hard-fails.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Protocol, runtime_checkable

from openai import AsyncOpenAI

from api.config.settings import settings
from api.services.ai_client import chat_complete
from api.services.config.runtime import get_config_value
from api.services.documents.errors import AIFailureError
from api.services.usage_logger import log_text_usage

log = logging.getLogger(__name__)


# ── Prompts ──────────────────────────────────────────────────────────────────

_SUMMARY_PROMPT_EN = (
    "You are a document-analysis expert. Summarize the following document in "
    "a clear, well-structured way. Capture the main thesis, key arguments, "
    "important sections, and notable conclusions. Output ONLY the summary "
    "text — no preamble, no markdown headers, no bullet points unless the "
    "document is itself a list.\n\nDOCUMENT:\n{text}"
)

_SUMMARY_PROMPT_AR = (
    "أنت خبير في تحليل الوثائق. قم بتلخيص الوثيقة التالية بشكل واضح ومنظم، "
    "مع إبراز الفكرة الرئيسية والحجج الأساسية والأقسام المهمة والاستنتاجات البارزة. "
    "أعد فقط نص التلخيص — دون مقدمات أو علامات تنسيق أو نقاط ما لم تكن الوثيقة قائمة في الأصل."
    "\n\nالوثيقة:\n{text}"
)

_STRUCTURED_PROMPT = (
    "You are a document-analysis expert.\n"
    "Analyze the following document and return VALID JSON only, no markdown "
    "fences, no commentary.\n\n"
    "Schema (every field required, use [] or \"\" for missing values):\n"
    "{{\n"
    "  \"title\":    \"\",\n"
    "  \"authors\":  [],\n"
    "  \"language\": \"\",\n"
    "  \"summary\":  \"\",\n"
    "  \"topics\":   [],\n"
    "  \"keywords\": [],\n"
    "  \"entities\": [{{\"name\": \"\", \"type\": \"\"}}],\n"
    "  \"chapters\": [{{\"title\": \"\", \"summary\": \"\"}}]\n"
    "}}\n\n"
    "Detected language: {language}\n"
    "Documents in Arabic: output JSON with Arabic values; document in English: "
    "output English. Do not translate.\n\n"
    "DOCUMENT:\n{text}"
)


# Per-call text cap.  The full extracted document can be hundreds of thousands
# of tokens — we send a leading slice to keep latency and cost bounded.
# Downstream consumers re-summarize from the structured output if needed.
_MAX_INPUT_CHARS_SUMMARY    = 60_000
_MAX_INPUT_CHARS_STRUCTURED = 40_000


# ── Provider Protocol ────────────────────────────────────────────────────────

@runtime_checkable
class TextAnalysisProvider(Protocol):
    name: str
    model: str

    async def generate_summary(self, text: str, language: str) -> str: ...
    async def generate_structured_json(self, text: str, language: str) -> dict[str, Any]: ...


# ── Shared JSON-parsing helper ───────────────────────────────────────────────

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


def _coerce_json(raw: str) -> dict[str, Any]:
    """
    Parse model output that's *supposed* to be JSON.  Strips markdown fences,
    finds the outermost {...} block, parses it.  Raises AIFailureError when
    nothing valid can be extracted.
    """
    s = (raw or "").strip()
    s = _JSON_FENCE_RE.sub("", s).strip()

    # Try direct parse first
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first '{' and try parsing onward
    first = s.find("{")
    last  = s.rfind("}")
    if first == -1 or last == -1 or last <= first:
        raise AIFailureError(
            "AI provider did not return JSON",
            detail={"sample": raw[:400]},
        )
    candidate = s[first:last + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise AIFailureError(
            f"AI provider returned malformed JSON: {exc}",
            detail={"sample": candidate[:400]},
        ) from exc


def _pick_prompt(text: str, language: str) -> str:
    template = _SUMMARY_PROMPT_AR if language == "ara" else _SUMMARY_PROMPT_EN
    return template.format(text=text[:_MAX_INPUT_CHARS_SUMMARY])


def _structured_prompt(text: str, language: str) -> str:
    return _STRUCTURED_PROMPT.format(
        text=text[:_MAX_INPUT_CHARS_STRUCTURED],
        language=language,
    )


# ── DeepSeek (OpenAI-compatible) ─────────────────────────────────────────────

class DeepSeekProvider:
    name = "deepseek"

    def __init__(self, model: str | None = None) -> None:
        if not settings.DEEPSEEK_API_KEY:
            raise AIFailureError(
                "DEEPSEEK_API_KEY is not set — add it to .env or switch "
                "DOC_AI_PROVIDER to 'openai' or 'claude'.",
                detail={"code": "missing_api_key"},
            )
        self.model = model or settings.DOC_AI_MODEL or "deepseek-chat"
        self._client = AsyncOpenAI(
            api_key=settings.DEEPSEEK_API_KEY,
            base_url=settings.DEEPSEEK_BASE_URL,
        )

    async def _chat(self, prompt: str, *, json_mode: bool) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3,
        )
        if json_mode:
            # DeepSeek supports OpenAI-style response_format for JSON.
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise AIFailureError(f"DeepSeek call failed: {exc}") from exc

        usage = getattr(resp, "usage", None)
        await log_text_usage(
            provider     = "deepseek",
            model        = self.model,
            input_tokens = getattr(usage, "prompt_tokens",     0) if usage else 0,
            output_tokens= getattr(usage, "completion_tokens", 0) if usage else 0,
        )
        return resp.choices[0].message.content or ""

    async def generate_summary(self, text: str, language: str) -> str:
        return (await self._chat(_pick_prompt(text, language), json_mode=False)).strip()

    async def generate_structured_json(self, text: str, language: str) -> dict[str, Any]:
        raw = await self._chat(_structured_prompt(text, language), json_mode=True)
        return _coerce_json(raw)


# ── Native OpenAI ────────────────────────────────────────────────────────────

class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str | None = None) -> None:
        if not settings.OPENAI_API_KEY:
            raise AIFailureError(
                "OPENAI_API_KEY is not set",
                detail={"code": "missing_api_key"},
            )
        self.model   = model or settings.DOC_AI_MODEL or "gpt-4.1-mini"
        self._client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def _chat(self, prompt: str, *, json_mode: bool) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=4000,
            temperature=0.3,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise AIFailureError(f"OpenAI call failed: {exc}") from exc

        usage = getattr(resp, "usage", None)
        await log_text_usage(
            provider     = "openai",
            model        = self.model,
            input_tokens = getattr(usage, "prompt_tokens",     0) if usage else 0,
            output_tokens= getattr(usage, "completion_tokens", 0) if usage else 0,
        )
        return resp.choices[0].message.content or ""

    async def generate_summary(self, text: str, language: str) -> str:
        return (await self._chat(_pick_prompt(text, language), json_mode=False)).strip()

    async def generate_structured_json(self, text: str, language: str) -> dict[str, Any]:
        raw = await self._chat(_structured_prompt(text, language), json_mode=True)
        return _coerce_json(raw)


# ── Claude (via the unified ai_client with fallback chain) ───────────────────

class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str | None = None) -> None:
        # Default to Sonnet for document analysis — better at long-form
        # structured output than Haiku.
        #
        # NOTE: We do NOT default to settings.DOC_AI_MODEL here — that field
        # may hold a value tailored for a DIFFERENT provider (e.g. when the
        # admin set DOC_AI_PROVIDER=deepseek + DOC_AI_MODEL=deepseek-chat
        # and we're falling back to Claude because the DeepSeek key is missing).
        # We only honor an explicitly-passed Claude/OpenRouter model.
        if model and (model.startswith("claude-") or "/" in model):
            self.model = model
        else:
            self.model = settings.MODEL_SONNET

    async def generate_summary(self, text: str, language: str) -> str:
        try:
            out = await chat_complete(
                model    = self.model,
                messages = [{"role": "user", "content": _pick_prompt(text, language)}],
                max_tokens=2500,
            )
        except Exception as exc:
            raise AIFailureError(f"Claude call failed: {exc}") from exc
        return out.strip()

    async def generate_structured_json(self, text: str, language: str) -> dict[str, Any]:
        # No native JSON mode on Anthropic — rely on prompt discipline + the
        # coercer to recover from minor format slips.
        try:
            out = await chat_complete(
                model    = self.model,
                messages = [{"role": "user", "content": _structured_prompt(text, language)}],
                max_tokens=4000,
            )
        except Exception as exc:
            raise AIFailureError(f"Claude call failed: {exc}") from exc
        return _coerce_json(out)


# ── OpenRouter (any vendor/model, including DeepSeek, Llama, Qwen, etc.) ────

class OpenRouterProvider:
    """
    Route any `vendor/model` identifier through OpenRouter.

    Why this exists in addition to DeepSeekProvider:
      - You already have OPENROUTER_API_KEY — no extra account / billing needed.
      - OpenRouter relays to every major lab (DeepSeek, Anthropic, OpenAI,
        Google, Meta, Mistral, Qwen, Cohere, etc.) under one key.
      - Best ratio of price-to-quality for document analysis is typically
        `deepseek/deepseek-chat` (≈$0.14 per 1M input tokens).

    Common admin pick:
        DOC_AI_PROVIDER = openrouter
        DOC_AI_MODEL    = deepseek/deepseek-chat
    """
    name = "openrouter"

    def __init__(self, model: str | None = None) -> None:
        if not settings.OPENROUTER_API_KEY:
            raise AIFailureError(
                "OPENROUTER_API_KEY is not set — add it to .env or switch "
                "DOC_AI_PROVIDER to 'openai' or 'claude'.",
                detail={"code": "missing_api_key"},
            )
        chosen = model or settings.DOC_AI_MODEL or "deepseek/deepseek-chat"
        # Sanity: must look like a `vendor/model` identifier for OpenRouter
        if "/" not in chosen:
            log.warning(
                "OpenRouterProvider got non-OpenRouter model %r — defaulting to "
                "deepseek/deepseek-chat", chosen,
            )
            chosen = "deepseek/deepseek-chat"
        self.model = chosen
        self._client = AsyncOpenAI(
            api_key  = settings.OPENROUTER_API_KEY,
            base_url = "https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://seeourbook.sa",
                "X-Title":      "SeeOurBook",
            },
        )

    async def _chat(self, prompt: str, *, json_mode: bool) -> str:
        kwargs: dict[str, Any] = dict(
            model       = self.model,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 4000,
            temperature = 0.3,
        )
        if json_mode:
            # Most OpenAI-compatible vendors on OpenRouter accept this.
            # If a specific model doesn't, the request still succeeds —
            # response_format is simply ignored, and _coerce_json mops up.
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:
            raise AIFailureError(f"OpenRouter call failed: {exc}") from exc

        usage = getattr(resp, "usage", None)
        await log_text_usage(
            provider     = "openrouter",
            model        = self.model,
            input_tokens = getattr(usage, "prompt_tokens",     0) if usage else 0,
            output_tokens= getattr(usage, "completion_tokens", 0) if usage else 0,
        )
        return resp.choices[0].message.content or ""

    async def generate_summary(self, text: str, language: str) -> str:
        return (await self._chat(_pick_prompt(text, language), json_mode=False)).strip()

    async def generate_structured_json(self, text: str, language: str) -> dict[str, Any]:
        raw = await self._chat(_structured_prompt(text, language), json_mode=True)
        return _coerce_json(raw)


# ── Factory ──────────────────────────────────────────────────────────────────

_PROVIDER_CLASSES: dict[str, type] = {
    "deepseek":   DeepSeekProvider,
    "openai":     OpenAIProvider,
    "claude":     ClaudeProvider,
    "openrouter": OpenRouterProvider,
}


async def get_provider(name: str | None = None) -> TextAnalysisProvider:
    """
    Resolve a provider by name + model, with admin-config override and
    graceful fallback.

    Smart routing
    ─────────────
    If `DOC_AI_MODEL` contains a "/" (e.g. "deepseek/deepseek-chat",
    "anthropic/claude-sonnet-4-6"), we auto-route to OpenRouterProvider
    regardless of what `DOC_AI_PROVIDER` says — that's what the admin
    obviously intended.  Saves admins from having to keep two settings
    in sync.

    Resolution order for provider:
        1. explicit `name` argument
        2. runtime config DOC_AI_PROVIDER
        3. settings.DOC_AI_PROVIDER (default 'deepseek')

    Resolution order for model:
        1. runtime config DOC_AI_MODEL
        2. settings.DOC_AI_MODEL

    If the chosen provider can't initialise (missing API key), we fall back
    to ClaudeProvider with ITS OWN default model — never pass through the
    original provider's model (which wouldn't work for Anthropic).
    """
    explicit = (name or await get_config_value("DOC_AI_PROVIDER", "")
                or settings.DOC_AI_PROVIDER or "deepseek").lower()
    model    = await get_config_value("DOC_AI_MODEL", settings.DOC_AI_MODEL) or None

    # ── Auto-route OpenRouter-style models regardless of the provider field ─
    if model and "/" in model and explicit != "openrouter":
        log.info(
            "DOC_AI_MODEL %r looks like an OpenRouter identifier — auto-routing "
            "via OpenRouterProvider (ignoring DOC_AI_PROVIDER=%s).",
            model, explicit,
        )
        try:
            return OpenRouterProvider(model=model)
        except AIFailureError as exc:
            log.warning(
                "OpenRouterProvider init failed (%s); falling back to Claude.", exc,
            )
            return ClaudeProvider()

    cls = _PROVIDER_CLASSES.get(explicit)
    if cls is None:
        log.warning(
            "Unknown DOC_AI_PROVIDER %r — falling back to ClaudeProvider (model=%s).",
            explicit, settings.MODEL_SONNET,
        )
        return ClaudeProvider()

    try:
        return cls(model=model)
    except AIFailureError as exc:
        log.warning(
            "Provider %s init failed (%s); falling back to ClaudeProvider with model=%s.",
            explicit, exc, settings.MODEL_SONNET,
        )
        return ClaudeProvider()
