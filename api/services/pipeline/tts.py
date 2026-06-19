"""
Text-to-Speech service.
Supports Deepgram, ElevenLabs, Cartesia, and Google Gemini (via OpenRouter).
Provider and voice are resolved from a runtime cfg dict (from admin panel) with
fallback to settings.py defaults — no restart needed when the admin switches providers.

Long-text handling
──────────────────
All providers have a per-call character limit (~2–5K).  A 10-minute book summary
is ~8–10K chars and would 4xx in a single request.  synthesize() splits long
text on sentence boundaries, synthesizes each chunk separately, and concatenates
the resulting MP3 byte streams.  The downstream audio-processing step re-encodes
to clean up any edge cases at chunk boundaries.

Arabic TTS providers (by recommendation order)
──────────────────────────────────────────────
  cartesia   — set CARTESIA_API_KEY + TTS_VOICE_AR=UUID, CARTESIA_MODEL=sonic-3.5-*
  gemini     — set OPENROUTER_API_KEY (uses google/gemini-2.5-flash-preview-tts)
  elevenlabs — set ELEVENLABS_API_KEY + ELEVENLABS_VOICE_AR (multilingual v2)
  deepgram   — ENGLISH ONLY, do not use for Arabic
"""
import base64
import logging
import os
import re
import httpx

_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

def _is_uuid(v: str | None) -> bool:
    return bool(v and _UUID_RE.match(v))
from api.config.settings import settings
from api.services.usage_logger import log_tts_usage
from api.services.openrouter_keys import (
    get_openrouter_key,
    rotate_openrouter_key,
    is_credit_error,
    openrouter_key_count,
)

log = logging.getLogger(__name__)

# Deepgram Aura model names that are English-only.
# Sending Arabic text to these produces garbled / unrecognisable audio.
_DEEPGRAM_ENGLISH_VOICE_PREFIXES = ("aura-",)

# OpenRouter TTS voice catalogs. Gemini models require native Gemini voice names;
# OpenAI audio models require OpenAI voice names. We hardcode the allowed names so
# a config mismatch (e.g. AR voice left as "fable" after switching to a Google
# model) is corrected automatically instead of producing a 500.
_OPENROUTER_GEMINI_VOICES = {
    "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe",
    "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir", "Gacrux",
    "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Puck", "Pulcherrima",
    "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat", "Umbriel",
    "Vindemiatrix", "Zephyr", "Zubenelgenubi",
}
_OPENROUTER_OPENAI_VOICES = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse",
    "ballad", "ash", "sage", "marin", "cedar",
}

# Per-provider character budget — the maximum size of a single TTS request body.
# Deepgram's published limit is 2000 but accounts on lower tiers 413 below that,
# so we use 1500 as a safe operational ceiling.  All other providers happily
# accept 3000+ chars, but staying low keeps audio chunks short enough that any
# per-chunk failure costs little to retry.
_PROVIDER_MAX_CHARS: dict[str, int] = {
    "deepgram":   1500,
    "elevenlabs": 2500,
    "cartesia":   2500,
    "gemini":     2500,
    "openrouter": 2500,
}
_DEFAULT_MAX_CHARS = 1500

# Gemini TTS handles long context in a SINGLE call, which avoids the voice/tone
# seams you get when a summary is split across several independent requests.
# BUT Gemini's real limit is ~8,192 INPUT TOKENS, not characters — and Arabic
# (especially with tashkeel) tokenises ~3-4x heavier than English. So the same
# character budget that fits English in one call will blow past the token limit
# in Arabic and Gemini silently narrates only the part that fits (truncated
# audio). We therefore keep a generous budget for English/Latin text and a much
# smaller one for Arabic. Override per deployment via TTS_MAX_CHARS_GEMINI /
# TTS_MAX_CHARS_GEMINI_AR. Tune DOWN if audio is cut short; UP for fewer seams.
_GEMINI_MAX_CHARS    = 2500   # English / Latin — Gemini silently truncates audio
# output when a single call would produce more than ~1:40 of speech; 2500 chars
# (~500 words, ~3:20 at audiobook pace) stays safely under that limit.
# The old value was 8000, which produced complete audio for short summaries but
# silently cut the end of longer ones. Lower if EN audio still sounds truncated.
_GEMINI_MAX_CHARS_AR = 2200   # Arabic with tashkeel — ~2-3 tokens/char, very dense


def _is_english_only_voice(voice: str) -> bool:
    return any(voice.startswith(p) for p in _DEEPGRAM_ENGLISH_VOICE_PREFIXES)


_VALID_AUDIO_STYLES = {"single", "multi", "podcast", "audiobook", "news", "bedtime", "custom"}

# Short bracket tags prepended to every TTS chunk *after the first*.
# Gemini TTS treats bracketed tags as delivery directions (not spoken words),
# so these help keep voice/tone consistent at chunk boundaries.
# "single" now gets a tag too — previously empty, leaving subsequent chunks
# without any continuity hint and causing the voice to drift.
_STYLE_TAGS: dict[str, str] = {
    "single":    "same voice",      # was "" — causes voice drift between chunks
    "audiobook": "narrator",
    "news":      "anchor",
    "bedtime":   "soothing",
    "custom":    "custom delivery",
    "multi":     "dialogue",
    "podcast":   "podcast",
}


def _gemini_voice_name(voice: str | None, fallback: str = "Kore") -> str:
    """Return a valid Gemini voice name, falling back if the input is missing/invalid."""
    voice = (voice or fallback).strip()
    if voice not in _OPENROUTER_GEMINI_VOICES:
        log.warning("Invalid Gemini voice %r; falling back to %r.", voice, fallback)
        return fallback
    return voice


def _resolve_speaker_voices(cfg: dict, language: str) -> tuple[str, str]:
    """Resolve Speaker 1 / Speaker 2 Gemini voices (with optional per-language overrides)."""
    lang = language.upper()
    v1 = cfg.get(f"GEMINI_TTS_SPEAKER1_VOICE_{lang}") or cfg.get("GEMINI_TTS_SPEAKER1_VOICE") or "Kore"
    v2 = cfg.get(f"GEMINI_TTS_SPEAKER2_VOICE_{lang}") or cfg.get("GEMINI_TTS_SPEAKER2_VOICE") or "Puck"
    return _gemini_voice_name(v1, "Kore"), _gemini_voice_name(v2, "Puck")


def _apply_audio_style(text: str, style: str | None, cfg: dict) -> tuple[str, bool, str]:
    """
    Apply a Gemini TTS style/profile to the transcript.

    Returns (styled_text, use_multi_speaker, chunk_prefix).  Multi-speaker
    styles rewrite the text as alternating Speaker1 / Speaker2 lines; other
    styles may prepend a natural-language direction prompt.

    chunk_prefix is a short bracketed tag (e.g. "[narrator]") that is prepended
    to every TTS chunk *except* the first one.  It reminds the model to keep the
    same delivery across chunk boundaries, which prevents sudden tone shifts
    ~40 seconds into a long audio file.

    Style control is prompt-based in Gemini TTS — there is no dedicated API
    field for pace/tone/profiles.  See:
    https://ai.google.dev/gemini-api/docs/speech-generation
    """
    style = (style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
    if style not in _VALID_AUDIO_STYLES:
        style = "single"

    tag = _STYLE_TAGS.get(style, "").strip()
    chunk_prefix = f"[{tag}]\n" if tag else ""

    # Custom style: admin provides the full direction prompt.
    if style == "custom":
        prompt = (cfg.get("GEMINI_TTS_STYLE_PROMPT") or "").strip()
        return (f"{prompt}\n\n{text}" if prompt else text), False, chunk_prefix

    # Multi-speaker / podcast: alternate sentences between two speakers.
    if style in ("multi", "podcast"):
        s1 = (cfg.get("GEMINI_TTS_SPEAKER1_NAME") or "Speaker1").strip()
        s2 = (cfg.get("GEMINI_TTS_SPEAKER2_NAME") or "Speaker2").strip()
        intro = ""
        if style == "podcast":
            intro = f"TTS the following podcast episode between {s1} and {s2}.\n\n"
        else:
            intro = f"TTS the following conversation between {s1} and {s2}.\n\n"
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
        if not sentences:
            return text, True, chunk_prefix
        lines = []
        for i, sent in enumerate(sentences):
            speaker = s1 if i % 2 == 0 else s2
            lines.append(f"{speaker}: {sent}")
        return intro + "\n".join(lines), True, chunk_prefix

    # Single-speaker styles: prepend a direction prompt.
    # "single" now also has a default direction so that the voice characteristics
    # set in chunk 0 are more likely to be reproduced by subsequent chunks
    # (which receive a "[same voice] continue" tag via chunk_prefix).
    prompts = {
        "single":    "Read the following text in a clear, natural, professional voice.",
        "audiobook": "Read the following text in a calm, immersive audiobook narration style.",
        "news":      "Read the following text as a professional news broadcast.",
        "bedtime":   "Read the following text in a soothing, gentle bedtime story style.",
    }
    prompt = prompts.get(style, "")
    # Allow admins to override or extend any style prompt.
    custom_prompt = (cfg.get("GEMINI_TTS_STYLE_PROMPT") or "").strip()
    if custom_prompt:
        prompt = custom_prompt
    return (f"{prompt}\n\n{text}" if prompt else text), False, chunk_prefix


# Sentence-ending punctuation in EN / AR / general — match end-of-sentence so we
# can split long text on natural boundaries.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟…])\s+")


def _split_text_for_tts(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    """
    Split text into TTS-sized chunks on sentence boundaries.

    Guarantees no chunk exceeds `max_chars`.  If a single sentence is longer
    than `max_chars` (rare), it falls back to splitting on whitespace.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
    chunks: list[str] = []
    buf = ""

    def flush():
        nonlocal buf
        if buf.strip():
            chunks.append(buf.strip())
        buf = ""

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue

        # Pathological case: one sentence longer than the budget — split on words
        if len(sent) > max_chars:
            flush()
            words = sent.split(" ")
            inner = ""
            for w in words:
                candidate = (inner + " " + w).strip() if inner else w
                if len(candidate) > max_chars:
                    if inner:
                        chunks.append(inner.strip())
                    inner = w
                else:
                    inner = candidate
            if inner.strip():
                buf = inner
            continue

        # Normal: try to append, flush if it would overflow
        candidate = (buf + " " + sent).strip() if buf else sent
        if len(candidate) > max_chars:
            flush()
            buf = sent
        else:
            buf = candidate

    flush()
    return chunks


async def synthesize(
    text: str,
    language: str,
    output_path: str,
    cfg: dict | None = None,
    audio_style: str | None = None,
) -> str:
    """
    Convert text to MP3 and save to output_path. Returns output_path.

    Text longer than `_MAX_TTS_CHARS` is split on sentence boundaries,
    each chunk is synthesized individually, and the resulting MP3 streams
    are concatenated.  The downstream audio-processing step re-encodes
    the file which cleans up any frame-boundary artefacts.
    """
    cfg = cfg or {}
    lang = language.upper()          # "EN" or "AR"

    # ── Spoken audio watermark ───────────────────────────────────────────────
    # Prepend a short branding intro that is read aloud at the start of the
    # audio (e.g. "Seeourbook تقدم لكم"). Configurable per language in
    # Admin → Settings → Watermarks. Empty = no intro.
    watermark = (cfg.get(f"AUDIO_WATERMARK_TEXT_{lang}") or "").strip()
    if watermark:
        # Trailing period + blank line gives the TTS engine a natural pause
        # before the content begins.
        text = f"{watermark}.\n\n{text}"

    provider = cfg.get(f"TTS_PROVIDER_{lang}") or (
        settings.TTS_PROVIDER_EN if language == "en" else settings.TTS_PROVIDER_AR
    )
    voice = cfg.get(f"TTS_VOICE_{lang}") or (
        settings.TTS_VOICE_EN if language == "en" else settings.TTS_VOICE_AR
    )

    # ── Gemini TTS style / profile ─────────────────────────────────────────────
    # Resolve the requested audio style (request option → admin config → single).
    # Multi-speaker styles rewrite the transcript as alternating speaker lines;
    # single-speaker styles may prepend a natural-language direction prompt.
    # These styles are only sent to Gemini-native or OpenRouter-Google models;
    # other providers receive plain text so prompts are not read aloud.
    is_gemini_tts = provider == "gemini" or (
        provider == "openrouter"
        and (cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL).lower().startswith("google/")
    )
    effective_style = "single"
    if is_gemini_tts:
        effective_style = (audio_style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
    styled_text, is_multi_style, chunk_prefix = _apply_audio_style(text, effective_style, cfg)

    # ── Warning: Deepgram Aura voices are English-only ───────────────────────
    # Sending Arabic text to an aura-* voice produces garbled audio because
    # these models have no Arabic phoneme training. We log a warning but still
    # proceed — the admin controls the provider choice, not the pipeline.
    if provider == "deepgram" and language == "ar" and _is_english_only_voice(voice):
        log.warning(
            "TTS warning: Deepgram voice %r is English-only. Arabic text will sound garbled. "
            "Switch TTS_PROVIDER_AR to 'elevenlabs' or 'cartesia' in Admin → Providers → Text-to-Speech.",
            voice,
        )

    # Provider-specific char budget — Deepgram is the strictest at ~1500 chars/req.
    # Gemini TTS gets a larger budget so the summary needs as few calls as
    # possible (consistent voice). The budget is LANGUAGE-AWARE: Arabic tokenises
    # much heavier than English, so it uses a smaller char budget to stay under
    # Gemini's ~8,192 input-token limit and avoid truncated audio.
    if is_gemini_tts:
        if language == "ar":
            cfg_key, default_budget = "TTS_MAX_CHARS_GEMINI_AR", _GEMINI_MAX_CHARS_AR
        else:
            cfg_key, default_budget = "TTS_MAX_CHARS_GEMINI", _GEMINI_MAX_CHARS
        try:
            max_chars = int(cfg.get(cfg_key) or default_budget)
        except (TypeError, ValueError):
            max_chars = default_budget
        max_chars = max(max_chars, 1000)
    else:
        max_chars = _PROVIDER_MAX_CHARS.get(provider, _DEFAULT_MAX_CHARS)

    # Reserve room for the per-chunk continuity tag so a chunk + tag never
    # exceeds the provider's per-request character budget.
    split_budget = max_chars
    if is_gemini_tts and chunk_prefix:
        split_budget = max(max_chars - len(chunk_prefix), 1000)

    chunks = _split_text_for_tts(styled_text, max_chars=split_budget)
    if not chunks:
        raise ValueError("synthesize() called with empty text")

    # Defensive guard — no chunk should ever exceed the provider's budget.
    oversize = [(i, len(c)) for i, c in enumerate(chunks) if len(c) > max_chars]
    if oversize:
        raise RuntimeError(
            f"TTS chunker produced oversize chunks for provider={provider} "
            f"(limit={max_chars}): {oversize[:3]}…"
        )

    log.info(
        "TTS: provider=%s style=%s multi=%s text=%d chars → %d chunk(s) (limit %d/chunk)",
        provider, effective_style, is_multi_style, len(styled_text), len(chunks), max_chars,
    )

    # ── Fast path: single chunk fits in one call ─────────────────────────────
    if len(chunks) == 1:
        await _dispatch_tts(chunks[0], provider, voice, language, cfg, output_path, audio_style=effective_style)
        await log_tts_usage(provider=provider, model=voice, characters=len(styled_text))
        return output_path

    # ── Slow path: synthesize each chunk to a part file, then concat ─────────
    part_paths: list[str] = []
    try:
        for i, chunk in enumerate(chunks):
            # Prepend a short continuity tag to every chunk after the first.
            # The first chunk already contains the full style prompt / intro;
            # subsequent chunks only contain content, so the tag reminds the
            # model to keep the same voice/tone across boundaries.
            if i > 0 and chunk_prefix:
                chunk = chunk_prefix + chunk
            part = f"{output_path}.part{i:03d}.mp3"
            await _dispatch_tts(chunk, provider, voice, language, cfg, part, audio_style=effective_style)
            part_paths.append(part)

        with open(output_path, "wb") as dst:
            for p in part_paths:
                with open(p, "rb") as src:
                    dst.write(src.read())
    finally:
        for p in part_paths:
            try:
                os.remove(p)
            except OSError:
                pass

    await log_tts_usage(provider=provider, model=voice, characters=len(styled_text))
    return output_path


async def _dispatch_tts(
    text: str,
    provider: str,
    voice: str,
    language: str,
    cfg: dict,
    output_path: str,
    audio_style: str | None = None,
) -> None:
    """Route a single (chunk-sized) TTS request to the chosen provider."""
    if provider == "deepgram":
        if not settings.DEEPGRAM_API_KEY:
            raise ValueError(
                "DEEPGRAM_API_KEY is not set. Add it to your .env file "
                "or switch TTS provider in the Admin panel."
            )
        await _deepgram(text, voice, output_path)
    elif provider == "elevenlabs":
        if not settings.ELEVENLABS_API_KEY:
            raise ValueError(
                "ELEVENLABS_API_KEY is not set. Add it to your .env file "
                "or switch TTS provider in the Admin panel."
            )
        await _elevenlabs(text, language, cfg, output_path)
    elif provider == "cartesia":
        if not settings.CARTESIA_API_KEY:
            raise ValueError(
                "CARTESIA_API_KEY is not set. Add it to your .env file "
                "or switch the TTS provider to 'elevenlabs' in Admin → Providers → Text-to-Speech."
            )
        # If no Cartesia voice is configured for this language, fall back to
        # OpenRouter TTS when an OpenRouter key is available.  This prevents a
        # missing AR voice from failing the whole translated-audio step.
        _cartesia_voice = (
            cfg.get(f"CARTESIA_VOICE_{language.upper()}")
            or voice
            or cfg.get("CARTESIA_VOICE_EN")
            or settings.CARTESIA_VOICE_EN
            or settings.CARTESIA_VOICE_AR
        )
        if not _is_uuid(_cartesia_voice) and get_openrouter_key():
            log.warning(
                "Cartesia voice missing for language %r (value=%r); falling back to OpenRouter TTS.",
                language, _cartesia_voice,
            )
            or_model = cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL
            lang_voice_key = f"OPENROUTER_TTS_VOICE_{language.upper()}"
            or_voice = (
                cfg.get(lang_voice_key)
                or cfg.get("OPENROUTER_TTS_VOICE")
                or settings.OPENROUTER_TTS_VOICE
            )
            await _openrouter_tts(text, or_voice, language, or_model, output_path, cfg=cfg, audio_style=audio_style)
        else:
            await _cartesia(text, voice, language, cfg, output_path)
    elif provider == "gemini":
        if not (settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY):
            raise ValueError(
                "GEMINI_API_KEY (or OPENROUTER_API_KEY as fallback) is not set. "
                "Gemini TTS uses the native Google API — get a key at https://aistudio.google.com/app/apikey"
            )
        gemini_model = cfg.get("GEMINI_TTS_MODEL") or settings.GEMINI_TTS_MODEL
        gemini_voice = cfg.get("GEMINI_TTS_VOICE") or settings.GEMINI_TTS_VOICE
        # Always use the Gemini-specific voice — ignore the generic TTS_VOICE_*
        # which may hold a Cartesia/ElevenLabs UUID incompatible with Gemini.
        await _gemini(text, gemini_voice, language, gemini_model, output_path, cfg=cfg, audio_style=audio_style)
    elif provider == "openrouter":
        if not get_openrouter_key():
            raise ValueError(
                "No OpenRouter API keys are configured. OpenRouter TTS requires "
                "OPENROUTER_API_KEY (plus optional OPENROUTER_API_KEY_2 / _3) in your .env file. "
                "Get one at https://openrouter.ai/keys."
            )
        or_model = cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL
        # Per-language voice (OPENROUTER_TTS_VOICE_EN / _AR) takes priority,
        # falling back to the shared OPENROUTER_TTS_VOICE, then the settings default.
        lang_voice_key = f"OPENROUTER_TTS_VOICE_{language.upper()}"
        or_voice = (
            cfg.get(lang_voice_key)
            or cfg.get("OPENROUTER_TTS_VOICE")
            or settings.OPENROUTER_TTS_VOICE
        )
        await _openrouter_tts(text, or_voice, language, or_model, output_path, cfg=cfg, audio_style=audio_style)
    else:
        raise ValueError(f"Unknown TTS provider: {provider!r}")


# ── Provider implementations ─────────────────────────────────────────────────

async def _deepgram(text: str, voice: str, output_path: str) -> None:
    """Deepgram TTS with retry logic for timeout errors."""
    import logging
    log = logging.getLogger(__name__)
    
    max_retries = 3
    base_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f"https://api.deepgram.com/v1/speak?model={voice}",
                    headers={
                        "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"text": text},
                )
                r.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(r.content)
            return  # Success - exit function
        except httpx.HTTPStatusError as e:
            # Check for 408 timeout or 429 rate limit
            if e.response.status_code in (408, 429) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff: 2, 4, 8 seconds
                log.warning(f"Deepgram TTS attempt {attempt + 1}/{max_retries} failed with {e.response.status_code}, retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            raise  # Re-raise if not retryable or last attempt
        except httpx.TimeoutException as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.warning(f"Deepgram TTS attempt {attempt + 1}/{max_retries} timed out, retrying in {delay}s...")
                await asyncio.sleep(delay)
                continue
            raise  # Re-raise on last attempt


async def _elevenlabs(text: str, language: str, cfg: dict, output_path: str) -> None:
    # Prefer the admin runtime config (set in the dashboard), fall back to .env.
    if language == "en":
        voice_id = cfg.get("ELEVENLABS_VOICE_EN") or settings.ELEVENLABS_VOICE_EN
    else:
        voice_id = cfg.get("ELEVENLABS_VOICE_AR") or settings.ELEVENLABS_VOICE_AR
    if not voice_id:
        raise ValueError(
            f"ELEVENLABS_VOICE_{'EN' if language == 'en' else 'AR'} is not set. "
            "Set it in Admin → Providers → Text-to-Speech, or add it to your .env "
            "file. Find voice IDs at https://elevenlabs.io/voice-library"
        )
    model_id = cfg.get("ELEVENLABS_MODEL") or "eleven_multilingual_v2"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": settings.ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": model_id,   # eleven_multilingual_v2 supports Arabic
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)


async def _cartesia(text: str, voice: str, language: str, cfg: dict, output_path: str) -> None:
    """
    Call Cartesia Sonic TTS.

    The `language` parameter is REQUIRED — without it Cartesia 400s when the
    voice supports a different language than the transcript. Use ISO codes:
    en, ar, fr, de, es, pt, zh, ja, hi, it, ko, nl, pl, ru, sv, tr.
    """
    model = cfg.get("CARTESIA_MODEL") or settings.CARTESIA_MODEL

    # Prefer the dedicated Cartesia voice id for this language, then fall back
    # to the generic TTS_VOICE_* value, then to a configured English voice (many
    # Cartesia voices are multilingual and can handle Arabic/others), then to
    # the settings default.
    _vkey = f"CARTESIA_VOICE_{language.upper()}"
    voice = (
        cfg.get(_vkey)
        or voice
        or cfg.get("CARTESIA_VOICE_EN")
        or settings.CARTESIA_VOICE_EN
        or settings.CARTESIA_VOICE_AR
    )

    # Guard: voice must be a UUID, not a model name.
    # "sonic-2024-10-19" is the MODEL id — passing it as a voice id returns 400.
    if not _is_uuid(voice):
        raise ValueError(
            f"Cartesia voice value '{voice}' is not a valid voice UUID for language '{language}'. "
            f"Voice IDs look like 'a0e99841-438c-4a64-b679-ae501e7d6091'. "
            f"Find a voice at https://play.cartesia.ai/voices, then set "
            f"{_vkey} (or CARTESIA_VOICE_EN as a multilingual fallback) in "
            f"Admin → Providers → Text-to-Speech."
        )

    payload = {
        "model_id":   model,
        "transcript": text,
        "voice":      {"mode": "id", "id": voice},
        "language":   language,          # ← required by Cartesia for proper pronunciation
        "output_format": {
            "container":   "mp3",
            "encoding":    "mp3",
            "sample_rate": 44100,
        },
    }

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            "https://api.cartesia.ai/tts/bytes",
            headers={
                "X-API-Key":        settings.CARTESIA_API_KEY,
                "Cartesia-Version": "2024-06-10",
                "Content-Type":     "application/json",
            },
            json=payload,
        )
        # Surface Cartesia's actual error body so the admin sees WHY it failed
        # (e.g. "voice does not support language ar", "model deprecated", etc.)
        if r.status_code >= 400:
            raise RuntimeError(
                f"Cartesia returned {r.status_code}: {r.text[:500]} "
                f"(model={model}, language={language}, voice={voice})"
            )
    with open(output_path, "wb") as f:
        f.write(r.content)


async def _gemini(
    text: str,
    voice: str,
    language: str,
    model: str,
    output_path: str,
    cfg: dict | None = None,
    audio_style: str | None = None,
) -> None:
    """
    Google Gemini TTS via the native Gemini API (generativelanguage.googleapis.com).

    Uses the Gemini 2.5 Flash native speech-generation endpoint.
    Gemini natively supports Arabic + 30+ other languages.

    Voices: Kore, Charon, Puck, Fenrir, Aoede, Leda, Orus, Zephyr (and more).
    See https://ai.google.dev/gemini-api/docs/speech-generation for the full list.

    Supports single-speaker styles (audiobook, news, bedtime, custom prompt) and
    multi-speaker styles (multi, podcast) via multiSpeakerVoiceConfig.

    Requires GEMINI_API_KEY to be set. Falls back to OPENROUTER_API_KEY if
    GEMINI_API_KEY is not available (for backward compatibility).
    """
    cfg = cfg or {}
    chosen_voice = _gemini_voice_name(voice, "Kore")

    # Use GEMINI_API_KEY if available, otherwise fall back to OPENROUTER_API_KEY
    api_key = settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY (or OPENROUTER_API_KEY as fallback) is not set. "
            "Get a Gemini API key at https://aistudio.google.com/app/apikey"
        )

    # Normalize model name — strip 'google/' prefix if present
    gemini_model = model.split("/", 1)[1] if "/" in model else model

    # Build speechConfig: single-speaker (prebuiltVoiceConfig) or
    # multi-speaker (multiSpeakerVoiceConfig) depending on the style.
    style = (audio_style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
    is_multi = style in ("multi", "podcast")
    if is_multi:
        s1_name, s2_name = (
            (cfg.get("GEMINI_TTS_SPEAKER1_NAME") or "Speaker1").strip(),
            (cfg.get("GEMINI_TTS_SPEAKER2_NAME") or "Speaker2").strip(),
        )
        s1_voice, s2_voice = _resolve_speaker_voices(cfg, language)
        speech_config = {
            "multiSpeakerVoiceConfig": {
                "speakerVoiceConfigs": [
                    {
                        "speaker": s1_name,
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": s1_voice}
                        },
                    },
                    {
                        "speaker": s2_name,
                        "voiceConfig": {
                            "prebuiltVoiceConfig": {"voiceName": s2_voice}
                        },
                    },
                ]
            }
        }
    else:
        speech_config = {
            "voiceConfig": {
                "prebuiltVoiceConfig": {"voiceName": chosen_voice}
            }
        }

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": text},
                ]
            }
        ],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": speech_config,
        },
    }

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent",
            headers={"Content-Type": "application/json"},
            params={"key": api_key},
            json=payload,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"Gemini TTS returned {r.status_code}: {r.text[:500]} "
                f"(model={gemini_model}, style={style}, voice={chosen_voice}, language={language})"
            )
        data = r.json()

    # Extract base64-encoded audio from Gemini native response shape:
    #   candidates[0].content.parts[0].inlineData.data
    audio_b64:   str | None = None
    audio_mime:  str | None = None
    try:
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            inline_data = part.get("inlineData")
            if inline_data and inline_data.get("mimeType", "").startswith("audio/"):
                audio_b64  = inline_data.get("data")
                audio_mime = inline_data.get("mimeType", "audio/mp3")
                break
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Gemini TTS returned an unexpected response shape: {data!r}"
        ) from exc

    if not audio_b64:
        raise RuntimeError(
            "Gemini TTS response did not contain audio data. "
            f"Make sure {gemini_model!r} supports audio output. "
            f"Response excerpt: {str(data)[:400]}"
        )

    raw_bytes = base64.b64decode(audio_b64)

    # Gemini native API often returns audio/pcm (raw L16 PCM at 24kHz mono).
    # Saving raw PCM as .mp3 makes ffmpeg mis-detect the format, causing the
    # "time_base 1/0" crash. Detect this and transcode to MP3 via ffmpeg so
    # the caller always receives a valid MP3 regardless of Gemini's output format.
    _is_pcm = (audio_mime or "").lower() in ("audio/pcm", "audio/l16", "audio/raw")
    if _is_pcm:
        import shutil, subprocess, tempfile
        tmp_pcm = output_path + ".pcm"
        try:
            with open(tmp_pcm, "wb") as f:
                f.write(raw_bytes)
            # PCM: signed 16-bit little-endian at 24 000 Hz mono (Gemini default)
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-f", "s16le", "-ar", "24000", "-ac", "1",
                    "-i", tmp_pcm,
                    "-ar", "44100", "-ac", "2", "-b:a", "128k",
                    output_path,
                ],
                check=True, capture_output=True,
            )
        finally:
            if os.path.exists(tmp_pcm):
                os.remove(tmp_pcm)
    else:
        with open(output_path, "wb") as f:
            f.write(raw_bytes)


# Valid OpenAI audio voices for gpt-audio / gpt-audio-mini
_OPENAI_AUDIO_VOICES: set[str] = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar",
}


def _transcode_to_mp3(input_path: str, output_path: str) -> None:
    """Transcode any ffmpeg-readable audio file to MP3 stereo 44100 Hz."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not installed. Cannot transcode non-MP3 TTS output."
        )

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", input_path,
            "-ar", "44100", "-ac", "2", "-b:a", "128k",
            output_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to transcode OpenRouter audio (exit {result.returncode}). "
            f"stderr: {result.stderr.decode('utf-8', errors='ignore')[:800]}"
        )


def _transcode_pcm_to_mp3(input_path: str, output_path: str) -> None:
    """Transcode raw signed 16-bit little-endian PCM (24 kHz mono) to MP3."""
    import shutil
    import subprocess

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is not installed. Cannot transcode PCM TTS output."
        )

    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", input_path,
            "-ar", "44100", "-ac", "2", "-b:a", "128k",
            output_path,
        ],
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed to transcode PCM audio (exit {result.returncode}). "
            f"stderr: {result.stderr.decode('utf-8', errors='ignore')[:800]}"
        )


def _is_mp3(data: bytes) -> bool:
    """Detect MP3 by magic bytes (ID3v2 or MPEG sync word)."""
    return data.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"))


def _detect_mime_type(data: bytes) -> str:
    """Best-effort MIME type from file magic bytes."""
    if data.startswith(b"ID3") or data.startswith((b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
        return "audio/mpeg"
    if data.startswith(b"RIFF"):
        return "audio/wav"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"fLaC"):
        return "audio/flac"
    if data.startswith(b"\x00\x00\x00 ") or data.startswith(b"ftyp"):
        return "audio/mp4"
    return "audio/mpeg"


async def _openrouter_tts(
    text: str,
    voice: str,
    language: str,
    model: str,
    output_path: str,
    cfg: dict | None = None,
    audio_style: str | None = None,
) -> None:
    """
    TTS via OpenRouter's /audio/speech endpoint.

    - OpenAI audio models (openai/gpt-audio…) accept OpenAI voice names
      (alloy, echo, …) and return MP3 by default.
    - Google Gemini TTS models (google/gemini-*-tts-*) accept native Gemini
      voice names (Kore, Aoede, …) and, when asked for response_format="pcm",
      return raw signed 16-bit little-endian PCM at 24 kHz mono.

    We always output a standard MP3 so the rest of the pipeline is format-agnostic.

    Rotates to the next configured OpenRouter key on credit/limit errors.
    """
    is_gemini = model.startswith("google/")
    chosen_voice = voice or ("Kore" if is_gemini else "alloy")

    # Correct mismatched voice names so switching OpenRouter TTS models doesn't
    # leave a stale voice from the previous vendor.
    if is_gemini and chosen_voice not in _OPENROUTER_GEMINI_VOICES:
        log.warning(
            "OpenRouter Gemini TTS got non-Gemini voice %r for model %s; "
            "falling back to 'Kore'.", chosen_voice, model,
        )
        chosen_voice = "Kore"
    elif not is_gemini and chosen_voice not in _OPENROUTER_OPENAI_VOICES:
        log.warning(
            "OpenRouter OpenAI-audio TTS got non-OpenAI voice %r for model %s; "
            "falling back to 'alloy'.", chosen_voice, model,
        )
        chosen_voice = "alloy"

    payload: dict = {
        "model": model,
        "input": text,
        "voice": chosen_voice,
    }
    if is_gemini:
        # Gemini models via OpenRouter return raw PCM; this matches the native
        # Gemini API and avoids broken/missing default encodings.
        payload["response_format"] = "pcm"

    last_error: Exception | None = None
    key_count = openrouter_key_count()
    # With only one key, retrying the same key on a credit error is pointless.
    max_attempts = max(1, key_count)

    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(max_attempts):
            key = get_openrouter_key()
            r = await client.post(
                "https://openrouter.ai/api/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type":  "application/json",
                    "HTTP-Referer":  "https://seeourbook.com",
                    "X-Title":       "SeeOurBook Summarizer",
                },
                json=payload,
            )

            if r.status_code < 400:
                break

            if is_credit_error(r.status_code, r.text):
                rotate_openrouter_key(key)
                log.warning(
                    "OpenRouter TTS credit/limit error (HTTP %s) on attempt %s; "
                    "rotating key and retrying.",
                    r.status_code,
                    attempt + 1,
                )
                last_error = RuntimeError(
                    f"OpenRouter TTS returned {r.status_code}: {r.text[:500]} "
                    f"(model={model}, voice={chosen_voice}, language={language})"
                )
                continue

            raise RuntimeError(
                f"OpenRouter TTS returned {r.status_code}: {r.text[:500]} "
                f"(model={model}, voice={chosen_voice}, language={language})"
            )
        else:
            raise last_error or RuntimeError(
                f"OpenRouter TTS failed on all configured keys "
                f"(model={model}, voice={chosen_voice}, language={language}). "
                f"Add OPENROUTER_API_KEY_2 / OPENROUTER_API_KEY_3 to your .env to enable automatic fallback."
            )

    content = r.content
    content_type = (r.headers.get("content-type") or "").lower()

    log.info(
        "OpenRouter TTS response: model=%s voice=%s lang=%s status=%s "
        "content-type=%s bytes=%d first_bytes=%r",
        model, chosen_voice, language, r.status_code, content_type, len(content),
        content[:40],
    )

    # Errors sometimes come back as 200 with a JSON/HTML body — surface them clearly.
    if not content or len(content) < 100:
        raise RuntimeError(
            f"OpenRouter TTS returned no audio (model={model}, voice={chosen_voice}, "
            f"language={language}, content-type={content_type}). Body: {content[:500]!r}"
        )

    import tempfile

    # Gemini models return raw PCM that must be wrapped/transcoded to MP3.
    if is_gemini:
        with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            with open(tmp_path, "wb") as f:
                f.write(content)
            _transcode_pcm_to_mp3(tmp_path, output_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return

    # OpenAI audio models return MP3 by default.
    if _is_mp3(content):
        with open(output_path, "wb") as f:
            f.write(content)
        return

    # Fallback: try to transcode whatever format was returned.
    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        try:
            _transcode_to_mp3(tmp_path, output_path)
        except RuntimeError as exc:
            raise RuntimeError(
                f"{exc}. Input first bytes: {content[:80]!r}"
            ) from exc
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
