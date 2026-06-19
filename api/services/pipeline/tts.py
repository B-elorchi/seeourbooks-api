
"""
Text-to-Speech service.
Supports Deepgram, ElevenLabs, Cartesia, and Google Gemini (via OpenRouter).
Provider and voice are resolved from a runtime cfg dict (from admin panel) with
fallback to settings.py defaults — no restart needed when the admin switches providers.

Long-text handling
──────────────────
All providers have a per-call character limit. A 10-minute book summary is ~8–10K
chars and would 4xx in a single request. synthesize() splits long text on sentence
boundaries, synthesizes each chunk separately, and concatenates the resulting MP3
files using ffmpeg concat demuxer for proper merging. The downstream audio-
processing step re-encodes to clean up any edge cases.

CRITICAL: Gemini TTS (native AND via OpenRouter) has a hard ~5-minute audio output
limit per request regardless of input length. Input can be 32K tokens, but the
generated audio silently truncates at ~5 minutes. We therefore keep chunk sizes
small enough that each chunk's spoken audio fits within this limit.
  • English ~130 WPM → 5 min ≈ 650 words ≈ ~3,800 chars (safe)
  • Arabic  ~100 WPM → 5 min ≈ 500 words ≈ ~3,200 chars (safe)

See: https://github.com/googleapis/python-genai/issues/922

Arabic TTS providers (by recommendation order)
──────────────────────────────────────────────
  cartesia   — set CARTESIA_API_KEY + TTS_VOICE_AR=UUID, CARTESIA_MODEL=sonic-3.5-*
  gemini     — set OPENROUTER_API_KEY (uses google/gemini-2.5-flash-preview-tts)
  elevenlabs — set ELEVENLABS_API_KEY + ELEVENLABS_VOICE_AR (multilingual v2)
  deepgram   — ENGLISH ONLY, do not use for Arabic
"""
import asyncio
import base64
import logging
import os
import re
import shutil
import subprocess
import tempfile
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
_DEEPGRAM_ENGLISH_VOICE_PREFIXES = ("aura-",)

# OpenRouter TTS voice catalogs.
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

# Per-provider character budget for a single TTS request.
_PROVIDER_MAX_CHARS: dict[str, int] = {
    "deepgram":   1500,
    "elevenlabs": 2500,
    "cartesia":   2500,
    "gemini":     2500,
    "openrouter": 2500,
}
_DEFAULT_MAX_CHARS = 1500

# ═══════════════════════════════════════════════════════════════════════════════
# GEMINI TTS CHUNK BUDGETS  (critical: ~5 min audio output hard limit)
# ═══════════════════════════════════════════════════════════════════════════════
# Gemini TTS (native AND OpenRouter) truncates audio at ~5 minutes per request.
# These budgets ensure each chunk's spoken audio fits within that limit.
#
# English: ~130 WPM, 5 min = ~650 words ≈ ~3,800 chars (conservative)
# Arabic:  ~100 WPM, 5 min = ~500 words ≈ ~3,200 chars (conservative,
#           tashkeel makes tokenisation heavy but does not affect speaking time)
#
# Override per deployment via TTS_MAX_CHARS_GEMINI / TTS_MAX_CHARS_GEMINI_AR.
# If audio is CUT SHORT → decrease. If you want fewer chunks → increase carefully.
# ═══════════════════════════════════════════════════════════════════════════════
_GEMINI_MAX_CHARS    = 3800   # English / Latin scripts
_GEMINI_MAX_CHARS_AR = 3200   # Arabic


def _is_english_only_voice(voice: str) -> bool:
    return any(voice.startswith(p) for p in _DEEPGRAM_ENGLISH_VOICE_PREFIXES)


_VALID_AUDIO_STYLES = {"single", "multi", "podcast", "audiobook", "news", "bedtime", "custom"}

# Continuity tags prepended to every TTS chunk after the first.
# The first chunk already contains the full style prompt; subsequent chunks need
# a strong reminder to maintain the SAME voice/tone/pacing.  Using a descriptive
# tag (not just a label) gives the model more context.
_STYLE_TAGS: dict[str, str] = {
    "single":    "",
    "audiobook": "narrator — same calm immersive voice, seamless continuation",
    "news":      "anchor — same professional broadcast voice, continuing",
    "bedtime":   "soothing — same gentle bedtime story voice, continuing",
    "custom":    "custom delivery — same voice and style as before, continuing",
    "multi":     "dialogue — same speakers and tone as before, continuing",
    "podcast":   "podcast — same hosts and energy as before, continuing",
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

    Returns (styled_text, use_multi_speaker, chunk_prefix).

    chunk_prefix is prepended to every TTS chunk *except* the first one.
    It reminds the model to keep the same delivery across chunk boundaries.
    """
    style = (style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
    if style not in _VALID_AUDIO_STYLES:
        style = "single"

    tag = _STYLE_TAGS.get(style, "").strip()
    chunk_prefix = f"[{tag}]\n\n" if tag else ""

    if style == "custom":
        prompt = (cfg.get("GEMINI_TTS_STYLE_PROMPT") or "").strip()
        return (f"{prompt}\n\n{text}" if prompt else text), False, chunk_prefix

    if style in ("multi", "podcast"):
        s1 = (cfg.get("GEMINI_TTS_SPEAKER1_NAME") or "Speaker1").strip()
        s2 = (cfg.get("GEMINI_TTS_SPEAKER2_NAME") or "Speaker2").strip()
        intro = (
            f"TTS the following podcast episode between {s1} and {s2}.\n\n"
            if style == "podcast"
            else f"TTS the following conversation between {s1} and {s2}.\n\n"
        )
        sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
        if not sentences:
            return text, True, chunk_prefix
        lines = []
        for i, sent in enumerate(sentences):
            speaker = s1 if i % 2 == 0 else s2
            lines.append(f"{speaker}: {sent}")
        return intro + "\n".join(lines), True, chunk_prefix

    prompts = {
        "audiobook": "Read the following text in a calm, immersive audiobook narration style.",
        "news":      "Read the following text as a professional news broadcast.",
        "bedtime":   "Read the following text in a soothing, gentle bedtime story style.",
    }
    prompt = prompts.get(style, "")
    custom_prompt = (cfg.get("GEMINI_TTS_STYLE_PROMPT") or "").strip()
    if custom_prompt:
        prompt = custom_prompt
    return (f"{prompt}\n\n{text}" if prompt else text), False, chunk_prefix


# Sentence-ending punctuation in EN / AR / general.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟…])\s+")


def _split_text_for_tts(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
    """
    Split text into TTS-sized chunks on sentence boundaries.
    Guarantees no chunk exceeds `max_chars`.
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

        candidate = (buf + " " + sent).strip() if buf else sent
        if len(candidate) > max_chars:
            flush()
            buf = sent
        else:
            buf = candidate

    flush()
    return chunks


def _concat_audio_ffmpeg(part_paths: list[str], output_path: str) -> None:
    """
    Properly concatenate MP3/audio files using ffmpeg concat demuxer.

    Raw byte concatenation of MP3 files creates malformed output with multiple
    ID3 headers, causing browser audio players to stop at the first header
    boundary and play only the first chunk. This function uses ffmpeg's concat
    demuxer with re-encoding to produce a single clean, valid MP3 file.
    """
    if not part_paths:
        raise ValueError("No audio parts to concatenate")

    if len(part_paths) == 1:
        shutil.copy2(part_paths[0], output_path)
        return

    if not shutil.which("ffmpeg"):
        raise RuntimeError(
            "ffmpeg is required for audio concatenation but is not installed."
        )

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list_path = f.name
        for p in part_paths:
            escaped = p.replace("'", "'\''")
            f.write(f"file '{escaped}'\n")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-ac", "2", "-ar", "44100", "-b:a", "128k",
                "-write_xing", "0",
                output_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            stderr = result.stderr.decode("utf-8", errors="ignore")[:800]
            raise RuntimeError(
                f"ffmpeg concat failed (exit {result.returncode}). stderr: {stderr}"
            )
    finally:
        try:
            os.unlink(concat_list_path)
        except OSError:
            pass


async def synthesize(
    text: str,
    language: str,
    output_path: str,
    cfg: dict | None = None,
    audio_style: str | None = None,
) -> str:
    """
    Convert text to MP3 and save to output_path. Returns output_path.

    Text longer than the provider's per-call limit is split on sentence
    boundaries, each chunk is synthesized individually, and the resulting MP3
    files are concatenated using ffmpeg concat demuxer for proper merging.
    """
    cfg = cfg or {}
    lang = language.upper()

    # ── Spoken audio watermark ───────────────────────────────────────────────
    watermark = (cfg.get(f"AUDIO_WATERMARK_TEXT_{lang}") or "").strip()
    if watermark:
        text = f"{watermark}.\n\n{text}"

    provider = cfg.get(f"TTS_PROVIDER_{lang}") or (
        settings.TTS_PROVIDER_EN if language == "en" else settings.TTS_PROVIDER_AR
    )
    voice = cfg.get(f"TTS_VOICE_{lang}") or (
        settings.TTS_VOICE_EN if language == "en" else settings.TTS_VOICE_AR
    )

    # ── Gemini TTS style / profile ───────────────────────────────────────────
    is_gemini_tts = provider == "gemini" or (
        provider == "openrouter"
        and (cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL).lower().startswith("google/")
    )
    effective_style = "single"
    if is_gemini_tts:
        effective_style = (audio_style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
    styled_text, is_multi_style, chunk_prefix = _apply_audio_style(text, effective_style, cfg)

    # ── Deepgram English-only warning ────────────────────────────────────────
    if provider == "deepgram" and language == "ar" and _is_english_only_voice(voice):
        log.warning(
            "TTS warning: Deepgram voice %r is English-only. Arabic text will sound garbled. "
            "Switch TTS_PROVIDER_AR to 'elevenlabs' or 'cartesia' in Admin → Providers → Text-to-Speech.",
            voice,
        )

    # ── Resolve per-provider character budget ────────────────────────────────
    if is_gemini_tts:
        # Gemini TTS has a hard ~5-minute audio output limit per request.
        # The budget must keep each chunk's spoken audio under that limit.
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

    # Reserve room for the per-chunk continuity tag.
    split_budget = max_chars
    if is_gemini_tts and chunk_prefix:
        split_budget = max(max_chars - len(chunk_prefix), 1000)

    chunks = _split_text_for_tts(styled_text, max_chars=split_budget)
    if not chunks:
        raise ValueError("synthesize() called with empty text")

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

    # ── Fast path: single chunk ──────────────────────────────────────────────
    if len(chunks) == 1:
        await _dispatch_tts(
            chunks[0], provider, voice, language, cfg, output_path,
            audio_style=effective_style,
        )
        await log_tts_usage(provider=provider, model=voice, characters=len(styled_text))
        return output_path

    # ── Slow path: synthesize each chunk, then concat with ffmpeg ────────────
    part_paths: list[str] = []
    try:
        for i, chunk in enumerate(chunks):
            if i > 0 and chunk_prefix:
                chunk = chunk_prefix + chunk
            part = f"{output_path}.part{i:03d}.mp3"
            await _dispatch_tts(
                chunk, provider, voice, language, cfg, part,
                audio_style=effective_style,
            )
            part_paths.append(part)

        _concat_audio_ffmpeg(part_paths, output_path)
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
                "Get a Gemini API key at https://aistudio.google.com/app/apikey"
            )
        gemini_model = cfg.get("GEMINI_TTS_MODEL") or settings.GEMINI_TTS_MODEL
        gemini_voice = cfg.get("GEMINI_TTS_VOICE") or settings.GEMINI_TTS_VOICE
        await _gemini(text, gemini_voice, language, gemini_model, output_path, cfg=cfg, audio_style=audio_style)
    elif provider == "openrouter":
        if not get_openrouter_key():
            raise ValueError(
                "No OpenRouter API keys are configured. OpenRouter TTS requires "
                "OPENROUTER_API_KEY (plus optional OPENROUTER_API_KEY_2 / _3). "
                "Get one at https://openrouter.ai/keys."
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
        raise ValueError(f"Unknown TTS provider: {provider!r}")


# ── Provider implementations ─────────────────────────────────────────────────

async def _deepgram(text: str, voice: str, output_path: str) -> None:
    """Deepgram TTS with retry logic for timeout/rate-limit errors."""
    max_retries = 3
    base_delay = 2

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
            return
        except httpx.HTTPStatusError as e:
            if e.response.status_code in (408, 429) and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.warning(
                    "Deepgram TTS attempt %d/%d failed with %d, retrying in %ds...",
                    attempt + 1, max_retries, e.response.status_code, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.warning(
                    "Deepgram TTS attempt %d/%d timed out, retrying in %ds...",
                    attempt + 1, max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue
            raise


async def _elevenlabs(text: str, language: str, cfg: dict, output_path: str) -> None:
    if language == "en":
        voice_id = cfg.get("ELEVENLABS_VOICE_EN") or settings.ELEVENLABS_VOICE_EN
    else:
        voice_id = cfg.get("ELEVENLABS_VOICE_AR") or settings.ELEVENLABS_VOICE_AR
    if not voice_id:
        raise ValueError(
            f"ELEVENLABS_VOICE_{'EN' if language == 'en' else 'AR'} is not set. "
            "Set it in Admin → Providers → Text-to-Speech, or add it to your .env."
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
                "model_id": model_id,
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
        )
        r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)


async def _cartesia(text: str, voice: str, language: str, cfg: dict, output_path: str) -> None:
    """Call Cartesia Sonic TTS."""
    model = cfg.get("CARTESIA_MODEL") or settings.CARTESIA_MODEL

    _vkey = f"CARTESIA_VOICE_{language.upper()}"
    voice = (
        cfg.get(_vkey)
        or voice
        or cfg.get("CARTESIA_VOICE_EN")
        or settings.CARTESIA_VOICE_EN
        or settings.CARTESIA_VOICE_AR
    )

    if not _is_uuid(voice):
        raise ValueError(
            f"Cartesia voice value '{voice}' is not a valid voice UUID for language '{language}'. "
            f"Voice IDs look like 'a0e99841-438c-4a64-b679-ae501e7d6091'. "
            f"Find a voice at https://play.cartesia.ai/voices, then set "
            f"{_vkey} (or CARTESIA_VOICE_EN as a multilingual fallback)."
        )

    payload = {
        "model_id":   model,
        "transcript": text,
        "voice":      {"mode": "id", "id": voice},
        "language":   language,
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
    Google Gemini TTS via the native Gemini API.

    Uses the Gemini 2.5 Flash native speech-generation endpoint.
    Gemini natively supports Arabic + 30+ other languages.

    CRITICAL: Gemini TTS has a hard ~5-minute audio output limit per request.
    Longer input text is silently truncated in the audio output. The chunking
    logic in synthesize() must keep each chunk under this limit.
    """
    cfg = cfg or {}
    chosen_voice = _gemini_voice_name(voice, "Kore")

    api_key = settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY (or OPENROUTER_API_KEY as fallback) is not set. "
            "Get a Gemini API key at https://aistudio.google.com/app/apikey"
        )

    gemini_model = model.split("/", 1)[1] if "/" in model else model

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
        "contents": [{"parts": [{"text": text}]}],
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

    _is_pcm = (audio_mime or "").lower() in ("audio/pcm", "audio/l16", "audio/raw")
    if _is_pcm:
        tmp_pcm = output_path + ".pcm"
        try:
            with open(tmp_pcm, "wb") as f:
                f.write(raw_bytes)
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
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed. Cannot transcode non-MP3 TTS output.")

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
            f"ffmpeg failed to transcode audio (exit {result.returncode}). "
            f"stderr: {result.stderr.decode('utf-8', errors='ignore')[:800]}"
        )


def _transcode_pcm_to_mp3(input_path: str, output_path: str) -> None:
    """Transcode raw signed 16-bit little-endian PCM (24 kHz mono) to MP3."""
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed. Cannot transcode PCM TTS output.")

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

    - OpenAI audio models accept OpenAI voice names and return MP3 by default.
    - Google Gemini TTS models accept native Gemini voice names and, when asked
      for response_format="pcm", return raw signed 16-bit little-endian PCM
      at 24 kHz mono.

    CRITICAL: Gemini TTS via OpenRouter has the same ~5-minute audio output
    hard limit as native Gemini. The chunking in synthesize() must keep each
    chunk small enough to fit.

    Rotates to the next configured OpenRouter key on credit/limit errors.
    """
    is_gemini = model.startswith("google/")
    chosen_voice = voice or ("Kore" if is_gemini else "alloy")

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
        payload["response_format"] = "pcm"

    last_error: Exception | None = None
    key_count = openrouter_key_count()
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
                    r.status_code, attempt + 1,
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
                f"Add OPENROUTER_API_KEY_2 / OPENROUTER_API_KEY_3 to your .env."
            )

    content = r.content
    content_type = (r.headers.get("content-type") or "").lower()

    log.info(
        "OpenRouter TTS response: model=%s voice=%s lang=%s status=%s "
        "content-type=%s bytes=%d first_bytes=%r",
        model, chosen_voice, language, r.status_code, content_type, len(content),
        content[:40],
    )

    if not content or len(content) < 100:
        raise RuntimeError(
            f"OpenRouter TTS returned no audio (model={model}, voice={chosen_voice}, "
            f"language={language}, content-type={content_type}). Body: {content[:500]!r}"
        )

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

    if _is_mp3(content):
        with open(output_path, "wb") as f:
            f.write(content)
        return

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