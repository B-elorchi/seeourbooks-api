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
from api.config.settings import settings
from api.services.usage_logger import log_tts_usage

log = logging.getLogger(__name__)

# Deepgram Aura model names that are English-only.
# Sending Arabic text to these produces garbled / unrecognisable audio.
_DEEPGRAM_ENGLISH_VOICE_PREFIXES = ("aura-",)

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


def _is_english_only_voice(voice: str) -> bool:
    return any(voice.startswith(p) for p in _DEEPGRAM_ENGLISH_VOICE_PREFIXES)


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

    provider = cfg.get(f"TTS_PROVIDER_{lang}") or (
        settings.TTS_PROVIDER_EN if language == "en" else settings.TTS_PROVIDER_AR
    )
    voice = cfg.get(f"TTS_VOICE_{lang}") or (
        settings.TTS_VOICE_EN if language == "en" else settings.TTS_VOICE_AR
    )

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
    max_chars = _PROVIDER_MAX_CHARS.get(provider, _DEFAULT_MAX_CHARS)

    chunks = _split_text_for_tts(text, max_chars=max_chars)
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
        "TTS: provider=%s text=%d chars → %d chunk(s) (limit %d/chunk)",
        provider, len(text), len(chunks), max_chars,
    )

    # ── Fast path: single chunk fits in one call ─────────────────────────────
    if len(chunks) == 1:
        await _dispatch_tts(chunks[0], provider, voice, language, cfg, output_path)
        await log_tts_usage(provider=provider, model=voice, characters=len(text))
        return output_path

    # ── Slow path: synthesize each chunk to a part file, then concat ─────────
    part_paths: list[str] = []
    try:
        for i, chunk in enumerate(chunks):
            part = f"{output_path}.part{i:03d}.mp3"
            await _dispatch_tts(chunk, provider, voice, language, cfg, part)
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

    await log_tts_usage(provider=provider, model=voice, characters=len(text))
    return output_path


async def _dispatch_tts(
    text: str,
    provider: str,
    voice: str,
    language: str,
    cfg: dict,
    output_path: str,
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
        await _gemini(text, gemini_voice, language, gemini_model, output_path)
    elif provider == "openrouter":
        if not settings.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. OpenRouter TTS requires it — "
                "add OPENROUTER_API_KEY to your .env file. Get one at https://openrouter.ai/keys."
            )
        or_model = cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL
        or_voice = cfg.get("OPENROUTER_TTS_VOICE") or settings.OPENROUTER_TTS_VOICE
        # Always use the OpenRouter-specific voice setting — ignore the generic
        # TTS_VOICE_* which may hold a Cartesia/ElevenLabs UUID incompatible with OpenRouter.
        await _openrouter_tts(text, or_voice, language, or_model, output_path)
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

    # Prefer the dedicated Cartesia voice id for this language, then fall back to
    # the generic TTS_VOICE_* value passed in.
    _vkey = f"CARTESIA_VOICE_{language.upper()}"
    voice = cfg.get(_vkey) or voice

    # Guard: voice must be a UUID, not a model name.
    # "sonic-2024-10-19" is the MODEL id — passing it as a voice id returns 400.
    import re
    _UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
    if not _UUID_RE.match(voice or ""):
        raise ValueError(
            f"Cartesia voice value '{voice}' is not a valid voice UUID. "
            f"Voice IDs look like 'a0e99841-438c-4a64-b679-ae501e7d6091'. "
            f"Find yours at https://play.cartesia.ai/voices, then set "
            f"{_vkey} in Admin → Providers → Text-to-Speech."
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


async def _gemini(text: str, voice: str, language: str, model: str, output_path: str) -> None:
    """
    Google Gemini TTS via the native Gemini API (generativelanguage.googleapis.com).

    Uses the Gemini 2.5 Flash native speech-generation endpoint.
    Gemini natively supports Arabic + 30+ other languages.

    Voices: Kore, Charon, Puck, Fenrir, Aoede, Leda, Orus, Zephyr (and more).
    See https://ai.google.dev/gemini-api/docs/speech-generation for the full list.

    Requires GEMINI_API_KEY to be set. Falls back to OPENROUTER_API_KEY if
    GEMINI_API_KEY is not available (for backward compatibility).
    """
    chosen_voice = voice or "Kore"   # Kore is the default Gemini voice

    # Use GEMINI_API_KEY if available, otherwise fall back to OPENROUTER_API_KEY
    api_key = settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY (or OPENROUTER_API_KEY as fallback) is not set. "
            "Get a Gemini API key at https://aistudio.google.com/app/apikey"
        )

    # Normalize model name — strip 'google/' prefix if present
    gemini_model = model.split("/", 1)[1] if "/" in model else model

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
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {
                        "voiceName": chosen_voice,
                    }
                }
            },
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
                f"(model={gemini_model}, voice={chosen_voice}, language={language})"
            )
        data = r.json()

    # Extract base64-encoded audio from Gemini native response shape:
    #   candidates[0].content.parts[0].inlineData.data
    audio_b64: str | None = None
    try:
        parts = data["candidates"][0]["content"]["parts"]
        for part in parts:
            inline_data = part.get("inlineData")
            if inline_data and inline_data.get("mimeType", "").startswith("audio/"):
                audio_b64 = inline_data.get("data")
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

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))


# Valid OpenAI audio voices for gpt-audio / gpt-audio-mini
_OPENAI_AUDIO_VOICES: set[str] = {
    "alloy", "echo", "fable", "onyx", "nova", "shimmer",
    "coral", "verse", "ballad", "ash", "sage", "marin", "cedar",
}


async def _openrouter_tts(text: str, voice: str, language: str, model: str, output_path: str) -> None:
    """
    TTS via OpenRouter's dedicated speech endpoint (OpenAI-compatible):

        POST https://openrouter.ai/api/v1/audio/speech
        { "model": ..., "input": ..., "voice": ... }

    The response body is the raw audio file (mp3) — NOT JSON. This endpoint
    supports both OpenAI audio models (openai/gpt-audio…) AND Google Gemini TTS
    models (google/gemini-*-tts-*) with OpenAI voice names like "alloy".

    NOTE: the chat/completions endpoint with modalities=["audio","text"] does
    NOT work for these models — it returns "No endpoints found that support the
    requested output modalities", which is why we use /audio/speech here.
    """
    chosen_voice = voice or "alloy"

    payload = {
        "model": model,
        "input": text,
        "voice": chosen_voice,
    }

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/audio/speech",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type":  "application/json",
                "HTTP-Referer":  "https://seeourbook.com",
                "X-Title":       "SeeOurBook Summarizer",
            },
            json=payload,
        )

    if r.status_code >= 400:
        raise RuntimeError(
            f"OpenRouter TTS returned {r.status_code}: {r.text[:500]} "
            f"(model={model}, voice={chosen_voice}, language={language})"
        )

    content = r.content
    if not content or len(content) < 100:
        # Some errors come back 200 with a tiny JSON body — surface it clearly.
        raise RuntimeError(
            f"OpenRouter TTS returned no audio (model={model}, voice={chosen_voice}, "
            f"language={language}). Body: {content[:300]!r}"
        )

    with open(output_path, "wb") as f:
        f.write(content)
