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
        await _elevenlabs(text, language, output_path)
    elif provider == "cartesia":
        if not settings.CARTESIA_API_KEY:
            raise ValueError(
                "CARTESIA_API_KEY is not set. Add it to your .env file "
                "or switch the TTS provider to 'elevenlabs' in Admin → Providers → Text-to-Speech."
            )
        await _cartesia(text, voice, language, cfg, output_path)
    elif provider == "gemini":
        if not settings.OPENROUTER_API_KEY:
            raise ValueError(
                "OPENROUTER_API_KEY is not set. Gemini TTS routes through OpenRouter — "
                "add OPENROUTER_API_KEY to your .env file. Get one at https://openrouter.ai/keys."
            )
        gemini_model = cfg.get("GEMINI_TTS_MODEL") or settings.GEMINI_TTS_MODEL
        await _gemini(text, voice, language, gemini_model, output_path)
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


async def _elevenlabs(text: str, language: str, output_path: str) -> None:
    voice_id = settings.ELEVENLABS_VOICE_EN if language == "en" else settings.ELEVENLABS_VOICE_AR
    if not voice_id:
        raise ValueError(
            f"ELEVENLABS_VOICE_{'EN' if language == 'en' else 'AR'} is not set. "
            "Add it to your .env file — find voice IDs at https://elevenlabs.io/voice-library"
        )
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={
                "xi-api-key": settings.ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={
                "text": text,
                "model_id": "eleven_multilingual_v2",   # supports Arabic natively
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

    # Guard: voice must be a UUID, not a model name.
    # "sonic-2024-10-19" is the MODEL id — passing it as a voice id returns 400.
    import re
    _UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)
    if not _UUID_RE.match(voice):
        raise ValueError(
            f"Cartesia TTS_VOICE value '{voice}' is not a valid voice UUID. "
            f"Voice IDs look like 'a0e99841-438c-4a64-b679-ae501e7d6091'. "
            f"Find yours at https://play.cartesia.ai/voices, then update "
            f"TTS_VOICE_AR in Admin → Providers → Text-to-Speech."
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
    Google Gemini Flash 2.5 TTS via OpenRouter.

    Uses the OpenAI-compatible chat/completions endpoint with audio modality.
    Gemini natively supports Arabic + 30+ other languages.

    Voices: Kore, Charon, Puck, Fenrir, Aoede, Leda, Orus, Zephyr (and more).
    See https://ai.google.dev/gemini-api/docs/speech-generation for the full list.

    The model parameter should be a Gemini TTS-capable variant, e.g.
    'google/gemini-2.5-flash-preview-tts' (default).
    """
    chosen_voice = voice or "Kore"   # Kore is the default Gemini voice

    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": text}],
        "modalities": ["audio"],
        "audio":    {"voice": chosen_voice, "format": "mp3"},
    }

    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization":  f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type":   "application/json",
                "HTTP-Referer":   "https://seeourbook.com",
                "X-Title":        "SeeOurBook Summarizer",
            },
            json=payload,
        )
        if r.status_code >= 400:
            raise RuntimeError(
                f"Gemini TTS via OpenRouter returned {r.status_code}: {r.text[:500]} "
                f"(model={model}, voice={chosen_voice}, language={language})"
            )
        data = r.json()

    # Extract base64-encoded audio. Standard OpenAI-compatible shape:
    #   choices[0].message.audio.data
    audio_b64: str | None = None
    try:
        msg = data["choices"][0]["message"]
        audio_b64 = msg.get("audio", {}).get("data")
        if not audio_b64:
            # Fallback: some providers return audio inside content blocks
            content = msg.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "audio":
                        audio_b64 = block.get("data") or block.get("audio", {}).get("data")
                        if audio_b64:
                            break
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            f"Gemini TTS returned an unexpected response shape: {data!r}"
        ) from exc

    if not audio_b64:
        raise RuntimeError(
            "Gemini TTS response did not contain audio data. "
            f"Make sure {model!r} supports audio output on OpenRouter. "
            f"Response excerpt: {str(data)[:400]}"
        )

    with open(output_path, "wb") as f:
        f.write(base64.b64decode(audio_b64))
