
# """
# Text-to-Speech service.
# Supports Deepgram, ElevenLabs, Cartesia, and Google Gemini (via OpenRouter).
# Provider and voice are resolved from a runtime cfg dict (from admin panel) with
# fallback to settings.py defaults — no restart needed when the admin switches providers.

# Long-text handling
# ──────────────────
# All providers have a per-call character limit. A 10-minute book summary is ~8–10K
# chars and would 4xx in a single request. synthesize() splits long text on sentence
# boundaries, synthesizes each chunk separately, and concatenates the resulting MP3
# files using ffmpeg concat demuxer for proper merging. The downstream audio-
# processing step re-encodes to clean up any edge cases.

# CRITICAL: Gemini TTS (native AND via OpenRouter) has a hard ~5-minute audio output
# limit per request regardless of input length. Input can be 32K tokens, but the
# generated audio silently truncates at ~5 minutes. We therefore keep chunk sizes
# small enough that each chunk's spoken audio fits within this limit.
#   • English ~130 WPM → 5 min ≈ 650 words ≈ ~3,800 chars (safe)
#   • Arabic  ~100 WPM → 5 min ≈ 500 words ≈ ~3,200 chars (safe)

# See: https://github.com/googleapis/python-genai/issues/922

# Arabic TTS providers (by recommendation order)
# ──────────────────────────────────────────────
#   cartesia   — set CARTESIA_API_KEY + TTS_VOICE_AR=UUID, CARTESIA_MODEL=sonic-3.5-*
#   gemini     — set OPENROUTER_API_KEY (uses google/gemini-2.5-flash-preview-tts)
#   elevenlabs — set ELEVENLABS_API_KEY + ELEVENLABS_VOICE_AR (multilingual v2)
#   deepgram   — ENGLISH ONLY, do not use for Arabic
# """
# import asyncio
# import base64
# import logging
# import os
# import re
# import shutil
# import subprocess
# import tempfile
# import httpx

# _UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)

# def _is_uuid(v: str | None) -> bool:
#     return bool(v and _UUID_RE.match(v))

# from api.config.settings import settings
# from api.services.usage_logger import log_tts_usage
# from api.services.openrouter_keys import (
#     get_openrouter_key,
#     rotate_openrouter_key,
#     is_credit_error,
#     openrouter_key_count,
# )

# log = logging.getLogger(__name__)

# # Deepgram Aura model names that are English-only.
# _DEEPGRAM_ENGLISH_VOICE_PREFIXES = ("aura-",)

# # OpenRouter TTS voice catalogs.
# _OPENROUTER_GEMINI_VOICES = {
#     "Achernar", "Achird", "Algenib", "Algieba", "Alnilam", "Aoede", "Autonoe",
#     "Callirrhoe", "Charon", "Despina", "Enceladus", "Erinome", "Fenrir", "Gacrux",
#     "Iapetus", "Kore", "Laomedeia", "Leda", "Orus", "Puck", "Pulcherrima",
#     "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat", "Umbriel",
#     "Vindemiatrix", "Zephyr", "Zubenelgenubi",
# }
# _OPENROUTER_OPENAI_VOICES = {
#     "alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse",
#     "ballad", "ash", "sage", "marin", "cedar",
# }

# # Per-provider character budget for a single TTS request.
# _PROVIDER_MAX_CHARS: dict[str, int] = {
#     "deepgram":   1500,
#     "elevenlabs": 2500,
#     "cartesia":   2500,
#     "gemini":     2500,
#     "openrouter": 2500,
# }
# _DEFAULT_MAX_CHARS = 1500

# # ═══════════════════════════════════════════════════════════════════════════════
# # GEMINI TTS CHUNK BUDGETS  (critical: ~5 min audio output hard limit)
# # ═══════════════════════════════════════════════════════════════════════════════
# # Gemini TTS (native AND OpenRouter) truncates audio at ~5 minutes per request.
# # These budgets ensure each chunk's spoken audio fits within that limit.
# #
# # English: ~130 WPM, 5 min = ~650 words ≈ ~3,800 chars (conservative)
# # Arabic:  ~100 WPM, 5 min = ~500 words ≈ ~3,200 chars (conservative,
# #           tashkeel makes tokenisation heavy but does not affect speaking time)
# #
# # Override per deployment via TTS_MAX_CHARS_GEMINI / TTS_MAX_CHARS_GEMINI_AR.
# # If audio is CUT SHORT → decrease. If you want fewer chunks → increase carefully.
# # ═══════════════════════════════════════════════════════════════════════════════
# _GEMINI_MAX_CHARS    = 3800   # English / Latin scripts
# _GEMINI_MAX_CHARS_AR = 3200   # Arabic


# def _is_english_only_voice(voice: str) -> bool:
#     return any(voice.startswith(p) for p in _DEEPGRAM_ENGLISH_VOICE_PREFIXES)


# _VALID_AUDIO_STYLES = {"single", "multi", "podcast", "audiobook", "news", "bedtime", "custom"}

# # Continuity tags prepended to every TTS chunk after the first.
# # The first chunk already contains the full style prompt; subsequent chunks need
# # a strong reminder to maintain the SAME voice/tone/pacing.  Using a descriptive
# # tag (not just a label) gives the model more context.
# _STYLE_TAGS: dict[str, str] = {
#     "single":    "",
#     "audiobook": "narrator — same calm immersive voice, seamless continuation",
#     "news":      "anchor — same professional broadcast voice, continuing",
#     "bedtime":   "soothing — same gentle bedtime story voice, continuing",
#     "custom":    "custom delivery — same voice and style as before, continuing",
#     "multi":     "dialogue — same speakers and tone as before, continuing",
#     "podcast":   "podcast — same hosts and energy as before, continuing",
# }


# def _gemini_voice_name(voice: str | None, fallback: str = "Kore") -> str:
#     """Return a valid Gemini voice name, falling back if the input is missing/invalid."""
#     voice = (voice or fallback).strip()
#     if voice not in _OPENROUTER_GEMINI_VOICES:
#         log.warning("Invalid Gemini voice %r; falling back to %r.", voice, fallback)
#         return fallback
#     return voice


# def _resolve_speaker_voices(cfg: dict, language: str) -> tuple[str, str]:
#     """Resolve Speaker 1 / Speaker 2 Gemini voices (with optional per-language overrides)."""
#     lang = language.upper()
#     v1 = cfg.get(f"GEMINI_TTS_SPEAKER1_VOICE_{lang}") or cfg.get("GEMINI_TTS_SPEAKER1_VOICE") or "Kore"
#     v2 = cfg.get(f"GEMINI_TTS_SPEAKER2_VOICE_{lang}") or cfg.get("GEMINI_TTS_SPEAKER2_VOICE") or "Puck"
#     return _gemini_voice_name(v1, "Kore"), _gemini_voice_name(v2, "Puck")


# def _apply_audio_style(text: str, style: str | None, cfg: dict) -> tuple[str, bool, str]:
#     """
#     Apply a Gemini TTS style/profile to the transcript.

#     Returns (styled_text, use_multi_speaker, chunk_prefix).

#     chunk_prefix is prepended to every TTS chunk *except* the first one.
#     It reminds the model to keep the same delivery across chunk boundaries.
#     """
#     style = (style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
#     if style not in _VALID_AUDIO_STYLES:
#         style = "single"

#     tag = _STYLE_TAGS.get(style, "").strip()
#     chunk_prefix = f"[{tag}]\n\n" if tag else ""

#     if style == "custom":
#         prompt = (cfg.get("GEMINI_TTS_STYLE_PROMPT") or "").strip()
#         return (f"{prompt}\n\n{text}" if prompt else text), False, chunk_prefix

#     if style in ("multi", "podcast"):
#         s1 = (cfg.get("GEMINI_TTS_SPEAKER1_NAME") or "Speaker1").strip()
#         s2 = (cfg.get("GEMINI_TTS_SPEAKER2_NAME") or "Speaker2").strip()
#         intro = (
#             f"TTS the following podcast episode between {s1} and {s2}.\n\n"
#             if style == "podcast"
#             else f"TTS the following conversation between {s1} and {s2}.\n\n"
#         )
#         sentences = [s.strip() for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
#         if not sentences:
#             return text, True, chunk_prefix
#         lines = []
#         for i, sent in enumerate(sentences):
#             speaker = s1 if i % 2 == 0 else s2
#             lines.append(f"{speaker}: {sent}")
#         return intro + "\n".join(lines), True, chunk_prefix

#     prompts = {
#         "audiobook": "Read the following text in a calm, immersive audiobook narration style.",
#         "news":      "Read the following text as a professional news broadcast.",
#         "bedtime":   "Read the following text in a soothing, gentle bedtime story style.",
#     }
#     prompt = prompts.get(style, "")
#     custom_prompt = (cfg.get("GEMINI_TTS_STYLE_PROMPT") or "").strip()
#     if custom_prompt:
#         prompt = custom_prompt
#     return (f"{prompt}\n\n{text}" if prompt else text), False, chunk_prefix


# # Sentence-ending punctuation in EN / AR / general.
# _SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?؟…])\s+")


# def _split_text_for_tts(text: str, max_chars: int = _DEFAULT_MAX_CHARS) -> list[str]:
#     """
#     Split text into TTS-sized chunks on sentence boundaries.
#     Guarantees no chunk exceeds `max_chars`.
#     """
#     text = (text or "").strip()
#     if not text:
#         return []
#     if len(text) <= max_chars:
#         return [text]

#     sentences = [s for s in _SENTENCE_BOUNDARY.split(text) if s.strip()]
#     chunks: list[str] = []
#     buf = ""

#     def flush():
#         nonlocal buf
#         if buf.strip():
#             chunks.append(buf.strip())
#         buf = ""

#     for sent in sentences:
#         sent = sent.strip()
#         if not sent:
#             continue

#         if len(sent) > max_chars:
#             flush()
#             words = sent.split(" ")
#             inner = ""
#             for w in words:
#                 candidate = (inner + " " + w).strip() if inner else w
#                 if len(candidate) > max_chars:
#                     if inner:
#                         chunks.append(inner.strip())
#                     inner = w
#                 else:
#                     inner = candidate
#             if inner.strip():
#                 buf = inner
#             continue

#         candidate = (buf + " " + sent).strip() if buf else sent
#         if len(candidate) > max_chars:
#             flush()
#             buf = sent
#         else:
#             buf = candidate

#     flush()
#     return chunks


# def _concat_audio_ffmpeg(part_paths: list[str], output_path: str) -> None:
#     """
#     Properly concatenate MP3/audio files using ffmpeg concat demuxer.

#     Raw byte concatenation of MP3 files creates malformed output with multiple
#     ID3 headers, causing browser audio players to stop at the first header
#     boundary and play only the first chunk. This function uses ffmpeg's concat
#     demuxer with re-encoding to produce a single clean, valid MP3 file.
#     """
#     if not part_paths:
#         raise ValueError("No audio parts to concatenate")

#     if len(part_paths) == 1:
#         shutil.copy2(part_paths[0], output_path)
#         return

#     if not shutil.which("ffmpeg"):
#         raise RuntimeError(
#             "ffmpeg is required for audio concatenation but is not installed."
#         )

#     with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
#         concat_list_path = f.name
#         for p in part_paths:
#             escaped = p.replace("'", "'\''")
#             f.write(f"file '{escaped}'\n")

#     try:
#         result = subprocess.run(
#             [
#                 "ffmpeg", "-y",
#                 "-f", "concat", "-safe", "0",
#                 "-i", concat_list_path,
#                 "-ac", "2", "-ar", "44100", "-b:a", "128k",
#                 "-write_xing", "0",
#                 output_path,
#             ],
#             capture_output=True,
#         )
#         if result.returncode != 0:
#             stderr = result.stderr.decode("utf-8", errors="ignore")[:800]
#             raise RuntimeError(
#                 f"ffmpeg concat failed (exit {result.returncode}). stderr: {stderr}"
#             )
#     finally:
#         try:
#             os.unlink(concat_list_path)
#         except OSError:
#             pass


# async def synthesize(
#     text: str,
#     language: str,
#     output_path: str,
#     cfg: dict | None = None,
#     audio_style: str | None = None,
# ) -> str:
#     """
#     Convert text to MP3 and save to output_path. Returns output_path.

#     Text longer than the provider's per-call limit is split on sentence
#     boundaries, each chunk is synthesized individually, and the resulting MP3
#     files are concatenated using ffmpeg concat demuxer for proper merging.
#     """
#     cfg = cfg or {}
#     lang = language.upper()

#     # ── Spoken audio watermark ───────────────────────────────────────────────
#     watermark = (cfg.get(f"AUDIO_WATERMARK_TEXT_{lang}") or "").strip()
#     if watermark:
#         text = f"{watermark}.\n\n{text}"

#     provider = cfg.get(f"TTS_PROVIDER_{lang}") or (
#         settings.TTS_PROVIDER_EN if language == "en" else settings.TTS_PROVIDER_AR
#     )
#     voice = cfg.get(f"TTS_VOICE_{lang}") or (
#         settings.TTS_VOICE_EN if language == "en" else settings.TTS_VOICE_AR
#     )

#     # ── Gemini TTS style / profile ───────────────────────────────────────────
#     is_gemini_tts = provider == "gemini" or (
#         provider == "openrouter"
#         and (cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL).lower().startswith("google/")
#     )
#     effective_style = "single"
#     if is_gemini_tts:
#         effective_style = (audio_style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
#     styled_text, is_multi_style, chunk_prefix = _apply_audio_style(text, effective_style, cfg)

#     # ── Deepgram English-only warning ────────────────────────────────────────
#     if provider == "deepgram" and language == "ar" and _is_english_only_voice(voice):
#         log.warning(
#             "TTS warning: Deepgram voice %r is English-only. Arabic text will sound garbled. "
#             "Switch TTS_PROVIDER_AR to 'elevenlabs' or 'cartesia' in Admin → Providers → Text-to-Speech.",
#             voice,
#         )

#     # ── Resolve per-provider character budget ────────────────────────────────
#     if is_gemini_tts:
#         # Gemini TTS has a hard ~5-minute audio output limit per request.
#         # The budget must keep each chunk's spoken audio under that limit.
#         if language == "ar":
#             cfg_key, default_budget = "TTS_MAX_CHARS_GEMINI_AR", _GEMINI_MAX_CHARS_AR
#         else:
#             cfg_key, default_budget = "TTS_MAX_CHARS_GEMINI", _GEMINI_MAX_CHARS
#         try:
#             max_chars = int(cfg.get(cfg_key) or default_budget)
#         except (TypeError, ValueError):
#             max_chars = default_budget
#         max_chars = max(max_chars, 1000)
#     else:
#         max_chars = _PROVIDER_MAX_CHARS.get(provider, _DEFAULT_MAX_CHARS)

#     # Reserve room for the per-chunk continuity tag.
#     split_budget = max_chars
#     if is_gemini_tts and chunk_prefix:
#         split_budget = max(max_chars - len(chunk_prefix), 1000)

#     chunks = _split_text_for_tts(styled_text, max_chars=split_budget)
#     if not chunks:
#         raise ValueError("synthesize() called with empty text")

#     oversize = [(i, len(c)) for i, c in enumerate(chunks) if len(c) > max_chars]
#     if oversize:
#         raise RuntimeError(
#             f"TTS chunker produced oversize chunks for provider={provider} "
#             f"(limit={max_chars}): {oversize[:3]}…"
#         )

#     log.info(
#         "TTS: provider=%s style=%s multi=%s text=%d chars → %d chunk(s) (limit %d/chunk)",
#         provider, effective_style, is_multi_style, len(styled_text), len(chunks), max_chars,
#     )

#     # ── Fast path: single chunk ──────────────────────────────────────────────
#     if len(chunks) == 1:
#         await _dispatch_tts(
#             chunks[0], provider, voice, language, cfg, output_path,
#             audio_style=effective_style,
#         )
#         await log_tts_usage(provider=provider, model=voice, characters=len(styled_text))
#         return output_path

#     # ── Slow path: synthesize each chunk, then concat with ffmpeg ────────────
#     part_paths: list[str] = []
#     try:
#         for i, chunk in enumerate(chunks):
#             if i > 0 and chunk_prefix:
#                 chunk = chunk_prefix + chunk
#             part = f"{output_path}.part{i:03d}.mp3"
#             await _dispatch_tts(
#                 chunk, provider, voice, language, cfg, part,
#                 audio_style=effective_style,
#             )
#             part_paths.append(part)

#         _concat_audio_ffmpeg(part_paths, output_path)
#     finally:
#         for p in part_paths:
#             try:
#                 os.remove(p)
#             except OSError:
#                 pass

#     await log_tts_usage(provider=provider, model=voice, characters=len(styled_text))
#     return output_path


# async def _dispatch_tts(
#     text: str,
#     provider: str,
#     voice: str,
#     language: str,
#     cfg: dict,
#     output_path: str,
#     audio_style: str | None = None,
# ) -> None:
#     """Route a single (chunk-sized) TTS request to the chosen provider."""
#     if provider == "deepgram":
#         if not settings.DEEPGRAM_API_KEY:
#             raise ValueError(
#                 "DEEPGRAM_API_KEY is not set. Add it to your .env file "
#                 "or switch TTS provider in the Admin panel."
#             )
#         await _deepgram(text, voice, output_path)
#     elif provider == "elevenlabs":
#         if not settings.ELEVENLABS_API_KEY:
#             raise ValueError(
#                 "ELEVENLABS_API_KEY is not set. Add it to your .env file "
#                 "or switch TTS provider in the Admin panel."
#             )
#         await _elevenlabs(text, language, cfg, output_path)
#     elif provider == "cartesia":
#         if not settings.CARTESIA_API_KEY:
#             raise ValueError(
#                 "CARTESIA_API_KEY is not set. Add it to your .env file "
#                 "or switch the TTS provider to 'elevenlabs' in Admin → Providers → Text-to-Speech."
#             )
#         _cartesia_voice = (
#             cfg.get(f"CARTESIA_VOICE_{language.upper()}")
#             or voice
#             or cfg.get("CARTESIA_VOICE_EN")
#             or settings.CARTESIA_VOICE_EN
#             or settings.CARTESIA_VOICE_AR
#         )
#         if not _is_uuid(_cartesia_voice) and get_openrouter_key():
#             log.warning(
#                 "Cartesia voice missing for language %r (value=%r); falling back to OpenRouter TTS.",
#                 language, _cartesia_voice,
#             )
#             or_model = cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL
#             lang_voice_key = f"OPENROUTER_TTS_VOICE_{language.upper()}"
#             or_voice = (
#                 cfg.get(lang_voice_key)
#                 or cfg.get("OPENROUTER_TTS_VOICE")
#                 or settings.OPENROUTER_TTS_VOICE
#             )
#             await _openrouter_tts(text, or_voice, language, or_model, output_path, cfg=cfg, audio_style=audio_style)
#         else:
#             await _cartesia(text, voice, language, cfg, output_path)
#     elif provider == "gemini":
#         if not (settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY):
#             raise ValueError(
#                 "GEMINI_API_KEY (or OPENROUTER_API_KEY as fallback) is not set. "
#                 "Get a Gemini API key at https://aistudio.google.com/app/apikey"
#             )
#         gemini_model = cfg.get("GEMINI_TTS_MODEL") or settings.GEMINI_TTS_MODEL
#         gemini_voice = cfg.get("GEMINI_TTS_VOICE") or settings.GEMINI_TTS_VOICE
#         await _gemini(text, gemini_voice, language, gemini_model, output_path, cfg=cfg, audio_style=audio_style)
#     elif provider == "openrouter":
#         if not get_openrouter_key():
#             raise ValueError(
#                 "No OpenRouter API keys are configured. OpenRouter TTS requires "
#                 "OPENROUTER_API_KEY (plus optional OPENROUTER_API_KEY_2 / _3). "
#                 "Get one at https://openrouter.ai/keys."
#             )
#         or_model = cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL
#         lang_voice_key = f"OPENROUTER_TTS_VOICE_{language.upper()}"
#         or_voice = (
#             cfg.get(lang_voice_key)
#             or cfg.get("OPENROUTER_TTS_VOICE")
#             or settings.OPENROUTER_TTS_VOICE
#         )
#         await _openrouter_tts(text, or_voice, language, or_model, output_path, cfg=cfg, audio_style=audio_style)
#     else:
#         raise ValueError(f"Unknown TTS provider: {provider!r}")


# # ── Provider implementations ─────────────────────────────────────────────────

# async def _deepgram(text: str, voice: str, output_path: str) -> None:
#     """Deepgram TTS with retry logic for timeout/rate-limit errors."""
#     max_retries = 3
#     base_delay = 2

#     for attempt in range(max_retries):
#         try:
#             async with httpx.AsyncClient(timeout=120) as client:
#                 r = await client.post(
#                     f"https://api.deepgram.com/v1/speak?model={voice}",
#                     headers={
#                         "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
#                         "Content-Type": "application/json",
#                     },
#                     json={"text": text},
#                 )
#                 r.raise_for_status()
#             with open(output_path, "wb") as f:
#                 f.write(r.content)
#             return
#         except httpx.HTTPStatusError as e:
#             if e.response.status_code in (408, 429) and attempt < max_retries - 1:
#                 delay = base_delay * (2 ** attempt)
#                 log.warning(
#                     "Deepgram TTS attempt %d/%d failed with %d, retrying in %ds...",
#                     attempt + 1, max_retries, e.response.status_code, delay,
#                 )
#                 await asyncio.sleep(delay)
#                 continue
#             raise
#         except httpx.TimeoutException:
#             if attempt < max_retries - 1:
#                 delay = base_delay * (2 ** attempt)
#                 log.warning(
#                     "Deepgram TTS attempt %d/%d timed out, retrying in %ds...",
#                     attempt + 1, max_retries, delay,
#                 )
#                 await asyncio.sleep(delay)
#                 continue
#             raise


# async def _elevenlabs(text: str, language: str, cfg: dict, output_path: str) -> None:
#     if language == "en":
#         voice_id = cfg.get("ELEVENLABS_VOICE_EN") or settings.ELEVENLABS_VOICE_EN
#     else:
#         voice_id = cfg.get("ELEVENLABS_VOICE_AR") or settings.ELEVENLABS_VOICE_AR
#     if not voice_id:
#         raise ValueError(
#             f"ELEVENLABS_VOICE_{'EN' if language == 'en' else 'AR'} is not set. "
#             "Set it in Admin → Providers → Text-to-Speech, or add it to your .env."
#         )
#     model_id = cfg.get("ELEVENLABS_MODEL") or "eleven_multilingual_v2"
#     async with httpx.AsyncClient(timeout=120) as client:
#         r = await client.post(
#             f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
#             headers={
#                 "xi-api-key": settings.ELEVENLABS_API_KEY,
#                 "Content-Type": "application/json",
#             },
#             json={
#                 "text": text,
#                 "model_id": model_id,
#                 "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
#             },
#         )
#         r.raise_for_status()
#     with open(output_path, "wb") as f:
#         f.write(r.content)


# async def _cartesia(text: str, voice: str, language: str, cfg: dict, output_path: str) -> None:
#     """Call Cartesia Sonic TTS."""
#     model = cfg.get("CARTESIA_MODEL") or settings.CARTESIA_MODEL

#     _vkey = f"CARTESIA_VOICE_{language.upper()}"
#     voice = (
#         cfg.get(_vkey)
#         or voice
#         or cfg.get("CARTESIA_VOICE_EN")
#         or settings.CARTESIA_VOICE_EN
#         or settings.CARTESIA_VOICE_AR
#     )

#     if not _is_uuid(voice):
#         raise ValueError(
#             f"Cartesia voice value '{voice}' is not a valid voice UUID for language '{language}'. "
#             f"Voice IDs look like 'a0e99841-438c-4a64-b679-ae501e7d6091'. "
#             f"Find a voice at https://play.cartesia.ai/voices, then set "
#             f"{_vkey} (or CARTESIA_VOICE_EN as a multilingual fallback)."
#         )

#     payload = {
#         "model_id":   model,
#         "transcript": text,
#         "voice":      {"mode": "id", "id": voice},
#         "language":   language,
#         "output_format": {
#             "container":   "mp3",
#             "encoding":    "mp3",
#             "sample_rate": 44100,
#         },
#     }

#     async with httpx.AsyncClient(timeout=120) as client:
#         r = await client.post(
#             "https://api.cartesia.ai/tts/bytes",
#             headers={
#                 "X-API-Key":        settings.CARTESIA_API_KEY,
#                 "Cartesia-Version": "2024-06-10",
#                 "Content-Type":     "application/json",
#             },
#             json=payload,
#         )
#         if r.status_code >= 400:
#             raise RuntimeError(
#                 f"Cartesia returned {r.status_code}: {r.text[:500]} "
#                 f"(model={model}, language={language}, voice={voice})"
#             )
#     with open(output_path, "wb") as f:
#         f.write(r.content)


# async def _gemini(
#     text: str,
#     voice: str,
#     language: str,
#     model: str,
#     output_path: str,
#     cfg: dict | None = None,
#     audio_style: str | None = None,
# ) -> None:
#     """
#     Google Gemini TTS via the native Gemini API.

#     Uses the Gemini 2.5 Flash native speech-generation endpoint.
#     Gemini natively supports Arabic + 30+ other languages.

#     CRITICAL: Gemini TTS has a hard ~5-minute audio output limit per request.
#     Longer input text is silently truncated in the audio output. The chunking
#     logic in synthesize() must keep each chunk under this limit.
#     """
#     cfg = cfg or {}
#     chosen_voice = _gemini_voice_name(voice, "Kore")

#     api_key = settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY
#     if not api_key:
#         raise ValueError(
#             "GEMINI_API_KEY (or OPENROUTER_API_KEY as fallback) is not set. "
#             "Get a Gemini API key at https://aistudio.google.com/app/apikey"
#         )

#     gemini_model = model.split("/", 1)[1] if "/" in model else model

#     style = (audio_style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
#     is_multi = style in ("multi", "podcast")
#     if is_multi:
#         s1_name, s2_name = (
#             (cfg.get("GEMINI_TTS_SPEAKER1_NAME") or "Speaker1").strip(),
#             (cfg.get("GEMINI_TTS_SPEAKER2_NAME") or "Speaker2").strip(),
#         )
#         s1_voice, s2_voice = _resolve_speaker_voices(cfg, language)
#         speech_config = {
#             "multiSpeakerVoiceConfig": {
#                 "speakerVoiceConfigs": [
#                     {
#                         "speaker": s1_name,
#                         "voiceConfig": {
#                             "prebuiltVoiceConfig": {"voiceName": s1_voice}
#                         },
#                     },
#                     {
#                         "speaker": s2_name,
#                         "voiceConfig": {
#                             "prebuiltVoiceConfig": {"voiceName": s2_voice}
#                         },
#                     },
#                 ]
#             }
#         }
#     else:
#         speech_config = {
#             "voiceConfig": {
#                 "prebuiltVoiceConfig": {"voiceName": chosen_voice}
#             }
#         }

#     payload = {
#         "contents": [{"parts": [{"text": text}]}],
#         "generationConfig": {
#             "responseModalities": ["AUDIO"],
#             "speechConfig": speech_config,
#         },
#     }

#     async with httpx.AsyncClient(timeout=180) as client:
#         r = await client.post(
#             f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent",
#             headers={"Content-Type": "application/json"},
#             params={"key": api_key},
#             json=payload,
#         )
#         if r.status_code >= 400:
#             raise RuntimeError(
#                 f"Gemini TTS returned {r.status_code}: {r.text[:500]} "
#                 f"(model={gemini_model}, style={style}, voice={chosen_voice}, language={language})"
#             )
#         data = r.json()

#     audio_b64:   str | None = None
#     audio_mime:  str | None = None
#     try:
#         parts = data["candidates"][0]["content"]["parts"]
#         for part in parts:
#             inline_data = part.get("inlineData")
#             if inline_data and inline_data.get("mimeType", "").startswith("audio/"):
#                 audio_b64  = inline_data.get("data")
#                 audio_mime = inline_data.get("mimeType", "audio/mp3")
#                 break
#     except (KeyError, IndexError, TypeError) as exc:
#         raise RuntimeError(
#             f"Gemini TTS returned an unexpected response shape: {data!r}"
#         ) from exc

#     if not audio_b64:
#         raise RuntimeError(
#             "Gemini TTS response did not contain audio data. "
#             f"Make sure {gemini_model!r} supports audio output. "
#             f"Response excerpt: {str(data)[:400]}"
#         )

#     raw_bytes = base64.b64decode(audio_b64)

#     _is_pcm = (audio_mime or "").lower() in ("audio/pcm", "audio/l16", "audio/raw")
#     if _is_pcm:
#         tmp_pcm = output_path + ".pcm"
#         try:
#             with open(tmp_pcm, "wb") as f:
#                 f.write(raw_bytes)
#             subprocess.run(
#                 [
#                     "ffmpeg", "-y",
#                     "-f", "s16le", "-ar", "24000", "-ac", "1",
#                     "-i", tmp_pcm,
#                     "-ar", "44100", "-ac", "2", "-b:a", "128k",
#                     output_path,
#                 ],
#                 check=True, capture_output=True,
#             )
#         finally:
#             if os.path.exists(tmp_pcm):
#                 os.remove(tmp_pcm)
#     else:
#         with open(output_path, "wb") as f:
#             f.write(raw_bytes)


# # Valid OpenAI audio voices for gpt-audio / gpt-audio-mini
# _OPENAI_AUDIO_VOICES: set[str] = {
#     "alloy", "echo", "fable", "onyx", "nova", "shimmer",
#     "coral", "verse", "ballad", "ash", "sage", "marin", "cedar",
# }


# def _transcode_to_mp3(input_path: str, output_path: str) -> None:
#     """Transcode any ffmpeg-readable audio file to MP3 stereo 44100 Hz."""
#     if not shutil.which("ffmpeg"):
#         raise RuntimeError("ffmpeg is not installed. Cannot transcode non-MP3 TTS output.")

#     result = subprocess.run(
#         [
#             "ffmpeg", "-y",
#             "-i", input_path,
#             "-ar", "44100", "-ac", "2", "-b:a", "128k",
#             output_path,
#         ],
#         capture_output=True,
#     )
#     if result.returncode != 0:
#         raise RuntimeError(
#             f"ffmpeg failed to transcode audio (exit {result.returncode}). "
#             f"stderr: {result.stderr.decode('utf-8', errors='ignore')[:800]}"
#         )


# def _transcode_pcm_to_mp3(input_path: str, output_path: str) -> None:
#     """Transcode raw signed 16-bit little-endian PCM (24 kHz mono) to MP3."""
#     if not shutil.which("ffmpeg"):
#         raise RuntimeError("ffmpeg is not installed. Cannot transcode PCM TTS output.")

#     result = subprocess.run(
#         [
#             "ffmpeg", "-y",
#             "-f", "s16le", "-ar", "24000", "-ac", "1",
#             "-i", input_path,
#             "-ar", "44100", "-ac", "2", "-b:a", "128k",
#             output_path,
#         ],
#         capture_output=True,
#     )
#     if result.returncode != 0:
#         raise RuntimeError(
#             f"ffmpeg failed to transcode PCM audio (exit {result.returncode}). "
#             f"stderr: {result.stderr.decode('utf-8', errors='ignore')[:800]}"
#         )


# def _is_mp3(data: bytes) -> bool:
#     """Detect MP3 by magic bytes (ID3v2 or MPEG sync word)."""
#     return data.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"))


# def _detect_mime_type(data: bytes) -> str:
#     """Best-effort MIME type from file magic bytes."""
#     if data.startswith(b"ID3") or data.startswith((b"\xff\xfb", b"\xff\xf3", b"\xff\xf2")):
#         return "audio/mpeg"
#     if data.startswith(b"RIFF"):
#         return "audio/wav"
#     if data.startswith(b"OggS"):
#         return "audio/ogg"
#     if data.startswith(b"fLaC"):
#         return "audio/flac"
#     if data.startswith(b"\x00\x00\x00 ") or data.startswith(b"ftyp"):
#         return "audio/mp4"
#     return "audio/mpeg"


# async def _openrouter_tts(
#     text: str,
#     voice: str,
#     language: str,
#     model: str,
#     output_path: str,
#     cfg: dict | None = None,
#     audio_style: str | None = None,
# ) -> None:
#     """
#     TTS via OpenRouter's /audio/speech endpoint.

#     - OpenAI audio models accept OpenAI voice names and return MP3 by default.
#     - Google Gemini TTS models accept native Gemini voice names and, when asked
#       for response_format="pcm", return raw signed 16-bit little-endian PCM
#       at 24 kHz mono.

#     CRITICAL: Gemini TTS via OpenRouter has the same ~5-minute audio output
#     hard limit as native Gemini. The chunking in synthesize() must keep each
#     chunk small enough to fit.

#     Rotates to the next configured OpenRouter key on credit/limit errors.
#     """
#     is_gemini = model.startswith("google/")
#     chosen_voice = voice or ("Kore" if is_gemini else "alloy")

#     if is_gemini and chosen_voice not in _OPENROUTER_GEMINI_VOICES:
#         log.warning(
#             "OpenRouter Gemini TTS got non-Gemini voice %r for model %s; "
#             "falling back to 'Kore'.", chosen_voice, model,
#         )
#         chosen_voice = "Kore"
#     elif not is_gemini and chosen_voice not in _OPENROUTER_OPENAI_VOICES:
#         log.warning(
#             "OpenRouter OpenAI-audio TTS got non-OpenAI voice %r for model %s; "
#             "falling back to 'alloy'.", chosen_voice, model,
#         )
#         chosen_voice = "alloy"

#     payload: dict = {
#         "model": model,
#         "input": text,
#         "voice": chosen_voice,
#     }
#     if is_gemini:
#         payload["response_format"] = "pcm"

#     last_error: Exception | None = None
#     key_count = openrouter_key_count()
#     max_attempts = max(1, key_count)

#     async with httpx.AsyncClient(timeout=180) as client:
#         for attempt in range(max_attempts):
#             key = get_openrouter_key()
#             r = await client.post(
#                 "https://openrouter.ai/api/v1/audio/speech",
#                 headers={
#                     "Authorization": f"Bearer {key}",
#                     "Content-Type":  "application/json",
#                     "HTTP-Referer":  "https://seeourbook.com",
#                     "X-Title":       "SeeOurBook Summarizer",
#                 },
#                 json=payload,
#             )

#             if r.status_code < 400:
#                 break

#             if is_credit_error(r.status_code, r.text):
#                 rotate_openrouter_key(key)
#                 log.warning(
#                     "OpenRouter TTS credit/limit error (HTTP %s) on attempt %s; "
#                     "rotating key and retrying.",
#                     r.status_code, attempt + 1,
#                 )
#                 last_error = RuntimeError(
#                     f"OpenRouter TTS returned {r.status_code}: {r.text[:500]} "
#                     f"(model={model}, voice={chosen_voice}, language={language})"
#                 )
#                 continue

#             raise RuntimeError(
#                 f"OpenRouter TTS returned {r.status_code}: {r.text[:500]} "
#                 f"(model={model}, voice={chosen_voice}, language={language})"
#             )
#         else:
#             raise last_error or RuntimeError(
#                 f"OpenRouter TTS failed on all configured keys "
#                 f"(model={model}, voice={chosen_voice}, language={language}). "
#                 f"Add OPENROUTER_API_KEY_2 / OPENROUTER_API_KEY_3 to your .env."
#             )

#     content = r.content
#     content_type = (r.headers.get("content-type") or "").lower()

#     log.info(
#         "OpenRouter TTS response: model=%s voice=%s lang=%s status=%s "
#         "content-type=%s bytes=%d first_bytes=%r",
#         model, chosen_voice, language, r.status_code, content_type, len(content),
#         content[:40],
#     )

#     if not content or len(content) < 100:
#         raise RuntimeError(
#             f"OpenRouter TTS returned no audio (model={model}, voice={chosen_voice}, "
#             f"language={language}, content-type={content_type}). Body: {content[:500]!r}"
#         )

#     if is_gemini:
#         with tempfile.NamedTemporaryFile(suffix=".pcm", delete=False) as tmp:
#             tmp_path = tmp.name
#         try:
#             with open(tmp_path, "wb") as f:
#                 f.write(content)
#             _transcode_pcm_to_mp3(tmp_path, output_path)
#         finally:
#             try:
#                 os.unlink(tmp_path)
#             except OSError:
#                 pass
#         return

#     if _is_mp3(content):
#         with open(output_path, "wb") as f:
#             f.write(content)
#         return

#     with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
#         tmp_path = tmp.name
#     try:
#         with open(tmp_path, "wb") as f:
#             f.write(content)
#         try:
#             _transcode_to_mp3(tmp_path, output_path)
#         except RuntimeError as exc:
#             raise RuntimeError(
#                 f"{exc}. Input first bytes: {content[:80]!r}"
#             ) from exc
#     finally:
#         try:
#             os.unlink(tmp_path)
#         except OSError:
#             pass



"""
Text-to-Speech service.
Supports Deepgram, ElevenLabs, Cartesia, and Google Gemini (via OpenRouter).
Provider and voice are resolved from a runtime cfg dict (from admin panel) with
fallback to settings.py defaults — no restart needed when the admin switches providers.

Long-text handling
──────────────────
All providers have a per-call character limit. A 10-minute book summary is ~8–10K
chars and would 4xx in a single request. synthesize() splits long text on sentence
boundaries, synthesizes each chunk separately, and concatenates the resulting audio
files using ffmpeg concat demuxer for proper merging.

OPENROUTER → GEMINI TTS  (special handling)
───────────────────────────────────────────
Gemini TTS (native AND via OpenRouter) has a hard ~5-minute audio output limit per
request AND frequently "early-stops" — returning audio that doesn't cover the full
input text. This is random, not purely size-based.

We defend against this with a robust strategy adapted from tts_speech.py:
  1. Small units: split text into <= max-chars pieces on sentence boundaries
  2. Retry + keep-longest: re-request a piece up to MAX_TRIES; keep the LONGEST
     audio (the complete render). Only if every try is too short do we split
     the piece and recurse.
  3. Completeness check: measure chars/sec. If too fast, audio was cut short.

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
# OPENROUTER → GEMINI TTS  ROBUSTNESS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
# Gemini TTS frequently "early-stops" — returning truncated audio. These constants
# control the retry/split strategy adapted from tts_speech.py.
# ═══════════════════════════════════════════════════════════════════════════════
_OR_GEMINI_MAX_CPS         = 18.0   # chars/sec threshold: faster = truncated
_OR_GEMINI_MAX_TRIES       = 3      # retry attempts per piece
_OR_GEMINI_MIN_SPLIT_CHARS = 150    # don't split pieces shorter than this
_OR_GEMINI_MAX_DEPTH       = 4      # max recursion depth for split-and-recurse
# Default max chars per piece for OpenRouter Gemini. Smaller = more reliable.
# The tts_speech.py script uses 260; we use 400 as a production balance.
_OR_GEMINI_DEFAULT_MAX_CHARS = 400


def _is_english_only_voice(voice: str) -> bool:
    return any(voice.startswith(p) for p in _DEEPGRAM_ENGLISH_VOICE_PREFIXES)


_VALID_AUDIO_STYLES = {"single", "multi", "podcast", "audiobook", "news", "bedtime", "custom"}

# Continuity tags prepended to every TTS chunk after the first.
# The first chunk already contains the full style prompt; subsequent chunks need
# a strong reminder to maintain the SAME voice/tone/pacing.
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


# ═══════════════════════════════════════════════════════════════════════════════
# OPENROUTER GEMINI TTS  ROBUST RENDERING (adapted from tts_speech.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _or_gemini_atomize(text: str, max_unit: int) -> list[str]:
    """Smallest pieces <= max_unit: sentences, long ones split at commas then spaces."""
    text = text.replace("\r", "")
    parts = re.split(r"(?<=[\.\!\?؟؛])\s+|\n+", text)
    parts = [p.strip() for p in parts if p.strip()]
    atoms = []
    for s in parts:
        if len(s) <= max_unit:
            atoms.append(s)
            continue
        for clause in re.split(r"(?<=[،,])\s+", s):
            while len(clause) > max_unit:
                cut = clause.rfind(" ", 0, max_unit)
                cut = cut if cut > 0 else max_unit
                atoms.append(clause[:cut].strip())
                clause = clause[cut:].strip()
            if clause:
                atoms.append(clause)
    return atoms


def _or_gemini_chunk_text(text: str, max_chars: int) -> list[str]:
    """Pack atoms greedily into pieces of <= max_chars (never cut mid-sentence)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    pieces, cur = [], ""
    for a in _or_gemini_atomize(text, max_chars):
        if not cur:
            cur = a
        elif len(cur) + 1 + len(a) <= max_chars:
            cur = cur + " " + a
        else:
            pieces.append(cur)
            cur = a
    if cur:
        pieces.append(cur)
    return pieces


def _or_gemini_pcm_seconds(pcm_bytes: bytes) -> float:
    """Calculate duration of raw PCM audio (24kHz / 16-bit / mono)."""
    SAMPLE_RATE = 24000
    SAMPLE_WIDTH = 2
    CHANNELS = 1
    return len(pcm_bytes) / (SAMPLE_RATE * SAMPLE_WIDTH * CHANNELS)


def _or_gemini_split_at_space(text: str) -> tuple[str, str]:
    """Split text roughly in half at a space boundary."""
    mid = len(text) // 2
    for off in range(mid):
        if text[mid - off] == " ":
            return text[:mid - off].strip(), text[mid - off:].strip()
        if mid + off < len(text) and text[mid + off] == " ":
            return text[:mid + off].strip(), text[mid + off:].strip()
    return text[:mid].strip(), text[mid:].strip()


async def _or_gemini_synth_piece(
    client: httpx.AsyncClient,
    text: str,
    api_key: str,
    voice: str,
    model: str,
    max_cps: float = _OR_GEMINI_MAX_CPS,
    max_tries: int = _OR_GEMINI_MAX_TRIES,
    min_split_chars: int = _OR_GEMINI_MIN_SPLIT_CHARS,
    max_depth: int = _OR_GEMINI_MAX_DEPTH,
    depth: int = 0,
    tag: str = "",
) -> bytes:
    """
    Render one piece of text completely via OpenRouter Gemini TTS.

    Strategy (from tts_speech.py):
      1. Try up to max_tries times, keeping the longest audio render.
      2. If the best render has a chars/sec rate > max_cps, the audio was
         truncated -> split the text in half and recurse.
      3. Only accept audio that passes the completeness check.

    Returns raw PCM bytes (24kHz / 16-bit / mono).
    """
    need = len(text) / max_cps
    best_pcm = b""
    best_secs = 0.0

    for attempt in range(1, max_tries + 1):
        try:
            payload = {
                "model": model,
                "input": text,
                "voice": voice,
                "response_format": "pcm",
            }
            r = await client.post(
                "https://openrouter.ai/api/v1/audio/speech",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://seeourbook.com",
                    "X-Title": "SeeOurBook Summarizer",
                },
                json=payload,
                timeout=300,
            )
            if r.status_code >= 400:
                log.warning(
                    "OpenRouter Gemini TTS attempt %d/%d failed: HTTP %d %s",
                    attempt, max_tries, r.status_code, r.text[:200],
                )
                if is_credit_error(r.status_code, r.text):
                    rotate_openrouter_key(api_key)
                continue

            pcm = r.content
            secs = _or_gemini_pcm_seconds(pcm)
            cps = len(text) / secs if secs > 0 else 999

            if secs > best_secs:
                best_pcm = pcm
                best_secs = secs

            if secs >= need:
                if attempt > 1:
                    log.info("OpenRouter Gemini TTS %sok on try %d (%.1fs)", tag, attempt, secs)
                return pcm

            log.warning(
                "OpenRouter Gemini TTS %stry %d: %dc -> %.1fs (%.0f cps) too short",
                tag, attempt, len(text), secs, cps,
            )
        except Exception as exc:
            log.warning("OpenRouter Gemini TTS %stry %d error: %s", tag, attempt, exc)

    # Every attempt was short -> split and recurse (or accept best if too small)
    if len(text) > min_split_chars and depth < max_depth:
        log.info("OpenRouter Gemini TTS %ssplitting %d chars at depth %d", tag, len(text), depth)
        a, b = _or_gemini_split_at_space(text)
        pcm_a = await _or_gemini_synth_piece(
            client, a, api_key, voice, model,
            max_cps, max_tries, min_split_chars, max_depth,
            depth + 1, tag + "a:",
        )
        pcm_b = await _or_gemini_synth_piece(
            client, b, api_key, voice, model,
            max_cps, max_tries, min_split_chars, max_depth,
            depth + 1, tag + "b:",
        )
        return pcm_a + pcm_b

    # Can't split further — return the best we have (may be truncated)
    if best_pcm:
        log.warning(
            "OpenRouter Gemini TTS %saccepting possibly truncated audio: "
            "%dc -> %.1fs (%.0f cps)", tag, len(text), best_secs, len(text) / best_secs if best_secs else 999,
        )
    return best_pcm


# ═══════════════════════════════════════════════════════════════════════════════
# AUDIO CONCATENATION
# ═══════════════════════════════════════════════════════════════════════════════

def _concat_audio_ffmpeg(part_paths: list[str], output_path: str) -> None:
    """
    Properly concatenate audio files using ffmpeg concat demuxer.

    Raw byte concatenation creates malformed output with multiple headers,
    causing players to stop at the first header boundary.
    """
    if not part_paths:
        raise ValueError("No audio parts to concatenate")

    if len(part_paths) == 1:
        shutil.copy2(part_paths[0], output_path)
        return

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for audio concatenation.")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list_path = f.name
        for p in part_paths:
            escaped = p.replace("'", "'\\''")
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
            raise RuntimeError(f"ffmpeg concat failed: {stderr}")
    finally:
        try:
            os.unlink(concat_list_path)
        except OSError:
            pass


def _concat_pcm_to_mp3(part_paths: list[str], output_path: str) -> None:
    """
    Concatenate raw PCM files and transcode to MP3.
    Used for OpenRouter Gemini TTS which returns raw PCM.
    """
    if not part_paths:
        raise ValueError("No PCM parts to concatenate")

    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for PCM concatenation.")

    # Build concat list for PCM files
    # For raw PCM, we need to specify format in the concat demuxer
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        concat_list_path = f.name
        for p in part_paths:
            escaped = p.replace("'", "'\\''")
            f.write(f"file '{escaped}'\n")

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", concat_list_path,
                "-f", "s16le", "-ar", "24000", "-ac", "1",
                "-ac", "2", "-ar", "44100", "-b:a", "128k",
                "-write_xing", "0",
                output_path,
            ],
            capture_output=True,
        )
        if result.returncode != 0:
            # Fallback: concatenate raw bytes then transcode
            raw_pcm = bytearray()
            for p in part_paths:
                with open(p, "rb") as src:
                    raw_pcm += src.read()
            tmp_pcm = output_path + ".raw"
            try:
                with open(tmp_pcm, "wb") as f:
                    f.write(raw_pcm)
                subprocess.run(
                    [
                        "ffmpeg", "-y",
                        "-f", "s16le", "-ar", "24000", "-ac", "1",
                        "-i", tmp_pcm,
                        "-ac", "2", "-ar", "44100", "-b:a", "128k",
                        output_path,
                    ],
                    check=True, capture_output=True,
                )
            finally:
                if os.path.exists(tmp_pcm):
                    os.remove(tmp_pcm)
    finally:
        try:
            os.unlink(concat_list_path)
        except OSError:
            pass


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SYNTHESIZE
# ═══════════════════════════════════════════════════════════════════════════════

async def synthesize(
    text: str,
    language: str,
    output_path: str,
    cfg: dict | None = None,
    audio_style: str | None = None,
) -> str:
    """
    Convert text to MP3 and save to output_path. Returns output_path.
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
            "Switch TTS_PROVIDER_AR to 'elevenlabs' or 'cartesia'.",
            voice,
        )

    # ── Resolve per-provider character budget ────────────────────────────────
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
        "TTS: provider=%s style=%s multi=%s text=%d chars -> %d chunk(s) (limit %d/chunk)",
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

    # ── Slow path: synthesize each chunk, then concat ────────────────────────
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
            raise ValueError("DEEPGRAM_API_KEY is not set.")
        await _deepgram(text, voice, output_path)
    elif provider == "elevenlabs":
        if not settings.ELEVENLABS_API_KEY:
            raise ValueError("ELEVENLABS_API_KEY is not set.")
        await _elevenlabs(text, language, cfg, output_path)
    elif provider == "cartesia":
        if not settings.CARTESIA_API_KEY:
            raise ValueError("CARTESIA_API_KEY is not set.")
        _cartesia_voice = (
            cfg.get(f"CARTESIA_VOICE_{language.upper()}")
            or voice
            or cfg.get("CARTESIA_VOICE_EN")
            or settings.CARTESIA_VOICE_EN
            or settings.CARTESIA_VOICE_AR
        )
        if not _is_uuid(_cartesia_voice) and get_openrouter_key():
            log.warning("Cartesia voice missing for %r; falling back to OpenRouter.", language)
            or_model = cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL
            lang_voice_key = f"OPENROUTER_TTS_VOICE_{language.upper()}"
            or_voice = cfg.get(lang_voice_key) or cfg.get("OPENROUTER_TTS_VOICE") or settings.OPENROUTER_TTS_VOICE
            await _openrouter_tts(text, or_voice, language, or_model, output_path, cfg=cfg, audio_style=audio_style)
        else:
            await _cartesia(text, voice, language, cfg, output_path)
    elif provider == "gemini":
        if not (settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY):
            raise ValueError("GEMINI_API_KEY (or OPENROUTER_API_KEY) is not set.")
        gemini_model = cfg.get("GEMINI_TTS_MODEL") or settings.GEMINI_TTS_MODEL
        gemini_voice = cfg.get("GEMINI_TTS_VOICE") or settings.GEMINI_TTS_VOICE
        await _gemini(text, gemini_voice, language, gemini_model, output_path, cfg=cfg, audio_style=audio_style)
    elif provider == "openrouter":
        if not get_openrouter_key():
            raise ValueError("No OpenRouter API keys are configured.")
        or_model = cfg.get("OPENROUTER_TTS_MODEL") or settings.OPENROUTER_TTS_MODEL
        lang_voice_key = f"OPENROUTER_TTS_VOICE_{language.upper()}"
        or_voice = cfg.get(lang_voice_key) or cfg.get("OPENROUTER_TTS_VOICE") or settings.OPENROUTER_TTS_VOICE
        await _openrouter_tts(text, or_voice, language, or_model, output_path, cfg=cfg, audio_style=audio_style)
    else:
        raise ValueError(f"Unknown TTS provider: {provider!r}")


# ── Provider implementations ─────────────────────────────────────────────────

async def _deepgram(text: str, voice: str, output_path: str) -> None:
    """Deepgram TTS with retry logic."""
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
                log.warning("Deepgram retry %d/%d in %ds (HTTP %d)", attempt + 1, max_retries, delay, e.response.status_code)
                await asyncio.sleep(delay)
                continue
            raise
        except httpx.TimeoutException:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.warning("Deepgram timeout retry %d/%d in %ds", attempt + 1, max_retries, delay)
                await asyncio.sleep(delay)
                continue
            raise


async def _elevenlabs(text: str, language: str, cfg: dict, output_path: str) -> None:
    voice_id = cfg.get(f"ELEVENLABS_VOICE_{language.upper()}") or (settings.ELEVENLABS_VOICE_EN if language == "en" else settings.ELEVENLABS_VOICE_AR)
    if not voice_id:
        raise ValueError(f"ELEVENLABS_VOICE_{language.upper()} is not set.")
    model_id = cfg.get("ELEVENLABS_MODEL") or "eleven_multilingual_v2"
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
            headers={"xi-api-key": settings.ELEVENLABS_API_KEY, "Content-Type": "application/json"},
            json={"text": text, "model_id": model_id, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        )
        r.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(r.content)


async def _cartesia(text: str, voice: str, language: str, cfg: dict, output_path: str) -> None:
    """Call Cartesia Sonic TTS."""
    model = cfg.get("CARTESIA_MODEL") or settings.CARTESIA_MODEL
    _vkey = f"CARTESIA_VOICE_{language.upper()}"
    voice = cfg.get(_vkey) or voice or cfg.get("CARTESIA_VOICE_EN") or settings.CARTESIA_VOICE_EN or settings.CARTESIA_VOICE_AR
    if not _is_uuid(voice):
        raise ValueError(f"Cartesia voice '{voice}' is not a valid UUID. Find voices at https://play.cartesia.ai/voices")
    payload = {
        "model_id": model,
        "transcript": text,
        "voice": {"mode": "id", "id": voice},
        "language": language,
        "output_format": {"container": "mp3", "encoding": "mp3", "sample_rate": 44100},
    }
    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post("https://api.cartesia.ai/tts/bytes", headers={"X-API-Key": settings.CARTESIA_API_KEY, "Cartesia-Version": "2024-06-10", "Content-Type": "application/json"}, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"Cartesia returned {r.status_code}: {r.text[:500]} (model={model}, language={language}, voice={voice})")
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
    """Google Gemini TTS via the native Gemini API."""
    cfg = cfg or {}
    chosen_voice = _gemini_voice_name(voice, "Kore")
    api_key = settings.GEMINI_API_KEY or settings.OPENROUTER_API_KEY
    if not api_key:
        raise ValueError("GEMINI_API_KEY (or OPENROUTER_API_KEY) is not set.")
    gemini_model = model.split("/", 1)[1] if "/" in model else model
    style = (audio_style or cfg.get("GEMINI_TTS_AUDIO_STYLE") or "single").strip().lower()
    is_multi = style in ("multi", "podcast")
    if is_multi:
        s1_name, s2_name = (cfg.get("GEMINI_TTS_SPEAKER1_NAME") or "Speaker1").strip(), (cfg.get("GEMINI_TTS_SPEAKER2_NAME") or "Speaker2").strip()
        s1_voice, s2_voice = _resolve_speaker_voices(cfg, language)
        speech_config = {"multiSpeakerVoiceConfig": {"speakerVoiceConfigs": [{"speaker": s1_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": s1_voice}}}, {"speaker": s2_name, "voiceConfig": {"prebuiltVoiceConfig": {"voiceName": s2_voice}}}]}}
    else:
        speech_config = {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": chosen_voice}}}
    payload = {"contents": [{"parts": [{"text": text}]}], "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": speech_config}}
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"https://generativelanguage.googleapis.com/v1beta/models/{gemini_model}:generateContent", headers={"Content-Type": "application/json"}, params={"key": api_key}, json=payload)
        if r.status_code >= 400:
            raise RuntimeError(f"Gemini TTS returned {r.status_code}: {r.text[:500]} (model={gemini_model}, voice={chosen_voice})")
        data = r.json()
    audio_b64 = None
    audio_mime = None
    try:
        for part in data["candidates"][0]["content"]["parts"]:
            inline_data = part.get("inlineData")
            if inline_data and inline_data.get("mimeType", "").startswith("audio/"):
                audio_b64 = inline_data.get("data")
                audio_mime = inline_data.get("mimeType", "audio/mp3")
                break
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Gemini TTS unexpected response: {data!r}") from exc
    if not audio_b64:
        raise RuntimeError(f"Gemini TTS no audio data. Response: {str(data)[:400]}")
    raw_bytes = base64.b64decode(audio_b64)
    _is_pcm = (audio_mime or "").lower() in ("audio/pcm", "audio/l16", "audio/raw")
    if _is_pcm:
        tmp_pcm = output_path + ".pcm"
        try:
            with open(tmp_pcm, "wb") as f:
                f.write(raw_bytes)
            subprocess.run(["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", tmp_pcm, "-ar", "44100", "-ac", "2", "-b:a", "128k", output_path], check=True, capture_output=True)
        finally:
            if os.path.exists(tmp_pcm):
                os.remove(tmp_pcm)
    else:
        with open(output_path, "wb") as f:
            f.write(raw_bytes)


# Valid OpenAI audio voices
_OPENAI_AUDIO_VOICES: set[str] = {"alloy", "echo", "fable", "onyx", "nova", "shimmer", "coral", "verse", "ballad", "ash", "sage", "marin", "cedar"}


def _transcode_to_mp3(input_path: str, output_path: str) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed.")
    result = subprocess.run(["ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "2", "-b:a", "128k", output_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.decode('utf-8', errors='ignore')[:800]}")


def _transcode_pcm_to_mp3(input_path: str, output_path: str) -> None:
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is not installed.")
    result = subprocess.run(["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", input_path, "-ar", "44100", "-ac", "2", "-b:a", "128k", output_path], capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg PCM failed: {result.stderr.decode('utf-8', errors='ignore')[:800]}")


def _is_mp3(data: bytes) -> bool:
    return data.startswith((b"ID3", b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"))


def _detect_mime_type(data: bytes) -> str:
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


# ═══════════════════════════════════════════════════════════════════════════════
# OPENROUTER TTS  —  ROBUST GEMINI PATH (adapted from tts_speech.py)
# ═══════════════════════════════════════════════════════════════════════════════

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

    For Gemini models: uses the robust retry+split strategy from tts_speech.py
    to handle Gemini's frequent "early-stop" truncation.

    For OpenAI audio models: standard MP3 path.
    """
    is_gemini = model.startswith("google/")
    chosen_voice = voice or ("Kore" if is_gemini else "alloy")

    if is_gemini and chosen_voice not in _OPENROUTER_GEMINI_VOICES:
        log.warning("OpenRouter Gemini TTS: invalid voice %r, falling back to 'Kore'.", chosen_voice)
        chosen_voice = "Kore"
    elif not is_gemini and chosen_voice not in _OPENROUTER_OPENAI_VOICES:
        log.warning("OpenRouter OpenAI TTS: invalid voice %r, falling back to 'alloy'.", chosen_voice)
        chosen_voice = "alloy"

    # ── GEMINI PATH: robust retry + split strategy ────────────────────────────
    if is_gemini:
        await _openrouter_gemini_tts(text, chosen_voice, model, output_path, cfg)
        return

    # ── OPENAI AUDIO PATH: standard MP3 ──────────────────────────────────────
    payload = {"model": model, "input": text, "voice": chosen_voice}
    last_error = None
    key_count = openrouter_key_count()
    max_attempts = max(1, key_count)

    async with httpx.AsyncClient(timeout=180) as client:
        for attempt in range(max_attempts):
            key = get_openrouter_key()
            r = await client.post(
                "https://openrouter.ai/api/v1/audio/speech",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json", "HTTP-Referer": "https://seeourbook.com", "X-Title": "SeeOurBook Summarizer"},
                json=payload,
            )
            if r.status_code < 400:
                break
            if is_credit_error(r.status_code, r.text):
                rotate_openrouter_key(key)
                log.warning("OpenRouter credit error HTTP %s, rotating key (attempt %d)", r.status_code, attempt + 1)
                last_error = RuntimeError(f"OpenRouter TTS {r.status_code}: {r.text[:500]}")
                continue
            raise RuntimeError(f"OpenRouter TTS {r.status_code}: {r.text[:500]}")
        else:
            raise last_error or RuntimeError("OpenRouter TTS failed on all keys")

    content = r.content
    if not content or len(content) < 100:
        raise RuntimeError(f"OpenRouter TTS returned no audio. Body: {content[:500]!r}")

    if _is_mp3(content):
        with open(output_path, "wb") as f:
            f.write(content)
        return

    with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        with open(tmp_path, "wb") as f:
            f.write(content)
        _transcode_to_mp3(tmp_path, output_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


async def _openrouter_gemini_tts(
    text: str,
    voice: str,
    model: str,
    output_path: str,
    cfg: dict | None = None,
) -> None:
    """
    OpenRouter -> Gemini TTS with robust retry+split strategy.

    Adapted from tts_speech.py:
      1. Split text into small pieces (default 400 chars) on sentence boundaries
      2. For each piece: retry up to 3 times, keep the longest audio render
      3. Detect truncation by chars/sec rate — if too fast, split and recurse
      4. Concatenate all PCM parts into final MP3

    This handles Gemini's frequent "early-stop" where it returns truncated
    audio even for short requests.
    """
    cfg = cfg or {}
    api_key = get_openrouter_key()
    if not api_key:
        raise ValueError("No OpenRouter API key available.")

    # Configurable max chars per piece (admin can override)
    max_chars = int(cfg.get("OPENROUTER_GEMINI_MAX_CHARS") or _OR_GEMINI_DEFAULT_MAX_CHARS)
    max_chars = max(max_chars, 100)

    # Split text into small pieces using the robust atomize+chunk strategy
    pieces = _or_gemini_chunk_text(text, max_chars)
    log.info("OpenRouter Gemini TTS: %d chars -> %d pieces (max %d chars/piece)", len(text), len(pieces), max_chars)

    # Render each piece completely
    pcm_parts: list[bytes] = []
    async with httpx.AsyncClient(timeout=300) as client:
        for i, piece in enumerate(pieces, 1):
            log.info("OpenRouter Gemini TTS: rendering piece %d/%d (%d chars)", i, len(pieces), len(piece))
            pcm = await _or_gemini_synth_piece(
                client=client,
                text=piece,
                api_key=api_key,
                voice=voice,
                model=model,
                tag=f"[{i}] ",
            )
            if not pcm:
                log.error("OpenRouter Gemini TTS: piece %d returned empty audio!", i)
                continue
            pcm_parts.append(pcm)
            secs = _or_gemini_pcm_seconds(pcm)
            log.info("OpenRouter Gemini TTS: piece %d -> %.1fs audio", i, secs)

    if not pcm_parts:
        raise RuntimeError("OpenRouter Gemini TTS: all pieces returned empty audio")

    # Concatenate all PCM parts and transcode to MP3
    total_pcm = b"".join(pcm_parts)
    total_secs = _or_gemini_pcm_seconds(total_pcm)
    overall_cps = len(text) / total_secs if total_secs > 0 else 0
    log.info(
        "OpenRouter Gemini TTS: total %.1fs audio, overall %.1f chars/sec (%s)",
        total_secs, overall_cps,
        "COMPLETE" if overall_cps <= _OR_GEMINI_MAX_CPS else "POSSIBLY TRUNCATED",
    )

    tmp_pcm = output_path + ".raw"
    try:
        with open(tmp_pcm, "wb") as f:
            f.write(total_pcm)
        _transcode_pcm_to_mp3(tmp_pcm, output_path)
    finally:
        if os.path.exists(tmp_pcm):
            os.remove(tmp_pcm)