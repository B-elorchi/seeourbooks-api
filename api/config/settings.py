from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── Database backend ──────────────────────────────────────────────────────
    # "supabase" → uses Supabase REST API (SUPABASE_URL + SUPABASE_SERVICE_KEY)
    # "postgres" → connects directly via asyncpg (DATABASE_URL)
    DB_BACKEND: str = "supabase"

    # ── Supabase (required when DB_BACKEND=supabase) ──────────────────────────
    SUPABASE_URL:         str = ""
    SUPABASE_SERVICE_KEY: str = ""

    # ── Postgres direct (required when DB_BACKEND=postgres) ───────────────────
    # Example: postgresql://user:pass@localhost:5432/seeourbook
    DATABASE_URL: str = ""

    # ── Anthropic ─────────────────────────────────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    MODEL_HAIKU:  str = "claude-haiku-4-5-20251001"
    MODEL_SONNET: str = "claude-sonnet-4-6"
    MODEL_OPUS:   str = "claude-opus-4-7"
    # Per-chunk summary model (Pass 1). Defaults to OpenRouter GPT-4.1-mini for cost efficiency.
    # Supports OpenRouter prefix (e.g., "openai/gpt-4.1-mini") or native Anthropic models.
    MODEL_CHUNK:  str = "openai/gpt-4.1-mini"

    # ── OpenRouter (OpenAI-compatible, supports both Claude + GPT models) ─────
    # Use OpenRouter model names with a vendor prefix, e.g.:
    #   anthropic/claude-haiku-4-5-20251001
    #   openai/gpt-4.1-mini
    # Setting any MODEL_* to an OpenRouter name automatically routes via OpenRouter.
    # Primary + optional spare keys.  When the primary hits a credit/limit error,
    # the pipeline rotates to OPENROUTER_API_KEY_2, then _3.
    OPENROUTER_API_KEY:   str = ""
    OPENROUTER_API_KEY_2: str = ""
    OPENROUTER_API_KEY_3: str = ""

    # Native Gemini API key (for TTS, image generation, etc.)
    # Get one at https://aistudio.google.com/app/apikey
    # Falls back to OPENROUTER_API_KEY if not set.
    GEMINI_API_KEY: str = ""

    # ── Text files ────────────────────────────────────────────────────────────
    TEXT_DIR: Path = Path("/path/to/text/files")

    # ── OpenAI ────────────────────────────────────────────────────────────────
    OPENAI_API_KEY:    str = ""
    # Legacy single-language model — kept as a fallback when the per-language
    # IMAGE_MODEL_{EN,AR} settings aren't configured in the admin panel yet.
    IMAGE_MODEL:       str = "google/gemini-2.5-flash-image"
    # Per-language image generation models. Either can be:
    #   - native OpenAI: dall-e-3, gpt-image-1, dall-e-2
    #   - OpenRouter:    any vendor/model name accepted by openrouter.ai (free-typed in admin)
    IMAGE_MODEL_EN:    str = "google/gemini-2.5-flash-image"
    IMAGE_MODEL_AR:    str = "google/gemini-2.5-flash-image"
    # Use "gpt-image-1" only if your OpenAI project has been granted access to it.
    IMAGE_QUALITY:     str = "high"          # high | standard | auto
    IMAGE_SIZE:        str = "1024x1536"   # gpt-image-1 portrait (1024x1792 is no longer valid)
    # Cover prompt size limits.  OpenRouter Gemini image models have ~32k context
    # windows, and image output consumes a large share of that.  Keep the text
    # prompt small (default 3 000 chars / ~750 tokens) to avoid context overflow.
    # Admin can override via IMAGE_PROMPT_MAX_CHARS in Admin → Settings → Cover Image.
    IMAGE_PROMPT_MAX_CHARS:  int = 3000
    IMAGE_SUMMARY_MAX_CHARS: int = 1200
    # Legacy aliases — kept for backward compatibility with existing .env files.
    COVER_MAX_PROMPT_CHARS:  int = 3000
    COVER_SUMMARY_MAX_CHARS: int = 1200
    # Mind map generation model (text, not image)
    MODEL_MINDMAP:     str = "gpt-4.1-mini"  # any chat model — supports OpenRouter prefix
    # Mind map output format: "mermaid" → SVG via mermaid.ink | "json" → structured JSON
    MINDMAP_FORMAT:    str = "mermaid"

    # ── TTS — per language ────────────────────────────────────────────────────
    TTS_PROVIDER_EN:   str = "deepgram"      # deepgram | elevenlabs | cartesia | openrouter | gemini
    TTS_PROVIDER_AR:   str = "cartesia"      # cartesia | elevenlabs | openrouter | gemini — Deepgram Aura is English-only
    TTS_VOICE_EN:      str = "aura-asteria-en"
    TTS_VOICE_AR:      str = ""              # set in admin: voice ID for chosen provider

    # Sample rate (Hz) to assume when a TTS provider returns headerless raw PCM
    # (Gemini TTS emits signed 16-bit little-endian PCM at 24 000 Hz mono).
    # Used by the audio post-processing step to transcode such output to MP3.
    TTS_PCM_SAMPLE_RATE:   int = 24000

    ELEVENLABS_API_KEY:    str = ""
    ELEVENLABS_VOICE_EN:   str = ""
    ELEVENLABS_VOICE_AR:   str = ""

    DEEPGRAM_API_KEY:      str = ""

    CARTESIA_API_KEY:      str = ""
    # sonic-3.5 supports 40+ languages including Arabic, French, Spanish, etc.
    # See https://docs.cartesia.ai/build-with-cartesia/models for current snapshots.
    CARTESIA_MODEL:        str = "sonic-3.5-2026-05-04"
    # Default Cartesia voice IDs.  The EN default is Cartesia's public
    # "Barbershop Man" voice.  Set AR to a multilingual/Arabic-capable voice
    # in Admin → Providers → Text-to-Speech, or leave it empty to fall back to EN.
    CARTESIA_VOICE_EN:     str = "a0e99841-438c-4a64-b679-ae501e7d6091"
    CARTESIA_VOICE_AR:     str = ""

    # Gemini TTS via native Google API — supports Arabic + 30+ languages natively.
    # Set TTS_PROVIDER_AR='gemini' to use. Voice defaults to 'Kore'.
    # Requires GEMINI_API_KEY (or falls back to OPENROUTER_API_KEY).
    # Model names: gemini-2.5-flash-preview-tts, gemini-3.1-flash-tts-preview
    GEMINI_TTS_MODEL:      str = "gemini-2.5-flash-preview-tts"
    GEMINI_TTS_VOICE:      str = "Kore"

    # OpenRouter TTS — any TTS-capable model on OpenRouter.
    # Recommended (good quality, multilingual, supports Arabic):
    #   google/gemini-2.5-flash-preview-tts   ← default
    #   google/gemini-2.5-flash-tts
    # Also supported (English/multilingual via OpenAI audio):
    #   openai/gpt-audio, openai/gpt-audio-mini
    OPENROUTER_TTS_MODEL:  str = "google/gemini-2.5-flash-preview-tts"
    # Default voice for OpenRouter TTS.
    # Gemini voices: Kore, Charon, Puck, Fenrir, Aoede, Leda, Orus, Zephyr (and more)
    # OpenAI voices: alloy, echo, fable, onyx, nova, shimmer, coral, verse, ballad, ash, sage, marin, cedar
    OPENROUTER_TTS_VOICE:  str = "Kore"

    # ── Alt text — per language ───────────────────────────────────────────────
    ALTTEXT_PROVIDER_EN:  str = "claude"     # claude | openai
    ALTTEXT_PROVIDER_AR:  str = "claude"
    ALTTEXT_MODEL_EN:     str = "claude-sonnet-4-6"
    ALTTEXT_MODEL_AR:     str = "claude-sonnet-4-6"

    # ── Storage ───────────────────────────────────────────────────────────────
    STORAGE_PROVIDER:     str = "spaces"     # spaces | minio
    DO_SPACES_KEY:        str = ""
    DO_SPACES_SECRET:     str = ""
    DO_SPACES_REGION:     str = "nyc3"
    DO_SPACES_BUCKET:     str = "seeourbook"
    DO_SPACES_CDN_URL:    str = ""

    MINIO_ENDPOINT:       str = "http://localhost:9000"
    MINIO_ACCESS_KEY:     str = ""
    MINIO_SECRET_KEY:     str = ""
    MINIO_BUCKET:         str = "seeourbook"

    # ── Pipeline step toggles ─────────────────────────────────────────────────
    PIPELINE_STEP_TTS:              bool = True
    PIPELINE_STEP_COVER:            bool = True
    PIPELINE_STEP_MINDMAP:          bool = True
    PIPELINE_STEP_ALTTEXT:          bool = True
    PIPELINE_STEP_AUDIO_PROCESSING: bool = True
    PIPELINE_STEP_INJECT_EPUB:      bool = True
    PIPELINE_STEP_VIDEO:            bool = True

    # ── EPUB sources ─────────────────────────────────────────────────────────
    # When set, the inject_epub step fetches the source EPUB from:
    #     {BOOK_FILES_BASE_URL}/books/english/{book_id}.epub
    #     {BOOK_FILES_BASE_URL}/books/arabic/{book_id}.epub
    # Leave empty to disable EPUB injection entirely.
    BOOK_FILES_BASE_URL: str = ""

    # ── Video generation (slideshow with TTS narration) ──────────────────────
    # Provider:
    #   moviepy      — CPU-only, ships everywhere, ~$0 marginal cost (default)
    #   svd          — Stable Video Diffusion (needs GPU host with 10GB+ VRAM)
    #   cogvideox    — CogVideoX-5B (needs GPU host)
    VIDEO_PROVIDER:    str = "moviepy"
    # Orientation: portrait (mobile / TikTok / Reels) or landscape (YouTube)
    VIDEO_ORIENTATION: str = "portrait"     # portrait | landscape
    # Pixel dimensions — overridden if VIDEO_ORIENTATION is set
    VIDEO_WIDTH:       int = 1080
    VIDEO_HEIGHT:      int = 1920
    # Frame rate.  24 fps looks cinematic, 30 fps matches YouTube/TikTok.
    VIDEO_FPS:         int = 30
    # Bitrate target (libx264 -b:v).  ~3 Mbps is plenty for slideshow.
    VIDEO_BITRATE:     str = "3500k"
    # Optional font file paths.  Empty → use bundled / system defaults.
    VIDEO_FONT_EN:     str = ""
    VIDEO_FONT_AR:     str = ""

    # ── Summary parameters ────────────────────────────────────────────────────
    CHUNK_SIZE_WORDS: int = 1500

    # ── Reliability / fallback ────────────────────────────────────────────────
    # When True, recoverable text-model failures (credit exhausted, rate-limited,
    # provider 5xx, network timeout) automatically retry the request against the
    # next model in the fallback chain defined in ai_client._DEFAULT_FALLBACK_CHAINS.
    # Per-model overrides can be set in the admin panel as FALLBACK_<model>=...
    ENABLE_MODEL_FALLBACK: bool = True

    # ── Documents pipeline (OCR → text → AI summary + structured JSON) ────────
    # Storage root for uploaded PDFs (originals + OCR'd outputs).
    # The processor writes:
    #   {DOCUMENTS_DIR}/{document_id}/original.pdf
    #   {DOCUMENTS_DIR}/{document_id}/ocr.pdf
    DOCUMENTS_DIR: Path = Path("/var/data/documents")

    # OCR languages handed to tesseract via ocrmypdf, e.g. "ara+eng" for Arabic + English.
    # Add more codes (fra, deu, spa, …) as needed; the corresponding tesseract
    # language packs must be installed on the host.
    DOC_OCR_LANGUAGES: str = "ara+eng"

    # Hard upload limit (bytes).  Large Arabic books can run 50–100 MB.
    DOC_MAX_UPLOAD_BYTES: int = 200 * 1024 * 1024     # 200 MB

    # Hard page-count limit — guards against accidental >1000-page uploads.
    DOC_MAX_PAGES: int = 2000

    # AI provider for summary + structured JSON.  Falls through ai_client fallback
    # chains when ENABLE_MODEL_FALLBACK is on.
    #
    # Default: route DeepSeek through OpenRouter so admins don't need a separate
    # DEEPSEEK_API_KEY — OPENROUTER_API_KEY covers it.  Set DOC_AI_MODEL to any
    # `vendor/model` and the factory auto-routes via OpenRouterProvider
    # regardless of the DOC_AI_PROVIDER value below.
    DOC_AI_PROVIDER: str = "openrouter"                # openrouter | deepseek | openai | claude
    DOC_AI_MODEL:    str = "deepseek/deepseek-chat"    # OR openai/gpt-4.1-mini, claude-sonnet-4-6, etc.

    # Chunk size for the knowledge base (used for future RAG search).
    DOC_CHUNK_SIZE_WORDS: int = 750

    # Embeddings — leave provider empty to skip embedding generation (chunks
    # are still stored without vectors and can be embedded later in batch).
    EMBEDDING_PROVIDER: str = ""                       # openai | deepseek | ""  (disabled)
    EMBEDDING_MODEL:    str = "text-embedding-3-small"

    # DeepSeek API key — OpenAI-compatible endpoint at api.deepseek.com
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # ── Supabase Auth ─────────────────────────────────────────────────────────
    # JWT secret for verifying Supabase-issued access tokens.
    # Get it from: Supabase Dashboard → Project Settings → API → JWT Secret.
    # When EMPTY, auth is DISABLED — all endpoints stay public (good for local dev).
    SUPABASE_JWT_SECRET: str = ""
    # Comma-separated emails that grant the "admin" role.
    # Anyone NOT in this list is treated as a normal user.
    # Leave empty to make every authenticated user an admin (single-tenant mode).
    ADMIN_EMAILS: str = ""

    # ── API Key Auth ──────────────────────────────────────────────────────────
    # When True, all /api/* routes (except /api/health, /api/auth/*) require a
    # valid X-API-Key header.  Set False to disable for local dev.
    API_KEY_AUTH_ENABLED: bool = False   # change to True in production .env

    # ── Watermarks ────────────────────────────────────────────────────────────
    # Text stamped on generated images (cover) and embedded in audio ID3 tags.
    WATERMARK_TEXT: str = "SeeOurBook.com"
    # Watermark corner for images: top-left | top-right | bottom-left | bottom-right
    WATERMARK_POSITION: str = "bottom-right"

    model_config = SettingsConfigDict(
        # Looks for .env in api/ first, then in the project root (seeourbook-summarizer-api/)
        env_file=(
            Path(__file__).parent.parent / ".env",
            Path(__file__).parent.parent.parent / ".env",
        ),
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

SUMMARY_LENGTHS: dict[str, int] = {
    "3min": 450,
    "5min": 750,
    "10min": 1500,
    "15min": 2250,
}
