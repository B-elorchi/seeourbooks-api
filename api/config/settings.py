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

    # ── OpenRouter (OpenAI-compatible, supports both Claude + GPT models) ─────
    # Use OpenRouter model names with a vendor prefix, e.g.:
    #   anthropic/claude-haiku-4-5-20251001
    #   openai/gpt-4.1-mini
    # Setting any MODEL_* to an OpenRouter name automatically routes via OpenRouter.
    OPENROUTER_API_KEY: str = ""

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
    # Mind map generation model (text, not image)
    MODEL_MINDMAP:     str = "gpt-4.1-mini"  # any chat model — supports OpenRouter prefix
    # Mind map output format: "mermaid" → SVG via mermaid.ink | "json" → structured JSON
    MINDMAP_FORMAT:    str = "mermaid"

    # ── TTS — per language ────────────────────────────────────────────────────
    TTS_PROVIDER_EN:   str = "deepgram"      # deepgram | elevenlabs | cartesia
    TTS_PROVIDER_AR:   str = "cartesia"      # cartesia | elevenlabs — Deepgram Aura is English-only
    TTS_VOICE_EN:      str = "aura-asteria-en"
    TTS_VOICE_AR:      str = ""              # set in admin: Cartesia voice UUID or ElevenLabs voice ID

    ELEVENLABS_API_KEY:    str = ""
    ELEVENLABS_VOICE_EN:   str = ""
    ELEVENLABS_VOICE_AR:   str = ""

    DEEPGRAM_API_KEY:      str = ""

    CARTESIA_API_KEY:      str = ""
    # sonic-3.5 supports 40+ languages including Arabic, French, Spanish, etc.
    # See https://docs.cartesia.ai/build-with-cartesia/models for current snapshots.
    CARTESIA_MODEL:        str = "sonic-3.5-2026-05-04"

    # Gemini TTS via OpenRouter — supports Arabic + 30+ languages natively.
    # Set TTS_PROVIDER_AR='gemini' to use. Voice defaults to 'Kore'.
    GEMINI_TTS_MODEL:      str = "google/gemini-2.5-flash-preview-tts"

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
    DOC_AI_PROVIDER: str = "deepseek"                  # deepseek | openai | claude
    DOC_AI_MODEL:    str = "deepseek-chat"             # OR openai/gpt-4.1-mini, claude-sonnet-4-6, etc.

    # Chunk size for the knowledge base (used for future RAG search).
    DOC_CHUNK_SIZE_WORDS: int = 750

    # Embeddings — leave provider empty to skip embedding generation (chunks
    # are still stored without vectors and can be embedded later in batch).
    EMBEDDING_PROVIDER: str = ""                       # openai | deepseek | ""  (disabled)
    EMBEDDING_MODEL:    str = "text-embedding-3-small"

    # DeepSeek API key — OpenAI-compatible endpoint at api.deepseek.com
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

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
