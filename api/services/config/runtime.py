"""
Runtime configuration — reads provider settings from Supabase provider_config table.
Falls back to hard-coded settings defaults when the table doesn't exist yet.
In-memory cache refreshes every 60 seconds so live admin changes take effect quickly.
"""
import time

from api.config.settings import settings
from api.services.db import find, upsert


# ── Defaults from env / settings.py ──────────────────────────────────────────

def _defaults() -> dict[str, str]:
    return {
        "MODEL_HAIKU":                    settings.MODEL_HAIKU,
        "MODEL_SONNET":                   settings.MODEL_SONNET,
        "MODEL_OPUS":                     settings.MODEL_OPUS,
        "TTS_PROVIDER_EN":                settings.TTS_PROVIDER_EN,
        "TTS_VOICE_EN":                   settings.TTS_VOICE_EN,
        "TTS_PROVIDER_AR":                settings.TTS_PROVIDER_AR,
        "TTS_VOICE_AR":                   settings.TTS_VOICE_AR,
        "CARTESIA_MODEL":                 settings.CARTESIA_MODEL,
        "GEMINI_TTS_MODEL":               settings.GEMINI_TTS_MODEL,
        "IMAGE_MODEL":                    settings.IMAGE_MODEL,
        "IMAGE_MODEL_EN":                 settings.IMAGE_MODEL_EN,
        "IMAGE_MODEL_AR":                 settings.IMAGE_MODEL_AR,
        "IMAGE_QUALITY":                  settings.IMAGE_QUALITY,
        "IMAGE_SIZE":                     settings.IMAGE_SIZE,
        "ALTTEXT_PROVIDER_EN":            settings.ALTTEXT_PROVIDER_EN,
        "ALTTEXT_MODEL_EN":               settings.ALTTEXT_MODEL_EN,
        "ALTTEXT_PROVIDER_AR":            settings.ALTTEXT_PROVIDER_AR,
        "ALTTEXT_MODEL_AR":               settings.ALTTEXT_MODEL_AR,
        "STORAGE_PROVIDER":               settings.STORAGE_PROVIDER,
        "PIPELINE_STEP_TTS":              str(settings.PIPELINE_STEP_TTS).lower(),
        "PIPELINE_STEP_COVER":            str(settings.PIPELINE_STEP_COVER).lower(),
        "PIPELINE_STEP_MINDMAP":          str(settings.PIPELINE_STEP_MINDMAP).lower(),
        "PIPELINE_STEP_ALTTEXT":          str(settings.PIPELINE_STEP_ALTTEXT).lower(),
        "PIPELINE_STEP_AUDIO_PROCESSING": str(settings.PIPELINE_STEP_AUDIO_PROCESSING).lower(),
        "ENABLE_MODEL_FALLBACK":          str(settings.ENABLE_MODEL_FALLBACK).lower(),
        # Documents pipeline
        "DOC_AI_PROVIDER":                settings.DOC_AI_PROVIDER,
        "DOC_AI_MODEL":                   settings.DOC_AI_MODEL,
        "DOC_OCR_LANGUAGES":              settings.DOC_OCR_LANGUAGES,
        "DOC_CHUNK_SIZE_WORDS":           str(settings.DOC_CHUNK_SIZE_WORDS),
        "EMBEDDING_PROVIDER":             settings.EMBEDDING_PROVIDER,
        "EMBEDDING_MODEL":                settings.EMBEDDING_MODEL,
        # EPUB injection step
        "PIPELINE_STEP_INJECT_EPUB":      str(settings.PIPELINE_STEP_INJECT_EPUB).lower(),
        "BOOK_FILES_BASE_URL":            settings.BOOK_FILES_BASE_URL,
        # Video step
        "PIPELINE_STEP_VIDEO":            str(settings.PIPELINE_STEP_VIDEO).lower(),
        "VIDEO_PROVIDER":                 settings.VIDEO_PROVIDER,
        "VIDEO_ORIENTATION":              settings.VIDEO_ORIENTATION,
        "VIDEO_FPS":                      str(settings.VIDEO_FPS),
        "VIDEO_BITRATE":                  settings.VIDEO_BITRATE,
    }


# ── In-memory cache ───────────────────────────────────────────────────────────

_cache: dict[str, str] = {}
_cache_ts: float = 0.0
_CACHE_TTL = 60.0   # seconds


async def get_all_config() -> dict[str, str]:
    """Return merged config: Supabase overrides on top of defaults."""
    global _cache, _cache_ts

    if _cache and (time.time() - _cache_ts) < _CACHE_TTL:
        return _cache

    defaults = _defaults()
    try:
        rows = await find("provider_config", select="key, value")
        db   = {r["key"]: r["value"] for r in rows}
        merged = {**defaults, **db}
    except Exception:
        merged = defaults   # graceful fallback — table may not exist yet

    _cache = merged
    _cache_ts = time.time()
    return _cache


async def get_config_value(key: str, fallback: str = "") -> str:
    cfg = await get_all_config()
    return cfg.get(key, fallback)


async def set_config_key(key: str, value: str) -> None:
    """Persist a setting to Supabase and update the in-memory cache."""
    global _cache, _cache_ts
    try:
        await upsert("provider_config", {"key": key, "value": value}, "key")
    except Exception:
        pass   # DB may not exist yet; still update the cache

    # Ensure cache is initialised before updating it
    if not _cache:
        _cache = _defaults()

    _cache[key] = value
    _cache_ts = time.time()
