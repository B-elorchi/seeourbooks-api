"""
Provider config auto-migrations.

Runs once at server startup. Fixes ONLY undeniably broken values — values
that cause hard errors regardless of what the admin intended. We NEVER change
provider choices (deepgram ↔ elevenlabs ↔ cartesia) because the admin chose
them deliberately; changing them silently breaks the admin panel contract.

Safe to migrate:
  ✓ A value that is technically invalid (model name used as a voice UUID)
  ✓ A config key that was renamed or moved
  ✓ A literal default typo we shipped in code

NOT safe to migrate:
  ✗ Provider selection — admin chose it intentionally
  ✗ Voice IDs that might still be valid in the admin's setup
"""
import logging

from api.services.db import find, upsert
from api.services.config import runtime as _runtime

log = logging.getLogger(__name__)

_CORRECTIONS: list[dict] = [
    # ── Cover image ───────────────────────────────────────────────────────────
    # openai/* prefix on image models routes to OpenRouter which returns HTML 404.
    # Strip the prefix — cover.py sends these to native OpenAI directly.
    {
        "key":       "IMAGE_MODEL",
        "old_value": "openai/dall-e-3",
        "new_value": "dall-e-3",
        "reason":    "openai/dall-e-3 via OpenRouter returns 404; stripping prefix for native OpenAI",
    },
    {
        "key":       "IMAGE_MODEL",
        "old_value": "openai/dall-e-2",
        "new_value": "dall-e-2",
        "reason":    "openai/dall-e-2 via OpenRouter returns 404; stripping prefix for native OpenAI",
    },
    {
        "key":       "IMAGE_MODEL",
        "old_value": "openai/gpt-image-1",
        "new_value": "gpt-image-1",
        "reason":    "openai/gpt-image-1 via OpenRouter returns 404; stripping prefix for native OpenAI",
    },
    # gpt-image-1 dropped 1024x1792 — it was never valid for that model.
    {
        "key":       "IMAGE_SIZE",
        "old_value": "1024x1792",
        "new_value": "1024x1536",
        "reason":    "1024x1792 is not a valid gpt-image-1 size; remapped to nearest portrait 1024x1536",
    },

    # ── Arabic TTS voice ──────────────────────────────────────────────────────
    # sonic-2024-10-19 is a Cartesia MODEL name, never a voice UUID.
    # Passing it as a voice ID always causes a 400 error regardless of setup.
    {
        "key":       "TTS_VOICE_AR",
        "old_value": "sonic-2024-10-19",
        "new_value": "",
        "reason":    "sonic-2024-10-19 is a Cartesia model name, not a voice UUID — always causes 400",
    },
    # ── Cartesia model ────────────────────────────────────────────────────────
    # sonic-2024-10-19 is the legacy English-only snapshot. It returns 400
    # "Invalid language for model" on any non-English transcript. The newer
    # sonic-3.5 line supports 40+ languages including Arabic.
    {
        "key":       "CARTESIA_MODEL",
        "old_value": "sonic-2024-10-19",
        "new_value": "sonic-3.5-2026-05-04",
        "reason":    "sonic-2024-10-19 is English-only; upgrading to sonic-3.5 multilingual snapshot",
    },

    # ── Book files CDN host ───────────────────────────────────────────────────
    # The production CDN lives at files.seeourbook.sa. The .com host does not
    # serve the EPUB/TXT files and returns 404, breaking ingest + epub injection.
    {
        "key":       "BOOK_FILES_BASE_URL",
        "old_value": "https://files.seeourbook.com",
        "new_value": "https://files.seeourbook.sa",
        "reason":    ".com host does not serve book files (404); correct CDN is files.seeourbook.sa",
    },
    {
        "key":       "BOOK_FILES_BASE_URL",
        "old_value": "https://files.seeourbook.com/",
        "new_value": "https://files.seeourbook.sa",
        "reason":    ".com host does not serve book files (404); correct CDN is files.seeourbook.sa",
    },
]


async def run_migrations() -> None:
    """Apply all pending corrections to provider_config. Called once at startup."""
    try:
        rows = await find("provider_config", select="key, value")
    except Exception as exc:
        log.warning("Config migrations skipped — could not read provider_config: %s", exc)
        return

    current: dict[str, str] = {r["key"]: r["value"] for r in rows}

    applied = False
    for m in _CORRECTIONS:
        key       = m["key"]
        old_value = m["old_value"]
        new_value = m["new_value"]
        reason    = m["reason"]

        if current.get(key) == old_value:
            try:
                await upsert("provider_config", {"key": key, "value": new_value}, "key")
                log.info("Migration applied [%s]: %r → %r  (%s)", key, old_value, new_value, reason)
                applied = True
            except Exception as exc:
                log.warning("Migration failed [%s]: %s", key, exc)

    # Bust the in-memory cache so corrected values take effect immediately
    if applied:
        _runtime._cache.clear()
        _runtime._cache_ts = 0.0
