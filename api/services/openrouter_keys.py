"""
OpenRouter API-key rotation.

Reads a list of keys from the environment and rotates to the next key when
OpenRouter reports low credits / limit exceeded (HTTP 402/403 with a credit
message, or HTTP 429).  This lets a pipeline job survive a single key running
out of quota by falling back to spare keys from .env.

Env sources (in order of preference):
  1. OPENROUTER_API_KEYS=key1,key2,key3   (comma-separated)
  2. OPENROUTER_API_KEY_1, OPENROUTER_API_KEY_2, ...   (indexed)
  3. OPENROUTER_API_KEY                   (single key)
"""
import logging
import os
from typing import Iterable

import httpx

from api.config.settings import settings

log = logging.getLogger(__name__)


def _load_keys() -> list[str]:
    """Build the key list from environment variables / settings."""
    keys: list[str] = []

    # 1. Explicit comma-separated list (highest precedence)
    raw = os.getenv("OPENROUTER_API_KEYS", "").strip()
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys

    # 2. Named keys from settings / env: OPENROUTER_API_KEY, _2, _3
    for k in (
        settings.OPENROUTER_API_KEY,
        settings.OPENROUTER_API_KEY_2,
        settings.OPENROUTER_API_KEY_3,
    ):
        if k and str(k).strip():
            keys.append(str(k).strip())
    if keys:
        return keys

    # 3. Indexed env vars OPENROUTER_API_KEY_1, _2, ...
    i = 1
    while True:
        k = os.getenv(f"OPENROUTER_API_KEY_{i}", "").strip()
        if not k:
            break
        keys.append(k)
        i += 1
    if keys:
        return keys

    return []


class OpenRouterKeyManager:
    """Round-robin key selector with per-key exhaustion tracking."""

    def __init__(self, keys: Iterable[str] | None = None) -> None:
        self._keys = list(keys or _load_keys())
        self._exhausted: set[int] = set()
        self._current_index = 0
        if not self._keys:
            log.warning("No OpenRouter API keys configured.")

    @property
    def keys(self) -> list[str]:
        return list(self._keys)

    @property
    def has_keys(self) -> bool:
        return bool(self._keys)

    def current(self) -> str:
        """Return the active (first non-exhausted) key."""
        if not self._keys:
            raise RuntimeError("No OpenRouter API keys configured")
        # If current is exhausted, advance.
        if self._current_index in self._exhausted:
            self.rotate()
        return self._keys[self._current_index]

    def rotate(self, exhausted_key: str | None = None) -> str:
        """
        Mark a key as exhausted and move to the next available one.
        Returns the new active key.
        """
        if not self._keys:
            raise RuntimeError("No OpenRouter API keys configured")

        if exhausted_key:
            try:
                idx = self._keys.index(exhausted_key)
                self._exhausted.add(idx)
                log.warning("OpenRouter key %s marked exhausted (low credits / limit).", idx + 1)
            except ValueError:
                pass

        # Find next non-exhausted key
        start = self._current_index
        for _ in range(len(self._keys)):
            self._current_index = (self._current_index + 1) % len(self._keys)
            if self._current_index not in self._exhausted:
                log.info("Rotated to OpenRouter key %s.", self._current_index + 1)
                return self._keys[self._current_index]

        # All keys exhausted — reset exhaustion and cycle back to key 1.
        # This lets the manager keep trying instead of permanently deadlocking;
        # the next call will fail with the provider's real error if no key works.
        log.error("All OpenRouter keys are exhausted; resetting exhaustion state.")
        self._exhausted.clear()
        self._current_index = 0
        return self._keys[0]

    def reset(self) -> None:
        """Clear exhaustion state (useful after admin adds credits)."""
        self._exhausted.clear()
        self._current_index = 0


# Module-level singleton — all OpenRouter callers share rotation state.
_key_manager = OpenRouterKeyManager()


def get_openrouter_key() -> str:
    return _key_manager.current()


def rotate_openrouter_key(exhausted_key: str | None = None) -> str:
    return _key_manager.rotate(exhausted_key)


def reset_openrouter_keys() -> None:
    _key_manager.reset()


def openrouter_key_count() -> int:
    return len(_key_manager.keys)


def is_credit_error(status_code: int | None, body: str) -> bool:
    """Heuristic: does this OpenRouter response mean the key is out of credits?"""
    if status_code in (402, 403, 429):
        text = (body or "").lower()
        if any(k in text for k in (
            "limit", "credits", "insufficient", "exceeded", "quota",
            "low credits", "key limit", "rate limit",
        )):
            return True
    return False


async def openrouter_key_has_credits(key: str | None = None) -> bool:
    """
    Ask OpenRouter whether the given key (or the current active key) still has
    credits / spending headroom.  Returns True when the key is usable, False
    when it is exhausted, invalid, or the check itself fails.
    """
    if not key:
        try:
            key = get_openrouter_key()
        except RuntimeError:
            return False

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {key}"},
            )
        if r.status_code >= 400:
            return False

        body = r.json()
        data = body.get("data") or {}

        # Try a few known shapes for limit / usage / remaining credits.
        limit = data.get("limit") or data.get("credit_limit") or data.get("total_credits")
        usage = data.get("usage") or data.get("credit_usage") or data.get("total_usage") or 0
        remaining = data.get("remaining") or data.get("credits") or data.get("credit_remaining")

        if remaining is not None:
            return float(remaining) > 0
        if limit is not None:
            return float(limit) - float(usage) > 0

        # No explicit limit → treat as usable (e.g. unlimited/key has credits).
        return True
    except Exception as exc:
        log.warning("Could not check OpenRouter credits: %s", exc)
        return False
