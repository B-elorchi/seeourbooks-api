import os
import redis.asyncio as redis
import logging

log = logging.getLogger(__name__)

# Use the same REDIS_URL as celery
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Create a global connection pool
_redis_client = None

def get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


async def get_chunk_summary_cache(chunk_id: str, language: str) -> str | None:
    """Retrieve a cached chunk summary from Redis."""
    try:
        client = get_redis()
        key = f"chunk_summary:{chunk_id}:{language}"
        summary = await client.get(key)
        if summary:
            return summary
    except Exception as exc:
        log.warning("Failed to read from Redis cache for %s: %s", chunk_id, exc)
    return None


async def set_chunk_summary_cache(chunk_id: str, language: str, summary: str, model: str) -> None:
    """Store a chunk summary in Redis cache for 24 hours."""
    try:
        client = get_redis()
        key = f"chunk_summary:{chunk_id}:{language}"
        # Cache for 24 hours (86400 seconds) - enough for any pipeline run
        await client.setex(key, 86400, summary)
    except Exception as exc:
        log.warning("Failed to write to Redis cache for %s: %s", chunk_id, exc)
