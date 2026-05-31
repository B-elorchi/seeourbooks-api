import httpx
from api.config.settings import settings

_HEADERS = {
    "apikey": settings.SUPABASE_SERVICE_KEY,
    "Authorization": f"Bearer {settings.SUPABASE_SERVICE_KEY}",
    "Content-Type": "application/json",
}

_BASE = f"{settings.SUPABASE_URL}/rest/v1"


async def sg(client: httpx.AsyncClient, path: str):
    """GET from Supabase REST."""
    r = await client.get(f"{_BASE}/{path}", headers=_HEADERS)
    r.raise_for_status()
    return r.json()


async def sp(client: httpx.AsyncClient, path: str, body: dict):
    """POST to Supabase REST — returns inserted row."""
    h = {**_HEADERS, "Prefer": "return=representation"}
    r = await client.post(f"{_BASE}/{path}", headers=h, json=body)
    r.raise_for_status()
    return r.json()


async def su(client: httpx.AsyncClient, path: str, body: dict, conflict: str):
    """Upsert to Supabase REST (merge on conflict)."""
    h = {**_HEADERS, "Prefer": "resolution=merge-duplicates,return=representation"}
    r = await client.post(f"{_BASE}/{path}?on_conflict={conflict}", headers=h, json=body)
    r.raise_for_status()
    return r.json()


async def spatch(client: httpx.AsyncClient, path: str, body: dict):
    """PATCH a Supabase row."""
    r = await client.patch(f"{_BASE}/{path}", headers=_HEADERS, json=body)
    r.raise_for_status()
