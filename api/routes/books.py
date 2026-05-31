import httpx
from fastapi import APIRouter
from api.services.db.supabase import sg

router = APIRouter()


@router.get("/summary/{book_id}")
async def get_summaries(book_id: str):
    """Return all cached summaries for a book."""
    async with httpx.AsyncClient(timeout=30) as client:
        return await sg(
            client,
            f"book_summaries?book_id=eq.{book_id}"
            f"&select=length,style,language,summary,word_count,audio_url,created_at",
        )
