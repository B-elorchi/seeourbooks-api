import json, os
from pathlib import Path
from typing import AsyncIterator

import anthropic
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Config (all values loaded from .env) ──────────────────────────────────────
SUPA_URL = os.environ["SUPABASE_URL"]          # https://your-project.supabase.co
SUPA_KEY = os.environ["SUPABASE_SERVICE_KEY"]  # service_role key
ANT_KEY  = os.environ["ANTHROPIC_API_KEY"]     # sk-ant-...
TEXT_DIR = Path(os.environ.get("TEXT_DIR", "/path/to/text/files"))

SUPA_HDR = {"apikey": SUPA_KEY, "Authorization": f"Bearer {SUPA_KEY}", "Content-Type": "application/json"}

# ── Models ────────────────────────────────────────────────────────────────────
# Haiku  → fast, cheap per-chapter summaries
# Sonnet → high-quality final summary
# Opus   → Arabic tashkeel (diacritics) pass for accurate TTS pronunciation
HAIKU  = "claude-haiku-4-5-20251001"
SONNET = "claude-sonnet-4-6"
OPUS   = "claude-opus-4-7"

WORDS   = {"3min": 450, "5min": 750, "10min": 1500, "15min": 2250}
CHUNK_W = 1500  # words per chunk


class SumReq(BaseModel):
    book_id:  str
    length:   str = "5min"    # 3min | 5min | 10min | 15min
    style:    str = "narrative"  # narrative | bullets | academic
    language: str = "en"     # en | ar


# ── Supabase REST helpers ─────────────────────────────────────────────────────

async def sg(client, path):
    """GET from Supabase REST."""
    r = await client.get(f"{SUPA_URL}/rest/v1/{path}", headers=SUPA_HDR)
    r.raise_for_status()
    return r.json()

async def sp(client, path, body):
    """POST to Supabase REST — returns inserted row."""
    h = {**SUPA_HDR, "Prefer": "return=representation"}
    r = await client.post(f"{SUPA_URL}/rest/v1/{path}", headers=h, json=body)
    r.raise_for_status()
    return r.json()

async def su(client, path, body, conflict):
    """Upsert to Supabase REST (merge on conflict)."""
    h = {**SUPA_HDR, "Prefer": "resolution=merge-duplicates,return=representation"}
    r = await client.post(f"{SUPA_URL}/rest/v1/{path}?on_conflict={conflict}", headers=h, json=body)
    r.raise_for_status()
    return r.json()

async def spatch(client, path, body):
    """PATCH a Supabase row."""
    r = await client.patch(f"{SUPA_URL}/rest/v1/{path}", headers=SUPA_HDR, json=body)
    r.raise_for_status()


# ── SSE helper ────────────────────────────────────────────────────────────────

def sse(ev, data):
    """Format a Server-Sent Event message."""
    return f"event: {ev}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── Text helpers ──────────────────────────────────────────────────────────────

def find_book_text(book_id: str) -> str | None:
    """Look for book text file in english/ or arabic/ subfolder."""
    for subdir in ["english", "arabic", ""]:
        for ext in [".txt", ".md"]:
            p = (TEXT_DIR / subdir / f"{book_id}{ext}") if subdir else (TEXT_DIR / f"{book_id}{ext}")
            if p.exists():
                return p.read_text(encoding="utf-8")
    return None


def chunk_text(text, max_words=CHUNK_W):
    """Split text into ~max_words chunks, breaking at sentence boundaries."""
    words, chunks = text.split(), []
    while words:
        part = " ".join(words[:max_words])
        words = words[max_words:]
        if words:
            cut = part.rfind(". ")
            if cut > len(part) * 0.7:
                words = part[cut + 2:].split() + words
                part = part[:cut + 1]
        chunks.append(part.strip())
    return [c for c in chunks if c]


# ── Main pipeline ─────────────────────────────────────────────────────────────

async def run(req: SumReq) -> AsyncIterator[str]:
    """
    Full summary pipeline — streams SSE events to the client.

    Flow:
      1. Cache check   → return instantly if already generated
      2. Create job    → track progress in summary_jobs table
      3. Load chunks   → from DB if already chunked, else from text file
      4. Haiku pass    → summarize each chunk (cached per language)
      5. Sonnet pass   → write final cohesive summary (streaming)
      6. Opus pass     → Arabic only: add full tashkeel for TTS accuracy
      7. Cache result  → save to book_summaries, mark job done
    """
    async with httpx.AsyncClient(timeout=180) as client:

        # ── 1. Cache check ────────────────────────────────────────────────────
        rows = await sg(client,
            f"book_summaries?book_id=eq.{req.book_id}"
            f"&length=eq.{req.length}&style=eq.{req.style}&language=eq.{req.language}"
            f"&select=summary,word_count&limit=1")
        if rows:
            yield sse("cached", {"summary": rows[0]["summary"], "word_count": rows[0]["word_count"]})
            return

        yield sse("status", {"msg": "Starting…"})

        # ── 2. Ensure book exists in catalog (FK requirement) ─────────────────
        await su(client, "books",
            {"book_id": req.book_id, "title": req.book_id, "status": "pending"}, "book_id")

        # ── 3. Create or reuse job ────────────────────────────────────────────
        existing = await sg(client,
            f"summary_jobs?book_id=eq.{req.book_id}"
            f"&length=eq.{req.length}&style=eq.{req.style}&language=eq.{req.language}"
            f"&status=in.(queued,processing)&limit=1")
        if existing:
            job_id = existing[0]["id"]
        else:
            j = await sp(client, "summary_jobs",
                {"book_id": req.book_id, "length": req.length,
                 "style": req.style, "language": req.language, "status": "processing"})
            job_id = (j[0] if isinstance(j, list) else j)["id"]

        await spatch(client, f"summary_jobs?id=eq.{job_id}", {"status": "processing"})

        # ── 4. Load chunks ────────────────────────────────────────────────────
        # Chunks are stored in DB after the first request — reused for all
        # subsequent summary lengths/styles, saving re-processing costs.
        chunks = await sg(client,
            f"chunks?book_id=eq.{req.book_id}&order=chunk_index.asc&select=id,chunk_index,content")
        if not chunks:
            yield sse("status", {"msg": "Loading book text…"})
            text = find_book_text(req.book_id)
            if not text:
                yield sse("error", {"msg": f"Book text not found for {req.book_id}"})
                await spatch(client, f"summary_jobs?id=eq.{job_id}",
                    {"status": "error", "error_msg": "text not found"})
                return
            parts = chunk_text(text)
            yield sse("status", {"msg": f"Chunked into {len(parts)} parts", "total": len(parts)})
            chunks = []
            for i, content in enumerate(parts):
                row = await su(client, "chunks",
                    {"book_id": req.book_id, "chunk_index": i,
                     "content": content, "token_count": len(content.split())},
                    "book_id,chunk_index")
                chunks.append(row[0] if isinstance(row, list) else row)

        total = len(chunks)
        yield sse("status", {"msg": f"Processing {total} chunks…", "total": total})

        # ── 5. Haiku pass — chunk summaries ───────────────────────────────────
        # Each chunk summary is cached per (chunk_id, language).
        # If user requests a different length later, Haiku is not called again.
        ai = anthropic.AsyncAnthropic(api_key=ANT_KEY)
        chunk_sums = []
        lang_name = "Arabic" if req.language == "ar" else "English"

        ar_tashkeel = (
            "\n\nمهم جداً: اكتب النص العربي بالتشكيل الكامل (الفتحة والضمة والكسرة والسكون والشدة والتنوين) "
            "على كل كلمة، وبشكل خاص على أواخر الكلمات (الإعراب) وعلى الكلمات التي تحتمل أكثر من قراءة. "
            "النص سيُستخدم مباشرةً في تحويل النص إلى صوت (TTS) لذا الدقة في النطق ضرورية."
        ) if req.language == "ar" else ""

        for chunk in chunks:
            cid, idx = chunk["id"], chunk["chunk_index"]
            yield ": keepalive\n\n"  # SSE comment — keeps connection alive through Nginx
            ex = await sg(client,
                f"chunk_summaries?chunk_id=eq.{cid}&language=eq.{req.language}&limit=1&select=summary")
            if ex:
                chunk_sums.append(ex[0]["summary"])
                yield sse("chunk_done", {"index": idx, "total": total})
                continue
            msg = await ai.messages.create(
                model=HAIKU, max_tokens=512,
                messages=[{"role": "user", "content":
                    f"Summarize this book excerpt in {lang_name} in 3-5 sentences. "
                    f"Focus on key ideas.{ar_tashkeel}\n\n{chunk['content']}"}])
            s = msg.content[0].text
            await su(client, "chunk_summaries",
                {"book_id": req.book_id, "chunk_id": cid, "chunk_index": idx,
                 "language": req.language, "summary": s, "model": HAIKU},
                "chunk_id,language")
            chunk_sums.append(s)
            yield sse("chunk_done", {"index": idx, "total": total})

        # ── 6. Sonnet pass — final summary (streaming) ────────────────────────
        yield sse("status", {"msg": "Generating final summary…"})
        target = WORDS.get(req.length, 750)
        style_map = {
            "narrative": "flowing narrative prose",
            "bullets":   "clear bullet points",
            "academic":  "formal academic style",
        }
        combined = "\n\n---\n\n".join(f"Section {i+1}:\n{s}" for i, s in enumerate(chunk_sums))
        prompt = (
            f"Create a book summary in {lang_name} for an audio presentation "
            f"of ~{req.length} (~{target} words).\n"
            f"Style: {style_map.get(req.style, 'narrative prose')}.\n"
            f"Based on these section summaries:\n\n{combined}\n\n"
            f"Write the complete summary in {lang_name}. Target: ~{target} words."
            f"{ar_tashkeel}"
        )
        full = ""
        async with ai.messages.stream(
            model=SONNET, max_tokens=target * 2,
            messages=[{"role": "user", "content": prompt}]
        ) as stream:
            async for tok in stream.text_stream:
                full += tok
                yield sse("token", {"text": tok})

        # ── 7. Opus pass — Arabic tashkeel ────────────────────────────────────
        # Arabic TTS mispronounces words without diacritics.
        # Opus rewrites the summary with full tashkeel on every letter.
        if req.language == "ar":
            yield sse("status", {"msg": "Applying full tashkeel…"})
            tashkeel_prompt = (
                "أنت متخصص في اللغة العربية الفصحى وعلم التشكيل. مهمتك إضافة التشكيل الكامل على النص التالي.\n\n"
                "القواعد الصارمة:\n"
                "١. ضَعْ حَرَكَةً على كُلِّ حَرْفٍ في كُلِّ كَلِمَة.\n"
                "٢. أضِفِ الشَّدَّة على كل حرف مُشَدَّد.\n"
                "٣. أضِفِ التَّنْوِين على نهايات الكلمات المُنَوَّنَة.\n"
                "٤. لا تُغَيِّر أي كلمة — فقط أضِفِ التشكيل.\n\n"
                f"النص:\n\n{full}"
            )
            tashkeel_msg = await ai.messages.create(
                model=OPUS, max_tokens=int(target * 2.5),
                messages=[{"role": "user", "content": tashkeel_prompt}])
            full = tashkeel_msg.content[0].text

        # ── 8. Cache result ───────────────────────────────────────────────────
        wc = len(full.split())
        await su(client, "book_summaries",
            {"book_id": req.book_id, "length": req.length, "style": req.style,
             "language": req.language, "summary": full, "word_count": wc, "model": SONNET},
            "book_id,length,style,language")
        await spatch(client, f"summary_jobs?id=eq.{job_id}", {"status": "done"})
        yield sse("done", {"summary": full, "word_count": wc, "job_id": job_id})


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/summarize")
async def summarize(req: SumReq):
    """Generate a summary — streams SSE events live."""
    return StreamingResponse(
        run(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.get("/api/summary/{book_id}")
async def get_summaries(book_id: str):
    """Return all cached summaries for a book."""
    async with httpx.AsyncClient(timeout=30) as client:
        return await sg(client,
            f"book_summaries?book_id=eq.{book_id}"
            f"&select=length,style,language,summary,word_count,audio_url,created_at")

@app.get("/api/job/{job_id}")
async def get_job(job_id: int):
    """Return job status."""
    async with httpx.AsyncClient(timeout=30) as client:
        rows = await sg(client, f"summary_jobs?id=eq.{job_id}&select=*&limit=1")
    if not rows:
        raise HTTPException(404, "Job not found")
    return rows[0]

@app.get("/api/health")
async def health():
    return {"status": "ok"}
