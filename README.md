# See Our Book — Book Summarizer API

An on-demand AI pipeline that generates book summaries and streams them live to the user. Built with FastAPI, Claude (Anthropic), and Supabase.

---

## How It Works

```
User Request
    │
    ▼
Cache check (book_summaries table)
    │ hit → return instantly (~200ms)
    │ miss ↓
    ▼
Load book text → split into ~1,500-word chunks → save to DB
    │
    ▼
Claude Haiku → summarize each chunk (cached per language)
    │
    ▼
Claude Sonnet → write final cohesive summary (streamed live)
    │
    ▼ (Arabic only)
Claude Opus → add full tashkeel (diacritics) for TTS accuracy
    │
    ▼
Save to book_summaries → return to user
```

### Caching layers

| Layer | Reused when |
|---|---|
| `chunks` | Same book, any future request |
| `chunk_summaries` (Haiku) | Same book + same language |
| `book_summaries` (final) | Same book + same length + style + language → instant |

This means: the first request for a book takes ~5–10 minutes. Every request after that for the same combination is **instant**.

---

## Project Structure

```
api/
  main.py            ← FastAPI app — full pipeline with SSE streaming
  requirements.txt   ← Python dependencies
  .env.example       ← Environment variables template

infra/
  summarizer.service ← systemd service definition
  nginx_api.conf     ← Nginx proxy config for /api/

API_DOCS.md          ← Full endpoint documentation with examples
```

---

## Database Schema (Supabase)

Five tables:

| Table | Purpose |
|---|---|
| `books` | Book catalog — title, author, language |
| `chunks` | Raw text chunks per book (~1,500 words each) |
| `chunk_summaries` | Haiku summary per chunk, per language |
| `book_summaries` | Cached final summaries — unique per `(book_id, length, style, language)` |
| `summary_jobs` | Request queue with status tracking |

`book_summaries` also has an `audio_url` column populated after TTS generation.

---

## API Endpoints

See [API_DOCS.md](./API_DOCS.md) for full documentation with request/response examples and a JavaScript integration snippet.

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/summarize` | Generate summary — streams SSE |
| GET | `/api/summary/{book_id}` | Get cached summaries for a book |
| GET | `/api/job/{job_id}` | Check job status |
| GET | `/api/health` | Health check |

### Summary options

| Field | Options |
|---|---|
| `length` | `3min` · `5min` · `10min` · `15min` |
| `style` | `narrative` · `bullets` · `academic` |
| `language` | `en` · `ar` |

### SSE events streamed

| Event | Description |
|---|---|
| `cached` | Instant return from cache |
| `status` | Progress updates |
| `chunk_done` | After each chapter is processed |
| `token` | Each word as Sonnet writes it |
| `done` | Generation complete |
| `error` | Something went wrong |

---

## Setup

### 1. Install dependencies

```bash
cd api
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your Supabase URL, service key, and Anthropic API key
```

### 3. Run the service

```bash
uvicorn main:app --host 127.0.0.1 --port 8080 --workers 2
```

Or with systemd:
```bash
cp infra/summarizer.service /etc/systemd/system/
systemctl enable summarizer
systemctl start summarizer
```

### 4. Configure Nginx

Add the contents of `infra/nginx_api.conf` inside your existing server block, then reload Nginx.

The key setting is `proxy_buffering off` — **required** for SSE to stream in real time.

---

## Models Used

| Step | Model | Why |
|---|---|---|
| Chapter summaries | `claude-haiku-4-5` | Fast and cheap — called once per chapter |
| Final summary | `claude-sonnet-4-6` | High quality narrative output |
| Arabic tashkeel | `claude-opus-4-7` | Best accuracy for full diacritics |

---

## Tech Stack

- **FastAPI** — async Python web framework
- **Anthropic Claude** — Haiku, Sonnet, Opus via official Python SDK
- **Supabase** — PostgreSQL database accessed via REST API
- **Nginx** — reverse proxy with SSE streaming support
- **systemd** — service management on Ubuntu
