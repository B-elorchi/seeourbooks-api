# SeeOurBook Pipeline API — Developer Guide

This is everything you need to summarize a book and pull the results.

The flow is **3 calls**:

1. **Start** a job → `POST /api/v2/pipeline/run` → you get a `job_id`.
2. **Poll** progress → `GET /api/pipeline/status/{job_id}` until `status` is `done` / `partial` / `failed`.
3. **Read** the final output → from the status response, or `GET /api/pipeline/output/{book_id}`.

Base URL: `https://<your-api-host>`  (all paths below are prefixed with `/api`).

---

## 1. Start a job

`POST /api/v2/pipeline/run`

One call does everything: it looks up the book, downloads + chunks the source EPUB/TXT if needed, then runs the pipeline in the background. It returns **immediately** with a `job_id`.

### Request body

```json
POST /api/v2/pipeline/run
{
  "book_id": "84",
  "language": "en",
  "source": "catalog",
  "steps": [
    "summarize",
    "translate",
    "audio_full",
    "audio_chapters",
    "audio_full_translate",
    "audio_chapters_translate",
    "cover",
    "alt_text",
    "mindmap",
    "mindmap_chapters",
    "mindmap_translate",
    "mindmap_chapters_translate",
    "inject_epub",
    "video"
  ],
  "options": {
    "length": "10min",
    "style": "narrative",
    "length_preset": "medium",
    "audio_style": "single"
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `book_id` | ✅ | The book id (matches the `books` table / Gutenberg id). |
| `language` | – | `"en"` or `"ar"`. Default `"en"`. |
| `source` | – | Free label stored on the job. Default `"catalog"`. |
| `steps` | – | `[]` = run **all** steps. Or pick a subset (see step list below). Dependencies are auto-added. |
| `options.length` | – | `"3min"` \| `"5min"` \| `"10min"` \| `"15min"`. |
| `options.style` | – | `"narrative"` \| `"bullets"` \| `"academic"`. |
| `options.length_preset` | – | `"small"` \| `"medium"` \| `"large"` \| `"custom"`. Overrides `options.length` char budgets. |
| `options.max_chars` | – | Required when `length_preset="custom"`. Hard cap on summary characters. |
| `options.audio_style` | – | `"single"` \| `"multi"` \| `"podcast"` \| `"audiobook"` \| `"news"` \| `"bedtime"` \| `"custom"`. Controls Gemini TTS delivery style. |

> Note: the final summary length is governed by `length_preset`/`max_chars` first, then by the legacy admin word overrides, then by `options.length`.

### Response — `202 Accepted`

```json
{
  "ok": true,
  "job_id": "9f3c2b7a1d4e4a91b0c2d3e4f5a6b7c8",
  "status": "queued",
  "status_url": "/api/pipeline/status/9f3c2b7a1d4e4a91b0c2d3e4f5a6b7c8",
  "book_id": "84",
  "title": "Frankenstein",
  "author": "Mary Wollstonecraft Shelley"
}
```

Keep the `job_id`.

---

## 2. Check progress

`GET /api/pipeline/status/{job_id}`

Poll every few seconds. While running, `result.steps` shows each step's live state and `result.current_step` shows what's running now.

### Response (running)

```json
{
  "id": "9f3c2b7a1d4e4a91b0c2d3e4f5a6b7c8",
  "book_id": "84",
  "status": "running",
  "retry_count": 0,
  "result": {
    "status": "running",
    "current_step": "audio_full",
    "running_steps": ["audio_full", "mindmap"],
    "steps": {
      "summarize": "done",
      "cover": "done",
      "audio_full": "running",
      "audio_chapters": "pending",
      "mindmap": "running",
      "mindmap_chapters": "pending",
      "alt_text": "done",
      "inject_epub": "pending",
      "video": "skipped"
    }
  }
}
```

### `status` values

| Status | Meaning |
|--------|---------|
| `queued` | Accepted, not started yet (or ingesting the source file). |
| `running` | In progress — check `result.steps`. |
| `done` | All requested steps succeeded. |
| `partial` | Some steps succeeded, some failed (see `result.errors`). |
| `failed` | Nothing usable produced. |
| `cancelled` | Stopped by an admin. |

### Per-step status (`result.steps[*]`)

`pending` → not started · `running` → in progress · `done` → ok · `partial` → some sub-items failed · `failed` → errored · `skipped` → not requested.

Stop polling when top-level `status` is `done`, `partial`, or `failed`.

---

## 3. Get the output

When finished, the full output is the `result` object in the status response. You can also fetch the latest completed result by book:

`GET /api/pipeline/output/{book_id}`

### Example output (`result`)

```json
{
  "book_id": "84",
  "status": "done",
  "generated_at": "2026-06-10T12:34:56Z",
  "processing_time": "3m 12s",

  "metadata": {
    "title": "Frankenstein",
    "author": "Mary Wollstonecraft Shelley",
    "cover_url": "https://cdn.../books/84/cover.jpg",
    "cover_alt_text": "A dark gothic laboratory under stormy skies…"
  },

  "quick_summary": "Mary Shelley's Frankenstein follows…",

  "summary_qa": {
    "score": 88,
    "passed": true,
    "threshold": 70,
    "missing": [],
    "model": "deepseek/deepseek-chat"
  },

  "summaries": {
    "10min_en": { "text": "Full English summary…", "word_count": 3950, "language": "en" },
    "10min_ar": { "text": "الملخص العربي الكامل…", "word_count": 3700, "language": "ar", "translated": true }
  },

  "audio": {
    "full_en": { "url": "https://cdn.../books/84/audio_en_10min.mp3", "duration": "26:41", "size_mb": 24.3 },
    "full_ar": { "url": "https://cdn.../books/84/audio_ar_10min.mp3", "duration": "28:10", "size_mb": 25.8 }
  },

  "mindmap": { "url": "https://cdn.../books/84/mindmap.svg" },

  "epub": { "enriched_en": { "url": "https://cdn.../books/84/84_en.epub" } },

  "chapters": [
    {
      "index": 1,
      "title": "Chapter I",
      "summary": "Walton writes to his sister…",
      "read_time_min": 2,
      "audio_en": "https://cdn.../books/84/chapters/ch_01_en.mp3",
      "mindmap_url": "https://cdn.../books/84/chapters/ch_01_mindmap.svg"
    }
  ],

  "files": {
    "cover": "https://cdn.../cover.jpg",
    "audio_full": "https://cdn.../audio_en_10min.mp3",
    "mindmap": "https://cdn.../mindmap.svg",
    "epub": "https://cdn.../84_en.epub",
    "video": null,
    "chapters": [
      { "index": 1, "title": "Chapter I", "audio_url": "…", "mindmap_url": "…" }
    ]
  },

  "cost": { "total_usd": 0.1392, "calls": 13, "by_provider": { "deepgram": 0.132, "openrouter": 0.007 } },

  "errors": {}
}
```

### Where to read each asset

| You want | Read from |
|----------|-----------|
| English summary text | `summaries["{length}_en"].text` |
| Arabic summary text | `summaries["{length}_ar"].text` |
| Full audio (per language) | `audio["full_en"].url` / `audio["full_ar"].url` |
| Cover image | `metadata.cover_url` (or `files.cover`) |
| Mind map | `mindmap.url` (or `files.mindmap`) |
| Enriched EPUB | `files.epub` |
| Per-chapter summary / audio / mindmap | `chapters[].summary` / `chapters[].audio_en` / `chapters[].mindmap_url` |
| Coverage QA score | `summary_qa.score` (0–100) |

---

## Pipeline steps

| Step | Produces | Depends on |
|------|----------|-----------|
| `summarize` | Full + per-chapter summaries | — |
| `translate` | Summary translated to the other language (EN↔AR) | `summarize` |
| `cover` | AI cover image | — |
| `alt_text` | Cover alt text (accessibility/SEO) | `cover` |
| `audio_full` | One MP3 of the full summary | `summarize` |
| `audio_chapters` | One MP3 per chapter | `summarize` |
| `audio_full_translate` | MP3 of the translated full summary | `translate` |
| `audio_chapters_translate` | MP3 per chapter from the translated summary | `translate`, `audio_chapters` |
| `mindmap` | Book mind map (SVG/JSON) | `summarize` |
| `mindmap_chapters` | Mind map per chapter | `summarize` |
| `mindmap_translate` | Mind map from the translated summary | `translate` |
| `mindmap_chapters_translate` | Mind map per chapter from the translated summary | `translate`, `mindmap_chapters` |
| `inject_epub` | Enriched EPUB (cover + summary + per-chapter insights) | `summarize` |
| `video` | Summary video | `summarize`, `audio_full` |

Send `"steps": []` to run them all. To run just a few, e.g. `"steps": ["summarize", "audio_full", "cover"]` — dependencies are added automatically.

> A summary **coverage check** (DeepSeek by default) scores how well the summary covers the whole book. If the score is below the admin threshold (default **70%**), `audio_full` / `audio_chapters` are blocked and reported in `errors` — improve the summary and retry.

---

## Quick examples

### cURL

```bash
# 1. start
curl -X POST https://<host>/api/v2/pipeline/run \
  -H "Content-Type: application/json" \
  -d '{
    "book_id":"84",
    "language":"en",
    "source":"catalog",
    "steps":[],
    "options":{
      "length":"10min",
      "style":"narrative",
      "length_preset":"medium",
      "audio_style":"podcast"
    }
  }'

# 2. poll  (repeat until status is done/partial/failed)
curl https://<host>/api/pipeline/status/<job_id>

# 3. output by book
curl https://<host>/api/pipeline/output/84
```

### JavaScript

```js
const host = "https://<host>";

// 1. start
const start = await fetch(`${host}/api/v2/pipeline/run`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    book_id: "84",
    language: "en",
    source: "catalog",
    steps: [],
    options: { length: "10min", style: "narrative", length_preset: "medium", audio_style: "podcast" },
  }),
}).then(r => r.json());

const jobId = start.job_id;

// 2. poll
async function waitForJob(id) {
  while (true) {
    const job = await fetch(`${host}/api/pipeline/status/${id}`).then(r => r.json());
    if (["done", "partial", "failed"].includes(job.status)) return job;
    await new Promise(r => setTimeout(r, 4000)); // 4s
  }
}

const job = await waitForJob(jobId);

// 3. use the output
const r = job.result;
console.log("EN summary:", r.summaries["10min_en"].text);
console.log("EN audio:",   r.audio["full_en"].url);
console.log("Cover:",      r.metadata.cover_url);
console.log("EPUB:",       r.files.epub);
```

---

## Other useful endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/pipeline/jobs?limit=50` | List recent jobs (newest first). |
| `GET` | `/api/pipeline/status/{job_id}` | Single job + live result. |
| `GET` | `/api/pipeline/output/{book_id}` | Latest completed result for a book. |
| `POST` | `/api/pipeline/jobs/{job_id}/cancel` | Request cancellation. |

## Common errors

| HTTP | When | Fix |
|------|------|-----|
| `404` (status) | Unknown `job_id` | Use the id returned from `/run`. |
| `404` (output) | No completed job for that book yet | Poll the job first. |
| `422` | Bad `steps` / missing `book_id` | Check the request body. |
| `result.errors.ingest` | Source EPUB/TXT not found for the book | Verify the book id + language, or upload the file. |







add whatermark in audios and images and mindmaps (json , mermaid)
add api key in the header for all the requests 
create new tables in supabase (user, permissions, api_keys) and link them to the existing jobs and books tables
add users and permissions  management (admin, editor, viewer) 
add table user_books to link users to uploaded books by user
add api key for each user and role
when book in arabic language the summary and audio should be in arabic + english and if the book in english the summary and audio should be in english + arabic
add cost per book and per user 


add ai check for summaries quality and coverage (e.g. using DeepSeek) and block audio generation if the score is below a certain threshold (e.g. 80%) — report the score and missing coverage areas in the output for debugging/improvement.


setup server by web proxy (caddy or nginx)
install apps (anything LLM , Qdrant )



options for future:

add user sold to upload her documents and books and get summaries and audio and mindmaps
add easyocr and tesseract for ocr and text extraction from images and books contains images
https://github.com/jaidedai/easyocr
https://github.com/tesseract-ocr/tesseract
