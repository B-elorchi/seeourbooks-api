# See Our Book — Summarizer API

**Base URL:** `https://files.seeourbook.sa/api`

All endpoints are HTTPS. No authentication required for API calls.

---

## Endpoints

### 1. Generate Summary (Streaming)

**`POST /api/summarize`**

Generates a book summary and streams it live as Server-Sent Events (SSE). If the summary was already generated before, it returns instantly from cache.

**Request body (JSON):**
```json
{
  "book_id":  "1342",
  "length":   "10min",
  "style":    "narrative",
  "language": "en"
}
```

| Field | Required | Options | Description |
|---|---|---|---|
| `book_id` | Yes | any Gutenberg ID | Book identifier (e.g. `84` for Frankenstein) |
| `length` | No | `3min` `5min` `10min` `15min` | Target summary length (default: `5min`) |
| `style` | No | `narrative` `bullets` `academic` | Writing style (default: `narrative`) |
| `language` | No | `en` `ar` | Summary language (default: `ar`) |

**Response:** `text/event-stream` — Server-Sent Events

The response streams a series of events. Each event has the format:
```
event: <event_name>
data: <json_payload>
```

**Events:**

| Event | When | Payload |
|---|---|---|
| `cached` | Summary already exists — returns instantly | `{ "summary": "...", "word_count": 452 }` |
| `status` | Progress updates during generation | `{ "msg": "Processing 53 chunks…", "total": 53 }` |
| `chunk_done` | After each chapter is processed | `{ "index": 5, "total": 53 }` |
| `token` | Each word as the final summary is written | `{ "text": "word " }` |
| `done` | Generation complete | `{ "summary": "...", "word_count": 750, "job_id": 12 }` |
| `error` | Something went wrong | `{ "msg": "Book text not found for 99999" }` |

---

**JavaScript example (for WordPress):**

```javascript
async function generateSummary(bookId, length, language) {
  const response = await fetch('https://files.seeourbook.sa/api/summarize', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      book_id:  bookId,
      length:   length,    // '5min', '10min', etc.
      language: language,  // 'en' or 'ar'
      style:    'narrative'
    })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const messages = buffer.split('\n\n');
    buffer = messages.pop();

    for (const msg of messages) {
      const eventMatch = msg.match(/^event: (.+)/m);
      const dataMatch  = msg.match(/^data: (.+)/m);
      if (!eventMatch || !dataMatch) continue;

      const event = eventMatch[1].trim();
      const data  = JSON.parse(dataMatch[1]);

      if (event === 'cached') {
        // Summary ready instantly
        document.getElementById('summary').textContent = data.summary;
      }

      if (event === 'token') {
        // Append each word as it streams
        document.getElementById('summary').textContent += data.text;
      }

      if (event === 'chunk_done') {
        // Update progress bar
        const pct = Math.round((data.index + 1) / data.total * 100);
        document.getElementById('progress').style.width = pct + '%';
      }

      if (event === 'done') {
        // Full summary available
        console.log('Done:', data.word_count, 'words');
      }

      if (event === 'error') {
        console.error('Error:', data.msg);
      }
    }
  }
}

// Example usage
generateSummary('1342', '10min', 'en');
```

---

### 2. Get Cached Summaries for a Book

**`GET /api/summary/{book_id}`**

Returns all summaries that have already been generated for a given book.

**Example request:**
```
GET https://files.seeourbook.sa/api/summary/1342
```

**Example response:**
```json
[
  {
    "length":     "10min",
    "style":      "narrative",
    "language":   "en",
    "summary":    "Pride and Prejudice opens in the Bennet household...",
    "word_count": 1376,
    "audio_url":  "https://files.seeourbook.sa/audio/1342_en_10min.mp3",
    "created_at": "2026-05-13T10:22:00+00:00"
  }
]
```

> Use this endpoint first to check if a summary already exists before calling `/api/summarize`.

---

### 3. Check Job Status

**`GET /api/job/{job_id}`**

Returns the status of a summary generation job.

**Example request:**
```
GET https://files.seeourbook.sa/api/job/12
```

**Example response:**
```json
{
  "id":         12,
  "book_id":    "1342",
  "length":     "10min",
  "language":   "en",
  "status":     "done",
  "created_at": "2026-05-13T10:20:00+00:00"
}
```

**Status values:** `queued` · `processing` · `done` · `error`

---

### 4. Health Check

**`GET /api/health`**

Returns `{"status": "ok"}` if the service is running.

---

## Audio Files

For books that already have audio, the MP3 URL follows this pattern:

```
https://files.seeourbook.sa/audio/{book_id}_en_10min.mp3
```

**Example:**
```html
<audio controls>
  <source src="https://files.seeourbook.sa/audio/1342_en_10min.mp3" type="audio/mpeg">
</audio>
```

---

## Recommended Integration Flow

```
1. Call GET /api/summary/{book_id}
   └── If results found → display instantly, done

2. If no cached summary → Call POST /api/summarize
   ├── Listen for "chunk_done" events → update progress bar
   ├── Listen for "token" events → stream text to screen
   └── Listen for "done" event → show final summary + audio player
```

---

## Notes

- First generation takes 2–10 minutes depending on book length (average 80 chapters)
- Second request for the same book/length/language is instant (cached)
- The `audio_url` field in the summary response is populated once audio has been generated
- Book IDs are Gutenberg IDs (e.g. `84` = Frankenstein, `1342` = Pride and Prejudice, `1513` = Romeo and Juliet)
