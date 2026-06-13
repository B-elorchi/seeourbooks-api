"""
Runtime configuration — reads provider settings from Supabase provider_config table.
Falls back to hard-coded settings defaults when the table doesn't exist yet.
In-memory cache refreshes every 60 seconds so live admin changes take effect quickly.
"""
import time

from api.config.settings import settings
from api.services.db import find, upsert

# ── Default prompt templates (editable via admin panel) ───────────────────────

PROMPT_COVER_DEFAULT = (
    "Design a professional, bookstore-quality book cover for the book titled "
    "\"{title}\" written by {author}.\n"
    "\n"
    "BOOK DETAILS:\n"
    "{details}\n"
    "\n"
    "WHAT THE BOOK IS ABOUT:\n"
    "{summary}\n"
    "\n"
    "VISUAL DIRECTION:\n"
    "- The illustration MUST visually represent the book's actual story, themes, and mood — "
    "use concrete symbols, characters, settings, or objects from the content above.\n"
    "- Reflect the {genre_hint} genre and emotional tone of the book.\n"
    "- Modern, premium publishing aesthetic — cinematic composition, rich atmospheric lighting, "
    "considered color palette that matches the book's mood.\n"
    "- Portrait orientation, single strong focal subject, balanced negative space in the upper "
    "third (where the title sits) and the lower strip (where the author name sits).\n"
    "- Photorealistic or high-quality illustrated style — NOT cartoon, NOT clip art.\n"
    "\n"
    "TEXT ON THE COVER (CRITICAL — render this text directly onto the image):\n"
    "- Render the TITLE prominently across the upper portion of the cover:\n"
    "      {title}\n"
    "  Use a large, bold, elegant serif or modern display typeface. The title must be the\n"
    "  largest text element, perfectly readable, with high contrast against the background.\n"
    "- Render the AUTHOR NAME in a smaller, refined typeface at the BOTTOM of the cover:\n"
    "      {author}\n"
    "  Author name should be roughly 25–35% the size of the title, centered, with letter-spacing\n"
    "  suitable for a professional cover.\n"
    "- Spell the title and author name EXACTLY as written above — every letter, accent, and word.\n"
    "- Do not invent any additional text, taglines, blurbs, publisher logos, or barcodes.\n"
    "\n"
    "STRICT RULES:\n"
    "- ONLY the title and author name appear as text — no other words, numbers, or labels.\n"
    "- NO watermarks, NO website URLs, NO frames around the image.\n"
    "- The final result must look like a finished, printable bookstore cover."
)

PROMPT_MINDMAP_MERMAID_DEFAULT = (
    "Create a Mermaid mind map diagram (graph TD) for the book '{title}'.\n"
    "Based on this summary:\n\n{summary}\n\n"
    "STRICT SYNTAX RULES (failure to follow these breaks the renderer):\n"
    "- First line MUST be exactly: graph TD\n"
    "- EVERY node MUST use the form  ID[Label]  — single-token ID followed by [bracketed label]\n"
    "- Edges MUST be:  A[Label] --> B[Label]   (with the brackets)\n"
    "- IDs are short alphanumeric tokens with no spaces (A, B, C, A1, B2, ROOT, etc.)\n"
    "- Labels go INSIDE the [] brackets, can have spaces, must be 2-4 English words\n"
    "- NO quotes, NO parentheses, NO Arabic, NO commas, NO special characters in labels\n"
    "- 5-7 main topic nodes branching from a single root\n"
    "- Each main node has 2-3 sub-nodes\n"
    "- Output ONLY the Mermaid code, no markdown fences, no explanation\n"
    "\n"
    "CORRECT example:\n"
    "  graph TD\n"
    "    ROOT[Atomic Habits] --> A[Small Changes]\n"
    "    ROOT --> B[Habit Loop]\n"
    "    A --> A1[Compound Effect]\n"
    "\n"
    "WRONG (do not do this):\n"
    "  Atomic Habits --> Small Changes      <-- NO brackets, will fail\n"
    "{lang_note}"
)

PROMPT_MINDMAP_JSON_DEFAULT = (
    "Create a mind map in JSON format for the book '{title}'.\n"
    "Based on this summary:\n\n{summary}\n\n"
    "Return ONLY valid JSON matching this exact structure:\n"
    '{{\n'
    '  "center_node": {{\n'
    '    "text": "<book title>",\n'
    '    "branches": [\n'
    '      {{"category": "Characters", "color": "orange", "sub_nodes": ["item1", "item2", "item3"]}},\n'
    '      {{"category": "Similar",    "color": "red",    "sub_nodes": ["item1", "item2", "item3"]}},\n'
    '      {{"category": "Impact",     "color": "green",  "sub_nodes": ["item1", "item2", "item3"]}},\n'
    '      {{"category": "Background", "color": "pink",   "sub_nodes": ["item1", "item2", "item3"]}},\n'
    '      {{"category": "Author",     "color": "blue",   "sub_nodes": ["item1", "item2", "item3"]}},\n'
    '      {{"category": "Quotations", "color": "purple", "sub_nodes": ["item1", "item2", "item3"]}}\n'
    '    ]\n'
    '  }}\n'
    '}}\n\n'
    "RULES:\n"
    "- center_node.text must be the book's title\n"
    "- Keep the 6 branch categories and colors exactly as shown\n"
    "- Each branch must have exactly 3 concise, meaningful sub_nodes\n"
    "- Output ONLY the JSON object, no markdown fences, no explanation\n"
    "{lang_note}"
)


# ── Defaults from env / settings.py ──────────────────────────────────────────

def _defaults() -> dict[str, str]:
    return {
        "MODEL_HAIKU":                    settings.MODEL_HAIKU,
        "MODEL_SONNET":                   settings.MODEL_SONNET,
        "MODEL_OPUS":                     settings.MODEL_OPUS,
        # Per-chunk summary model (Pass 1). Defaults to OpenRouter GPT-4.1-mini
        # for cost efficiency. Supports OpenRouter prefix (openai/gpt-4.1-mini)
        # or native Anthropic models (claude-haiku-4-5-20251001).
        "MODEL_CHUNK":                    settings.MODEL_CHUNK,
        # Concurrency knobs for the parallelised steps.
        "HAIKU_CONCURRENCY":             "6",
        "MINDMAP_CONCURRENCY":           "4",
        # ── Chunking & summary length knobs (per language) ───────────────────
        # Words per ingest chunk. Bigger = fewer chunks (cheaper, less granular).
        "CHUNK_WORDS_EN":                str(settings.CHUNK_SIZE_WORDS),
        "CHUNK_WORDS_AR":                str(settings.CHUNK_SIZE_WORDS),
        # Max words for the FULL book summary. Default 4000 words per book.
        # Set 0 to fall back to the length preset (3min=450 … 15min=2250).
        "SUMMARY_MAX_WORDS_EN":          "4000",
        "SUMMARY_MAX_WORDS_AR":          "4000",
        # Max words for each per-chapter summary (Pass 1 / Haiku). 0 = default.
        "CHAPTER_SUMMARY_MAX_WORDS":     "0",
        # ── Summary quality / coverage check (gates audio generation) ────────
        # An independent model scores how well the summary covers the whole
        # book (0-100). audio_full / audio_chapters only run when the score
        # meets SUMMARY_QA_THRESHOLD. Set ENABLED=false to skip the check.
        "SUMMARY_QA_ENABLED":            "true",
        "SUMMARY_QA_MODEL":              "deepseek/deepseek-chat",
        "SUMMARY_QA_THRESHOLD":          "70",
        # ── Cross-language translation ───────────────────────────────────────
        # Always produce BOTH an English and Arabic summary. Translation is on
        # by default (required); audio in the translated/target language is
        # opt-in via TARGET_LANG_AUDIO_ENABLED.
        "TRANSLATE_SUMMARY_ENABLED":     "true",
        "TRANSLATE_MODEL":               settings.MODEL_SONNET,
        "TARGET_LANG_AUDIO_ENABLED":     "false",
        "TTS_PROVIDER_EN":                settings.TTS_PROVIDER_EN,
        "TTS_VOICE_EN":                   settings.TTS_VOICE_EN,
        "TTS_PROVIDER_AR":                settings.TTS_PROVIDER_AR,
        "TTS_VOICE_AR":                   settings.TTS_VOICE_AR,
        "CARTESIA_MODEL":                 settings.CARTESIA_MODEL,
        "CARTESIA_VOICE_EN":              "",
        "CARTESIA_VOICE_AR":              "",
        "GEMINI_TTS_MODEL":               settings.GEMINI_TTS_MODEL,
        "OPENROUTER_TTS_MODEL":           settings.OPENROUTER_TTS_MODEL,
        "OPENROUTER_TTS_VOICE":           settings.OPENROUTER_TTS_VOICE,
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
        # AI prompt templates (editable via admin panel)
        "PROMPT_COVER":                   PROMPT_COVER_DEFAULT,
        "PROMPT_MINDMAP_MERMAID":         PROMPT_MINDMAP_MERMAID_DEFAULT,
        "PROMPT_MINDMAP_JSON":            PROMPT_MINDMAP_JSON_DEFAULT,
        # Mindmap generation settings
        # Set to 0 or empty for unlimited tokens (no max_tokens limit sent to API)
        "MINDMAP_JSON_MAX_TOKENS":        "0",
        # ── Security ─────────────────────────────────────────────────────────
        "API_KEY_AUTH_ENABLED":           "false",
        # ── Watermarks ────────────────────────────────────────────────────────
        "WATERMARK_TEXT":                 settings.WATERMARK_TEXT,
        "WATERMARK_POSITION":             settings.WATERMARK_POSITION,
        # Spoken intro read at the start of generated audio, per language.
        # Empty = no intro.
        "AUDIO_WATERMARK_TEXT_EN":        "SeeOurBook presents",
        "AUDIO_WATERMARK_TEXT_AR":        "Seeourbook تقدم لكم",
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
