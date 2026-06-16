"""
Mind map generation.
Supports two formats controlled by MINDMAP_FORMAT setting:
  - "mermaid" (default): AI generates Mermaid graph TD code → SVG via mermaid.ink/kroki.io
  - "json": AI generates structured JSON with center_node + branches

Renderer chain for Mermaid (tried in order):
  1. mermaid.ink  — primary, fast, no rate limit
  2. kroki.io     — fallback, reliable, different infrastructure

Model is configurable via MODEL_MINDMAP — supports any chat model,
including OpenRouter prefixed names (e.g. "openai/gpt-4.1-mini").
"""
import asyncio
import base64
import json as json_module
import logging
import re
import httpx

from api.config.settings import settings
from api.services.ai_client import chat_complete
from api.services.config.runtime import (
    get_config_value,
    PROMPT_MINDMAP_MERMAID_DEFAULT,
    PROMPT_MINDMAP_JSON_DEFAULT,
)

log = logging.getLogger(__name__)


async def generate_mermaid_code(title: str, summary: str, language: str, model: str | None = None) -> str:
    """Use an AI model to generate Mermaid graph TD code for the book."""
    model = model or settings.MODEL_MINDMAP

    # Output language instruction: the diagram text should match the book's language.
    lang_note = (
        "IMPORTANT: The book is in Arabic. ALL node labels, the root title, and any "
        "other text in the diagram MUST be written in Arabic. Do not use English."
        if language == "ar" else
        "Write all node labels in English."
    )

    template = await get_config_value("PROMPT_MINDMAP_MERMAID", PROMPT_MINDMAP_MERMAID_DEFAULT)
    prompt = template.format(
        title   = title,
        summary = summary[:1500],
        lang_note = lang_note,
    )

    code = await chat_complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )

    code = code.strip()
    # Strip markdown fences if model adds them anyway
    if code.startswith("```"):
        code = "\n".join(code.split("\n")[1:])
    if code.endswith("```"):
        code = "\n".join(code.split("\n")[:-1])
    return _sanitize_mermaid(code.strip())


_EDGE_RE = re.compile(
    r"^(\s*)(.+?)\s*(--?>|---|--)\s*(.+?)\s*$"
)


def _wrap_node(token: str, id_counter: list[int]) -> str:
    """
    Ensure a node reference uses the form  ID[Label]  required by Mermaid.

    - If token already has [..] or (..) or {..} shape  → leave as-is
    - If token is a plain Latin single-word ID             → leave as-is
    - Otherwise (spaces, Arabic, CJK, symbols, ...)        → wrap as N{counter}["Label"]
    """
    token = token.strip()
    if not token:
        return token
    # Already bracketed in any shape Mermaid understands
    if any(c in token for c in "[(){<"):
        return token
    # Plain single-word Latin ID — valid as-is
    if " " not in token and token.isascii() and token[0].isalpha():
        return token
    # Everything else gets a generated ID + quoted label (quotes protect Arabic/CJK)
    id_counter[0] += 1
    safe_label = token.replace('"', '\\"')
    return f'N{id_counter[0]}["{safe_label}"]'


def _sanitize_mermaid(code: str) -> str:
    """
    Post-process LLM output to fix the most common Mermaid syntax mistakes —
    specifically: multi-word or non-Latin node names used directly in edges
    without brackets. Mermaid 400s on  'Atomic Habits --> Small Changes' unless wrapped.
    """
    fixed_lines: list[str] = []
    id_counter = [0]
    for line in code.splitlines():
        m = _EDGE_RE.match(line)
        if not m:
            fixed_lines.append(line)
            continue
        indent, lhs, arrow, rhs = m.groups()
        # Don't touch the header line  'graph TD' / 'graph LR' etc.
        if lhs.lower().startswith("graph") and arrow == "--":
            fixed_lines.append(line)
            continue
        fixed_lines.append(f"{indent}{_wrap_node(lhs, id_counter)} {arrow} {_wrap_node(rhs, id_counter)}")
    return "\n".join(fixed_lines)


async def generate_json_mindmap(title: str, summary: str, language: str, model: str | None = None) -> dict:
    """
    Use an AI model to generate a structured JSON mind map for the book.

    Retries up to 3 times on empty or truncated responses (the most common
    failure mode — the model hits max_tokens mid-JSON or returns nothing).
    """
    model = model or settings.MODEL_MINDMAP

    lang_note = (
        "IMPORTANT: The book is in Arabic. ALL text values in the JSON "
        "(center_node.text and every sub_node) MUST be written in Arabic. "
        "Do not use English."
        if language == "ar" else
        "Write all text values in English."
    )

    template = await get_config_value("PROMPT_MINDMAP_JSON", PROMPT_MINDMAP_JSON_DEFAULT)
    prompt = template.format(
        title     = title,
        summary   = summary[:1500],
        lang_note = lang_note,
    )

    # Read max_tokens from config — 0 or empty means unlimited (no limit)
    max_tokens_str = await get_config_value("MINDMAP_JSON_MAX_TOKENS", "0")
    max_tokens = int(max_tokens_str) if max_tokens_str and max_tokens_str.strip() else 0

    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            # Build kwargs — only include max_tokens if > 0 (unlimited mode)
            kwargs: dict = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if max_tokens > 0:
                kwargs["max_tokens"] = max_tokens
            
            raw = await chat_complete(**kwargs)
            return _parse_json_mindmap(raw)
        except ValueError as exc:
            last_exc = exc
            log.warning(
                "json mindmap attempt %d/3 failed for %r: %s",
                attempt + 1, title[:40], exc,
            )
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)   # 0s, 2s, 4s backoff
    raise last_exc  # type: ignore[misc]


def _repair_truncated_json(text: str) -> str | None:
    """
    Attempt to repair truncated JSON by closing unclosed brackets/braces.
    Returns repaired string or None if cannot repair.
    """
    # Count open/close braces and brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')
    
    # If it looks complete, don't modify
    if open_braces <= 0 and open_brackets <= 0:
        return None
    
    repaired = text
    
    # Close strings that are left open (odd number of unescaped quotes)
    # Simple heuristic: count quotes not preceded by backslash
    quote_count = 0
    i = 0
    while i < len(repaired):
        if repaired[i] == '"' and (i == 0 or repaired[i-1] != '\\'):
            quote_count += 1
        i += 1
    
    # If odd number of quotes, close the last string
    if quote_count % 2 == 1:
        # Find the last unclosed quote and close it
        repaired = repaired.rstrip() + '"'
    
    # Close brackets first (they're inside braces usually)
    for _ in range(open_brackets):
        repaired = repaired.rstrip() + ']'
    
    # Close braces
    for _ in range(open_braces):
        repaired = repaired.rstrip() + '}'
    
    return repaired


def _parse_json_mindmap(raw: str) -> dict:
    """
    Robustly parse a JSON mind map from an LLM response.

    Handles the common failure modes that caused
    'Expecting value: line 1 column 1 (char 0)':
      • empty / whitespace-only responses
      • ```json ... ``` markdown fences
      • leading prose before the JSON object ("Here is the mind map: { ... }")
      • truncated JSON (incomplete due to max_tokens cutoff)
    """
    raw = (raw or "").strip()
    if not raw:
        raise ValueError("mind map model returned an empty response")

    # Strip markdown fences (```json ... ``` or ``` ... ```)
    if raw.startswith("```"):
        lines = raw.split("\n")
        lines = lines[1:]                      # drop opening fence (``` or ```json)
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]                 # drop closing fence
        raw = "\n".join(lines).strip()

    # Try parsing as-is first
    try:
        return json_module.loads(raw)
    except json_module.JSONDecodeError:
        pass

    # Fallback 1: extract the outermost { ... } object from surrounding prose.
    start = raw.find("{")
    end   = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = raw[start:end + 1]
        try:
            return json_module.loads(candidate)
        except json_module.JSONDecodeError:
            pass  # Try repair next
    
    # Fallback 2: try to repair truncated JSON
    if start != -1:
        # Extract from first { to end
        candidate = raw[start:]
        repaired = _repair_truncated_json(candidate)
        if repaired:
            try:
                result = json_module.loads(repaired)
                log.info("Successfully repaired truncated JSON mindmap")
                return result
            except json_module.JSONDecodeError:
                pass
    
    # Final error with helpful context
    if start == -1:
        raise ValueError(
            f"mind map response contained no JSON object. First 200 chars: {raw[:200]!r}"
        )
    
    # Show what we tried to parse
    candidate = raw[start:end + 1] if end > start else raw[start:]
    raise ValueError(
        f"mind map response was not valid JSON (tried repair). "
        f"First 200 chars: {raw[:200]!r}. "
        f"Extracted JSON length: {len(candidate)}"
    )


async def render_mermaid_svg(mermaid_code: str, output_path: str) -> str:
    """
    Render Mermaid code to SVG and save to output_path.

    Tries mermaid.ink first (up to 2 attempts), then falls back to
    kroki.io if mermaid.ink is unavailable (503 / 5xx / timeout).
    """
    encoded = base64.urlsafe_b64encode(mermaid_code.encode()).decode()

    # ── Attempt 1 & 2: mermaid.ink ───────────────────────────────────────────
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.get(f"https://mermaid.ink/svg/{encoded}")
                r.raise_for_status()
            with open(output_path, "wb") as f:
                f.write(r.content)
            return output_path
        except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.RequestError) as exc:
            last_exc = exc
            is_server_error = (
                isinstance(exc, httpx.HTTPStatusError)
                and exc.response.status_code >= 500
            )
            if is_server_error or isinstance(exc, (httpx.TimeoutException, httpx.RequestError)):
                if attempt == 0:
                    log.warning("mermaid.ink attempt 1 failed (%s), retrying in 3s…", exc)
                    await asyncio.sleep(3)
                continue
            # 4xx errors (bad diagram syntax) — no point retrying
            raise

    # ── Fallback: kroki.io ────────────────────────────────────────────────────
    log.warning("mermaid.ink unavailable (%s) — falling back to kroki.io", last_exc)
    try:
        async with httpx.AsyncClient(timeout=45) as client:
            r = await client.post(
                "https://kroki.io/mermaid/svg",
                content=mermaid_code.encode(),
                headers={"Content-Type": "text/plain"},
            )
            r.raise_for_status()
        with open(output_path, "wb") as f:
            f.write(r.content)
        log.info("Mind map rendered via kroki.io fallback")
        return output_path
    except Exception as kroki_exc:
        # Both renderers failed — raise original error with context
        raise RuntimeError(
            f"Mind map rendering failed on both mermaid.ink and kroki.io. "
            f"mermaid.ink: {last_exc}  |  kroki.io: {kroki_exc}"
        ) from kroki_exc
