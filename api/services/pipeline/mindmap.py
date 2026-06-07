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

    # mermaid.ink cannot render Arabic / RTL text — node labels must always be
    # short English keywords regardless of the book's language.
    lang_note = (
        "The book is in Arabic. Use short English keywords for all node labels "
        "(concepts, not translations). The summary is in Arabic for context only."
        if language == "ar" else ""
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


def _wrap_node(token: str) -> str:
    """
    Ensure a node reference uses the form  ID[Label]  required by Mermaid.

    - If token already has [..] or (..) or {..} shape  → leave as-is
    - If token is a single word                        → leave as-is (valid ID)
    - If token has spaces                              → wrap as N_<id>[Label]
    """
    token = token.strip()
    if not token:
        return token
    # Already bracketed in any shape Mermaid understands
    if any(c in token for c in "[(){<"):
        return token
    if " " not in token:
        return token   # plain ID, valid as-is
    # Multi-word bare label → convert to ID[Label]
    safe_id = re.sub(r"[^A-Za-z0-9]", "_", token)[:40] or "N"
    return f"{safe_id}[{token}]"


def _sanitize_mermaid(code: str) -> str:
    """
    Post-process LLM output to fix the most common Mermaid syntax mistakes —
    specifically: multi-word node names used directly in edges without brackets.
    Mermaid 400s on  'Atomic Habits --> Small Changes'  unless wrapped.
    """
    fixed_lines: list[str] = []
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
        fixed_lines.append(f"{indent}{_wrap_node(lhs)} {arrow} {_wrap_node(rhs)}")
    return "\n".join(fixed_lines)


async def generate_json_mindmap(title: str, summary: str, language: str, model: str | None = None) -> dict:
    """Use an AI model to generate a structured JSON mind map for the book."""
    model = model or settings.MODEL_MINDMAP

    lang_note = (
        "The book is in Arabic. Use short English keywords for all text values."
        if language == "ar" else ""
    )

    template = await get_config_value("PROMPT_MINDMAP_JSON", PROMPT_MINDMAP_JSON_DEFAULT)
    prompt = template.format(
        title     = title,
        summary   = summary[:1500],
        lang_note = lang_note,
    )

    raw = await chat_complete(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=800,
    )

    raw = raw.strip()
    if raw.startswith("```"):
        raw = "\n".join(raw.split("\n")[1:])
    if raw.endswith("```"):
        raw = "\n".join(raw.split("\n")[:-1])
    return json_module.loads(raw.strip())


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
