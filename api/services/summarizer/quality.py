"""
Summary quality / coverage scoring.

After the full summary is built, an independent model (configurable — e.g. a
DeepSeek model via OpenRouter) checks whether the summary actually covers the
whole book. It returns a 0-100 coverage score plus the key points it judged
missing. The pipeline uses this to gate audio generation: audio_full /
audio_chapters only run when the score meets the admin-set threshold.
"""
import json as json_module
import logging

from api.config.settings import settings
from api.services.ai_client import chat_complete

log = logging.getLogger(__name__)


_QA_PROMPT = (
    "You are a strict book-summary quality reviewer.\n\n"
    "Below are per-chapter notes that represent the ACTUAL content of the book, "
    "followed by a FULL SUMMARY that is supposed to cover the entire book.\n\n"
    "Judge how completely the FULL SUMMARY covers the book's key content, "
    "characters, arguments, and progression described in the chapter notes.\n\n"
    "=== CHAPTER NOTES (ground truth) ===\n{chapters}\n\n"
    "=== FULL SUMMARY (to score) ===\n{summary}\n\n"
    "Return ONLY valid JSON, no markdown, in this exact shape:\n"
    '{{"score": <integer 0-100>, "covered": <true|false>, '
    '"missing": ["short point", "short point"], "reason": "one sentence"}}\n'
    "Where score = percentage of the book's key content present in the summary."
)


async def score_summary_coverage(
    full_summary: str,
    chapter_summaries: list[str],
    language: str,
    model: str | None = None,
    threshold: int = 70,
) -> dict:
    """
    Return {"score": int, "passed": bool, "covered": bool,
            "missing": [...], "reason": str, "model": str}.

    Never raises — on any failure it returns a permissive result (passed=True,
    score=-1) so a QA hiccup can't block the whole pipeline.
    """
    model = model or settings.MODEL_SONNET
    result = {
        "score": -1, "passed": True, "covered": True,
        "missing": [], "reason": "", "model": model,
    }
    if not full_summary or not chapter_summaries:
        result["reason"] = "no summary or chapter notes to score"
        return result

    # Cap the chapter notes so we don't blow the context window.
    joined = "\n\n".join(
        f"Chapter {i + 1}: {s}" for i, s in enumerate(chapter_summaries) if s
    )[:12000]

    prompt = _QA_PROMPT.format(chapters=joined, summary=full_summary[:12000])

    try:
        raw = await chat_complete(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
        )
    except Exception as exc:
        log.warning("summary QA call failed (%s) — allowing pipeline to continue", exc)
        result["reason"] = f"QA model error: {exc}"
        return result

    data = _parse_json(raw)
    if not data:
        result["reason"] = "QA model returned unparseable output"
        return result

    try:
        score = int(data.get("score", -1))
    except (TypeError, ValueError):
        score = -1

    result["score"]   = score
    result["covered"] = bool(data.get("covered", score >= threshold))
    result["missing"] = data.get("missing") or []
    result["reason"]  = str(data.get("reason") or "")
    # Only fail when we got a real score below the threshold.
    result["passed"]  = score < 0 or score >= threshold
    return result


def _parse_json(raw: str) -> dict | None:
    raw = (raw or "").strip()
    if not raw:
        return None
    if raw.startswith("```"):
        lines = raw.split("\n")[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    try:
        return json_module.loads(raw)
    except json_module.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json_module.loads(raw[start:end + 1])
        except json_module.JSONDecodeError:
            return None
    return None
