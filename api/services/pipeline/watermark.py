"""
Watermark helpers for SeeOurBook assets.

  stamp_image(path)   — writes text + logo corner badge to a JPEG/PNG in-place
  stamp_audio(path)   — writes ID3 comment + copyright tag to an MP3 in-place
  stamp_mindmap_json  — adds a _watermark key to a JSON dict
  stamp_mindmap_mermaid — prepends a %% comment line to Mermaid source

All functions are no-ops (log + return) when the watermark text is empty
or when an optional dependency (Pillow / mutagen) is missing, so the rest
of the pipeline is never blocked.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)


def _text(cfg: dict | None = None) -> str:
    if cfg:
        return (cfg.get("WATERMARK_TEXT") or "").strip()
    from api.config.settings import settings  # noqa: PLC0415
    return (settings.WATERMARK_TEXT or "").strip()


def _position(cfg: dict | None = None) -> str:
    if cfg:
        return (cfg.get("WATERMARK_POSITION") or "bottom-right").lower()
    from api.config.settings import settings  # noqa: PLC0415
    return (settings.WATERMARK_POSITION or "bottom-right").lower()


# ── Image watermark ───────────────────────────────────────────────────────────

def stamp_image(path: str | Path, cfg: dict | None = None) -> None:
    """
    Burn a semi-transparent text badge into the image at `path`.
    Uses Pillow (PIL) — gracefully skips if not installed.
    """
    text = _text(cfg)
    if not text:
        return
    try:
        from PIL import Image, ImageDraw, ImageFont  # noqa: PLC0415
    except ImportError:
        log.debug("Pillow not installed — image watermark skipped")
        return

    path = Path(path)
    if not path.exists():
        return

    try:
        img = Image.open(path).convert("RGBA")
        w, h = img.size
        pos = _position(cfg)

        # Overlay layer (transparent)
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Font — try to load a system font, fall back to default
        font_size = max(18, w // 40)
        font = None
        for candidate in (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            "C:/Windows/Fonts/Arial.ttf",
            "C:/Windows/Fonts/calibri.ttf",
        ):
            if os.path.exists(candidate):
                try:
                    from PIL import ImageFont as _IF  # noqa: PLC0415
                    font = _IF.truetype(candidate, font_size)
                    break
                except Exception:
                    pass
        if font is None:
            font = ImageFont.load_default()

        # Measure text
        bbox = draw.textbbox((0, 0), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]

        pad = 12
        pos_map = {
            "top-left":     (pad, pad),
            "top-right":    (w - tw - pad, pad),
            "bottom-left":  (pad, h - th - pad),
            "bottom-right": (w - tw - pad, h - th - pad),
        }
        x, y = pos_map.get(pos, pos_map["bottom-right"])

        # Shadow for legibility
        draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 160))
        # White text with alpha
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 200))

        # Merge onto original
        img = Image.alpha_composite(img, overlay).convert("RGB")
        img.save(path, quality=92)
        log.debug("Image watermarked: %s", path)
    except Exception as exc:
        log.warning("Image watermark failed (%s): %s", path, exc)


# ── Audio watermark (ID3 tags) ────────────────────────────────────────────────

def stamp_audio(path: str | Path, cfg: dict | None = None) -> None:
    """
    Write ID3 comment + copyright tags to an MP3 file.
    Uses mutagen — gracefully skips if not installed.
    """
    text = _text(cfg)
    if not text:
        return
    try:
        from mutagen.id3 import ID3, COMM, TCOP, error as ID3Error  # noqa: PLC0415
    except ImportError:
        log.debug("mutagen not installed — audio watermark skipped")
        return

    path = Path(path)
    if not path.exists():
        return

    try:
        try:
            tags = ID3(str(path))
        except ID3Error:
            tags = ID3()

        # Comment tag (shown in most players)
        tags.add(COMM(encoding=3, lang="eng", desc="", text=text))
        # Copyright tag
        tags.add(TCOP(encoding=3, text=text))
        tags.save(str(path))
        log.debug("Audio watermarked: %s", path)
    except Exception as exc:
        log.warning("Audio watermark failed (%s): %s", path, exc)


# ── Mindmap watermarks ────────────────────────────────────────────────────────

def stamp_mindmap_json(data: dict, cfg: dict | None = None) -> dict:
    """Add a _watermark key to a mindmap JSON dict. Returns the modified dict."""
    text = _text(cfg)
    if text:
        data["_watermark"] = text
    return data


def stamp_mindmap_mermaid(source: str, cfg: dict | None = None) -> str:
    """Prepend a %% comment line to Mermaid source."""
    text = _text(cfg)
    if not text:
        return source
    return f"%% {text}\n{source}"
