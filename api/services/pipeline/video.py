"""
Video generation for the SeeOurBook pipeline.

Public surface
──────────────
    generate_book_video(...)        — main entry point.  Returns path to MP4.
    VideoProvider Protocol          — for swapping in GPU-based generators later.
    get_video_provider(name) -> VideoProvider

Built-in providers
──────────────────
    MoviePyProvider               — CPU-only slideshow, production-ready (DEFAULT)
                                    Cover Ken-Burns + chapter cards + mindmap reveal
                                    + outro card + burned-in subtitles.
    StableVideoDiffusionProvider  — Stub.  Real implementation requires GPU host
                                    with 10GB+ VRAM and the SVD diffusers pipeline.
                                    Raises NotImplementedError until the GPU host
                                    is provisioned.
    CogVideoXProvider             — Stub.  Same story — CogVideoX-5B is amazing
                                    but needs proper GPU infrastructure.

Why an abstraction?
───────────────────
The MoviePy slideshow ships TODAY on the existing CPU server with $0 marginal
cost.  When the client wants higher-fidelity AI-generated visuals later, we
swap one config value (VIDEO_PROVIDER) and add a GPU host — no orchestrator
changes.  The Protocol lets us demonstrate this design to the client now
while staying pragmatic about what runs in production today.

Visual style
────────────
    - Vertical 1080×1920 (default — mobile-first, fits TikTok/Reels/Shorts)
      or horizontal 1920×1080 via VIDEO_ORIENTATION setting
    - Indigo accents matching the SeeOurBook brand
    - Soft vignette + drop shadow behind text for accessibility contrast
    - Bilingual: Arabic uses proper RTL shaping (arabic-reshaper + python-bidi)
    - Captions burned into the bottom third — synced to TTS audio duration
"""
from __future__ import annotations

import asyncio
import functools
import logging
import math
import os
import re
import textwrap
from io import BytesIO
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from api.config.settings import settings

log = logging.getLogger(__name__)


# ── Errors ───────────────────────────────────────────────────────────────────

class VideoError(Exception):
    """Generic video-step failure."""


class VideoProviderUnavailable(VideoError):
    """The selected provider can't run on this host (e.g., GPU required)."""


# ── Protocol ─────────────────────────────────────────────────────────────────

@runtime_checkable
class VideoProvider(Protocol):
    name: str

    async def generate(
        self,
        *,
        title:         str,
        author:        str,
        summary_text:  str,
        language:      str,
        audio_path:    str | None,    # None → silent slideshow
        cover_path:    str | None,
        mindmap_path:  str | None,
        chapters:      list[dict],
        output_path:   str,
    ) -> dict: ...


# ── Helpers: Arabic shaping + font lookup ────────────────────────────────────

def _is_arabic(lang: str | None) -> bool:
    return (lang or "en").lower() == "ar"


def _shape_arabic(text: str) -> str:
    """Reshape + bidi-flip Arabic text so PIL can render it correctly."""
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
    except ImportError:
        return text
    return get_display(arabic_reshaper.reshape(text))


# Font candidates per language.  First match wins.  Linux + macOS + Windows.
_FONT_FALLBACKS_AR = [
    "/usr/share/fonts/truetype/amiri/Amiri-Bold.ttf",
    "/usr/share/fonts/truetype/amiri/Amiri-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoNaskhArabic-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansArabic-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
]
_FONT_FALLBACKS_EN = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/Library/Fonts/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arialbd.ttf",
    "C:/Windows/Fonts/arial.ttf",
]


def _resolve_font_path(language: str) -> str | None:
    # Admin override wins
    override = settings.VIDEO_FONT_AR if _is_arabic(language) else settings.VIDEO_FONT_EN
    if override and Path(override).is_file():
        return override
    candidates = _FONT_FALLBACKS_AR if _is_arabic(language) else _FONT_FALLBACKS_EN
    for p in candidates:
        if Path(p).is_file():
            return p
    return None


def _font(language: str, size: int):
    from PIL import ImageFont
    path = _resolve_font_path(language)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            pass
    return ImageFont.load_default()


# ── Helpers: image composition (PIL) ─────────────────────────────────────────

# Brand colors
_BG_DARK     = (12, 14, 23)        # near-black indigo
_BRAND       = (99, 102, 241)      # indigo-500
_BRAND_SOFT  = (165, 180, 252)     # indigo-300
_TEXT_LIGHT  = (240, 240, 245)
_TEXT_DIM    = (160, 165, 180)
_OVERLAY     = (0, 0, 0, 180)      # subtitle background


def _video_size() -> tuple[int, int]:
    """Return (W, H) honoring VIDEO_ORIENTATION."""
    if (settings.VIDEO_ORIENTATION or "portrait").lower() == "landscape":
        return (1920, 1080)
    return (settings.VIDEO_WIDTH, settings.VIDEO_HEIGHT)


def _gradient_bg(size: tuple[int, int]) -> "Image.Image":
    """Soft vertical gradient background — dark with subtle indigo glow."""
    from PIL import Image
    W, H = size
    img = Image.new("RGB", size, _BG_DARK)
    px = img.load()
    for y in range(H):
        # Subtle indigo glow at center, fading to dark at top/bottom
        center_dist = abs(y - H / 2) / (H / 2)
        glow = max(0, 1 - center_dist) * 0.35
        r = int(_BG_DARK[0] + (_BRAND[0] - _BG_DARK[0]) * glow * 0.25)
        g = int(_BG_DARK[1] + (_BRAND[1] - _BG_DARK[1]) * glow * 0.25)
        b = int(_BG_DARK[2] + (_BRAND[2] - _BG_DARK[2]) * glow * 0.25)
        for x in range(W):
            px[x, y] = (r, g, b)
    return img


def _wrap_text(text: str, font, max_width_px: int) -> list[str]:
    """Wrap a string to lines that fit max_width_px using the given font."""
    from PIL import ImageDraw, Image
    if not text:
        return [""]
    draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    words = text.split()
    lines: list[str] = []
    cur = ""
    for w in words:
        candidate = (cur + " " + w).strip() if cur else w
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if (bbox[2] - bbox[0]) <= max_width_px:
            cur = candidate
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _draw_centered_text(
    img,
    lines: list[str],
    font,
    *,
    color: tuple,
    y_start: int,
    line_spacing: float = 1.3,
    shadow: bool = True,
) -> int:
    """Draw centered, multi-line text. Returns the Y after the last line."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    W = img.size[0]
    y = y_start
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        w  = bbox[2] - bbox[0]
        h  = bbox[3] - bbox[1]
        x  = (W - w) // 2
        if shadow:
            draw.text((x + 3, y + 3), line, font=font, fill=(0, 0, 0, 200))
        draw.text((x, y), line, font=font, fill=color)
        y += int(h * line_spacing)
    return y


def _add_brand_strip(img, language: str):
    """Bottom indigo accent strip + small SeeOurBook brand text."""
    from PIL import ImageDraw
    draw = ImageDraw.Draw(img)
    W, H = img.size
    strip_h = max(6, H // 200)
    draw.rectangle([(0, H - strip_h), (W, H)], fill=_BRAND)
    # Small brand tag right above the strip
    font = _font(language, max(18, H // 80))
    brand = "SeeOurBook.com"
    bbox = draw.textbbox((0, 0), brand, font=font)
    draw.text(
        ((W - (bbox[2] - bbox[0])) // 2, H - strip_h - (bbox[3] - bbox[1]) - 20),
        brand, font=font, fill=_TEXT_DIM,
    )


def _render_title_card(title: str, author: str, language: str) -> "np.ndarray":
    """Title + author centered on a gradient bg with brand strip."""
    from PIL import Image
    W, H = _video_size()
    img = _gradient_bg((W, H))

    is_ar    = _is_arabic(language)
    badge    = "ملخص" if is_ar else "BOOK SUMMARY"
    title_t  = _shape_arabic(title) if is_ar else title
    author_t = _shape_arabic(f"بقلم {author}" if author else "") if is_ar \
               else (f"by {author}" if author else "")

    # Sizes scale with shorter dimension so portrait + landscape both look good
    short = min(W, H)
    badge_f   = _font(language, short // 24)
    title_f   = _font(language, short // 14)
    author_f  = _font(language, short // 28)

    # Brand badge near top
    _draw_centered_text(img, [badge], badge_f, color=_BRAND_SOFT, y_start=int(H * 0.18))

    # Title (wrapped to ~70% width)
    max_w = int(W * 0.78)
    lines = _wrap_text(title_t, title_f, max_w)
    y_after = _draw_centered_text(
        img, lines, title_f,
        color=_TEXT_LIGHT, y_start=int(H * 0.32), line_spacing=1.15,
    )

    if author_t:
        _draw_centered_text(
            img, [author_t], author_f,
            color=_TEXT_DIM, y_start=y_after + int(H * 0.02),
        )

    _add_brand_strip(img, language)
    return np.array(img)


def _render_chapter_card(idx: int, ch_title: str, snippet: str,
                         language: str) -> "np.ndarray":
    """Chapter card: 'CHAPTER N' + title + short excerpt."""
    from PIL import Image
    W, H = _video_size()
    img = _gradient_bg((W, H))

    is_ar = _is_arabic(language)
    label  = f"الفصل {idx}" if is_ar else f"CHAPTER {idx}"
    label_t = _shape_arabic(label) if is_ar else label
    title_t = _shape_arabic(ch_title) if is_ar else ch_title
    snip_t  = _shape_arabic(snippet)  if is_ar else snippet

    short = min(W, H)
    label_f   = _font(language, short // 28)
    title_f   = _font(language, short // 16)
    snippet_f = _font(language, short // 32)

    _draw_centered_text(img, [label_t], label_f, color=_BRAND_SOFT,
                        y_start=int(H * 0.16))

    title_lines = _wrap_text(title_t, title_f, int(W * 0.82))
    y_after = _draw_centered_text(
        img, title_lines, title_f,
        color=_TEXT_LIGHT, y_start=int(H * 0.26),
    )

    snip_lines = _wrap_text(snip_t, snippet_f, int(W * 0.82))[:6]
    _draw_centered_text(
        img, snip_lines, snippet_f,
        color=_TEXT_DIM, y_start=y_after + int(H * 0.04),
        line_spacing=1.5,
    )

    _add_brand_strip(img, language)
    return np.array(img)


def _render_outro_card(language: str) -> "np.ndarray":
    """Outro card — call-to-action + brand."""
    from PIL import Image
    W, H = _video_size()
    img = _gradient_bg((W, H))

    is_ar = _is_arabic(language)
    line1 = "اقرأ الملخص الكامل" if is_ar else "READ THE FULL SUMMARY"
    line2 = "SeeOurBook.com"

    short = min(W, H)
    f1 = _font(language, short // 18)
    f2 = _font(language, short // 12)

    _draw_centered_text(
        img,
        [_shape_arabic(line1) if is_ar else line1],
        f1, color=_BRAND_SOFT, y_start=int(H * 0.40),
    )
    _draw_centered_text(
        img, [line2], f2,
        color=_TEXT_LIGHT, y_start=int(H * 0.48),
    )

    _add_brand_strip(img, language)
    return np.array(img)


# ── Subtitles ────────────────────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences for subtitle alignment (en + ar)."""
    parts = re.split(r"(?<=[.!?؟…])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _allocate_subtitle_times(
    sentences: list[str], total_duration: float,
) -> list[tuple[float, float, str]]:
    """
    Distribute sentence start/end timestamps across the audio duration
    weighted by sentence length.  Not word-perfect but visually solid.
    """
    if not sentences or total_duration <= 0:
        return []
    weights = [max(1, len(s)) for s in sentences]
    total   = sum(weights)
    out: list[tuple[float, float, str]] = []
    t = 0.0
    for sent, w in zip(sentences, weights):
        dur = total_duration * (w / total)
        out.append((t, t + dur, sent))
        t += dur
    # Snap the last end to exact duration
    if out:
        s, _, txt = out[-1]
        out[-1] = (s, total_duration, txt)
    return out


def _render_subtitle_image(text: str, language: str) -> "np.ndarray":
    """Render a subtitle line as a 1080×W PNG with semi-transparent background."""
    from PIL import Image
    W, H = _video_size()
    # Subtitle band is bottom ~22% of the frame
    band_h = int(H * 0.22)
    img = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))

    is_ar = _is_arabic(language)
    short = min(W, H)
    fnt = _font(language, short // 28)

    shown = _shape_arabic(text) if is_ar else text
    lines = _wrap_text(shown, fnt, int(W * 0.86))[:3]

    # Center the lines vertically within the band
    line_h    = fnt.size + 12
    block_h   = len(lines) * line_h
    y_start   = (band_h - block_h) // 2

    # Soft background pill behind the lines
    from PIL import ImageDraw
    overlay = Image.new("RGBA", (W, band_h), (0, 0, 0, 0))
    draw_o  = ImageDraw.Draw(overlay)
    pad_x   = int(W * 0.05)
    draw_o.rounded_rectangle(
        [pad_x, y_start - 18, W - pad_x, y_start + block_h + 18],
        radius=18, fill=_OVERLAY,
    )
    img = Image.alpha_composite(img, overlay)

    _draw_centered_text(
        img, lines, fnt, color=_TEXT_LIGHT, y_start=y_start, line_spacing=1.15,
    )
    return np.array(img)


# ── MoviePy provider ─────────────────────────────────────────────────────────

class MoviePyProvider:
    """
    Production CPU video provider — uses MoviePy + ffmpeg.

    Composition (durations scale to audio length):
       [ Title card 6% ]
       [ Cover Ken-Burns 14% ]
       [ Chapter cards split 60% ]
       [ Mindmap reveal 14% ]
       [ Outro card 6% ]
    + Subtitles burned in for the entire run
    """
    name = "moviepy"

    def _sync_generate(
        self,
        *,
        title:         str,
        author:        str,
        summary_text:  str,
        language:      str,
        audio_path:    str | None,
        cover_path:    str | None,
        mindmap_path:  str | None,
        chapters:      list[dict],
        output_path:   str,
    ) -> dict:
        try:
            from moviepy.editor import (
                AudioFileClip, ImageClip, CompositeVideoClip,
                concatenate_videoclips, ColorClip,
            )
        except ImportError as exc:
            raise VideoProviderUnavailable(
                "moviepy is not installed — run `pip install moviepy==1.0.3`"
            ) from exc

        # ── Audio (optional) ────────────────────────────────────────────────
        # If audio is missing, we generate a SILENT slideshow with a duration
        # proportional to the number of chapters.  This is the path local
        # tests take when PIPELINE_STEP_TTS is disabled.
        audio = None
        if audio_path and Path(audio_path).is_file():
            try:
                audio = AudioFileClip(audio_path)
                total = float(audio.duration)
                if total < 1:
                    audio.close()
                    audio = None
            except Exception as exc:
                log.warning("audio load failed (%s) — falling back to silent video", exc)
                audio = None

        if audio is None:
            # Silent slideshow: 5s title + 5s cover + 5s per chapter + 5s mindmap + 5s outro
            n_chapters = min(10, max(1, sum(1 for c in chapters if (c.get("summary") or "").strip())))
            total = 5 + 5 + (5 * n_chapters) + (5 if mindmap_path else 0) + 5
            log.info("video: SILENT mode — duration=%.1fs (no audio_path supplied)", total)

        W, H = _video_size()
        log.info("video: total_duration=%.1fs size=%dx%d audio=%s",
                 total, W, H, "yes" if audio else "silent")

        # ── Stage budgets (clamped so titles always have a min duration) ────
        budget_title  = max(4.0, total * 0.06)
        budget_cover  = max(6.0, total * 0.14)
        budget_outro  = max(4.0, total * 0.06)
        budget_mind   = max(0.0, total * 0.14) if mindmap_path else 0.0
        budget_chaps  = max(
            0.0,
            total - budget_title - budget_cover - budget_outro - budget_mind,
        )

        usable_chapters = [c for c in chapters if (c.get("summary") or "").strip()][:10]
        per_chapter = (budget_chaps / max(1, len(usable_chapters))) if usable_chapters else 0.0

        clips: list = []

        # ── 1. Title card ──────────────────────────────────────────────────
        title_arr = _render_title_card(title, author, language)
        clips.append(
            ImageClip(title_arr)
            .set_duration(budget_title)
            .fadein(0.5).fadeout(0.4)
        )

        # ── 2. Cover with Ken-Burns zoom ───────────────────────────────────
        if cover_path and Path(cover_path).is_file():
            try:
                cover_clip = self._ken_burns_cover(cover_path, budget_cover, (W, H))
                clips.append(cover_clip)
            except Exception as exc:
                log.warning("cover render failed (%s) — skipping cover stage", exc)
                budget_chaps += budget_cover
                per_chapter = (budget_chaps / max(1, len(usable_chapters))) \
                              if usable_chapters else 0.0
        else:
            # Fall back: extra time on chapter cards
            budget_chaps += budget_cover
            per_chapter = (budget_chaps / max(1, len(usable_chapters))) \
                          if usable_chapters else 0.0

        # ── 3. Chapter cards ───────────────────────────────────────────────
        for ch in usable_chapters:
            idx     = ch.get("index") or (usable_chapters.index(ch) + 1)
            ch_t    = (ch.get("title") or f"Chapter {idx}").strip()
            snippet = (ch.get("summary") or "").strip()
            # Trim snippet to a readable size
            if len(snippet) > 300:
                snippet = snippet[:300].rsplit(" ", 1)[0] + "…"
            arr = _render_chapter_card(idx, ch_t, snippet, language)
            clips.append(
                ImageClip(arr).set_duration(per_chapter)
                              .fadein(0.4).fadeout(0.4)
            )

        # ── 4. Mindmap reveal ──────────────────────────────────────────────
        if mindmap_path and Path(mindmap_path).is_file() and budget_mind > 0:
            try:
                mm_clip = self._mindmap_clip(mindmap_path, budget_mind, (W, H), language)
                clips.append(mm_clip)
            except Exception as exc:
                log.warning("mindmap render failed (%s) — skipping mindmap stage", exc)

        # ── 5. Outro card ──────────────────────────────────────────────────
        outro_arr = _render_outro_card(language)
        clips.append(
            ImageClip(outro_arr).set_duration(budget_outro)
                                 .fadein(0.4).fadeout(0.6)
        )

        # ── Stitch + overlay subtitles ─────────────────────────────────────
        if not clips:
            if audio:
                try: audio.close()
                except Exception: pass
            raise VideoError("nothing to render — no clips produced")

        video = concatenate_videoclips(clips, method="compose")
        # Make sure video matches the target duration exactly
        if video.duration < total:
            # Pad with a still frame of the outro
            tail = ImageClip(outro_arr).set_duration(total - video.duration)
            video = concatenate_videoclips([video, tail], method="compose")
        elif video.duration > total:
            video = video.subclip(0, total)

        # Subtitle layer — works in both narrated and silent modes
        subs = _allocate_subtitle_times(_split_sentences(summary_text), total)
        if subs:
            sub_clips = []
            for start, end, line in subs:
                img_arr = _render_subtitle_image(line, language)
                sub_clip = (ImageClip(img_arr)
                            .set_start(start)
                            .set_duration(max(0.1, end - start))
                            .set_position(("center", "bottom")))
                sub_clips.append(sub_clip)
            video = CompositeVideoClip([video, *sub_clips], size=(W, H))

        # Bind audio (only when present)
        if audio is not None:
            video = video.set_audio(audio)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        log.info("video: encoding %s", output_path)
        try:
            video.write_videofile(
                output_path,
                fps            = settings.VIDEO_FPS,
                codec          = "libx264",
                audio_codec    = "aac" if audio else None,
                audio          = audio is not None,
                bitrate        = settings.VIDEO_BITRATE,
                preset         = "medium",
                threads        = max(1, (os.cpu_count() or 2) // 2),
                logger         = None,           # silence moviepy's bar
                verbose        = False,
                temp_audiofile = (
                    str(Path(output_path).with_suffix(".tmp.m4a")) if audio else None
                ),
                remove_temp    = True,
            )
        finally:
            try:    video.close()
            except Exception: pass
            if audio is not None:
                try:    audio.close()
                except Exception: pass

        size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        return {
            "duration_seconds": int(total),
            "size_mb":          round(size_mb, 2),
            "width":            W,
            "height":           H,
            "provider":         self.name,
            "silent":           audio is None,
        }

    # ── Per-stage helpers ───────────────────────────────────────────────────

    def _ken_burns_cover(self, cover_path: str, duration: float,
                         size: tuple[int, int]):
        """Cover image with slow zoom + drift centered on the canvas."""
        from PIL import Image
        from moviepy.editor import ImageClip
        W, H = size

        cover = Image.open(cover_path).convert("RGB")
        cw, ch = cover.size
        # Fit cover to ~80% of frame height keeping aspect ratio
        target_h = int(H * 0.80)
        target_w = int(cw * (target_h / ch))
        if target_w > int(W * 0.86):
            target_w = int(W * 0.86)
            target_h = int(ch * (target_w / cw))
        cover_resized = cover.resize((target_w, target_h), Image.LANCZOS)

        bg = _gradient_bg((W, H))
        bg.paste(cover_resized, ((W - target_w) // 2, (H - target_h) // 2))
        # Subtle vignette / brand strip
        _add_brand_strip(bg, "en")
        arr = np.array(bg)

        clip = (ImageClip(arr)
                .set_duration(duration)
                .resize(lambda t: 1.0 + 0.04 * (t / max(1e-6, duration)))
                .fadein(0.5).fadeout(0.5))
        return clip

    def _mindmap_clip(self, mindmap_path: str, duration: float,
                      size: tuple[int, int], language: str):
        """Display the mindmap (PNG/JPG/SVG-rasterized) with a gentle zoom."""
        from PIL import Image
        from moviepy.editor import ImageClip
        W, H = size

        ext = Path(mindmap_path).suffix.lower()
        if ext == ".svg":
            # MoviePy can't open SVGs — convert via Pillow if cairosvg available
            try:
                import cairosvg
                png_bytes = cairosvg.svg2png(
                    url=str(mindmap_path),
                    output_width=int(W * 0.9),
                )
                mm = Image.open(BytesIO(png_bytes)).convert("RGBA")
            except Exception as exc:
                raise VideoError(f"could not rasterize SVG mindmap: {exc}") from exc
        else:
            mm = Image.open(mindmap_path).convert("RGBA")

        # Fit mindmap into ~88% of frame
        mw, mh = mm.size
        target_w = int(W * 0.88)
        target_h = int(mh * (target_w / mw))
        if target_h > int(H * 0.72):
            target_h = int(H * 0.72)
            target_w = int(mw * (target_h / mh))
        mm_resized = mm.resize((target_w, target_h), Image.LANCZOS)

        bg = _gradient_bg((W, H))
        bg_rgba = bg.convert("RGBA")
        bg_rgba.paste(mm_resized, ((W - target_w) // 2, (H - target_h) // 2), mm_resized)
        bg = bg_rgba.convert("RGB")

        # Caption above the mindmap
        is_ar = _is_arabic(language)
        cap   = "خريطة ذهنية" if is_ar else "MIND MAP"
        cap_t = _shape_arabic(cap) if is_ar else cap
        cap_f = _font(language, min(W, H) // 26)
        _draw_centered_text(bg, [cap_t], cap_f, color=_BRAND_SOFT,
                            y_start=int(H * 0.08))
        _add_brand_strip(bg, language)

        clip = (ImageClip(np.array(bg))
                .set_duration(duration)
                .resize(lambda t: 1.0 + 0.02 * (t / max(1e-6, duration)))
                .fadein(0.4).fadeout(0.5))
        return clip

    async def generate(self, **kwargs) -> dict:
        loop = asyncio.get_running_loop()
        fn = functools.partial(self._sync_generate, **kwargs)
        return await loop.run_in_executor(None, fn)


# ── GPU provider stubs ───────────────────────────────────────────────────────

class StableVideoDiffusionProvider:
    """
    Stable Video Diffusion — high-fidelity text/image-to-video.

    Real implementation requires:
      - GPU host with 10GB+ VRAM (A10 / A100 / RTX 4090)
      - diffusers + torch + xformers
      - SVD checkpoint downloaded (~9.5 GB)

    When configured + the host has GPU, swap in the actual pipeline call
    here.  Until then it raises VideoProviderUnavailable so the orchestrator
    can fall back to MoviePy automatically.
    """
    name = "svd"

    async def generate(self, **kwargs) -> dict:  # noqa: ARG002
        raise VideoProviderUnavailable(
            "Stable Video Diffusion requires a GPU host with 10GB+ VRAM and "
            "the diffusers pipeline installed.  Switch VIDEO_PROVIDER to "
            "'moviepy' for CPU-based slideshow generation."
        )


class CogVideoXProvider:
    """
    CogVideoX-5B — open-source long-form video diffusion (THUDM).

    Real implementation requires:
      - GPU host with 24GB+ VRAM (A100 / H100)
      - diffusers + torch + flash-attn
      - CogVideoX checkpoint (~12 GB)
      - 5–10 minutes per 8-second clip on A100

    Stubbed until the GPU host is provisioned.
    """
    name = "cogvideox"

    async def generate(self, **kwargs) -> dict:  # noqa: ARG002
        raise VideoProviderUnavailable(
            "CogVideoX-5B requires a GPU host with 24GB+ VRAM and the "
            "diffusers pipeline installed.  Switch VIDEO_PROVIDER to "
            "'moviepy' for CPU-based slideshow generation."
        )


# ── Factory ──────────────────────────────────────────────────────────────────

_PROVIDERS: dict[str, type[VideoProvider]] = {
    "moviepy":   MoviePyProvider,
    "svd":       StableVideoDiffusionProvider,
    "cogvideox": CogVideoXProvider,
}


def get_video_provider(name: str | None = None) -> VideoProvider:
    """Resolve a video provider by name. Falls back to MoviePy on unknown values."""
    chosen = (name or settings.VIDEO_PROVIDER or "moviepy").lower()
    cls = _PROVIDERS.get(chosen)
    if cls is None:
        log.warning("Unknown VIDEO_PROVIDER %r — using MoviePy", chosen)
        return MoviePyProvider()
    return cls()


# ── Public entry point ───────────────────────────────────────────────────────

async def generate_book_video(
    *,
    title:         str,
    author:        str,
    summary_text:  str,
    language:      str,
    audio_path:    str | None,        # None → silent slideshow
    cover_path:    str | None,
    mindmap_path:  str | None,
    chapters:      list[dict],
    output_path:   str,
    provider_name: str | None = None,
) -> dict:
    """
    Generate a book-summary video and save to `output_path`.

    Returns a dict:
        {duration_seconds, size_mb, width, height, provider}

    On VideoProviderUnavailable (GPU-required provider on CPU host), we
    automatically retry with MoviePy so the pipeline still produces a video.
    """
    provider = get_video_provider(provider_name)

    try:
        return await provider.generate(
            title=title, author=author, summary_text=summary_text,
            language=language, audio_path=audio_path,
            cover_path=cover_path, mindmap_path=mindmap_path,
            chapters=chapters, output_path=output_path,
        )
    except VideoProviderUnavailable as exc:
        if provider.name == "moviepy":
            raise   # nothing to fall back to
        log.warning("Video provider %s unavailable (%s) — falling back to MoviePy",
                    provider.name, exc)
        moviepy = MoviePyProvider()
        return await moviepy.generate(
            title=title, author=author, summary_text=summary_text,
            language=language, audio_path=audio_path,
            cover_path=cover_path, mindmap_path=mindmap_path,
            chapters=chapters, output_path=output_path,
        )
