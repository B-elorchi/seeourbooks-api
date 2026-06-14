"""
Audio post-processing via FFmpeg.
Normalizes loudness to EBU R128 (-16 LUFS) and writes ID3 tags.

If ffmpeg is not installed on the host, every audio step would crash with
`FileNotFoundError: 'ffmpeg'`.  Instead of failing, we detect that case
once, log a clear warning telling the admin to install ffmpeg, and fall
back to the raw TTS output (no normalization / no ID3 tags).  Audio still
plays — it just isn't loudness-corrected.

Install on Debian/Ubuntu hosts with:
    sudo apt-get update && sudo apt-get install -y ffmpeg
"""
import logging
import os
import shutil
import subprocess

from api.config.settings import settings

log = logging.getLogger(__name__)

_FFMPEG_AVAILABLE: bool | None = None    # cached after first check
_WARNED: bool = False                    # so we only log the warning once


def _ffmpeg_available() -> bool:
    """Cache and return whether ffmpeg is on PATH.  Logs a one-time warning if not."""
    global _FFMPEG_AVAILABLE, _WARNED
    if _FFMPEG_AVAILABLE is None:
        _FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None
        if not _FFMPEG_AVAILABLE and not _WARNED:
            log.warning(
                "ffmpeg is NOT installed on this host. Audio steps will skip "
                "loudness normalization and ID3 tagging — TTS output is uploaded as-is. "
                "Install with `sudo apt-get install -y ffmpeg` to enable processing."
            )
            _WARNED = True
    return _FFMPEG_AVAILABLE


def _fallback_passthrough(input_path: str, output_path: str) -> dict:
    """When ffmpeg is unavailable (or disabled), just move the raw TTS file."""
    # Use shutil.move so it works across filesystems too (rename is intra-fs only)
    if input_path != output_path:
        shutil.move(input_path, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return {"duration_seconds": 0, "size_mb": round(size_mb, 2)}


def process_audio(input_path: str, output_path: str, title: str, author: str) -> dict:
    """
    Normalize audio and write ID3 tags.
    Returns dict with duration_seconds and size_mb.

    Falls back to a no-op passthrough when:
      - PIPELINE_STEP_AUDIO_PROCESSING is disabled in settings, or
      - ffmpeg is not installed on the host.
    """
    if not settings.PIPELINE_STEP_AUDIO_PROCESSING:
        return _fallback_passthrough(input_path, output_path)

    if not _ffmpeg_available():
        return _fallback_passthrough(input_path, output_path)

    _meta = [
        "-metadata", f"title={title}",
        "-metadata", f"artist={author}",
        "-metadata", f"album=SeeOurBook Summary",
    ]

    # Primary: loudness-normalize to EBU R128.
    # -fflags +genpts  regenerates presentation timestamps — fixes the
    # "time_base 1/0" crash that occurs when Gemini TTS returns raw PCM
    # or concatenated chunks with missing/zero frame headers.
    # -ar 44100 on INPUT forces ffmpeg to assume a sane sample rate before
    # the filter graph tries to configure itself.
    cmd_normalize = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-ar", "44100",
        "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "128k",
        *_meta,
        output_path,
    ]

    # Fallback: plain re-encode without the loudnorm filter. Runs when the
    # input audio is so malformed that the filter graph fails to init. Still
    # better than raw TTS bytes — we at least get clean MP3 framing and tags.
    cmd_reencode = [
        "ffmpeg", "-y",
        "-fflags", "+genpts",
        "-i", input_path,
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "128k",
        *_meta,
        output_path,
    ]

    def _run(cmd: list[str]) -> None:
        subprocess.run(cmd, check=True, capture_output=True)

    try:
        _run(cmd_normalize)
    except FileNotFoundError:
        log.warning("ffmpeg vanished mid-run, falling back to passthrough")
        return _fallback_passthrough(input_path, output_path)
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[-600:]
        log.warning(
            "ffmpeg loudnorm failed (exit %s): %s — retrying without filter",
            exc.returncode, stderr,
        )
        try:
            _run(cmd_reencode)
        except subprocess.CalledProcessError as exc2:
            stderr2 = (exc2.stderr or b"").decode("utf-8", errors="replace")[-500:]
            log.error("ffmpeg re-encode also failed (exit %s): %s", exc2.returncode, stderr2)
            return _fallback_passthrough(input_path, output_path)

    # Get duration via ffprobe (also optional — if it fails, just skip)
    duration = 0
    try:
        if shutil.which("ffprobe"):
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", output_path],
                capture_output=True, text=True,
            )
            if probe.stdout.strip():
                duration = float(probe.stdout.strip())
    except (subprocess.SubprocessError, ValueError):
        pass

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return {"duration_seconds": round(duration), "size_mb": round(size_mb, 2)}
