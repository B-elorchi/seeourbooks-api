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
    if input_path != output_path:
        shutil.move(input_path, output_path)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return {"duration_seconds": 0, "size_mb": round(size_mb, 2)}


def _looks_like_container(path: str) -> bool:
    """
    Return True if the file starts with a recognised *compressed/container*
    audio signature (MP3, WAV, OGG, FLAC, MP4/M4A).  Return False when no
    known header is present — which means the bytes are almost certainly raw
    headerless PCM.

    Why not ffprobe?  ffprobe leniently mis-detects headerless PCM as "mp3"
    (it sees a byte that looks like an MP3 sync word), then ffmpeg blows up
    with `time_base 1/0` / `Decode error rate 1`.  Reading the magic bytes
    ourselves is deterministic and correct for the TTS outputs we deal with.

    Gemini TTS (native API and via OpenRouter) returns raw s16le PCM with no
    header, so this function returns False for it → the caller transcodes it
    with an explicit `-f s16le` input.
    """
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError:
        return False

    if len(head) < 4:
        return False

    # MP3: ID3 tag, or MPEG audio frame sync (0xFF 0xEx/0xFx)
    if head[:3] == b"ID3":
        return True
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return True
    # WAV / AIFF
    if head[:4] == b"RIFF" or head[:4] == b"FORM":
        return True
    # OGG
    if head[:4] == b"OggS":
        return True
    # FLAC
    if head[:4] == b"fLaC":
        return True
    # MP4 / M4A (ftyp box at offset 4)
    if head[4:8] == b"ftyp":
        return True

    return False


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


def process_audio(input_path: str, output_path: str, title: str, author: str) -> dict:
    """
    Normalize audio and write ID3 tags.
    Returns dict with duration_seconds and size_mb.

    Falls back to a no-op passthrough when:
      - PIPELINE_STEP_AUDIO_PROCESSING is disabled in settings, or
      - ffmpeg is not installed on the host.

    Handles two TTS output formats automatically:
      1. Valid MP3/AAC  → loudnorm → re-encode fallback → passthrough
      2. Raw PCM (s16le 24kHz mono, Gemini TTS) → forced decode → loudnorm
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

    # Raw PCM input args for the headerless-PCM case. Gemini TTS emits signed
    # 16-bit little-endian PCM at 24 000 Hz mono. These can be overridden via
    # settings if a provider uses a different raw layout.
    pcm_rate = str(getattr(settings, "TTS_PCM_SAMPLE_RATE", 24000) or 24000)
    pcm_in = ["-f", "s16le", "-ar", pcm_rate, "-ac", "1"]

    is_pcm = not _looks_like_container(input_path)
    if is_pcm:
        log.info(
            "audio: input has no audio-container header — treating as raw PCM "
            "(s16le %s Hz mono) and transcoding explicitly", pcm_rate,
        )

    # Build the input-side args once based on detection.
    in_args = pcm_in if is_pcm else ["-fflags", "+genpts"]

    try:
        # Primary: with loudness normalisation.
        _run([
            "ffmpeg", "-y",
            *in_args, "-i", input_path,
            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
            "-ar", "44100", "-ac", "2", "-b:a", "128k",
            *_meta, output_path,
        ])

    except FileNotFoundError:
        log.warning("ffmpeg vanished mid-run, falling back to passthrough")
        return _fallback_passthrough(input_path, output_path)

    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[-500:]
        log.warning("ffmpeg loudnorm failed (exit %s): %s — retrying without filter", exc.returncode, stderr)

        # Second attempt: skip loudnorm and force raw-PCM input. Both failure
        # modes resolve here — a PCM file that choked on loudnorm, and a file we
        # mistook for a container that's actually headerless PCM.
        try:
            _run([
                "ffmpeg", "-y",
                *pcm_in, "-i", input_path,
                "-ar", "44100", "-ac", "2", "-b:a", "128k",
                *_meta, output_path,
            ])
            log.info("audio: recovered via raw-PCM re-encode fallback")
        except subprocess.CalledProcessError as exc2:
            log.error(
                "ffmpeg re-encode also failed (exit %s): %s",
                exc2.returncode,
                (exc2.stderr or b"").decode("utf-8", errors="replace")[-400:],
            )
            return _fallback_passthrough(input_path, output_path)

    # Get duration via ffprobe (best-effort)
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
