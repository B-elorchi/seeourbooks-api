


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

FFmpeg attempt strategy (3 stages):
  Stage 1 — auto-detected input format  + loudnorm filter
  Stage 2 — forced raw-PCM input        + loudnorm filter   ← catches false-positive container detection
  Stage 3 — forced raw-PCM input        (no loudnorm)       ← bare-minimum transcode fallback
If all three fail, the raw TTS file is passed through unchanged.

Background on the false-positive problem:
  Gemini TTS emits raw s16le PCM with no header.  The first two bytes of that
  PCM can be 0xFF 0xFB (or similar), which matches the MPEG sync-word pattern
  used by magic-byte sniffers — including ffprobe, which then reports
  time_base=1/0 and a 100 % decode-error rate.  Forcing `-f s16le` in Stage 2
  bypasses that misdetection reliably.
"""
import logging
import os
import shutil
import subprocess

from api.config.settings import settings

log = logging.getLogger(__name__)

_FFMPEG_AVAILABLE: bool | None = None   # cached after first check
_WARNED: bool = False                   # log the missing-ffmpeg warning only once


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

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
    """When ffmpeg is unavailable (or all retries fail), just move the raw TTS file."""
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
    with an explicit `-f s16le` input flag.

    Note: this check can still produce a false positive if the first two bytes
    of raw PCM happen to look like an MPEG sync word (0xFF 0xEx/0xFx).  That
    is why process_audio() has a Stage-2 retry that forces PCM input even when
    this function returns True.
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


def _ffmpeg_attempt(
    input_path: str,
    output_path: str,
    in_flags: list[str],
    af_args: list[str],
    meta: list[str],
    stage: int,
) -> bool:
    """
    Run a single ffmpeg invocation.

    Returns True on success, False on CalledProcessError (logs the stderr).
    Re-raises FileNotFoundError so the caller can handle ffmpeg disappearing.
    """
    cmd = [
        "ffmpeg", "-y",
        *in_flags, "-i", input_path,
        *af_args,
        "-ar", "44100", "-ac", "2", "-b:a", "128k",
        *meta,
        output_path,
    ]
    try:
        _run(cmd)
        log.info("audio: ffmpeg Stage %d succeeded", stage)
        return True
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace")[-600:]
        log.warning(
            "audio: ffmpeg Stage %d failed (exit %s):\n%s",
            stage, exc.returncode, stderr,
        )
        return False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_audio(input_path: str, output_path: str, title: str, author: str) -> dict:
    """
    Normalize audio loudness (EBU R128, -16 LUFS) and write ID3 tags.

    Returns a dict with ``duration_seconds`` and ``size_mb``.

    Falls back to a no-op passthrough when:
      - ``PIPELINE_STEP_AUDIO_PROCESSING`` is disabled in settings, or
      - ffmpeg is not installed on the host, or
      - all three ffmpeg retry stages fail.

    Three-stage ffmpeg strategy
    ---------------------------
    Stage 1  Auto-detected input format  +  loudnorm filter
             Normal path for MP3/WAV/AAC/FLAC.

    Stage 2  Forced raw-PCM input  +  loudnorm filter
             Catches the case where ``_looks_like_container()`` returned True
             for a headerless PCM file whose first bytes looked like an MPEG
             sync word.  The ``time_base 1/0`` / ``Decode error rate 1`` errors
             in ffmpeg logs are the signature of this false-positive.

    Stage 3  Forced raw-PCM input  —  *no* loudnorm filter
             Last resort: produce any valid MP3 without loudness correction.
             Covers rare PCM variants that confuse the loudnorm graph even
             after the input is correctly typed.
    """
    if not settings.PIPELINE_STEP_AUDIO_PROCESSING:
        return _fallback_passthrough(input_path, output_path)

    if not _ffmpeg_available():
        return _fallback_passthrough(input_path, output_path)

    meta = [
        "-metadata", f"title={title}",
        "-metadata", f"artist={author}",
        "-metadata", f"album=SeeOurBook Summary",
    ]

    # Raw PCM descriptor — Gemini TTS emits s16le at 24 000 Hz mono.
    # Override TTS_PCM_SAMPLE_RATE in settings if your provider differs.
    pcm_rate = str(getattr(settings, "TTS_PCM_SAMPLE_RATE", 24000) or 24000)
    pcm_in   = ["-f", "s16le", "-ar", pcm_rate, "-ac", "1"]
    loudnorm = ["-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]

    is_pcm = not _looks_like_container(input_path)
    log.info(
        "audio: input detected as %s (%s)",
        "raw PCM" if is_pcm else "container (MP3/WAV/…)",
        input_path,
    )
    auto_in = pcm_in if is_pcm else ["-fflags", "+genpts"]

    try:
        # ------------------------------------------------------------------
        # Stage 1: auto-detected input + loudnorm
        # ------------------------------------------------------------------
        if _ffmpeg_attempt(input_path, output_path, auto_in, loudnorm, meta, stage=1):
            pass  # success — fall through to duration probe

        # ------------------------------------------------------------------
        # Stage 2: force raw-PCM input + loudnorm
        #   Fixes false-positive container detection (time_base 1/0 errors).
        # ------------------------------------------------------------------
        elif _ffmpeg_attempt(input_path, output_path, pcm_in, loudnorm, meta, stage=2):
            log.info("audio: Stage 2 (forced PCM + loudnorm) recovered the file")

        # ------------------------------------------------------------------
        # Stage 3: force raw-PCM input, NO loudnorm
        #   Bare-minimum transcode — produces valid MP3 without normalisation.
        # ------------------------------------------------------------------
        elif _ffmpeg_attempt(input_path, output_path, pcm_in, [], meta, stage=3):
            log.info("audio: Stage 3 (forced PCM, no loudnorm) recovered the file")

        else:
            log.error(
                "audio: all three ffmpeg stages failed for %s — "
                "falling back to raw TTS passthrough",
                input_path,
            )
            return _fallback_passthrough(input_path, output_path)

    except FileNotFoundError:
        log.warning("ffmpeg vanished mid-run, falling back to passthrough")
        return _fallback_passthrough(input_path, output_path)

    # ----------------------------------------------------------------------
    # Duration probe (best-effort — never blocks delivery)
    # ----------------------------------------------------------------------
    duration = 0
    try:
        if shutil.which("ffprobe"):
            probe = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    output_path,
                ],
                capture_output=True,
                text=True,
            )
            raw = probe.stdout.strip()
            if raw:
                duration = float(raw)
    except (subprocess.SubprocessError, ValueError):
        pass

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    return {"duration_seconds": round(duration), "size_mb": round(size_mb, 2)}