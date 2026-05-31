"""
Audio post-processing via FFmpeg.
Normalizes loudness to EBU R128 (-16 LUFS) and writes ID3 tags.
"""
import os
import subprocess
from api.config.settings import settings


def process_audio(input_path: str, output_path: str, title: str, author: str) -> dict:
    """
    Normalize audio and write ID3 tags.
    Returns dict with duration_seconds and size_mb.
    """
    if not settings.PIPELINE_STEP_AUDIO_PROCESSING:
        os.rename(input_path, output_path)
        size = os.path.getsize(output_path) / (1024 * 1024)
        return {"duration_seconds": 0, "size_mb": round(size, 2)}

    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-ar", "44100",
        "-ac", "2",
        "-b:a", "128k",
        "-metadata", f"title={title}",
        "-metadata", f"artist={author}",
        "-metadata", f"album=SeeOurBook Summary",
        output_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)

    # Get duration via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", output_path],
        capture_output=True, text=True,
    )
    duration = float(probe.stdout.strip()) if probe.stdout.strip() else 0
    size = os.path.getsize(output_path) / (1024 * 1024)

    return {"duration_seconds": round(duration), "size_mb": round(size, 2)}
