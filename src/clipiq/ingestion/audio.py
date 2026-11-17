import subprocess
from pathlib import Path

from clipiq.config import AUDIO_SAMPLE_RATE


def extract_audio(video_path, output_path, sample_rate=AUDIO_SAMPLE_RATE):
    """
    Extract audio from a video and save it as a WAV file.

    Uses FFmpeg for audio extraction.
    """

    video_path = Path(video_path)
    output_path = Path(output_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    command = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-ac", "1",
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(output_path)
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Audio extraction failed: {result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError("Audio extraction produced no output file.")

    return output_path
