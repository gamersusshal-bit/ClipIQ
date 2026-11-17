class _LazyWhisper:
    def load_model(self, *args, **kwargs):
        try:
            import whisper as engine
        except ImportError as exc:
            raise RuntimeError("Install speech support: pip install -e '.[speech]'") from exc
        return engine.load_model(*args, **kwargs)


whisper = _LazyWhisper()

from pathlib import Path

from clipiq.config import WHISPER_MODEL


def transcribe_audio(audio_path, model_name=WHISPER_MODEL):
    """
    Transcribe audio using OpenAI Whisper.

    Returns transcription text, language, and timestamped segments.
    """

    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = whisper.load_model(model_name)

    result = model.transcribe(
        str(audio_path),
        verbose=False
    )

    return {
        "text": result["text"],
        "language": result["language"],
        "segments": [
            {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }
            for segment in result["segments"]
        ],
        "segment_count": len(result["segments"])
    }
