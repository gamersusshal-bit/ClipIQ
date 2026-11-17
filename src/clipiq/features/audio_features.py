import numpy as np
import librosa

from clipiq.config import AUDIO_SAMPLE_RATE


def analyze_audio(
    audio_path,
    sample_rate=AUDIO_SAMPLE_RATE,
    frame_length=2048,
    hop_length=512
):
    """
    Extract RMS audio energy features using librosa.
    """

    audio, sr = librosa.load(
        str(audio_path),
        sr=sample_rate,
        mono=True
    )

    rms = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=hop_length
    )[0]

    times = librosa.frames_to_time(
        np.arange(len(rms)),
        sr=sr,
        hop_length=hop_length
    )

    return {
        "rms": rms,
        "times": times,
        "sample_rate": sr,
        "duration": len(audio) / sr,
        "average_energy": float(np.mean(rms)) if len(rms) else 0.0,
        "maximum_energy": float(np.max(rms)) if len(rms) else 0.0
    }


def get_audio_energy_for_block(audio_features, start, end):
    """
    Calculate average and maximum RMS energy within a time interval.
    """

    rms = np.asarray(audio_features["rms"])
    times = np.asarray(audio_features["times"])

    mask = (times >= start) & (times <= end)
    block_energy = rms[mask]

    if len(block_energy) == 0:
        return {
            "average_energy": 0.0,
            "maximum_energy": 0.0
        }

    return {
        "average_energy": float(np.mean(block_energy)),
        "maximum_energy": float(np.max(block_energy))
    }
