import numpy as np


def detect_audio_peaks(audio_features, percentile=95):
    """
    Detect high-energy audio events using an RMS percentile threshold.
    """

    rms = np.asarray(audio_features["rms"])
    times = audio_features["times"]

    if len(rms) == 0 or np.max(rms) <= 1e-8:
        return []

    threshold = np.percentile(rms, percentile)

    active_indices = np.where(rms >= threshold)[0]

    return [
        {
            "time": float(times[index]),
            "energy": float(rms[index])
        }
        for index in active_indices
    ]


def filter_audio_peaks(peaks, min_energy=None):
    """Filter detected peaks using an optional energy threshold."""

    if min_energy is None:
        return peaks

    return [
        peak for peak in peaks
        if peak["energy"] >= min_energy
    ]


def group_audio_peaks(peaks, max_gap=2.0):
    """Group nearby audio peaks into continuous events."""

    if not peaks:
        return []

    peaks = sorted(peaks, key=lambda item: item["time"])

    groups = []
    current = {
        "start": peaks[0]["time"],
        "end": peaks[0]["time"],
        "peak_count": 1,
        "maximum_energy": peaks[0]["energy"]
    }

    for peak in peaks[1:]:
        if peak["time"] - current["end"] <= max_gap:
            current["end"] = peak["time"]
            current["peak_count"] += 1
            current["maximum_energy"] = max(
                current["maximum_energy"],
                peak["energy"]
            )
        else:
            groups.append(current)
            current = {
                "start": peak["time"],
                "end": peak["time"],
                "peak_count": 1,
                "maximum_energy": peak["energy"]
            }

    groups.append(current)
    return groups


def generate_audio_speech_candidates(
    audio_events,
    speech_blocks,
    video_duration,
    speech_gap=3.0,
    padding=3.0
):
    """
    Connect audio events to nearby speech blocks and create candidates.
    """

    candidates = []

    for event in audio_events:
        matching_speech = None
        closest_gap = float("inf")

        for block in speech_blocks:
            if block["end"] < event["start"]:
                gap = event["start"] - block["end"]
            elif block["start"] > event["end"]:
                gap = block["start"] - event["end"]
            else:
                gap = 0

            if gap <= speech_gap and gap < closest_gap:
                matching_speech = block
                closest_gap = gap

        if matching_speech is not None:
            start = min(event["start"], matching_speech["start"]) - padding
            end = max(event["end"], matching_speech["end"]) + padding
            speech = matching_speech["text"]
        else:
            start = event["start"] - padding
            end = event["end"] + padding
            speech = ""

        start = max(0.0, start)
        end = min(float(video_duration), end)

        if end <= start:
            continue

        candidates.append({
            "start": start,
            "end": end,
            "duration": end - start,
            "audio_event_count": event["peak_count"],
            "maximum_audio_energy": event["maximum_energy"],
            "speech": speech
        })

    return candidates
