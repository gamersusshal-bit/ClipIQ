import numpy as np


def percentile_normalize(values):
    """Convert values into percentile ranks between 0 and 1."""

    values = np.asarray(values, dtype=float)

    if len(values) == 0:
        return values


    if len(values) == 1:
        return np.ones(1)

    # Average tied ranks; constant signals contain no relative activity.
    if np.ptp(values) <= 1e-12:
        return np.zeros_like(values)
    from scipy.stats import rankdata
    return (rankdata(values, method="average") - 1) / (len(values) - 1)


def build_activity_timeline(audio_features, visual_features):
    """
    Combine audio energy and visual activity on a shared time axis.
    """

    audio_times = np.asarray(audio_features["times"])
    audio_values = np.asarray(audio_features["rms"])

    visual_times = np.asarray(visual_features["times"])
    visual_values = np.asarray(visual_features["visual_activity"])

    if len(audio_times) == 0 or len(visual_times) == 0:
        raise ValueError("Audio or visual activity timeline is empty.")

    start = max(audio_times.min(), visual_times.min())
    end = min(audio_times.max(), visual_times.max())

    if end <= start:
        raise ValueError("Audio and visual timelines do not overlap.")

    timeline = np.unique(
        np.concatenate([
            audio_times[(audio_times >= start) & (audio_times <= end)],
            visual_times[(visual_times >= start) & (visual_times <= end)]
        ])
    )

    aligned_audio = np.interp(timeline, audio_times, audio_values)
    aligned_visual = np.interp(timeline, visual_times, visual_values)

    audio_percentile = percentile_normalize(aligned_audio)
    visual_percentile = percentile_normalize(aligned_visual)

    combined_activity = (
        0.5 * audio_percentile +
        0.5 * visual_percentile
    )

    return {
        "times": timeline,
        "audio_activity": aligned_audio,
        "visual_activity": aligned_visual,
        "audio_percentile": audio_percentile,
        "visual_percentile": visual_percentile,
        "combined_activity": combined_activity
    }


def detect_activity_windows(activity_timeline, percentile=90, max_gap=2.0):
    """
    Detect high combined-activity windows and merge nearby timestamps.
    """

    times = activity_timeline["times"]
    activity = activity_timeline["combined_activity"]

    if len(times) == 0 or np.max(activity) <= 1e-12:
        return []

    threshold = np.percentile(activity, percentile)
    active_indices = np.where(activity >= threshold)[0]

    if len(active_indices) == 0:
        return []

    windows = []
    current = {
        "start": float(times[active_indices[0]]),
        "end": float(times[active_indices[0]]),
        "activity_values": [float(activity[active_indices[0]])]
    }

    for index in active_indices[1:]:
        timestamp = float(times[index])

        if timestamp - current["end"] <= max_gap:
            current["end"] = timestamp
            current["activity_values"].append(float(activity[index]))
        else:
            windows.append({
                "start": current["start"],
                "end": current["end"],
                "average_activity": float(np.mean(current["activity_values"])),
                "maximum_activity": float(np.max(current["activity_values"]))
            })

            current = {
                "start": timestamp,
                "end": timestamp,
                "activity_values": [float(activity[index])]
            }

    windows.append({
        "start": current["start"],
        "end": current["end"],
        "average_activity": float(np.mean(current["activity_values"])),
        "maximum_activity": float(np.max(current["activity_values"]))
    })

    return windows
