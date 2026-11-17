from clipiq.candidates.activity import (
    build_activity_timeline,
    detect_activity_windows
)


def generate_activity_candidates(
    audio_features,
    visual_features,
    percentile=90,
    max_gap=2.0,
    padding=3.0
):
    """
    Generate candidates from combined audio and visual activity.

    This is an independent candidate-generation strategy.
    """

    timeline = build_activity_timeline(
        audio_features,
        visual_features
    )

    windows = detect_activity_windows(
        timeline,
        percentile=percentile,
        max_gap=max_gap
    )

    candidates = []

    for index, window in enumerate(windows):
        start = max(0.0, window["start"] - padding)
        end = min(
            audio_features["duration"],
            window["end"] + padding
        )

        if end <= start:
            continue

        candidates.append({
            "candidate_id": index,
            "start": start,
            "end": end,
            "duration": end - start,
            "speech": "",
            "audio_event_count": window.get("event_count", 0),
            "generation_method": "multimodal_activity"
        })

    return candidates
