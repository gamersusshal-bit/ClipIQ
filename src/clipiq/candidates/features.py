import pandas as pd

from clipiq.features.audio_features import get_audio_energy_for_block
from clipiq.features.visual_features import get_visual_activity_for_block


def extract_candidate_features(
    candidates,
    audio_features,
    visual_features
):
    """
    Build a structured feature table for detected candidates.
    """

    records = []

    for candidate in candidates:
        start = candidate["start"]
        end = candidate["end"]

        audio = get_audio_energy_for_block(
            audio_features, start, end
        )

        visual = get_visual_activity_for_block(
            visual_features, start, end
        )

        duration = max(0.0, end - start)

        records.append({
            "candidate_id": candidate.get("candidate_id"),
            "source_candidate_ids": candidate.get("source_candidate_ids", [candidate.get("candidate_id")]),
            "start": start,
            "end": end,
            "duration": duration,
            "audio_event_count": candidate.get("audio_event_count", 0),
            "avg_audio_energy": audio["average_energy"],
            "max_audio_energy": audio["maximum_energy"],
            "avg_visual_activity": visual["average_visual_activity"],
            "max_visual_activity": visual["maximum_visual_activity"],
            "speech": candidate.get("speech", "")
        })

    return pd.DataFrame(records)
