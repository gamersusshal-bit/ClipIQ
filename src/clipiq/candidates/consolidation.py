def consolidate_candidates(candidates, overlap_threshold=0.5):
    """
    Merge overlapping candidates only when both contain matching,
    non-empty speech.
    """

    if not candidates:
        return []

    candidates = sorted(candidates, key=lambda item: item["start"])
    consolidated = []

    for original_index, candidate in enumerate(candidates):
        candidate = candidate.copy()
        candidate["source_candidate_ids"] = list(candidate.get("source_candidate_ids", [candidate.get("candidate_id", original_index)]))
        matched = False

        candidate_speech = candidate.get("speech", "").strip().lower()

        for existing in consolidated:
            existing_speech = existing.get("speech", "").strip().lower()

            if not candidate_speech or not existing_speech:
                continue

            intersection = max(
                0.0,
                min(existing["end"], candidate["end"])
                - max(existing["start"], candidate["start"])
            )

            union = max(existing["end"], candidate["end"]) - min(
                existing["start"], candidate["start"]
            )

            overlap = intersection / union if union > 0 else 0.0

            if (
                candidate_speech == existing_speech
                and overlap >= overlap_threshold
            ):
                existing["start"] = min(
                    existing["start"], candidate["start"]
                )
                existing["end"] = max(
                    existing["end"], candidate["end"]
                )
                existing["duration"] = (
                    existing["end"] - existing["start"]
                )

                existing["audio_event_count"] = (
                    existing.get("audio_event_count", 0)
                    + candidate.get("audio_event_count", 0)
                )

                existing["maximum_audio_energy"] = max(
                    existing.get("maximum_audio_energy", 0.0),
                    candidate.get("maximum_audio_energy", 0.0)
                )

                existing["source_candidate_ids"] = list(dict.fromkeys(
                    existing["source_candidate_ids"] + candidate["source_candidate_ids"]))
                matched = True
                break

        if not matched:
            consolidated.append(candidate)

    for index, candidate in enumerate(consolidated):
        candidate["candidate_id"] = index

    return consolidated
