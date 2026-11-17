import pandas as pd


DEFAULT_WEIGHTS = {
    "avg_audio_energy": 0.15,
    "max_audio_energy": 0.30,
    "avg_visual_activity": 0.10,
    "max_visual_activity": 0.25,
    "audio_event_count": 0.20
}


def rank_candidates(features, weights=None):
    """
    Rank candidates using percentile-normalized feature scores.

    This is a heuristic baseline, not a trained ML model.
    """

    if features.empty:
        return features.copy()

    weights = weights or DEFAULT_WEIGHTS

    ranked = features.copy()

    score_columns = []

    for feature, weight in weights.items():
        if feature not in ranked.columns:
            raise ValueError(f"Missing ranking feature: {feature}")

        score_column = f"{feature}_percentile"

        ranked[score_column] = ranked[feature].rank(
            method="average",
            pct=True
        )

        score_columns.append((score_column, weight))

    ranked["ranking_score"] = sum(
        ranked[column] * weight
        for column, weight in score_columns
    )

    ranked = ranked.sort_values(
        "ranking_score",
        ascending=False
    ).reset_index(drop=True)

    ranked["rank"] = ranked.index + 1

    return ranked
