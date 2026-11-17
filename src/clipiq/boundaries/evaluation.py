"""Compare predicted boundaries with separate human annotations (never train on them)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable


def evaluate_boundaries(
    proposals: Iterable[dict[str, Any]],
    annotation_csv: str | Path,
    *,
    video_id: str | None = None,
) -> dict[str, Any]:
    """Return per-annotation errors and aggregates, or None if no reviewed values.

    This is descriptive evaluation on reviewed examples, not an unbiased
    population estimate. One candidate can belong to a merged proposal.
    """
    rows = list(proposals)
    with Path(annotation_csv).open(encoding="utf-8", newline="") as handle:
        human = list(csv.DictReader(handle))
    details: list[dict[str, Any]] = []
    for annotation in human:
        if video_id is not None and annotation.get("video_id") != video_id:
            continue
        cid = str(annotation["candidate_id"])
        match = next((c for c in rows if cid in {str(x) for x in c.get("source_candidate_ids", [])}), None)
        if match is None:
            details.append({"candidate_id": cid, "matched": False, "start_abs_error_s": None, "end_abs_error_s": None})
            continue
        actual_start, actual_end = annotation.get("review_start", ""), annotation.get("review_end", "")
        start_error = abs(float(match["start"]) - float(actual_start)) if actual_start.strip() else None
        end_error = abs(float(match["end"]) - float(actual_end)) if actual_end.strip() else None
        details.append({
            "candidate_id": cid,
            "matched": True,
            "proposal_clip_id": match["clip_id"],
            "predicted_start": match["start"],
            "predicted_end": match["end"],
            "start_abs_error_s": round(start_error, 3) if start_error is not None else None,
            "end_abs_error_s": round(end_error, 3) if end_error is not None else None,
        })
    start_errors = [r["start_abs_error_s"] for r in details if r.get("start_abs_error_s") is not None]
    end_errors = [r["end_abs_error_s"] for r in details if r.get("end_abs_error_s") is not None]
    return {
        "compared_annotation_count": len(details),
        "start_error_count": len(start_errors),
        "end_error_count": len(end_errors),
        "mean_start_abs_error_s": round(sum(start_errors) / len(start_errors), 3) if start_errors else None,
        "mean_end_abs_error_s": round(sum(end_errors) / len(end_errors), 3) if end_errors else None,
        "details": details,
    }
