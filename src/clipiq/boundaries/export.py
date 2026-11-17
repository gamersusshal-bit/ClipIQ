"""JSON/CSV exports and optional FFmpeg rendering for proposed boundary clips."""
from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable


SUMMARY_COLUMNS = (
    "clip_id", "start", "end", "duration", "score", "source_candidate_ids",
    "source_start", "source_end", "reason_codes", "review_required",
    "review_notes", "transcript_excerpt",
)


def export_boundary_report(clips: Iterable[dict[str, Any]], directory: str | Path) -> tuple[Path, Path]:
    """Export human-inspectable proposals. Does not update any original CSV."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    data = list(clips)
    json_file = path / "boundary_proposals.json"
    csv_file = path / "boundary_proposals.csv"
    json_file.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    with csv_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in data:
            serialized = dict(row)
            for key in ("source_candidate_ids", "reason_codes", "review_notes"):
                serialized[key] = " | ".join(str(s) for s in row.get(key, []))
            writer.writerow(serialized)
    return json_file, csv_file


def _render_one(video_path: Path, start: float, end: float, output_path: Path, ffmpeg: str) -> None:
    if not (0 <= start < end):
        raise ValueError(f"Invalid clip interval: {start}..{end}")
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
        "-ss", str(start), "-i", str(video_path),
        "-t", str(end - start),
        "-map", "0:v:0", "-map", "0:a?", "-sn",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
        "-c:a", "aac", "-movflags", "+faststart", str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise RuntimeError(f"FFmpeg did not create a clip: {output_path}")


def render_refined_clips(
    video_path: str | Path,
    clips: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    limit: int | None = 5,
    ffmpeg: str = "ffmpeg",
) -> list[Path]:
    """Render proposed clips with accurate re-encoding; no source video changes.

    ``limit=None`` renders all proposals. By default, only the top five are rendered.
    """
    video = Path(video_path)
    if not video.is_file():
        raise FileNotFoundError(f"Cannot find source video: {video}")
    if shutil.which(ffmpeg) is None:
        raise FileNotFoundError(f"FFmpeg not found: {ffmpeg}")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >=0 or None")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    selected = list(clips)
    if limit is not None:
        selected = selected[:limit]
    rendered: list[Path] = []
    for row in selected:
        clip_id = str(row["clip_id"])
        if not clip_id.replace("_", "").isalnum():
            raise ValueError(f"Unexpected clip_id: {clip_id!r}")
        output = folder / f"{clip_id}.mp4"
        _render_one(video, float(row["start"]), float(row["end"]), output, ffmpeg)
        rendered.append(output)
    return rendered
