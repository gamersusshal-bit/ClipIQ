#!/usr/bin/env python3
"""Batch-mode adapter for ClipIQ boundary refinement.

Example:
python tools/run_boundary_review.py --video data/raw/test_video2.mp4 \
    --candidates-csv data/results/candidates.csv \
    --transcript-json data/analysis/transcription.json \
    --out data/validation/boundary_review --render 5
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path

from clipiq.boundaries import (
    BoundaryConfig, export_boundary_report, refine_candidates, render_refined_clips,
)


def video_duration(path: Path) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        capture_output=True, check=True, text=True,
    )
    return float(result.stdout.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description="Review ClipIQ candidate boundaries")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--candidates-csv", type=Path, required=True)
    parser.add_argument("--transcript-json", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--render", type=int, default=0, help="Number of video clips to render")
    parser.add_argument("--end-lookahead", type=float, default=22.0)
    args = parser.parse_args()
    if not args.video.is_file():
        parser.error(f"Video does not exist: {args.video}")
    with args.candidates_csv.open(newline="", encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    transcript = []
    if args.transcript_json:
        with args.transcript_json.open(encoding="utf-8") as f:
            transcript = json.load(f)
    duration = video_duration(args.video)
    proposed = refine_candidates(
        candidates, transcript, duration,
        BoundaryConfig(end_lookahead_s=args.end_lookahead),
    )
    json_file, csv_file = export_boundary_report(proposed, args.out)
    print(f"Generated {len(proposed)} clip proposals")
    print(f"JSON: {json_file}\nCSV: {csv_file}")
    for clip in proposed[:5]:
        print(f"  {clip['clip_id']}: {clip['start']:.2f}–{clip['end']:.2f}s "
              f"(sources {','.join(clip['source_candidate_ids'])})")
    if args.render:
        rendered = render_refined_clips(args.video, proposed, args.out / "clips", limit=args.render)
        for file in rendered:
            print(f"Clip: {file}")


if __name__ == "__main__":
    main()
