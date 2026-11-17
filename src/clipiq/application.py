"""Integrated workflow: cached analysis -> evidence selection -> safe exports."""
from __future__ import annotations
from dataclasses import asdict
from datetime import datetime, timezone
import logging
from pathlib import Path
import time
import uuid

from clipiq.cache import atomic_json
from clipiq.persistent import analyze
from clipiq.selection import SelectionConfig, propose
from clipiq.media.production import render, captions
from clipiq.boundaries import export_boundary_report
from clipiq.evaluation.export import export_pipeline_results

LOG = logging.getLogger(__name__)


def run(video, output="data/processed/clipiq", *, top_k=5, max_duration=60.,
        model="base", no_speech=False, allow_download=False, transcript=None,
        legacy_dir=None, force=False, portrait=False, with_captions=False,
        render_clips=True, semantic_adapter=None):
    if top_k < 1:
        raise ValueError("top_k must be positive")
    config = SelectionConfig(max_duration=max_duration)
    video, output = Path(video).expanduser().resolve(), Path(output).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    started = time.monotonic()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    folder = output / "runs" / run_id
    folder.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "run_id": run_id, "video_path": str(video),
                "output_dir": str(folder), "status": "processing", "clips": [],
                "parameters": {**asdict(config), "top_k": top_k, "model": model,
                               "no_speech": no_speech, "portrait": portrait,
                               "captions": with_captions, "render": render_clips},
                "warnings": [], "errors": []}
    manifest_path = folder / "manifest.json"
    atomic_json(manifest_path, manifest)
    try:
        results = analyze(video, output / "cache", model=model, no_speech=no_speech,
                          allow_download=allow_download, transcript=transcript,
                          legacy_dir=legacy_dir, force=force)
        LOG.info("Selecting proposals from both candidate strategies")
        proposals = propose(results, config, semantic_adapter)
        manifest.update(video_duration=results["video_duration"],
                        source_fingerprint=results["source_fingerprint"], cache=results["cache_info"],
                        warnings=results["warnings"], proposal_count=len(proposals))
        # Preserve both baseline ranked tables for honest comparison.
        export_pipeline_results(results, folder / "baseline")
        export_boundary_report(proposals, folder / "reports")
        atomic_json(folder / "transcription.json", results["transcription"])
        manifest["clips"] = proposals
        for row in proposals:
            row["status"] = "proposal"
        eligible = [r for r in proposals if r["eligible_for_export"]]
        for row in eligible[:top_k] if render_clips else []:
            LOG.info("Rendering %s (%.2f–%.2fs)", row["clip_id"], row["start"], row["end"])
            try:
                row["export"] = render(video, row["start"], row["end"],
                    folder / "clips" / f"{row['clip_id']}.mp4", duration=results["video_duration"],
                    maximum=max_duration)
                row["status"] = "rendered"
                if portrait:
                    row["portrait_export"] = render(video, row["start"], row["end"],
                        folder / "portrait" / f"{row['clip_id']}.mp4", duration=results["video_duration"],
                        maximum=max_duration, portrait=True)
                if with_captions:
                    subtitle = folder / "clips" / f"{row['clip_id']}.srt"
                    if captions(results["transcription"]["segments"], row["start"], row["end"], subtitle):
                        row["captions_path"] = str(subtitle)
                        row["captions_precision"] = "segment-level, not word-aligned"
                    else:
                        row["review_notes"].append("No transcript for captions; video exported without captions")
            except (ValueError, OSError, RuntimeError) as exc:
                row["status"] = "export_failed"
                row["error"] = str(exc)
                manifest["errors"].append({"clip_id": row["clip_id"], "error": str(exc)})
            atomic_json(manifest_path, manifest)
        manifest["status"] = "partial_failure" if manifest["errors"] else "complete"
        if not proposals:
            manifest["warnings"].append("No candidate events detected; no highlights were invented.")
        if portrait:
            manifest["warnings"].append("Portrait output uses padded 9:16 framing; subject-aware cropping is unavailable.")
        manifest["elapsed_seconds"] = round(time.monotonic()-started, 3)
        atomic_json(manifest_path, manifest)
        atomic_json(output / "latest.json", {"manifest": str(manifest_path)})
        return manifest
    except Exception as exc:
        manifest.update(status="failed", elapsed_seconds=round(time.monotonic()-started, 3))
        manifest["errors"].append({"error": str(exc)})
        atomic_json(manifest_path, manifest)
        raise
