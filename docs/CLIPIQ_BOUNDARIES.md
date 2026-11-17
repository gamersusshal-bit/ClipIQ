# ClipIQ boundary refinement — non-invasive add-on

## Status and scope

This is a **working conservative v1 post-processor** that can use the data already produced by ClipIQ. It is not an LLM/video-language model and cannot reliably understand a complete story, wordless actions, or whether an explanation is interesting. It deliberately flags every proposal as requiring review.

It provides:

- High-score-first grouping of **overlapping** candidates, preserving all source candidate IDs.
- A maximum candidate-cluster duration to prevent an overlap chain from merging an entire video.
- Speech-aware alignment when a cut lands **inside** a Whisper segment.
- Limited continuation through contiguous unfinished speech segments (punctuation + pauses as weak signals).
- Conservative deduplication of near-identical proposed intervals, avoiding blind removal of short events nested inside a much longer segment.
- Review reasons, a short transcript excerpt, JSON and CSV exports.
- Accurate FFmpeg re-encoding of optional review clips.
- An integration API that reads your existing `results` dictionary without changing production code.

It **does not** claim to autonomously find the true complete narrative ending. Candidate 6's manually reviewed 125.768s ending is recorded as validation ground truth, not hardcoded into the algorithm.

## Install in your existing Mac project

1. Unzip the archive **into the ClipIQ project root**, keeping the contained `src/clipiq/boundaries/`, `tools/`, `tests/`, `docs/`, and `data/validation/` subdirectories. The archive deliberately does **not** include or overwrite `src/clipiq/__init__.py` or any existing production code.
2. Activate your existing `.venv312` environment. Your normal editable installation (`pip install -e .`) should expose these files; if it is already editable, no reinstall is usually needed.
3. Run the included tests:

   ```bash
   python -m unittest discover -s tests -p 'test_boundary_refiner.py' -v
   ```

## Use directly in your existing Jupyter notebook

This is the recommended first run because `results` (Airsoft) already exists:

```python
from pathlib import Path
import pandas as pd
from clipiq.boundaries import (
    BoundaryConfig,
    refine_from_results,
    export_boundary_report,
    render_refined_clips,
)

# 'results' is the existing output of process_video(test_video2.mp4)
proposals = refine_from_results(results)

summary = pd.DataFrame(proposals)
print(summary[["clip_id", "start", "end", "duration", "source_candidate_ids", "reason_codes"]].head(10).to_string(index=False))

folder = Path(project_root) / "data" / "validation" / "boundary_review_v1"
export_boundary_report(proposals, folder)

# Render the top 5 *proposals* to inspect them (not automatically publish them):
render_refined_clips(results["video_path"], proposals, folder / "clips", limit=5)
```

The proposal IDs (`refined_001`, etc.) are not the original candidate IDs. Use `source_candidate_ids` to compare against original annotations.

### Tunable behaviour

```python
config = BoundaryConfig(
    max_cluster_span_s=52,
    start_lookback_s=8,
    end_lookahead_s=22,
    max_clip_s=120,
)
proposals = refine_from_results(results, config)
```

`end_lookahead_s` is a search limit, **not** an automatic amount added. It only grows a clip when it detects a mid-speech cut. Increasing it does not grant the engine an understanding of a coherent 90-second topic.

### Command-line mode (for exported CSV/transcript JSON)

```bash
python tools/run_boundary_review.py \
  --video data/raw/test_video2.mp4 \
  --candidates-csv path/to/ranked_candidates.csv \
  --transcript-json path/to/transcription.json \
  --out data/validation/boundary_review_v1 \
  --render 5
```

Replace these video/CSV/transcript paths with the real paths. `--transcript-json` is optional, but with no transcript the engine cannot identify cut-off speech.

## How to read the audit flags

- `merged_overlapping_candidates`: two or more source candidates were clustered due to substantial overlap; this doesn't guarantee one storyline.
- `completed_cut_speech_segment`: the candidate ended within a recognized speech segment and the cut was moved to its end.
- `followed_unfinished_utterance`: continuation of an utterance that lacked terminal punctuation across adjacent segments.
- `aligned_start_to_speech`: an original cut began inside a speech segment.
- `long_clip_review`: a long proposal could contain multiple independent events.
- `possible_long_tail`: candidate ends a while after the last transcribed speech; don't auto-trim non-speech action.
- `no_transcript` / `no_speech_near_candidate`: need visual inspection; signal alone cannot certify an ending.
- `extension_rejected_duration_cap`: rejected a suggested extension rather than knowingly making an overly long clip.

## Known limitations (important)

1. **Whisper segments are not reliable semantic sentence/topic boundaries.** It may miss the ending or stop at a plausible sentence long before the *full* sequence completes.
2. **Overlapping candidates need not be the same event.** This is a conservative grouping heuristic, not learned narrative segmentation.
3. No learned semantic model, object/action tracking, replay/event detector, automatic vertical reframing, or captions are included.
4. No quality validation against the user's actual three source videos was possible here: the local MP4s and repository are not accessible in this environment.
5. Existing manual corrections for Airsoft 6 and 32 are in `data/validation/boundary_review_seed.csv`. These are held-out review references, not features or hardcoded interval outputs. Candidate 6's starting boundary was never individually confirmed, so the seed CSV intentionally leaves it blank.
6. Proposals are not publication-ready until reviewed, especially for wordless videos, sports finishes, topic changes and story completions.

## Next engineering milestone

A separate semantic event-level segmenter (using targeted transcript windows plus video/audio evidence) is required to determine *where a full idea/action ends*, not just whether Whisper cut inside a speech segment. The review dataset should be expanded with independent labelled start/end pairs to measure boundary error. Do not re-label synthetic tests as empirical model performance.

## Measure against human-reviewed Airsoft boundaries

Run **after** generating proposals:

```python
from clipiq.boundaries import evaluate_boundaries

review_file = Path(project_root) / "data" / "validation" / "boundary_review_seed.csv"
metrics = evaluate_boundaries(proposals, review_file, video_id="test_video2")
print(metrics)
```

For annotations with known corrections, this measures absolute difference between the proposal and the reviewed boundary. Unknown/blank corrections are not treated as zero error. Since we have only two validated examples, these numbers are **diagnostic**, not a scientifically valid performance estimate. Expect the speech-only engine to fail on many long explanations and visually driven highlights; these are the next models to build.
