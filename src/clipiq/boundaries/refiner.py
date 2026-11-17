"""Conservative, reviewable post-processing for ClipIQ candidates.

This engine is NOT a semantic video-understanding model. It merges highly
intersecting detections, aligns cuts to observed speech segments when possible,
and keeps an audit trail of each modification.

The persistent ClipIQ workflow calls this engine for individual event boundaries.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import isfinite
from typing import Any, Iterable, Mapping, Sequence
import re


@dataclass(frozen=True)
class BoundaryConfig:
    overlap_threshold: float = 0.28
    max_cluster_span_s: float = 52.0
    start_lookback_s: float = 8.0
    end_lookahead_s: float = 22.0
    speech_edge_tolerance_s: float = 0.35
    speech_padding_s: float = 0.15
    sentence_pause_s: float = 0.50
    max_clip_s: float = 120.0
    long_clip_review_s: float = 45.0
    tail_review_s: float = 3.0
    near_duplicate_overlap: float = 0.80

    def __post_init__(self) -> None:
        if not 0 <= self.overlap_threshold <= 1:
            raise ValueError("overlap_threshold must be between 0 and 1")
        if not 0 <= self.near_duplicate_overlap <= 1:
            raise ValueError("near_duplicate_overlap must be between 0 and 1")
        for name, value in vars(self).items():
            if not isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name.endswith("_s") and value < 0:
                raise ValueError(f"{name} must be nonnegative")
        if self.max_clip_s <= 0 or self.max_cluster_span_s <= 0:
            raise ValueError("maximum durations must be positive")


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    start: float
    end: float
    ranking_score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class SpeechSegment:
    start: float
    end: float
    text: str


@dataclass
class RefinedClip:
    clip_id: str
    start: float
    end: float
    score: float
    source_candidate_ids: list[str]
    source_start: float
    source_end: float
    reason_codes: list[str] = field(default_factory=list)
    review_notes: list[str] = field(default_factory=list)
    review_required: bool = True
    transcript_excerpt: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["duration"] = round(self.duration, 3)
        for key in ("start", "end", "source_start", "source_end", "score"):
            data[key] = round(data[key], 5)
        return data


def _value(row: Mapping[str, Any], key: str, default: Any = None) -> Any:
    val = row.get(key, default)
    # Avoid pandas NaN / NA on optional fields.
    try:
        return default if val is None or (isinstance(val, float) and not isfinite(val)) else val
    except (TypeError, ValueError):
        return default


def _rows(source: Any) -> list[Mapping[str, Any]]:
    if source is None:
        return []
    if hasattr(source, "to_dict") and hasattr(source, "columns"):
        return list(source.to_dict(orient="records"))
    if isinstance(source, Mapping):
        if "ranked" in source:
            return _rows(source["ranked"])
        if "segments" in source:
            return _rows(source["segments"])
        raise ValueError("expected a sequence of rows or a 'ranked'/'segments' wrapper")
    return list(source)


def _float(value: Any, *, name: str) -> float:
    try:
        out = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric, received {value!r}") from exc
    if not isfinite(out):
        raise ValueError(f"{name} must be finite")
    return out


def _normalize_candidates(data: Any, duration: float) -> list[Candidate]:
    seen: set[str] = set()
    output: list[Candidate] = []
    for i, row in enumerate(_rows(data)):
        if not isinstance(row, Mapping):
            raise ValueError(f"candidate row {i} must be a mapping")
        key = str(_value(row, "candidate_id", i))
        if key in seen:
            raise ValueError(f"duplicate candidate_id {key!r}")
        seen.add(key)
        start = _float(row.get("start"), name="candidate start")
        end = _float(row.get("end"), name="candidate end")
        if start < 0 or end <= start or start >= duration:
            raise ValueError(f"invalid candidate interval {key}: {start}..{end}")
        score = _float(_value(row, "ranking_score", 0.0), name="ranking_score")
        output.append(Candidate(key, max(0.0, start), min(duration, end), score))
    return output


def _normalize_segments(data: Any, duration: float) -> list[SpeechSegment]:
    output: list[SpeechSegment] = []
    for i, row in enumerate(_rows(data)):
        if not isinstance(row, Mapping):
            raise ValueError(f"speech segment {i} must be a mapping")
        start = _float(row.get("start"), name="speech start")
        end = _float(row.get("end"), name="speech end")
        if start < 0 or end < start:
            raise ValueError(f"invalid speech interval at row {i}")
        if start >= duration or end <= start:
            continue
        output.append(SpeechSegment(start, min(end, duration), str(row.get("text") or "").strip()))
    return sorted(output, key=lambda s: (s.start, s.end))


def _overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def _shorter_overlap(a: Candidate, b: Candidate) -> float:
    common = _overlap(a.start, a.end, b.start, b.end)
    small, large = min(a.duration, b.duration), max(a.duration, b.duration)
    # A brief payoff fully nested in a lengthy context clip should not be
    # merged away just because it is temporally contained.
    if small / large < 0.35 and common / small >= 0.90:
        return 0.0
    return common / small


def group_overlapping_candidates(
    candidates: list[Candidate], config: BoundaryConfig
) -> list[list[Candidate]]:
    """High-score-first grouping; the maximum group span prevents chain collapse.

    Deliberately does NOT merge merely adjacent clips: temporal proximity is not
    proof that they belong to the same narrative event.
    """
    remaining = sorted(candidates, key=lambda c: (-c.ranking_score, c.start, c.candidate_id))
    groups: list[list[Candidate]] = []
    while remaining:
        group = [remaining.pop(0)]
        changed = True
        while changed:
            changed = False
            for candidate in list(remaining):
                span = max(x.end for x in [*group, candidate]) - min(x.start for x in [*group, candidate])
                if span > config.max_cluster_span_s:
                    continue
                if any(_shorter_overlap(candidate, member) >= config.overlap_threshold for member in group):
                    group.append(candidate)
                    remaining.remove(candidate)
                    changed = True
        groups.append(sorted(group, key=lambda c: (c.start, c.candidate_id)))
    return groups


def _nearest_sentence_start(
    start: float, segments: Sequence[SpeechSegment], config: BoundaryConfig
) -> tuple[float, bool]:
    """Only align a start when it falls *inside* a speech segment.

    Enlarging arbitrary starts to older sentences frequently adds unwanted setup.
    """
    covering = [
        s for s in segments
        if s.start + 0.1 < start < s.end - 0.05
        and start - s.start <= config.start_lookback_s
    ]
    if not covering:
        return start, False
    s = min(covering, key=lambda x: x.start)
    corrected = max(0.0, s.start - config.speech_padding_s)
    return corrected, corrected < start - 0.05


def _sentence_end(text: str) -> bool:
    return bool(re.search(r'[.!?][\s"\'\u201d\u2019]*$', text.strip()))


def _speech_end(
    end: float, segments: Sequence[SpeechSegment], config: BoundaryConfig, duration: float
) -> tuple[float, list[str], list[str]]:
    """Extend *an incomplete utterance* to a defensible speech boundary.

    Pauses are supporting evidence only. This does not claim to determine when
    visually driven actions, a long explanation or a whole narrative ends.
    """
    codes: list[str] = []
    notes: list[str] = []
    if not segments:
        return end, ["no_transcript"], ["No speech segments available; inspect boundaries manually"]

    containing = [
        (idx, s) for idx, s in enumerate(segments)
        if s.start < end - 0.05 and s.end > end + config.speech_edge_tolerance_s
    ]
    if not containing:
        # Do not extend if already outside speech or aligned with its end.
        trailing = [s for s in segments if 0 <= end - s.end <= 20.0]
        if trailing and end - max(s.end for s in trailing) > config.tail_review_s:
            notes.append("Possible empty tail after speech: review for non-speech action")
            codes.append("possible_long_tail")
        return end, codes, notes

    idx, seg = max(containing, key=lambda pair: pair[1].end)
    limit = min(duration, end + config.end_lookahead_s)
    new_end = min(limit, seg.end + config.speech_padding_s)
    codes.append("completed_cut_speech_segment")
    if seg.end > limit:
        notes.append("Speech extends beyond configured lookahead; sentence may still be cut")
        codes.append("lookahead_cap_reached")
        return new_end, codes, notes

    # Continue *only* if Whisper's segment has no terminal punctuation and
    # the next segment is nearly contiguous. Stop after an observed pause and
    # punctuation, or when the lookahead is exhausted.
    if not _sentence_end(seg.text):
        for nxt_idx in range(idx + 1, len(segments)):
            nxt = segments[nxt_idx]
            if nxt.start > limit or nxt.start - seg.end > config.sentence_pause_s:
                break
            new_end = min(limit, nxt.end + config.speech_padding_s)
            seg = nxt
            if seg.end >= limit or _sentence_end(seg.text):
                break
        codes.append("followed_unfinished_utterance")
        if not _sentence_end(seg.text):
            notes.append("No confident sentence ending detected; review manually")

    return new_end, codes, notes


def _excerpt(start: float, end: float, segments: Sequence[SpeechSegment], limit: int = 280) -> str:
    value = " ".join(s.text for s in segments if s.text and _overlap(start, end, s.start, s.end) > 0)
    return (value[:limit - 3] + "...") if len(value) > limit else value


def _refine_group(
    group: list[Candidate], segments: Sequence[SpeechSegment], duration: float, config: BoundaryConfig, number: int
) -> RefinedClip:
    start, end = min(x.start for x in group), max(x.end for x in group)
    codes: list[str] = []
    notes: list[str] = []
    if len(group) > 1:
        codes.append("merged_overlapping_candidates")
    new_start, aligned = _nearest_sentence_start(start, segments, config)
    if aligned:
        codes.append("aligned_start_to_speech")
        start = new_start
    new_end, end_codes, end_notes = _speech_end(end, segments, config, duration)
    codes.extend(end_codes)
    notes.extend(end_notes)
    end = min(duration, max(end, new_end))
    if end - start > config.max_clip_s:
        # Never knowingly cut the end of a word to satisfy max_clip_s.
        # Instead keep the original span only if that fits and flag any
        # proposed extension that cannot be applied safely.
        original_start, original_end = min(x.start for x in group), max(x.end for x in group)
        if original_end - original_start <= config.max_clip_s:
            start, end = original_start, original_end
            codes.append("extension_rejected_duration_cap")
            notes.append("Proposed correction exceeded max_clip_s; kept original interval")
        else:
            codes.append("already_exceeds_duration_cap")
            notes.append("Original candidate cluster exceeds max_clip_s; human segmentation required")
    if end - start >= config.long_clip_review_s:
        codes.append("long_clip_review")
        notes.append("Long sequence; verify topic continuity and whether it belongs in one clip")
    if not any(_overlap(start, end, s.start, s.end) > 0 for s in segments):
        codes.append("no_speech_near_candidate")
        notes.append("Non-speech content: transcript alone cannot determine action completion")

    return RefinedClip(
        clip_id=f"refined_{number:03d}", start=start, end=end,
        score=max(x.ranking_score for x in group),
        source_candidate_ids=[x.candidate_id for x in group],
        source_start=min(x.start for x in group),
        source_end=max(x.end for x in group),
        reason_codes=list(dict.fromkeys(codes)),
        review_notes=list(dict.fromkeys(notes)),
        review_required=True,  # Current engine cannot certify visual or semantic completeness.
        transcript_excerpt=_excerpt(start, end, segments),
    )


def _dedup_refined(clips: list[RefinedClip], threshold: float) -> list[RefinedClip]:
    """Suppress near-identical final windows, preserve their candidate provenance.

    Distinct partially-overlapping events are kept for manual review.
    """
    winners: list[RefinedClip] = []
    for clip in sorted(clips, key=lambda c: (-c.score, c.start)):
        match = next((w for w in winners if (
            _overlap(clip.start, clip.end, w.start, w.end) / max(clip.duration, w.duration)
        ) >= threshold), None)
        if match is not None:
            match.source_candidate_ids = list(dict.fromkeys(
                [*match.source_candidate_ids, *clip.source_candidate_ids]
            ))
            match.reason_codes.append("suppressed_near_duplicate_proposal")
            match.review_notes.append("Suppressed a near-identical proposal; verify that no unique payoff was lost")
        else:
            winners.append(clip)
    return winners


def refine_candidates(
    candidates: Any,
    transcript_segments: Any,
    video_duration: float,
    config: BoundaryConfig | None = None,
) -> list[dict[str, Any]]:
    """Refine ClipIQ's ranked candidate rows and Whisper segments.

    Parameters
    ----------
    candidates: pandas DataFrame, list[dict], or {'ranked': DataFrame}
    transcript_segments: list[dict], or Whisper {'segments': list[dict]}
    video_duration: seconds, e.g. results['video_duration']
    config: conservative, explicitly configurable thresholds

    Returned rows contain proposed boundaries, provenance and reasons. These
    are proposals for reviewing/rendering, not certified semantic highlights.
    """
    config = config or BoundaryConfig()
    duration = _float(video_duration, name="video_duration")
    if duration <= 0:
        raise ValueError("video_duration must be positive")
    all_candidates = _normalize_candidates(candidates, duration)
    speech = _normalize_segments(transcript_segments, duration)
    if not all_candidates:
        return []
    groups = group_overlapping_candidates(all_candidates, config)
    proposed = [_refine_group(group, speech, duration, config, i + 1) for i, group in enumerate(groups)]
    distinct = _dedup_refined(proposed, config.near_duplicate_overlap)
    ordered = sorted(distinct, key=lambda c: (-c.score, c.start))
    for number, clip in enumerate(ordered, start=1):
        clip.clip_id = f"refined_{number:03d}"
    return [clip.to_dict() for clip in ordered]


def refine_from_results(results: Mapping[str, Any], config: BoundaryConfig | None = None) -> list[dict[str, Any]]:
    """Direct bridge to the ClipIQ ``process_video`` results dictionary."""
    activity = results["multimodal_activity"]
    transcript = results.get("transcription") or {"segments": []}
    duration = results["video_duration"]
    return refine_candidates(activity["ranked"], transcript, duration, config)
