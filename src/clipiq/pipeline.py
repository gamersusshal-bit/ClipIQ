from pathlib import Path

from clipiq.config import WHISPER_MODEL

from clipiq.ingestion.video import load_video, release_video
from clipiq.ingestion.audio import extract_audio

from clipiq.transcription.whisper_engine import transcribe_audio

from clipiq.features.audio_features import analyze_audio
from clipiq.features.visual_features import analyze_visuals

from clipiq.analysis.speech import (
    group_speech_segments,
    score_speech_block
)

from clipiq.candidates.audio_speech import (
    detect_audio_peaks,
    group_audio_peaks,
    generate_audio_speech_candidates
)

from clipiq.candidates.activity_pipeline import (
    generate_activity_candidates
)

from clipiq.candidates.consolidation import consolidate_candidates
from clipiq.candidates.features import extract_candidate_features

from clipiq.ranking.scoring import rank_candidates


def process_video(video_path, output_dir, model_name=WHISPER_MODEL):
    """
    Process a video using both independent candidate-generation methods.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    video_data = load_video(video_path)

    try:
        video_duration = video_data["duration"]
    finally:
        release_video(video_data)

    audio_path = output_dir / "extracted_audio.wav"
    extract_audio(video_path, audio_path)

    transcription = transcribe_audio(audio_path, model_name=model_name)

    speech_blocks = group_speech_segments(transcription["segments"])

    for block in speech_blocks:
        block.update(score_speech_block(block["text"]))

    audio_features = analyze_audio(audio_path)
    visual_features = analyze_visuals(video_path)

    return assemble_results(video_path, video_duration, audio_path, transcription,
                            speech_blocks, audio_features, visual_features)


def assemble_results(video_path, video_duration, audio_path, transcription,
                     speech_blocks, audio_features, visual_features):
    """Shared candidate/ranking stages for notebook and persistent workflows."""
    audio_peaks = detect_audio_peaks(audio_features)
    audio_events = group_audio_peaks(audio_peaks)

    # Strategy 1: Audio events combined with speech
    audio_speech_candidates = generate_audio_speech_candidates(
        audio_events=audio_events,
        speech_blocks=speech_blocks,
        video_duration=video_duration
    )

    audio_speech_candidates = consolidate_candidates(
        audio_speech_candidates
    )

    # Strategy 2: Combined audio and visual activity
    activity_candidates = generate_activity_candidates(
        audio_features=audio_features,
        visual_features=visual_features
    )

    activity_candidates = consolidate_candidates(activity_candidates)

    # Extract and rank each strategy independently
    audio_speech_features = extract_candidate_features(
        audio_speech_candidates,
        audio_features,
        visual_features
    )

    activity_features = extract_candidate_features(
        activity_candidates,
        audio_features,
        visual_features
    )

    audio_speech_ranked = rank_candidates(audio_speech_features)
    activity_ranked = rank_candidates(activity_features)

    return {
        "video_path": str(video_path),
        "video_duration": video_duration,
        "audio_path": str(audio_path),
        "transcription": transcription,
        "speech_blocks": speech_blocks,
        "audio_features": audio_features,
        "visual_features": visual_features,
        "audio_events": audio_events,

        "audio_speech": {
            "candidates": audio_speech_candidates,
            "features": audio_speech_features,
            "ranked": audio_speech_ranked
        },

        "multimodal_activity": {
            "candidates": activity_candidates,
            "features": activity_features,
            "ranked": activity_ranked
        }
    }
