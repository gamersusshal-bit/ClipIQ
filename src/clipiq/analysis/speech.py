import re


TEXT_CUES = {
    "excitement": [
        "wow", "amazing", "incredible", "unbelievable",
        "huge", "massive", "incredible", "let's go"
    ],
    "reaction": [
        "oh", "no way", "what", "are you kidding",
        "oh my god", "come on"
    ],
    "anticipation": [
        "here we go", "coming up", "about to",
        "watch this", "wait for it", "this could be"
    ],
    "payoff": [
        "there it is", "he did it", "what a move",
        "that's it", "got him", "it's over"
    ]
}


def group_speech_segments(segments, max_gap=2.0):
    """
    Group nearby Whisper segments into larger speech blocks.
    """

    if not segments:
        return []

    blocks = []
    current = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"].strip()
    }

    for segment in segments[1:]:
        gap = segment["start"] - current["end"]

        if gap <= max_gap:
            current["end"] = segment["end"]
            current["text"] += " " + segment["text"].strip()
        else:
            blocks.append(current)
            current = {
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            }

    blocks.append(current)
    return blocks


def score_speech_block(text):
    """
    Score a speech block using configurable language cues.

    This is a heuristic score, not a trained ML prediction.
    """

    text = text.lower()

    scores = {}

    for category, cues in TEXT_CUES.items():
        scores[category] = sum(
            len(re.findall(r"\b" + re.escape(cue) + r"\b", text))
            for cue in cues
        )

    scores["total"] = sum(scores.values())

    return scores
