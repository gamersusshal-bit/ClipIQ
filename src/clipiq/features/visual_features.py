import cv2
import numpy as np


def analyze_visuals(video_path, sample_interval=1):
    """
    Analyze visual activity by measuring frame-to-frame differences.

    sample_interval: Number of seconds between analyzed frames.
    """

    if sample_interval <= 0:
        raise ValueError("sample_interval must be positive")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        cap.release()
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = (
        cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        if fps > 0 else 0
    )

    differences = []
    times = []
    previous_frame = None
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            current_time = frame_index / fps if fps > 0 else 0

            if frame_index % max(1, int(fps * sample_interval)) == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 180))

                if previous_frame is not None:
                    difference = np.mean(
                        cv2.absdiff(gray, previous_frame)
                    )

                    differences.append(float(difference))
                    times.append(current_time)

                previous_frame = gray

            frame_index += 1


    finally:
        cap.release()

    if not differences:
        raise ValueError("Not enough frames to analyze visual activity.")

    if frame_index + 2 < duration * fps * .95:
        raise ValueError("Video decoding stopped before the expected end")

    differences = np.array(differences)

    return {
        "average_visual_difference": float(np.mean(differences)),
        "maximum_visual_difference": float(np.max(differences)),
        "visual_frames_analyzed": len(differences),
        "duration_analyzed": duration,
        "visual_activity": differences,
        "times": np.array(times)
    }


def get_visual_activity_for_block(visual_features, start, end):
    """
    Calculate average and maximum visual activity within a time block.
    """

    times = visual_features["times"]
    activity = visual_features["visual_activity"]

    mask = (times >= start) & (times <= end)
    block_activity = activity[mask]

    if len(block_activity) == 0:
        return {
            "average_visual_activity": 0.0,
            "maximum_visual_activity": 0.0
        }

    return {
        "average_visual_activity": float(np.mean(block_activity)),
        "maximum_visual_activity": float(np.max(block_activity))
    }
