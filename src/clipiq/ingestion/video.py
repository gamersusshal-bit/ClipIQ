import cv2
from pathlib import Path


def load_video(video_path):
    """
    Open a video and return its metadata.

    Parameters
    ----------
    video_path : str or Path
        Path to the input video.

    Returns
    -------
    dict
        Video capture object and metadata.
    """

    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    duration = frame_count / fps if fps > 0 else 0

    return {
        "capture": cap,
        "path": video_path,
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration": duration
    }


def release_video(video_data):
    """Release the video capture resource."""

    capture = video_data.get("capture")

    if capture is not None:
        capture.release()
