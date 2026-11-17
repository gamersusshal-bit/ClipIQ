from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Output directories
FEATURES_DIR = PROCESSED_DIR / "features"
CLIPS_DIR = PROCESSED_DIR / "clips"
ANNOTATIONS_DIR = PROCESSED_DIR / "annotations"

# Default processing settings
AUDIO_SAMPLE_RATE = 16000
AUDIO_HOP_LENGTH = 512
AUDIO_FRAME_LENGTH = 2048

WHISPER_MODEL = "base"

# Video extensions
SUPPORTED_VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".mkv"]
