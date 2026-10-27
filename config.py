import os
import tempfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

def _resolve_data_dir() -> Path:
    candidate = BASE_DIR / "data"
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        test_file = candidate / ".write_test"
        test_file.touch()
        test_file.unlink()
        return candidate
    except Exception:
        # Fallback to /tmp for serverless/read-only container environments (e.g. AWS Lambda / Vercel)
        tmp_dir = Path(tempfile.gettempdir()) / "multimodal_attendance_data"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        return tmp_dir

DATA_DIR = _resolve_data_dir()
EXPORTS_DIR = DATA_DIR / "exports"
MODELS_DIR = DATA_DIR / "models"

try:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

class Config:
    BASE_DIR = BASE_DIR
    DATA_DIR = DATA_DIR
    EXPORTS_DIR = EXPORTS_DIR
    MODELS_DIR = MODELS_DIR

    # App
    SECRET_KEY = os.environ.get("SECRET_KEY", "multimodal-ai-classroom-secret-key-2026")
    DEBUG = False
    HOST = "0.0.0.0"
    PORT = 5000

    # Database
    DATABASE_PATH = str(DATA_DIR / "attendance.db")
    SQLALCHEMY_DATABASE_URI = f"sqlite:///{DATABASE_PATH}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    EMBEDDINGS_FILE = str(DATA_DIR / "embeddings.json")

    # Vision & Face Recognition
    FACE_SIMILARITY_THRESHOLD = 0.45  # Cosine similarity threshold for InsightFace / FaceNet
    PROCESS_EVERY_N_FRAMES = 2        # Process 1 out of every N frames to maintain high FPS
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
    INSIGHTFACE_MODEL = "buffalo_l"
    PREFER_GPU = True                 # CUDA execution provider with CPU fallback
    YOLO_MODEL_PATH = str(BASE_DIR / "models" / "yolov8n.pt")
    PHONE_CONF_THRESHOLD = 0.22       # Confidence threshold for cell phone detection
    
    # Face Enrollment Quality Thresholds
    ENROLLMENT_SAMPLES_REQUIRED = 5
    MIN_FACE_SIZE_PX = 75             # Minimum face bounding box width/height
    MIN_LAPLACIAN_VAR = 25.0          # Blur rejection threshold (variance of Laplacian)
    MIN_BRIGHTNESS = 30.0             # Minimum mean brightness (0-255)
    MAX_BRIGHTNESS = 240.0            # Maximum mean brightness (overexposure)

    # Attendance Sustained Presence Logic
    ROLLING_WINDOW_SECONDS = 60       # 60s rolling window
    SUSTAINED_PRESENCE_RATIO = 0.70   # Seen in >= 70% of frames over the rolling window
    LATE_GRACE_PERIOD_SECONDS = 300   # 5 minutes into session is marked Late instead of Present

    # Audio Engine
    AUDIO_SAMPLE_RATE = 16000
    AUDIO_CHUNK_DURATION = 0.03       # 30ms chunks for webrtcvad
    AUDIO_VAD_MODE = 2                # 0-3 aggressiveness
    AUDIO_RMS_THRESHOLD = 0.003       # Minimum RMS threshold for ambient speech (optimized for laptop mics)
    DISCUSSION_BURST_THRESHOLD = 0.65 # Speech ratio threshold for group discussion

    # Multimodal Engagement Fusion (30-second intervals)
    ENGAGEMENT_INTERVAL_SECONDS = 30
    WEIGHT_ATTENTION = 0.35
    WEIGHT_MOVEMENT = 0.25
    WEIGHT_INACTIVITY = 0.20
    WEIGHT_AUDIO = 0.20
    
    # Head Pose Pitch/Yaw Bounds for "Looking at Board" (degrees)
    BOARD_YAW_MAX = 28.0              # Deviations beyond ±28 deg count as looking away
    BOARD_PITCH_MAX = 22.0            # Deviations beyond ±22 deg count as looking down/up

    # Anomaly Detection
    ATTENTION_DROP_THRESHOLD = 0.20   # 3-interval moving average falls >20% below session baseline
    INTERACTION_PEAK_THRESHOLD = 0.25 # Interval audio + movement spikes >25% above baseline

    # AI Teaching Agent
    OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL = "llama3.1:8b"
    OLLAMA_FALLBACK_MODEL = "phi3"
    OLLAMA_REQUEST_TIMEOUT = 12       # seconds

    # Privacy & Safety
    PRIVACY_STORE_NO_RAW_FRAMES = True
    PRIVACY_STORE_NO_RAW_AUDIO = True
    ANONYMOUS_MODE_DEFAULT = False
