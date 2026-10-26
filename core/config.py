"""
config.py – Central configuration for the Sign Language Translator.
Change a value here and it takes effect across all scripts instantly.
"""
import os

# Project root is one level up from core/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_FILE  = os.path.join(BASE_DIR, "data",  "gestures_data.pickle")
MODEL_FILE = os.path.join(BASE_DIR, "model", "sign_classifier.p")
CM_FILE    = os.path.join(BASE_DIR, "model", "confusion_matrix.png")

# ── Data collection ────────────────────────────────────────────────────────
SAMPLES_PER_GESTURE = 60   # Frames captured per sign class

# ── MediaPipe ──────────────────────────────────────────────────────────────
MP_MAX_HANDS              = 2
MP_DETECTION_CONFIDENCE   = 0.7
MP_TRACKING_CONFIDENCE    = 0.7

# ── Model ──────────────────────────────────────────────────────────────────
N_ESTIMATORS   = 100   # Random Forest trees
TEST_SIZE      = 0.2   # Train/test split ratio

# ── Real-time translator ───────────────────────────────────────────────────
CONFIDENCE_THRESHOLD   = 0.60   # Min prediction probability to accept a sign
PAUSE_BETWEEN_GESTURES = 1.0    # Cooldown (seconds) after a sign is committed
HOLD_FRAMES_TO_COMMIT  = 15     # Frames a sign must stay stable to auto-commit
TTS_RATE               = 150    # Text-to-speech words-per-minute
