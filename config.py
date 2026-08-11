import os
import sys

CURR_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(CURR_DIR) == "backend":
    BASE_DIR = os.path.dirname(CURR_DIR)
else:
    BASE_DIR = CURR_DIR

if CURR_DIR not in sys.path:
    sys.path.insert(0, CURR_DIR)
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

VENDOR_DIR = os.path.join(BASE_DIR, ".vendor312")
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.insert(0, VENDOR_DIR)

APP_VERSION = "arcface-guided-register-v5-2026-04-26"

STUDENT_FILE = os.path.join(BASE_DIR, "StudentDetails", "StudentDetails.csv")
DB_FILE = os.path.join(BASE_DIR, "attendance.db")
TRAINING_DIR = os.path.join(BASE_DIR, "TrainingImage")
MODEL_FILE = os.path.join(BASE_DIR, "TrainingImageLabel", "Trainner.yml")
FALLBACK_MODEL_FILE = os.path.join(BASE_DIR, "TrainingImageLabel", "face_model.npz")
ARCFACE_MODEL_FILE = os.path.join(BASE_DIR, "TrainingImageLabel", "arcface_embeddings.npz")
ATTENDANCE_DIR = os.path.join(BASE_DIR, "Attendance")
ARCFACE_MODEL_ROOT = os.path.join(BASE_DIR, "FaceModelStore")
ARCFACE_MODEL_NAME = "buffalo_l"

SAMPLE_COUNT = 15
FACE_SIZE = (220, 220)
FACE_DETECT_SCALE_FACTOR = 1.06
FACE_DETECT_MIN_NEIGHBORS = 3
FACE_DETECT_MIN_SIZE = (48, 48)

CONFIDENCE_LIMIT = 50
AVG_CONFIDENCE_LIMIT = 48
FALLBACK_CONFIDENCE_LIMIT = 55
FALLBACK_AVG_CONFIDENCE_LIMIT = 52
ARCFACE_SIMILARITY_THRESHOLD = 0.42
ARCFACE_SECOND_BEST_MARGIN = 0.03

MIN_CONFIDENT_FRAMES = 3
MIN_WIN_RATIO = 0.55
FAST_MATCH_TARGET = 4

REMOVED_STUDENT_IDS = set()
REMOVED_FACE_CONFIDENCE_LIMIT = 48
REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT = 35
REMOVED_FACE_ARCFACE_THRESHOLD = 0.38
REMOVED_FACE_MIN_FRAMES = 3
CAMERA_FRAME_RETRY_LIMIT = 20


def ensure_folders():
    os.makedirs(os.path.join(BASE_DIR, "StudentDetails"), exist_ok=True)
    os.makedirs(TRAINING_DIR, exist_ok=True)
    os.makedirs(os.path.join(BASE_DIR, "TrainingImageLabel"), exist_ok=True)
    os.makedirs(ATTENDANCE_DIR, exist_ok=True)
