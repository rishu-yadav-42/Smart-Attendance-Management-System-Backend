from collections import Counter, defaultdict
import os
import cv2
import numpy as np
from PIL import Image

try:
    from insightface.app import FaceAnalysis
except ImportError:
    FaceAnalysis = None

try:
    from backend.config import (
        ARCFACE_MODEL_FILE,
        ARCFACE_MODEL_NAME,
        ARCFACE_MODEL_ROOT,
        ARCFACE_SECOND_BEST_MARGIN,
        ARCFACE_SIMILARITY_THRESHOLD,
        AVG_CONFIDENCE_LIMIT,
        CONFIDENCE_LIMIT,
        FALLBACK_AVG_CONFIDENCE_LIMIT,
        FALLBACK_CONFIDENCE_LIMIT,
        FALLBACK_MODEL_FILE,
        FACE_DETECT_MIN_NEIGHBORS,
        FACE_DETECT_MIN_SIZE,
        FACE_DETECT_SCALE_FACTOR,
        FACE_SIZE,
        MIN_CONFIDENT_FRAMES,
        MIN_WIN_RATIO,
        MODEL_FILE,
        REMOVED_FACE_ARCFACE_THRESHOLD,
        REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT,
        REMOVED_STUDENT_IDS,
        TRAINING_DIR,
        ensure_folders,
    )
    from backend.helpers import normalize_id, parse_training_id, training_file_student_id
    from backend.services import load_students
except ImportError:
    from config import (
        ARCFACE_MODEL_FILE,
        ARCFACE_MODEL_NAME,
        ARCFACE_MODEL_ROOT,
        ARCFACE_SECOND_BEST_MARGIN,
        ARCFACE_SIMILARITY_THRESHOLD,
        AVG_CONFIDENCE_LIMIT,
        CONFIDENCE_LIMIT,
        FALLBACK_AVG_CONFIDENCE_LIMIT,
        FALLBACK_CONFIDENCE_LIMIT,
        FALLBACK_MODEL_FILE,
        FACE_DETECT_MIN_NEIGHBORS,
        FACE_DETECT_MIN_SIZE,
        FACE_DETECT_SCALE_FACTOR,
        FACE_SIZE,
        MIN_CONFIDENT_FRAMES,
        MIN_WIN_RATIO,
        MODEL_FILE,
        REMOVED_FACE_ARCFACE_THRESHOLD,
        REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT,
        REMOVED_STUDENT_IDS,
        TRAINING_DIR,
        ensure_folders,
    )
    from helpers import normalize_id, parse_training_id, training_file_student_id
    from services import load_students

def load_face_cascade():
    candidates = [
        os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml"),
        os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml") if hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades") else "",
        os.path.join(os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"),
        "haarcascade_frontalface_default.xml",
    ]
    for path in candidates:
        if path and os.path.exists(path):
            cascade = cv2.CascadeClassifier(path)
            if not cascade.empty():
                print(f"[DEBUG] Loaded valid CascadeClassifier from: {path}")
                return cascade

    return cv2.CascadeClassifier("haarcascade_frontalface_default.xml")


face_cascade = load_face_cascade()

arcface_app = None
arcface_init_error = None
_arcface_onnx_session = None


def has_lbph():
    return hasattr(cv2, "face") and hasattr(cv2.face, "LBPHFaceRecognizer_create")


def insightface_available():
    return FaceAnalysis is not None


def ensure_color(face_image):
    if face_image is None:
        return None
    if len(face_image.shape) == 2:
        return cv2.cvtColor(face_image, cv2.COLOR_GRAY2BGR)
    return face_image


def cosine_similarity(vec_a, vec_b):
    return float(np.dot(vec_a, vec_b))


def normalize_embedding(embedding):
    embedding = np.asarray(embedding, dtype="float32").flatten()
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return None
    return embedding / norm


def get_arcface_onnx_session():
    global _arcface_onnx_session
    if _arcface_onnx_session is not None:
        return _arcface_onnx_session
    model_path = os.path.join(ARCFACE_MODEL_ROOT, "models", ARCFACE_MODEL_NAME, "w600k_r50.onnx")
    if not os.path.exists(model_path):
        print(f"[DEBUG] ArcFace ONNX model not found at {model_path}")
        return None
    try:
        import onnxruntime as ort
        _arcface_onnx_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        print("[DEBUG] ArcFace ONNX session loaded successfully (no insightface needed)")
        return _arcface_onnx_session
    except Exception as exc:
        print(f"[DEBUG] ArcFace ONNX load failed: {exc}")
        return None


def extract_arcface_embedding_onnx(face_image):
    session = get_arcface_onnx_session()
    if session is None:
        return None
    try:
        img = cv2.resize(face_image, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img = (img - 127.5) / 127.5
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, axis=0)
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: img})
        embedding = outputs[0].flatten()
        return normalize_embedding(embedding)
    except Exception as exc:
        print(f"[DEBUG] ONNX ArcFace extraction failed: {exc}")
        return None


def get_arcface_app():
    global arcface_app, arcface_init_error

    if arcface_app is not None:
        return arcface_app
    if not insightface_available():
        print("[DEBUG] ArcFace skip: insightface not installed")
        return None
    if arcface_init_error is not None:
        print(f"[DEBUG] ArcFace skip: previous init error: {arcface_init_error}")
        return None

    try:
        os.makedirs(ARCFACE_MODEL_ROOT, exist_ok=True)
        print(f"[DEBUG] Initializing ArcFace with root={os.path.abspath(ARCFACE_MODEL_ROOT)}, model={ARCFACE_MODEL_NAME}")
        arcface_app = FaceAnalysis(
            name=ARCFACE_MODEL_NAME,
            root=os.path.abspath(ARCFACE_MODEL_ROOT),
            providers=["CPUExecutionProvider"],
        )
        arcface_app.prepare(ctx_id=0, det_size=(640, 640))
        print("[DEBUG] ArcFace initialized successfully")
    except Exception as exc:
        arcface_init_error = str(exc)
        print(f"[DEBUG] ArcFace init FAILED: {exc}")
        arcface_app = None

    return arcface_app


def extract_arcface_embedding(face_image):
    color_face = ensure_color(face_image)
    if color_face is None:
        return None

    # Try insightface first
    app = get_arcface_app()
    if app is not None:
        try:
            faces = app.get(color_face)
            if faces:
                best_face = max(
                    faces,
                    key=lambda item: (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1]),
                )
                embedding = normalize_embedding(best_face.embedding)
                if embedding is not None:
                    return embedding
        except Exception:
            pass

        try:
            recognition_model = getattr(app, "models", {}).get("recognition")
            if recognition_model is not None and hasattr(recognition_model, "get_feat"):
                aligned = cv2.resize(color_face, (112, 112))
                embedding = normalize_embedding(recognition_model.get_feat(aligned))
                if embedding is not None:
                    return embedding
        except Exception:
            pass

    # Fallback to pure onnxruntime
    return extract_arcface_embedding_onnx(color_face)


def preprocess_face(gray_face):
    face = cv2.resize(gray_face, FACE_SIZE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    face = clahe.apply(face)
    return face


def crop_face_with_padding(image, x, y, w, h, padding_ratio=0.25):
    if image is None or image.size == 0:
        return None

    img_h, img_w = image.shape[:2]
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h)
    half_side = int(round((side * (1.0 + 2.0 * padding_ratio)) / 2.0))

    x1 = int(round(cx - half_side))
    y1 = int(round(cy - half_side))
    x2 = int(round(cx + half_side))
    y2 = int(round(cy + half_side))

    pad_left = max(0, -x1)
    pad_top = max(0, -y1)
    pad_right = max(0, x2 - img_w)
    pad_bottom = max(0, y2 - img_h)

    x1_clamped = max(0, x1)
    y1_clamped = max(0, y1)
    x2_clamped = min(img_w, x2)
    y2_clamped = min(img_h, y2)

    crop = image[y1_clamped:y2_clamped, x1_clamped:x2_clamped]
    if crop.size == 0:
        return None

    if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
        crop = cv2.copyMakeBorder(
            crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    return crop


def face_descriptor(face):
    face = cv2.resize(face, (120, 120)).astype("float32") / 255.0
    mean = float(face.mean())
    std = float(face.std()) or 1.0
    return ((face - mean) / std).reshape(-1)


class SimpleFaceRecognizer:
    def __init__(self, descriptors=None, labels=None):
        self.descriptors = np.array([] if descriptors is None else descriptors, dtype="float32")
        self.labels = np.array([] if labels is None else labels, dtype=str)

    def predict(self, face):
        if len(self.descriptors) == 0:
            return None, 999.0, 999.0

        descriptor = face_descriptor(face)
        distances = np.sqrt(((self.descriptors - descriptor) ** 2).mean(axis=1))
        nearest = np.argsort(distances)[:8]
        scores = defaultdict(list)

        for index in nearest:
            scores[str(self.labels[index])].append(float(distances[index]))

        ordered = sorted(
            ((label, sum(values) / len(values)) for label, values in scores.items()),
            key=lambda item: item[1],
        )
        best_label, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else 999.0
        return best_label, round(best_score * 100, 2), round(second_score * 100, 2)


class ArcFaceRecognizer:
    def __init__(self, embeddings=None, labels=None):
        self.embeddings = np.array([] if embeddings is None else embeddings, dtype="float32")
        self.labels = np.array([] if labels is None else labels, dtype=str)

    def predict(self, face):
        embedding = extract_arcface_embedding(face)
        if embedding is None or len(self.embeddings) == 0:
            return None, -1.0, -1.0

        similarities = self.embeddings @ embedding
        nearest = np.argsort(similarities)[::-1][:8]
        scores = defaultdict(list)

        for index in nearest:
            scores[str(self.labels[index])].append(float(similarities[index]))

        ordered = sorted(
            ((label, sum(values[:3]) / len(values[:3])) for label, values in scores.items()),
            key=lambda item: item[1],
            reverse=True,
        )
        best_label, best_score = ordered[0]
        second_score = ordered[1][1] if len(ordered) > 1 else -1.0
        return best_label, round(best_score, 4), round(second_score, 4)


def build_simple_recognizer(allowed_ids):
    descriptors = []
    labels = []

    if not os.path.exists(TRAINING_DIR):
        return None

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        student_id = training_file_student_id(file)
        if student_id not in allowed_ids:
            continue

        img_path = os.path.join(TRAINING_DIR, file)
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue

        arr = np.array(img, "uint8")
        for f in (arr, np.fliplr(arr)):
            descriptors.append(face_descriptor(preprocess_face(f)))
            labels.append(student_id)

    if not descriptors:
        return None

    return SimpleFaceRecognizer(descriptors, labels)


def build_arcface_recognizer(allowed_ids):
    embeddings = []
    labels = []

    arcface_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if not arcface_ready:
        print("[DEBUG] build_arcface_recognizer: No ArcFace runtime available (insightface or onnxruntime)")
        return None

    total = 0
    skipped = 0
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        student_id = training_file_student_id(file)
        if student_id not in allowed_ids:
            continue

        image = cv2.imread(os.path.join(TRAINING_DIR, file))
        if image is None:
            skipped += 1
            continue

        total += 1
        for var in (image, cv2.flip(image, 1)):
            embedding = extract_arcface_embedding(var)
            if embedding is None:
                skipped += 1
                continue
            embeddings.append(embedding)
            labels.append(student_id)

    print(f"[DEBUG] build_arcface_recognizer: total={total}, success={len(embeddings)}, skipped={skipped}")
    if not embeddings:
        return None

    return ArcFaceRecognizer(embeddings, labels)


def get_removed_student_ids():
    if not os.path.exists(TRAINING_DIR):
        return set()

    registered_ids = set(load_students()["Id"].astype(str))
    training_ids = set()

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        student_id = training_file_student_id(file)
        if student_id:
            training_ids.add(student_id)

    return (training_ids - registered_ids) | (set(REMOVED_STUDENT_IDS) - registered_ids)


def training_sample_counts():
    counts = Counter()
    if not os.path.exists(TRAINING_DIR):
        return {}

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        student_id = training_file_student_id(file)
        if student_id:
            counts[student_id] += 1

    return dict(counts)


def delete_student_training_images(student_id):
    removed = 0
    if not os.path.exists(TRAINING_DIR):
        return removed

    normalized_id = normalize_id(student_id)
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if training_file_student_id(file) != normalized_id:
            continue
        try:
            os.remove(os.path.join(TRAINING_DIR, file))
            removed += 1
        except OSError:
            continue
    return removed


def clear_model_files():
    for path in (MODEL_FILE, FALLBACK_MODEL_FILE, ARCFACE_MODEL_FILE):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                continue


def save_simple_model(faces, ids):
    descriptors = [face_descriptor(face) for face in faces]
    np.savez_compressed(FALLBACK_MODEL_FILE, descriptors=np.array(descriptors, dtype="float32"), labels=np.array(ids, dtype=str))


def save_arcface_model(recognizer):
    np.savez_compressed(
        ARCFACE_MODEL_FILE,
        embeddings=np.array(recognizer.embeddings, dtype="float32"),
        labels=np.array(recognizer.labels, dtype=str),
    )


def load_face_recognizer():
    arcface_runtime_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if os.path.exists(ARCFACE_MODEL_FILE) and arcface_runtime_ready:
        data = np.load(ARCFACE_MODEL_FILE)
        recognizer = ArcFaceRecognizer(data["embeddings"], data["labels"])
        print("[DEBUG] load_face_recognizer: Loaded ArcFace model")
        return recognizer, "arcface"

    if has_lbph() and os.path.exists(MODEL_FILE):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_FILE)
        print("[DEBUG] load_face_recognizer: Loaded LBPH model")
        return recognizer, "lbph"

    if os.path.exists(FALLBACK_MODEL_FILE):
        data = np.load(FALLBACK_MODEL_FILE)
        recognizer = SimpleFaceRecognizer(data["descriptors"], data["labels"])
        print("[DEBUG] load_face_recognizer: Loaded fallback (Simple) model")
        return recognizer, "fallback"

    print("[DEBUG] load_face_recognizer: No model found!")
    return None, None


def recognizer_student_count(recognizer, fallback_count):
    labels = getattr(recognizer, "labels", None)
    if labels is None:
        return fallback_count

    unique_labels = {str(label) for label in labels if str(label).strip()}
    return len(unique_labels) or fallback_count


def build_removed_face_recognizer():
    removed_ids = get_removed_student_ids()
    if not removed_ids:
        return None
    if insightface_available():
        recognizer = build_arcface_recognizer(removed_ids)
        if recognizer is not None:
            return recognizer, "arcface"
    recognizer = build_simple_recognizer(removed_ids)
    if recognizer is not None:
        return recognizer, "fallback"
    return None


def predict_with_backend(recognizer, backend, face_gray, face_color):
    if backend == "arcface":
        res1 = recognizer.predict(face_color)
        if face_color is not None:
            res2 = recognizer.predict(cv2.flip(face_color, 1))
            if res2[1] > res1[1]:
                return res2
        return res1
    if backend == "lbph":
        label, confidence = recognizer.predict(preprocess_face(face_gray))
        return str(label), float(confidence), 999.0
    res1 = recognizer.predict(face_gray)
    if face_gray is not None:
        res2 = recognizer.predict(cv2.flip(face_gray, 1))
        if res2[1] < res1[1]:
            return res2
    return res1


def arcface_threshold_for_count(registered_count):
    if registered_count <= 3:
        return 0.40
    return ARCFACE_SIMILARITY_THRESHOLD


def attendance_match_ok(backend, best_score, second_score, registered_count):
    if backend == "arcface":
        threshold = arcface_threshold_for_count(registered_count)
        margin = 0.02 if registered_count <= 3 else ARCFACE_SECOND_BEST_MARGIN
        if best_score < threshold:
            return False
        if second_score > -1 and (best_score - second_score) < margin:
            return False
        return True

    if backend == "lbph":
        return best_score <= CONFIDENCE_LIMIT

    return best_score <= FALLBACK_CONFIDENCE_LIMIT


def stable_match_ok(backend, vote_count, total_votes, avg_score, registered_count):
    if backend == "arcface":
        required_frames = 2 if registered_count <= 3 else MIN_CONFIDENT_FRAMES
        required_win_ratio = 0.45 if registered_count <= 3 else MIN_WIN_RATIO
    else:
        required_frames = 6 if registered_count <= 3 else MIN_CONFIDENT_FRAMES + 2
        required_win_ratio = 0.65 if registered_count <= 3 else 0.70

    if vote_count < required_frames or vote_count / total_votes < required_win_ratio:
        return False

    if backend == "arcface":
        return avg_score >= arcface_threshold_for_count(registered_count)

    if backend == "lbph":
        return avg_score <= AVG_CONFIDENCE_LIMIT

    fallback_limit = 160 if registered_count == 1 else FALLBACK_AVG_CONFIDENCE_LIMIT
    return avg_score <= fallback_limit


def _process_single_frame(img):
    students = load_students()
    if students.empty:
        return {"status": "error", "message": "No student is registered."}

    id_to_name = dict(zip(students["Id"].astype(str), students["Name"]))
    recognizer, backend = load_face_recognizer()
    if recognizer is None:
        return {"status": "error", "message": "Model is not trained. Please run 'Train Model' first."}

    registered_count = recognizer_student_count(recognizer, len(id_to_name))
    removed_recognizer = build_removed_face_recognizer()

    h_img, w_img = img.shape[:2]
    max_dim = 480
    if max(h_img, w_img) > max_dim:
        scale = max_dim / max(h_img, w_img)
        img_small = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
        scale_factor = 1.0 / scale
    else:
        img_small = img
        scale_factor = 1.0

    try:
        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=FACE_DETECT_SCALE_FACTOR,
            minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
            minSize=(int(FACE_DETECT_MIN_SIZE[0] * scale_factor), int(FACE_DETECT_MIN_SIZE[1] * scale_factor)),
        )
    except Exception:
        return {"status": "no_face", "message": "No face detected."}

    if len(faces) == 0:
        return {"status": "no_face", "message": "No face detected."}

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    x_orig = int(x * scale_factor)
    y_orig = int(y * scale_factor)
    w_orig = int(w * scale_factor)
    h_orig = int(h * scale_factor)

    face_color = crop_face_with_padding(img, x_orig, y_orig, w_orig, h_orig, 0.22)
    if face_color is None:
        return {"status": "no_face", "message": "No face detected."}

    try:
        face_gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        face_gray = clahe.apply(face_gray)
        face_gray = cv2.GaussianBlur(face_gray, (3, 3), 0)
    except Exception:
        return {"status": "no_face", "message": "No face detected."}

    if removed_recognizer is not None:
        removed_recognizer_model, removed_backend = removed_recognizer
        _, removed_score, _ = predict_with_backend(removed_recognizer_model, removed_backend, face_gray, face_color)
        removed_match = (
            removed_score >= REMOVED_FACE_ARCFACE_THRESHOLD
            if removed_backend == "arcface"
            else removed_score <= REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT
        )
        if removed_match:
            return {"status": "removed", "message": "Removed face detected.", "confidence": float(removed_score)}

    label, best_score, second_score = predict_with_backend(recognizer, backend, face_gray, face_color)
    student_id = str(label) if label is not None else None

    if student_id in id_to_name and attendance_match_ok(backend, best_score, second_score, registered_count):
        return {
            "status": "matched",
            "student_id": student_id,
            "name": id_to_name[student_id],
            "confidence": float(best_score),
            "backend": backend,
        }

    return {"status": "unmatched", "message": "Face match failed. Unrecognized face."}


def train_model():
    ensure_folders()
    if not os.path.exists(TRAINING_DIR) or len(os.listdir(TRAINING_DIR)) == 0:
        return False, "No training images found."

    registered_ids = set(load_students()["Id"].astype(str))
    if not registered_ids:
        return False, "No registered students found."

    counts = training_sample_counts()
    faces = []
    ids = []

    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        label = parse_training_id(file)
        if label is None:
            continue
        if str(label) not in registered_ids:
            continue

        img_path = os.path.join(TRAINING_DIR, file)
        try:
            img = Image.open(img_path).convert("L")
        except Exception:
            continue

        arr = np.array(img, "uint8")
        for f in (arr, np.fliplr(arr)):
            faces.append(preprocess_face(f))
            ids.append(label)

    if not faces:
        return False, "No matching training images were found for the registered students. Please add the student again using **Register Student**."

    save_simple_model(faces, ids)
    trained_counts = Counter(str(student_id) for student_id in ids)
    summary = ", ".join(f"ID {student_id}: {count}" for student_id, count in sorted(trained_counts.items()))

    arcface_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if arcface_ready:
        arcface_recognizer = build_arcface_recognizer(registered_ids)
        if arcface_recognizer is not None and len(arcface_recognizer.labels) > 0:
            save_arcface_model(arcface_recognizer)
            arcface_counts = Counter(str(student_id) for student_id in arcface_recognizer.labels)
            arcface_summary = ", ".join(
                f"ID {student_id}: {count}" for student_id, count in sorted(arcface_counts.items())
            )
            return True, f"ArcFace model trained successfully. {arcface_summary}"

    if has_lbph():
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.train(faces, np.array(ids))
        recognizer.save(MODEL_FILE)
        return True, f"Model trained with {len(faces)} face samples. {summary}"

    return True, f"Model trained with {len(faces)} face samples. {summary}. Using OpenCV fallback model."


def active_model_file():
    arcface_runtime_ready = get_arcface_app() is not None or get_arcface_onnx_session() is not None
    if arcface_runtime_ready and os.path.exists(ARCFACE_MODEL_FILE):
        return ARCFACE_MODEL_FILE
    if has_lbph() and os.path.exists(MODEL_FILE):
        return MODEL_FILE
    if os.path.exists(FALLBACK_MODEL_FILE):
        return FALLBACK_MODEL_FILE
    if insightface_available():
        return ARCFACE_MODEL_FILE
    if has_lbph():
        return MODEL_FILE
    return FALLBACK_MODEL_FILE


def latest_registered_training_mtime(registered_ids):
    if not os.path.exists(TRAINING_DIR):
        return 0

    latest_mtime = 0
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        student_id = training_file_student_id(file)
        if student_id not in registered_ids:
            continue
        try:
            latest_mtime = max(latest_mtime, os.path.getmtime(os.path.join(TRAINING_DIR, file)))
        except OSError:
            continue
    return latest_mtime


def model_needs_training():
    students = load_students()
    registered_ids = set(students["Id"].astype(str))
    if not registered_ids:
        return False

    recognizer, _ = load_face_recognizer()
    active_model = active_model_file()
    if recognizer is None or not os.path.exists(active_model):
        return True

    sample_counts = training_sample_counts()
    if any(sample_counts.get(student_id, 0) == 0 for student_id in registered_ids):
        return True

    if os.path.getmtime(active_model) < latest_registered_training_mtime(registered_ids):
        return True

    return False
