from collections import Counter, defaultdict
from datetime import datetime
import os
import cv2
import numpy as np
import pandas as pd
from flask import Blueprint, Response, jsonify, redirect, render_template, request, session, url_for

try:
    from backend.camera import (
        clear_frame_buffer,
        gen_frames,
        get_buffered_frames,
        open_camera,
        stop_camera_stream,
    )
    from backend.config import (
        APP_VERSION,
        ARCFACE_MODEL_FILE,
        ATTENDANCE_DIR,
        FAST_MATCH_TARGET,
        FACE_DETECT_MIN_NEIGHBORS,
        FACE_DETECT_MIN_SIZE,
        FACE_DETECT_SCALE_FACTOR,
        REMOVED_FACE_ARCFACE_THRESHOLD,
        REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT,
        REMOVED_FACE_MIN_FRAMES,
        SAMPLE_COUNT,
        TRAINING_DIR,
        ensure_folders,
    )
    from backend.database import get_db_connection, init_db, legacy_date_to_iso
    from backend.face_recognition import (
        _process_single_frame,
        arcface_init_error,
        build_removed_face_recognizer,
        crop_face_with_padding,
        face_cascade,
        get_arcface_app,
        get_removed_student_ids,
        has_lbph,
        insightface_available,
        load_face_recognizer,
        model_needs_training,
        predict_with_backend,
        recognizer_student_count,
        stable_match_ok,
        train_model,
        training_sample_counts,
    )
    from backend.helpers import (
        clean_name,
        normalize_id,
        safe_file_name,
        training_file_student_id,
        valid_student_id,
    )
    from backend.services import (
        add_student,
        delete_student,
        latest_attendance_rows,
        load_students,
        student_attendance_stat,
        student_exists,
        attendance_summary,
        total_attendance_count,
    )
except ImportError:
    from camera import (
        clear_frame_buffer,
        gen_frames,
        get_buffered_frames,
        open_camera,
        stop_camera_stream,
    )
    from config import (
        APP_VERSION,
        ARCFACE_MODEL_FILE,
        ATTENDANCE_DIR,
        FAST_MATCH_TARGET,
        FACE_DETECT_MIN_NEIGHBORS,
        FACE_DETECT_MIN_SIZE,
        FACE_DETECT_SCALE_FACTOR,
        REMOVED_FACE_ARCFACE_THRESHOLD,
        REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT,
        REMOVED_FACE_MIN_FRAMES,
        SAMPLE_COUNT,
        TRAINING_DIR,
        ensure_folders,
    )
    from database import get_db_connection, init_db, legacy_date_to_iso
    from face_recognition import (
        _process_single_frame,
        arcface_init_error,
        build_removed_face_recognizer,
        crop_face_with_padding,
        face_cascade,
        get_arcface_app,
        get_removed_student_ids,
        has_lbph,
        insightface_available,
        load_face_recognizer,
        model_needs_training,
        predict_with_backend,
        recognizer_student_count,
        stable_match_ok,
        train_model,
        training_sample_counts,
    )
    from helpers import (
        clean_name,
        normalize_id,
        safe_file_name,
        training_file_student_id,
        valid_student_id,
    )
    from services import (
        add_student,
        delete_student,
        latest_attendance_rows,
        load_students,
        student_attendance_stat,
        student_exists,
        attendance_summary,
        total_attendance_count,
    )

bp = Blueprint("main", __name__)


def safe_render_template(template_name, **context):
    try:
        return render_template(template_name, **context)
    except Exception:
        return jsonify({
            "status": "online",
            "message": "Smart Attendance Management System Backend API is running.",
            "version": APP_VERSION,
            "api_endpoints": {
                "health": "/health",
                "app_status": "/app_status",
                "process_frame": "/process_frame",
                "mark_attendance": "/mark_attendance_ajax",
                "capture_frame": "/capture_register_frame",
                "process_register": "/process_register_ajax"
            }
        })


@bp.route("/video_feed")
def video_feed():
    max_duration = request.args.get("max_duration", type=int, default=15)
    return Response(gen_frames(max_duration=max_duration), mimetype="multipart/x-mixed-replace; boundary=frame")


@bp.route("/stop_camera_preview", methods=["POST"])
def stop_camera_preview():
    stop_camera_stream()
    return ("", 204)


@bp.route("/camera")
def camera():
    return safe_render_template("camera.html")


@bp.route("/register_camera")
def register_camera():
    if "admin" not in session:
        return redirect(url_for("main.login"))
    return safe_render_template("register_camera.html")


@bp.route("/train")
def train():
    if "admin" not in session:
        return redirect(url_for("main.login"))

    success, message = train_model()
    return safe_render_template("result.html", success=success, title="Training", message=message, back_url=url_for("main.home"))


@bp.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        if request.form.get("username") == "Rishu" and request.form.get("password") == "Rishu@123":
            session["admin"] = True
            return redirect(url_for("main.home"))
        error = "Invalid username or password."

    return safe_render_template("login.html", error=error)


@bp.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("main.login"))


@bp.route("/")
def home():
    try:
        students = load_students()
        attendance_rows = latest_attendance_rows(limit=10)
        attendance_total_count = total_attendance_count()
        attendance_stats = attendance_summary()
        model_ready = not model_needs_training()
        recognizer, backend = load_face_recognizer()
        backend_name = backend if backend else "none"
        return safe_render_template(
            "index.html",
            students=students.to_dict("records"),
            attendance_rows=attendance_rows,
            attendance_total_count=attendance_total_count,
            attendance_stats=attendance_stats,
            model_ready=model_ready,
            student_count=len(students),
            backend=backend_name,
        )
    except Exception as exc:
        return jsonify({
            "status": "online",
            "message": "Smart Attendance Management System Backend API is running.",
            "version": APP_VERSION,
            "api_endpoints": {
                "health": "/health",
                "app_status": "/app_status",
                "process_frame": "/process_frame",
                "mark_attendance": "/mark_attendance_ajax",
                "capture_frame": "/capture_register_frame",
                "process_register": "/process_register_ajax"
            }
        })


@bp.route("/app_status")
def app_status():
    try:
        students = load_students()
        arcface_runtime_ready = get_arcface_app() is not None if insightface_available() else False
        return {
            "version": APP_VERSION,
            "students": students.to_dict("records"),
            "model_needs_training": model_needs_training(),
            "removed_student_ids": sorted(get_removed_student_ids()),
            "face_backend": "arcface" if insightface_available() else ("opencv_lbph" if has_lbph() else "numpy_fallback"),
            "training_sample_counts": training_sample_counts(),
            "arcface_ready": os.path.exists(ARCFACE_MODEL_FILE),
            "arcface_runtime_ready": arcface_runtime_ready,
            "arcface_error": arcface_init_error,
        }
    except Exception as exc:
        return {
            "version": APP_VERSION,
            "status": "online",
            "message": str(exc),
        }


@bp.route("/result")
def result_page():
    success = request.args.get("success", "0") == "1"
    title = request.args.get("title", "Result")
    message = request.args.get("message", "")
    return safe_render_template("result.html", success=success, title=title, message=message, back_url=url_for("main.home"))


@bp.route("/process_frame", methods=["POST"])
def process_frame():
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No image received"})

    file = request.files["image"]
    img_bytes = file.read()
    if not img_bytes:
        return jsonify({"status": "error", "message": "Empty image received"})

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"status": "error", "message": "Invalid image format"})

    result = _process_single_frame(img)
    return jsonify(result)


@bp.route("/mark_attendance_ajax", methods=["POST"])
def mark_attendance_ajax():
    data = request.get_json(force=True, silent=True) or {}
    student_id = str(data.get("student_id", "")).strip()
    name = str(data.get("name", "")).strip()
    confidence = data.get("confidence", 0)

    if not student_id or not name:
        return jsonify({"success": False, "message": "Invalid student data."})

    if not student_exists(student_id):
        return jsonify({"success": False, "message": "Student registered nahi hai."})

    now = datetime.now()
    attendance_date = now.strftime("%Y-%m-%d")
    attendance_time = now.strftime("%H:%M:%S")
    display_date = now.strftime("%d-%m-%Y")

    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND attendance_date = ?",
            (student_id, attendance_date),
        ).fetchone()
        already_marked = existing is not None
        if not already_marked:
            conn.execute(
                """
                INSERT INTO attendance
                (student_id, name, attendance_date, attendance_time, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, name, attendance_date, attendance_time, round(float(confidence), 2)),
            )
            try:
                ensure_folders()
                excel_filename = f"Attendance_{display_date}.xlsx"
                excel_path = os.path.join(ATTENDANCE_DIR, excel_filename)
                new_row = pd.DataFrame([{
                    "Id": student_id,
                    "Name": name,
                    "Date": display_date,
                    "Time": attendance_time,
                    "Confidence": round(float(confidence), 2),
                }])
                if os.path.exists(excel_path):
                    existing_df = pd.read_excel(excel_path, dtype={"Id": str})
                    combined = pd.concat([existing_df, new_row], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["Id", "Date"], keep="last")
                    combined.to_excel(excel_path, index=False)
                else:
                    new_row.to_excel(excel_path, index=False)
            except Exception as exc:
                print(f"[DEBUG] Excel save failed: {exc}")

    return jsonify({"success": True, "message": f"Attendance marked for {name} (ID: {student_id}) at {attendance_time}."})


@bp.route("/capture_register_frame", methods=["POST"])
def capture_register_frame():
    try:
        data = request.get_json(force=True, silent=True) or {}
        student_id = str(request.form.get("student_id") or data.get("student_id") or session.get("temp_id") or "").strip()
        name = str(request.form.get("name") or data.get("name") or session.get("temp_name") or "").strip()

        if not student_id or not name:
            return jsonify({"success": False, "message": "Missing student ID or name. Please start registration again."})

        if "image" not in request.files:
            return jsonify({"success": False, "message": "No image received"})

        file = request.files["image"]
        img_bytes = file.read()
        if not img_bytes:
            return jsonify({"success": False, "message": "Empty image received"})

        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({"success": False, "message": "Invalid image format"})

        h_img, w_img = img.shape[:2]
        max_dim = 480
        if max(h_img, w_img) > max_dim:
            scale = max_dim / max(h_img, w_img)
            img_small = cv2.resize(img, (int(w_img * scale), int(h_img * scale)))
            scale_factor = 1.0 / scale
        else:
            img_small = img
            scale_factor = 1.0

        gray = cv2.cvtColor(img_small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(int(70 * scale_factor), int(70 * scale_factor)),
        )

        if len(faces) == 0:
            return jsonify({"success": False, "message": "No face detected in frame"})

        x, y, w, h = faces[0]
        x_orig = int(x * scale_factor)
        y_orig = int(y * scale_factor)
        w_orig = int(w * scale_factor)
        h_orig = int(h * scale_factor)
        padded_face = crop_face_with_padding(img, x_orig, y_orig, w_orig, h_orig, 0.22)
        if padded_face is None:
            return jsonify({"success": False, "message": "Face crop failed"})

        face = cv2.resize(padded_face, (224, 224))
        ensure_folders()

        existing_samples = [
            f for f in os.listdir(TRAINING_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and training_file_student_id(f) == str(student_id)
        ]
        sample_num = len(existing_samples) + 1

        filename = f"{safe_file_name(name)}.{student_id}.{sample_num}.jpg"
        filepath = os.path.join(TRAINING_DIR, filename)
        cv2.imwrite(filepath, face)

        return jsonify({"success": True, "sample": sample_num})
    except Exception as exc:
        print(f"[ERROR] capture_register_frame exception: {exc}")
        return jsonify({"success": False, "message": f"Server capture exception: {str(exc)}"})


@bp.route("/process_register_ajax", methods=["POST"])
def process_register_ajax():
    try:
        data = request.get_json(force=True, silent=True) or {}
        student_id = str(request.form.get("student_id") or data.get("student_id") or session.get("temp_id") or "").strip()
        name = str(request.form.get("name") or data.get("name") or session.get("temp_name") or "").strip()

        if not student_id or not name:
            return jsonify({"success": False, "message": "Registration session expired or missing parameters."})

        if student_exists(student_id):
            session.pop("temp_id", None)
            session.pop("temp_name", None)
            return jsonify({"success": False, "message": f"ID {student_id} already registered."})

        ensure_folders()
        sample_count = sum(
            1 for f in os.listdir(TRAINING_DIR)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and training_file_student_id(f) == str(student_id)
        )

        if sample_count < 12:
            return jsonify({"success": False, "message": f"Only {sample_count} samples captured. Need at least 12. Please try again with better lighting."})

        add_student(student_id, name)
        success, train_message = train_model()
        session.pop("temp_id", None)
        session.pop("temp_name", None)

        return jsonify({
            "success": success,
            "message": f"{name} (ID: {student_id}) registered successfully. {train_message}"
        })
    except Exception as exc:
        print(f"[ERROR] process_register_ajax exception: {exc}")
        return jsonify({"success": False, "message": f"Registration processing error: {str(exc)}"})


@bp.route("/process_attendance")
def process_attendance():
    if model_needs_training():
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="The model needs to be updated. Please log in as admin first and run **Train Model**.",
            back_url=url_for("main.home"),
        )

    students = load_students()
    if students.empty:
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="No student is registered.",
            back_url=url_for("main.home"),
        )

    id_to_name = dict(zip(students["Id"].astype(str), students["Name"]))
    recognizer, backend = load_face_recognizer()
    if recognizer is None:
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="The model could not be loaded. Please log in as admin first and run **Train Model**.",
            back_url=url_for("main.home"),
        )

    removed_recognizer = build_removed_face_recognizer()
    registered_count = recognizer_student_count(recognizer, len(id_to_name))

    def _process_attendance_frame(img):
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = cv2.equalizeHist(gray)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=FACE_DETECT_SCALE_FACTOR,
                minNeighbors=FACE_DETECT_MIN_NEIGHBORS,
                minSize=FACE_DETECT_MIN_SIZE,
            )
        except Exception:
            return "no_face", None, None

        if len(faces) == 0:
            return "no_face", None, None

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        face_color = crop_face_with_padding(img, x, y, w, h, 0.22)
        if face_color is None:
            return "no_face", None, None

        try:
            face_gray = cv2.cvtColor(face_color, cv2.COLOR_BGR2GRAY)
            face_gray = cv2.equalizeHist(face_gray)
        except Exception:
            return "no_face", None, None

        if removed_recognizer is not None:
            removed_recognizer_model, removed_backend = removed_recognizer
            _, removed_score, _ = predict_with_backend(removed_recognizer_model, removed_backend, face_gray, face_color)
            removed_match = (
                removed_score >= REMOVED_FACE_ARCFACE_THRESHOLD
                if removed_backend == "arcface"
                else removed_score <= REMOVED_FACE_FALLBACK_CONFIDENCE_LIMIT
            )
            if removed_match:
                return "removed", None, removed_score

        label, best_score, second_score = predict_with_backend(recognizer, backend, face_gray, face_color)
        student_id = str(label) if label is not None else None

        try:
            from backend.face_recognition import attendance_match_ok
        except ImportError:
            from face_recognition import attendance_match_ok

        if student_id in id_to_name and attendance_match_ok(backend, best_score, second_score, registered_count):
            return "matched", student_id, best_score

        return "unmatched", None, None

    cam = None
    votes = Counter()
    confidences = defaultdict(list)
    removed_votes = 0
    frames_checked = 0
    face_detected_frames = 0

    buffered_frames = get_buffered_frames()
    use_buffered = len(buffered_frames) >= 8

    if use_buffered:
        for img in buffered_frames:
            status, student_id, best_score = _process_attendance_frame(img)

            if status == "removed":
                removed_votes += 1
                if removed_votes >= REMOVED_FACE_MIN_FRAMES:
                    break
                continue

            if status in ("matched", "unmatched"):
                face_detected_frames += 1

            if status == "matched":
                votes[student_id] += 1
                confidences[student_id].append(best_score)
                current_votes = votes[student_id]
                current_avg = sum(confidences[student_id]) / len(confidences[student_id])
                if stable_match_ok(backend, current_votes, sum(votes.values()), current_avg, registered_count) and current_votes >= FAST_MATCH_TARGET:
                    break

            if sum(votes.values()) >= FAST_MATCH_TARGET + 1:
                break

        frames_checked = len(buffered_frames)
    else:
        for camera_attempt in range(3):
            stop_camera_stream()
            import time
            time.sleep(1.0 + camera_attempt * 0.6)
            cam = open_camera(0)
            if cam is None:
                continue

            stabilized = 0
            for _ in range(15):
                try:
                    ret, _ = cam.read()
                    if ret:
                        stabilized += 1
                except Exception:
                    pass
                time.sleep(0.07)

            if stabilized < 5:
                cam.release()
                time.sleep(0.3)
                continue

            votes = Counter()
            confidences = defaultdict(list)
            removed_votes = 0
            frames_checked = 0
            face_detected_frames = 0

            for _ in range(150):
                try:
                    ret, img = cam.read()
                except Exception:
                    time.sleep(0.05)
                    continue
                if not ret or img is None or getattr(img, "size", 0) == 0:
                    time.sleep(0.05)
                    continue

                frames_checked += 1
                status, student_id, best_score = _process_attendance_frame(img)

                if status == "removed":
                    removed_votes += 1
                    if removed_votes >= REMOVED_FACE_MIN_FRAMES:
                        break
                    continue

                if status in ("matched", "unmatched"):
                    face_detected_frames += 1

                if status == "matched":
                    votes[student_id] += 1
                    confidences[student_id].append(best_score)
                    current_votes = votes[student_id]
                    current_avg = sum(confidences[student_id]) / len(confidences[student_id])
                    if stable_match_ok(backend, current_votes, sum(votes.values()), current_avg, registered_count) and current_votes >= FAST_MATCH_TARGET:
                        break

                if sum(votes.values()) >= FAST_MATCH_TARGET + 1:
                    break

            cam.release()

            if removed_votes >= REMOVED_FACE_MIN_FRAMES:
                break
            if frames_checked >= 8:
                break

    print(f"[DEBUG] Backend={backend}, buffered={len(buffered_frames)}, use_buffered={use_buffered}, frames_checked={frames_checked}, face_detected={face_detected_frames}, votes={dict(votes)}, removed_votes={removed_votes}")

    if not use_buffered and cam is None:
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="Camera open nahi ho pa raha. Dusri app me camera use ho raha ho to usse band karke phir try karein.",
            back_url=url_for("main.home"),
        )

    if removed_votes >= REMOVED_FACE_MIN_FRAMES:
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="This face matches a removed student ID. Attendance has not been marked.",
            back_url=url_for("main.home"),
        )

    if frames_checked < 8:
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="Camera frames stable nahi mil rahe. Camera preview ya dusri app band karke phir try karein.",
            back_url=url_for("main.home"),
        )

    if face_detected_frames == 0:
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="Face detect nahi ho paaya. Camera ke saamne seedha aayein aur light improve karein.",
            back_url=url_for("main.home"),
        )

    if not votes:
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="The face was not recognized. Please look straight at the camera, ensure proper lighting, and try again.",
            back_url=url_for("main.home"),
        )

    student_id, vote_count = votes.most_common(1)[0]
    total_votes = sum(votes.values())
    avg_conf = sum(confidences[student_id]) / len(confidences[student_id])

    if not stable_match_ok(backend, vote_count, total_votes, avg_conf, registered_count):
        return safe_render_template(
            "result.html",
            success=False,
            title="Attendance",
            message="A stable face match was not found. Please come a bit closer and try again in proper lighting.",
            back_url=url_for("main.home"),
        )

    name = id_to_name[student_id]
    attendance_date = datetime.now().strftime("%Y-%m-%d")
    attendance_time = datetime.now().strftime("%H:%M:%S")
    display_date = datetime.now().strftime("%d-%m-%Y")

    init_db()
    with get_db_connection() as conn:
        existing = conn.execute(
            "SELECT 1 FROM attendance WHERE student_id = ? AND attendance_date = ?",
            (student_id, attendance_date),
        ).fetchone()
        already_marked = existing is not None
        if not already_marked:
            conn.execute(
                """
                INSERT INTO attendance
                (student_id, name, attendance_date, attendance_time, confidence)
                VALUES (?, ?, ?, ?, ?)
                """,
                (student_id, name, attendance_date, attendance_time, round(avg_conf, 2)),
            )
            try:
                ensure_folders()
                excel_filename = f"Attendance_{display_date}.xlsx"
                excel_path = os.path.join(ATTENDANCE_DIR, excel_filename)
                new_row = pd.DataFrame([{
                    "Id": student_id,
                    "Name": name,
                    "Date": display_date,
                    "Time": attendance_time,
                    "Confidence": round(avg_conf, 2),
                }])
                if os.path.exists(excel_path):
                    existing_df = pd.read_excel(excel_path, dtype={"Id": str})
                    combined = pd.concat([existing_df, new_row], ignore_index=True)
                    combined = combined.drop_duplicates(subset=["Id", "Date"], keep="last")
                    combined.to_excel(excel_path, index=False)
                else:
                    new_row.to_excel(excel_path, index=False)
            except Exception as exc:
                print(f"[DEBUG] Excel save failed: {exc}")

    stat = student_attendance_stat(student_id)
    return safe_render_template(
        "result.html",
        success=True,
        title="Attendance",
        message=(
            f"{name} (ID: {student_id}), your today's attendance is already marked."
            if already_marked
            else f"{name} (ID: {student_id}) attendance marked."
        ),
        back_url=url_for("main.home"),
        stats=[
            {"label": "Present Days", "value": stat["PresentDays"]},
            {"label": "Total Days", "value": stat["TotalDays"]},
            {"label": "Attendance", "value": f"{stat['Percentage']}%"},
        ],
    )


@bp.route("/register", methods=["POST"])
def register():
    if "admin" not in session:
        return redirect(url_for("main.login"))

    student_id = request.form.get("id", "").strip()
    name = clean_name(request.form.get("name", ""))

    if not valid_student_id(student_id):
        return safe_render_template(
            "result.html",
            success=False,
            title="Register Student",
            message="The Student ID should contain only numbers.",
            back_url=url_for("main.home"),
        )

    student_id = normalize_id(student_id)
    if student_exists(student_id):
        return safe_render_template(
            "result.html",
            success=False,
            title="Register Student",
            message=f"ID {student_id} already registered .Each student must have a unique ID.",
            back_url=url_for("main.home"),
        )

    session["temp_id"] = student_id
    session["temp_name"] = name
    return redirect(url_for("main.register_camera"))


@bp.route("/process_register")
def process_register():
    if "admin" not in session:
        return redirect(url_for("main.login"))

    student_id = session.get("temp_id")
    name = session.get("temp_name")

    if not student_id or not name:
        return redirect(url_for("main.home"))

    if student_exists(student_id):
        return safe_render_template(
            "result.html",
            success=False,
            title="Register Student",
            message=f"ID {student_id} Already registered. The registration has been canceled.",
            back_url=url_for("main.home"),
        )

    cam = None
    import time
    for camera_attempt in range(3):
        stop_camera_stream()
        time.sleep(0.5 + camera_attempt * 0.4)
        cam = open_camera(0)
        if cam is not None:
            break

    if cam is None:
        session.pop("temp_id", None)
        session.pop("temp_name", None)
        return safe_render_template(
            "result.html",
            success=False,
            title="Register Student",
            message="Camera open nahi ho pa raha. Camera ko use karne wali dusri app band karke phir try karein.",
            back_url=url_for("main.home"),
        )
    sample = 0
    ensure_folders()

    for _ in range(650):
        ret, img = cam.read()
        if not ret:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=5, minSize=(70, 70))

        for (x, y, w, h) in faces[:1]:
            sample += 1
            padded_face = crop_face_with_padding(img, x, y, w, h, 0.22)
            if padded_face is None:
                continue
            face = cv2.resize(padded_face, (224, 224))
            cv2.imwrite(os.path.join(TRAINING_DIR, f"{safe_file_name(name)}.{student_id}.{sample}.jpg"), face)

        if sample >= SAMPLE_COUNT:
            break

    cam.release()
    cv2.destroyAllWindows()

    if sample < 12:
        return safe_render_template(
            "result.html",
            success=False,
            title="Register Student",
            message="Not enough face samples were captured (need 12+). Please check the lighting and camera angle, then try again.",
            back_url=url_for("main.home"),
        )

    add_student(student_id, name)
    success, train_message = train_model()
    session.pop("temp_id", None)
    session.pop("temp_name", None)

    return safe_render_template(
        "result.html",
        success=success,
        title="Register Student",
        message=f"{name} (ID: {student_id}) add ho gaya. {train_message}",
        back_url=url_for("main.home"),
    )


@bp.route("/delete_student", methods=["POST"])
def delete_student_route():
    if "admin" not in session:
        return redirect(url_for("main.login"))

    student_id = request.form.get("student_id", "").strip()
    if not valid_student_id(student_id):
        return safe_render_template(
            "result.html",
            success=False,
            title="Delete Student",
            message="The selected student ID is invalid.",
            back_url=url_for("main.home"),
        )

    success, message = delete_student(student_id)
    if session.get("temp_id") == normalize_id(student_id):
        session.pop("temp_id", None)
        session.pop("temp_name", None)

    return safe_render_template(
        "result.html",
        success=success,
        title="Delete Student",
        message=message,
        back_url=url_for("main.home"),
    )


@bp.route("/retrain")
def retrain_model():
    if "admin" not in session:
        return redirect(url_for("main.login"))
    students = load_students()
    if students.empty:
        return safe_render_template(
            "result.html",
            success=False,
            title="Retrain Model",
            message="Koyi student registered nahi hai. Pehle student register karein.",
            back_url=url_for("main.home"),
        )
    success, train_message = train_model()
    return safe_render_template(
        "result.html",
        success=success,
        title="Retrain Model",
        message=train_message,
        back_url=url_for("main.home"),
    )


@bp.route("/sync_attendance")
def sync_attendance_route():
    if "admin" not in session:
        return redirect(url_for("main.login"))
    init_db()
    count = 0
    if os.path.exists(ATTENDANCE_DIR):
        for filename in os.listdir(ATTENDANCE_DIR):
            if not filename.startswith("Attendance_") or not filename.endswith(".xlsx"):
                continue
            path = os.path.join(ATTENDANCE_DIR, filename)
            try:
                df = pd.read_excel(path, dtype={"Id": str})
            except Exception:
                continue
            if df.empty or "Id" not in df.columns or "Name" not in df.columns:
                continue
            with get_db_connection() as conn:
                for _, row in df.iterrows():
                    student_id = str(row.get("Id", "")).strip()
                    if not valid_student_id(student_id):
                        continue
                    attendance_date = legacy_date_to_iso(row.get("Date", ""))
                    attendance_time = str(row.get("Time", "")).strip() or "00:00:00"
                    if not attendance_date:
                        continue
                    confidence = row.get("Confidence")
                    try:
                        confidence = float(confidence) if confidence == confidence else None
                    except Exception:
                        confidence = None
                    conn.execute(
                        """
                        INSERT OR IGNORE INTO attendance
                        (student_id, name, attendance_date, attendance_time, confidence)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            normalize_id(student_id),
                            clean_name(row.get("Name", "")),
                            attendance_date,
                            attendance_time,
                            confidence,
                        ),
                    )
                    count += 1
    return safe_render_template(
        "result.html",
        success=True,
        title="Sync Attendance",
        message=f"Attendance sync complete. {count} records imported from Excel files.",
        back_url=url_for("main.home"),
    )


@bp.route("/db_view")
def db_view():
    if "admin" not in session:
        return redirect(url_for("main.login"))
    init_db()
    with get_db_connection() as conn:
        students_rows = conn.execute("SELECT * FROM students ORDER BY CAST(id AS INTEGER), id").fetchall()
        attendance_rows = conn.execute(
            """
            SELECT id, student_id, name,
                   strftime('%d-%m-%Y', attendance_date) AS date,
                   attendance_time AS time,
                   ROUND(COALESCE(confidence, 0), 2) AS confidence
            FROM attendance
            ORDER BY attendance_date DESC, attendance_time DESC, id DESC
            LIMIT 300
            """
        ).fetchall()
        total_attendance = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        distinct_dates = conn.execute("SELECT COUNT(DISTINCT attendance_date) FROM attendance").fetchone()[0]
    return safe_render_template(
        "db_view.html",
        students=[dict(row) for row in students_rows],
        attendance=[dict(row) for row in attendance_rows],
        total_attendance=total_attendance,
        total_students=total_students,
        distinct_dates=distinct_dates,
    )


@bp.route("/health")
def health():
    return {"status": "ok", "version": APP_VERSION}
