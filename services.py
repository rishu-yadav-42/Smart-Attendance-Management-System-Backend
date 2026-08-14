import os
import pandas as pd

try:
    from backend.config import (
        ARCFACE_MODEL_FILE,
        ATTENDANCE_DIR,
        FALLBACK_MODEL_FILE,
        MODEL_FILE,
        STUDENT_FILE,
        TRAINING_DIR,
    )
    from backend.database import get_db_connection, init_db
    from backend.helpers import clean_name, normalize_id, training_file_student_id
except ImportError:
    from config import (
        ARCFACE_MODEL_FILE,
        ATTENDANCE_DIR,
        FALLBACK_MODEL_FILE,
        MODEL_FILE,
        STUDENT_FILE,
        TRAINING_DIR,
    )
    from database import get_db_connection, init_db
    from helpers import clean_name, normalize_id, training_file_student_id


def load_students():
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT id AS Id, name AS Name FROM students ORDER BY CAST(id AS INTEGER), id"
        ).fetchall()
    if not rows:
        return pd.DataFrame(columns=["Id", "Name"])
    return pd.DataFrame([dict(row) for row in rows], columns=["Id", "Name"])


def save_students(df):
    init_db()
    df = df[["Id", "Name"]].drop_duplicates(subset=["Id"], keep="last").copy()
    df["Id"] = df["Id"].astype(str).map(normalize_id)
    df["Name"] = df["Name"].astype(str).map(clean_name)
    with get_db_connection() as conn:
        conn.execute("DELETE FROM students")
        conn.executemany(
            "INSERT INTO students (id, name) VALUES (?, ?)",
            [(row["Id"], row["Name"]) for _, row in df.iterrows()],
        )
        conn.commit()
    try:
        os.makedirs(os.path.dirname(STUDENT_FILE), exist_ok=True)
        df.to_csv(STUDENT_FILE, index=False)
    except Exception as exc:
        print(f"[WARNING] Could not save students to CSV: {exc}")


def student_exists(student_id):
    students = load_students()
    return normalize_id(student_id) in set(students["Id"].astype(str))


def sync_student_to_csv(student_id, name):
    try:
        os.makedirs(os.path.dirname(STUDENT_FILE), exist_ok=True)
        if os.path.exists(STUDENT_FILE):
            try:
                df = pd.read_csv(STUDENT_FILE, dtype={"Id": str, "Name": str})
            except Exception:
                df = pd.DataFrame(columns=["Id", "Name"])
        else:
            df = pd.DataFrame(columns=["Id", "Name"])

        if "Id" not in df.columns or "Name" not in df.columns:
            df = pd.DataFrame(columns=["Id", "Name"])

        sid = str(student_id).strip()
        sname = str(name).strip()
        df["Id"] = df["Id"].astype(str).str.strip()
        df = df[df["Id"] != sid]
        new_row = pd.DataFrame([{"Id": sid, "Name": sname}])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(STUDENT_FILE, index=False)
    except Exception as exc:
        print(f"[WARNING] Could not sync student to CSV: {exc}")


def add_student(student_id, name):
    init_db()
    sid = normalize_id(student_id)
    sname = clean_name(name)
    with get_db_connection() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO students (id, name) VALUES (?, ?)",
            (sid, sname),
        )
        conn.commit()
    sync_student_to_csv(sid, sname)


def remove_student_files(student_id):
    if not os.path.exists(TRAINING_DIR):
        return 0

    removed = 0
    for file in os.listdir(TRAINING_DIR):
        if not file.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        if training_file_student_id(file) != student_id:
            continue
        try:
            os.remove(os.path.join(TRAINING_DIR, file))
            removed += 1
        except OSError:
            continue
    return removed


def remove_model_files():
    for path in (MODEL_FILE, FALLBACK_MODEL_FILE, ARCFACE_MODEL_FILE):
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                continue


def remove_student_from_csv(student_id):
    if not os.path.exists(STUDENT_FILE):
        return
    try:
        df = pd.read_csv(STUDENT_FILE, dtype={"Id": str, "Name": str})
        if df.empty or "Id" not in df.columns:
            return
        df = df[df["Id"].astype(str).str.strip() != student_id]
        df.to_csv(STUDENT_FILE, index=False)
    except Exception:
        pass


def delete_student(student_id):
    try:
        from backend.face_recognition import train_model
    except ImportError:
        from face_recognition import train_model

    student_id = normalize_id(student_id)
    students = load_students()
    match = students[students["Id"].astype(str) == student_id]
    if match.empty:
        return False, "Student record not found."

    student_name = match.iloc[0]["Name"]
    removed_samples = remove_student_files(student_id)
    remove_student_from_csv(student_id)

    init_db()
    with get_db_connection() as conn:
        conn.execute("DELETE FROM attendance WHERE student_id = ?", (student_id,))
        conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        conn.commit()

    remaining_students = load_students()
    if remaining_students.empty:
        remove_model_files()
        model_message = "No students are left, so the trained face model files were cleared."
    else:
        success, train_message = train_model()
        model_message = train_message if success else f"Student removed, but model retraining needs attention: {train_message}"

    return True, f"{student_name} (ID: {student_id}) deleted. Removed {removed_samples} face samples. {model_message}"


def latest_attendance_rows(limit=200):
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT
                student_id AS Id,
                name AS Name,
                strftime('%d-%m-%Y', attendance_date) AS Date,
                attendance_time AS Time,
                ROUND(COALESCE(confidence, 0), 2) AS Confidence
            FROM attendance
            ORDER BY attendance_date DESC, attendance_time DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]


def total_attendance_count():
    init_db()
    with get_db_connection() as conn:
        return conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]


def attendance_files():
    init_db()
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT attendance_date FROM attendance ORDER BY attendance_date"
        ).fetchall()
    return [row["attendance_date"] for row in rows]


def attendance_summary():
    students = load_students()
    total_days = len(attendance_files())
    if students.empty:
        return []

    with get_db_connection() as conn:
        present_rows = conn.execute(
            """
            SELECT student_id, COUNT(DISTINCT attendance_date) AS present_days
            FROM attendance
            GROUP BY student_id
            """
        ).fetchall()
    present_map = {row["student_id"]: row["present_days"] for row in present_rows}

    summary = []
    for _, row in students.iterrows():
        student_id = str(row["Id"])
        present_days = int(present_map.get(student_id, 0))
        summary.append(
            {
                "Id": student_id,
                "Name": row["Name"],
                "PresentDays": present_days,
                "TotalDays": total_days,
                "Percentage": round((present_days / total_days) * 100, 2) if total_days else 0,
            }
        )

    return summary


def student_attendance_stat(student_id):
    student_id = normalize_id(student_id)
    for row in attendance_summary():
        if row["Id"] == student_id:
            return row
    return {"Id": student_id, "PresentDays": 0, "TotalDays": len(attendance_files()), "Percentage": 0}
