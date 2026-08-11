from datetime import datetime
import os
import sqlite3
import pandas as pd

try:
    from backend.config import ATTENDANCE_DIR, DB_FILE, STUDENT_FILE, ensure_folders
    from backend.helpers import clean_name, normalize_id, valid_student_id
except ImportError:
    from config import ATTENDANCE_DIR, DB_FILE, STUDENT_FILE, ensure_folders
    from helpers import clean_name, normalize_id, valid_student_id


def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def legacy_date_to_iso(value):
    value = str(value).strip()
    if not value:
        return None
    for fmt in ("%d-%m-%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return None


def init_db():
    ensure_folders()
    with get_db_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id TEXT NOT NULL,
                name TEXT NOT NULL,
                attendance_date TEXT NOT NULL,
                attendance_time TEXT NOT NULL,
                confidence REAL,
                UNIQUE(student_id, attendance_date),
                FOREIGN KEY(student_id) REFERENCES students(id)
            )
            """
        )

        student_count = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
        if student_count == 0 and os.path.exists(STUDENT_FILE):
            df = pd.read_csv(STUDENT_FILE, dtype={"Id": str, "Name": str})
            if not df.empty:
                df = df.dropna(subset=["Id", "Name"])
                df["Id"] = df["Id"].astype(str).str.strip()
                df["Name"] = df["Name"].astype(str).map(clean_name)
                df = df[df["Id"].map(valid_student_id)]
                df["Id"] = df["Id"].map(normalize_id)
                df = df.drop_duplicates(subset=["Id"], keep="last")
                conn.executemany(
                    "INSERT OR REPLACE INTO students (id, name) VALUES (?, ?)",
                    [(row["Id"], row["Name"]) for _, row in df.iterrows()],
                )

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
