import os
import re


def valid_student_id(student_id):
    return bool(re.fullmatch(r"\d+", str(student_id).strip()))


def normalize_id(student_id):
    return str(int(str(student_id).strip()))


def clean_name(name):
    return re.sub(r"\s+", " ", str(name).strip())


def safe_file_name(name):
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", clean_name(name))
    return safe.strip("_") or "Student"


def parse_training_id(filename):
    stem = os.path.splitext(filename)[0]
    parts = stem.rsplit(".", 2)
    if len(parts) != 3:
        return None
    try:
        return int(parts[1])
    except ValueError:
        return None


def training_file_student_id(filename):
    label = parse_training_id(filename)
    if label is None:
        return None
    return str(label)
