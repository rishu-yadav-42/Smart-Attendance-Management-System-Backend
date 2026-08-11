# Smart Attendance Management System - Backend API

This repository contains the backend Python Flask API for the Smart Attendance Management System.

## Features
- **Face Recognition**: Powered by OpenCV Haar Cascades and ArcFace ONNX embeddings.
- **SQLite & Excel Database**: Automated attendance logging with dual-database persistence.
- **RESTful Endpoints & SSE Streaming**: Camera live streaming (`/video_feed`), frame classification (`/process_frame`), and attendance marking (`/mark_attendance_ajax`).
- **CORS Support**: Cross-Origin Resource Sharing enabled for independent frontend hosting.

## Project Structure
```
backend/
├── app.py                # Primary entry point (gunicorn / flask app execution)
├── __init__.py           # Flask app factory with CORS headers
├── config.py             # Configuration parameters and file paths
├── database.py           # SQLite database connection and table initialization
├── face_recognition.py   # OpenCV & ArcFace embedding recognition pipelines
├── camera.py             # Video capture, thread-safe buffering, and streaming
├── services.py           # Student details & attendance summary services
├── routes.py             # Web routes and REST API endpoints
├── helpers.py            # Validation and string utility helpers
├── requirements.txt      # Python package dependencies
├── Procfile              # Deployment web process definition
└── runtime.txt           # Python runtime version specification
```

## Running Locally

```bash
pip install -r requirements.txt
python app.py
```

## Cloud Deployment (Render / Railway / Heroku)

1. Connect this repository to Render / Railway / Heroku.
2. Build Command: `pip install -r requirements.txt`
3. Start Command: `gunicorn app:app`
