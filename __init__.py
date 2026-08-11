import os
from flask import Flask

try:
    from backend.config import BASE_DIR, ensure_folders
    from backend.database import init_db
except ImportError:
    from config import BASE_DIR, ensure_folders
    from database import init_db


def create_app():
    frontend_dir = os.path.join(BASE_DIR, "frontend")
    frontend_static = os.path.join(frontend_dir, "static")

    app = Flask(
        __name__,
        template_folder=frontend_dir if os.path.exists(os.path.join(frontend_dir, "index.html")) else BASE_DIR,
        static_folder=frontend_static if os.path.exists(frontend_static) else BASE_DIR,
    )
    app.secret_key = os.getenv("SECRET_KEY", "secret123")

    @app.after_request
    def add_cors_headers(response):
        allowed_origin = os.getenv("FRONTEND_URL") or os.getenv("CORS_ORIGIN") or "*"
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE, PUT"
        return response

    try:
        from backend.routes import bp
    except ImportError:
        from routes import bp

    app.register_blueprint(bp)

    try:
        ensure_folders()
        init_db()
    except Exception as exc:
        print(f"[WARNING] Startup initialization issue: {exc}")

    return app


app = create_app()
