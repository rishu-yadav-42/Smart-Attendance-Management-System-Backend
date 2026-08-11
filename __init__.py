import os
from flask import Flask

from backend.config import BASE_DIR, ensure_folders
from backend.database import init_db


def create_app():
    frontend_dir = os.path.join(BASE_DIR, "frontend")
    frontend_static = os.path.join(frontend_dir, "static")

    app = Flask(
        __name__,
        template_folder=frontend_dir,
        static_folder=frontend_static,
    )
    app.secret_key = "secret123"

    # Add CORS headers for separate frontend-backend deployments
    @app.after_request
    def add_cors_headers(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE, PUT"
        return response

    from backend.routes import bp
    app.register_blueprint(bp)

    try:
        ensure_folders()
        init_db()
    except Exception as exc:
        print(f"[WARNING] Startup initialization issue: {exc}")

    return app


app = create_app()
