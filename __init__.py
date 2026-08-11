import os
from flask import Flask, jsonify

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

    @app.before_request
    def handle_options_preflight():
        from flask import request
        if request.method == "OPTIONS":
            origin = request.headers.get("Origin") or "*"
            response = jsonify({"status": "ok"})
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE, PUT"
            response.headers["Access-Control-Allow-Credentials"] = "true"
            return response, 200

    @app.after_request
    def add_cors_headers(response):
        from flask import request
        origin = request.headers.get("Origin") or "*"
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With, Accept"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS, DELETE, PUT"
        response.headers["Access-Control-Allow-Credentials"] = "true"
        return response

    @app.errorhandler(Exception)
    def handle_exception(e):
        return jsonify({
            "success": False,
            "status": "error",
            "message": f"Backend Exception: {str(e)}"
        }), 500

    @app.errorhandler(404)
    def handle_404(e):
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Endpoint not found on backend server."
        }), 404

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
