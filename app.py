import os
import sys

# Ensure both current directory and parent directory are in Python path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

for d in (CURRENT_DIR, PARENT_DIR):
    if d not in sys.path:
        sys.path.insert(0, d)

try:
    from backend import app
except ImportError:
    from __init__ import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
