"""
ASL Sign Language Detection — Flask Web Application.

Provides real-time webcam detection via WebSocket and video upload processing.
"""

import os
import sys
import tempfile
import traceback
import time

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from detection import DetectionEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload (Render safe)

CORS(app)

# Use gevent for production (Render), threading for local dev
# gevent is the most reliable async mode for gunicorn + WebSocket
async_mode = "gevent" if "gunicorn" in os.environ.get("SERVER_SOFTWARE", "") else "threading"

# Fallback: detect if gevent is importable
try:
    import gevent  # noqa: F401
    async_mode = "gevent"
except ImportError:
    async_mode = "threading"

print(f"[App] Using async_mode: {async_mode}")

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode=async_mode,
    max_http_buffer_size=10 * 1024 * 1024,
    ping_timeout=60,
    ping_interval=25,
)

# Shared detection engine (single instance, thread-safe via lock)
import threading
engine_lock = threading.Lock()
shared_engine = None

# Per-client sentence state
client_state = {}


def get_engine():
    global shared_engine
    if shared_engine is None:
        shared_engine = DetectionEngine()
    return shared_engine


def get_client_state(sid):
    if sid not in client_state:
        client_state[sid] = {
            "sentence": [],
            "current_letter": None,
            "letter_start_time": None,
            "last_locked_time": 0,
        }
    return client_state[sid]


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    """Health check endpoint for Render monitoring."""
    return jsonify({"status": "ok", "timestamp": time.time()})


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file
    ext = os.path.splitext(video.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=tempfile.gettempdir())
    try:
        video.save(tmp.name)
        tmp.close()
        file_size = os.path.getsize(tmp.name)
        print(f"[Video Upload] Saved: {tmp.name} ({file_size / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"[Video Upload] Error saving file: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Failed to save video: {str(e)}"}), 500

    try:
        engine = DetectionEngine()
        start_time = time.time()
        result = engine.process_video(tmp.name)
        elapsed = time.time() - start_time
        engine.close()
        print(f"[Video Upload] Done in {elapsed:.1f}s: '{result.get('transcription', '')}'")
        return jsonify(result)
    except Exception as e:
        print(f"[Video Upload] Processing error: {e}")
        traceback.print_exc()
        return jsonify({"error": f"Video processing failed: {str(e)}"}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


# ── WebSocket Events ────────────────────────────────────────────────────────

@socketio.on("connect")
def on_connect():
    sid = request.sid
    get_client_state(sid)
    print(f"[WS] Client connected: {sid}")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    if sid in client_state:
        del client_state[sid]
    print(f"[WS] Client disconnected: {sid}")


@socketio.on("frame")
def on_frame(data):
    sid = request.sid

    try:
        engine = get_engine()
        image_data = data.get("data", "")
        result = engine.process_frame(image_data)

        # Merge per-client sentence state
        state = get_client_state(sid)
        result["sentence"] = "".join(state.get("sentence", []))

        emit("prediction", result)
    except Exception as e:
        print(f"[WS] Frame error for {sid}: {e}")
        emit("prediction", {"hand_detected": False, "error": str(e)})


@socketio.on("reset_sentence")
def on_reset():
    engine = get_engine()
    engine.reset_sentence()
    emit("prediction", {"sentence": "", "hand_detected": False})


@socketio.on("backspace")
def on_backspace():
    engine = get_engine()
    engine.backspace()
    emit("prediction", {"sentence": "".join(engine.sentence), "hand_detected": False})


@socketio.on("add_space")
def on_space():
    engine = get_engine()
    engine.add_space()
    emit("prediction", {"sentence": "".join(engine.sentence), "hand_detected": False})


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ASL Sign Language Detection — Web Application")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
