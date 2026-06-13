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

# Use threading for local dev (most reliable with flask-socketio).
# gevent/eventlet only needed for production WSGI servers.
async_mode = "threading"
try:
    # If running under gunicorn, use gevent
    import gevent  # noqa: F401
    if "gunicorn" in os.environ.get("SERVER_SOFTWARE", ""):
        async_mode = "gevent"
except ImportError:
    pass

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
        engine = get_engine()
        client_state[sid] = engine.create_client_state()
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
        # Reuse the shared engine — avoids reloading MediaPipe + ONNX
        # from disk on every request (was the main cause of video upload
        # failures due to slow initialization)
        engine = get_engine()
        start_time = time.time()
        result = engine.process_video(tmp.name)
        elapsed = time.time() - start_time
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
        state = get_client_state(sid)
        result = engine.process_frame(image_data, state=state)
        emit("prediction", result)
    except Exception as e:
        print(f"[WS] Frame error for {sid}: {e}")
        emit("prediction", {"hand_detected": False, "error": str(e)})


@socketio.on("reset_sentence")
def on_reset():
    sid = request.sid
    state = get_client_state(sid)
    engine = get_engine()
    engine.reset_client_state(state)
    emit("prediction", {"sentence": "", "hand_detected": False})


@socketio.on("backspace")
def on_backspace():
    sid = request.sid
    state = get_client_state(sid)
    if state["sentence"]:
        state["sentence"].pop()
    emit("prediction", {"sentence": "".join(state["sentence"]), "hand_detected": False})


@socketio.on("add_space")
def on_space():
    sid = request.sid
    state = get_client_state(sid)
    if not state["sentence"] or state["sentence"][-1] != " ":
        state["sentence"].append(" ")
    emit("prediction", {"sentence": "".join(state["sentence"]), "hand_detected": False})


# ── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  ASL Sign Language Detection — Web Application")
    print("  Open http://localhost:5000 in your browser")
    print("=" * 60)
    socketio.run(app, host="0.0.0.0", port=5000, debug=False,
                 allow_unsafe_werkzeug=True)
