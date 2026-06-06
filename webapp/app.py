"""
ASL Sign Language Detection — Flask Web Application.

Provides real-time webcam detection via WebSocket and video upload processing.
"""

import os
import tempfile
import threading

from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
from flask_cors import CORS

from detection import DetectionEngine

app = Flask(__name__)
app.config["SECRET_KEY"] = os.urandom(24).hex()
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB max upload

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading",
                    max_http_buffer_size=10 * 1024 * 1024)

# Per-client detection engines and processing locks
engines = {}
locks = {}


def get_engine(sid):
    if sid not in engines:
        engines[sid] = DetectionEngine()
        locks[sid] = threading.Lock()
    return engines[sid]


def get_lock(sid):
    if sid not in locks:
        locks[sid] = threading.Lock()
    return locks[sid]


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "No video file uploaded"}), 400

    video = request.files["video"]
    if video.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Save to temp file
    ext = os.path.splitext(video.filename)[1] or ".mp4"
    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
    try:
        video.save(tmp.name)
        tmp.close()
        print(f"[Video Upload] Saved temp file: {tmp.name} ({os.path.getsize(tmp.name)} bytes)")
    except Exception as e:
        print(f"[Video Upload] Error saving file: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Failed to save video: {str(e)}"}), 500

    try:
        engine = DetectionEngine()
        result = engine.process_video(tmp.name)
        engine.close()
        print(f"[Video Upload] Processing complete: {result.get('transcription', '')}")
        return jsonify(result)
    except Exception as e:
        print(f"[Video Upload] Processing error: {e}")
        import traceback
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
    print(f"[WS] Client connected: {request.sid}")


@socketio.on("disconnect")
def on_disconnect():
    sid = request.sid
    if sid in engines:
        engines[sid].close()
        del engines[sid]
    if sid in locks:
        del locks[sid]
    print(f"[WS] Client disconnected: {sid}")


@socketio.on("frame")
def on_frame(data):
    sid = request.sid
    lock = get_lock(sid)

    # Skip if still processing previous frame (prevents backlog)
    if not lock.acquire(blocking=False):
        return

    try:
        engine = get_engine(sid)
        image_data = data.get("data", "")
        result = engine.process_frame(image_data)
        emit("prediction", result)
    finally:
        lock.release()


@socketio.on("reset_sentence")
def on_reset():
    engine = get_engine(request.sid)
    engine.reset_sentence()
    emit("prediction", {"sentence": "", "hand_detected": False})


@socketio.on("backspace")
def on_backspace():
    engine = get_engine(request.sid)
    engine.backspace()
    emit("prediction", {"sentence": "".join(engine.sentence), "hand_detected": False})


@socketio.on("add_space")
def on_space():
    engine = get_engine(request.sid)
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
