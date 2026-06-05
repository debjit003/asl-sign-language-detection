"""
ASL Sign Language Detection Engine.

Handles MediaPipe hand detection, 170-feature extraction, ONNX inference,
and prediction smoothing. This is the proven Python pipeline — identical
to the training code in extract_landmarks.py.
"""

import base64
import json
import time
import tempfile
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
try:
    mp_hands = mp.solutions.hands
except AttributeError:
    import mediapipe.solutions.hands as mp_hands
import numpy as np
import onnxruntime as ort

# ── MediaPipe landmark indices ──────────────────────────────────────────────
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

FINGERS = [
    (THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP),
    (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
]
FINGERTIPS = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
FINGER_MCPS = [THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

KEY_DISTANCE_PAIRS = [
    (THUMB_TIP, INDEX_TIP), (THUMB_TIP, MIDDLE_TIP),
    (THUMB_TIP, RING_TIP), (THUMB_TIP, PINKY_TIP),
    (INDEX_TIP, MIDDLE_TIP), (MIDDLE_TIP, RING_TIP),
    (RING_TIP, PINKY_TIP), (INDEX_TIP, RING_TIP),
    (INDEX_TIP, PINKY_TIP), (MIDDLE_TIP, PINKY_TIP),
    (THUMB_TIP, INDEX_MCP), (THUMB_TIP, MIDDLE_MCP),
    (THUMB_TIP, RING_MCP), (THUMB_TIP, PINKY_MCP),
    (INDEX_PIP, RING_PIP), (INDEX_DIP, MIDDLE_DIP),
    (THUMB_IP, INDEX_PIP), (THUMB_TIP, WRIST),
    (INDEX_TIP, WRIST), (PINKY_TIP, WRIST),
]


# ── Helper functions ────────────────────────────────────────────────────────
def _pt(coords, idx):
    return coords[idx * 3: idx * 3 + 3]

def _dist(p1, p2):
    return np.sqrt(np.sum((p1 - p2) ** 2))

def _angle(coords, i1, i2, i3):
    p1, p2, p3 = _pt(coords, i1), _pt(coords, i2), _pt(coords, i3)
    v1, v2 = p1 - p2, p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))

def _palm_center(coords):
    indices = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return np.array([_pt(coords, i) for i in indices]).mean(axis=0)


# ── Feature extraction (170 features — identical to training) ───────────────
def extract_features(raw_landmarks):
    """Extract the 170-feature vector. Exact copy of extract_landmarks.py."""
    if raw_landmarks is None or len(raw_landmarks) != 63:
        return None
    coords = raw_landmarks.astype(np.float64).copy()

    # Normalize: center on wrist, scale by max distance
    wrist = coords[0:3].copy()
    for i in range(21):
        coords[i*3:i*3+3] -= wrist
    max_dist = max(np.sqrt(np.sum(coords[i*3:i*3+3]**2)) for i in range(21))
    if max_dist > 0:
        coords /= max_dist

    features = []

    # 63 normalized coordinates
    features.extend(coords.tolist())

    # 20 key pairwise distances
    for i1, i2 in KEY_DISTANCE_PAIRS:
        features.append(_dist(_pt(coords, i1), _pt(coords, i2)))

    # 15 finger joint angles
    for mcp, pip, dip, tip in FINGERS:
        features.append(_angle(coords, WRIST, mcp, pip))
        features.append(_angle(coords, mcp, pip, dip))
        features.append(_angle(coords, pip, dip, tip))

    # 5 fingertip-to-palm distances
    pc = _palm_center(coords)
    for tip_idx in FINGERTIPS:
        features.append(_dist(_pt(coords, tip_idx), pc))

    # 5 finger extension ratios
    ext_ratios = []
    for mcp, pip, dip, tip in FINGERS:
        direct = _dist(_pt(coords, tip), _pt(coords, mcp))
        bone = _dist(_pt(coords, mcp), _pt(coords, pip)) + _dist(_pt(coords, pip), _pt(coords, dip)) + _dist(_pt(coords, dip), _pt(coords, tip))
        r = direct / (bone + 1e-8)
        ext_ratios.append(r)
        features.append(r)

    # 1 bbox aspect ratio
    xs = [coords[i*3] for i in range(21)]
    ys = [coords[i*3+1] for i in range(21)]
    features.append((max(xs) - min(xs)) / (max(ys) - min(ys) + 1e-8))

    # 10 pairwise fingertip distances
    for i in range(len(FINGERTIPS)):
        for j in range(i+1, len(FINGERTIPS)):
            features.append(_dist(_pt(coords, FINGERTIPS[i]), _pt(coords, FINGERTIPS[j])))

    # 4 thumb cross-over
    thumb_tip = _pt(coords, THUMB_TIP)
    for k in range(1, len(FINGER_MCPS)):
        features.append(thumb_tip[0] - _pt(coords, FINGER_MCPS[k])[0])

    # 5 curl indicators
    wrist_pt = _pt(coords, WRIST)
    for tip_idx in FINGERTIPS:
        features.append(_dist(_pt(coords, tip_idx), wrist_pt))

    # 4 MCP splay
    for a, b, c in [(THUMB_CMC, WRIST, INDEX_MCP), (INDEX_MCP, WRIST, MIDDLE_MCP),
                     (MIDDLE_MCP, WRIST, RING_MCP), (RING_MCP, WRIST, PINKY_MCP)]:
        features.append(_angle(coords, a, b, c))

    # 4 TIP splay
    for a, b, c in [(THUMB_TIP, WRIST, INDEX_TIP), (INDEX_TIP, WRIST, MIDDLE_TIP),
                     (MIDDLE_TIP, WRIST, RING_TIP), (RING_TIP, WRIST, PINKY_TIP)]:
        features.append(_angle(coords, a, b, c))

    # 2 thumb position
    features.append(_pt(coords, THUMB_TIP)[1] - _pt(coords, INDEX_MCP)[1])
    features.append(_pt(coords, THUMB_TIP)[2] - _pt(coords, MIDDLE_MCP)[2])

    # 12 enhanced spread
    tip_indices = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    for k in range(len(tip_indices) - 1):
        p1, p2 = _pt(coords, tip_indices[k]), _pt(coords, tip_indices[k+1])
        features.append(abs(p1[0] - p2[0]))
    dip_indices = [THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]
    for k in range(len(dip_indices) - 1):
        features.append(_dist(_pt(coords, dip_indices[k]), _pt(coords, dip_indices[k+1])))
    total_spread = sum(_dist(_pt(coords, tip_indices[k]), _pt(coords, tip_indices[k+1])) for k in range(len(tip_indices)-1))
    features.append(total_spread)
    tip_xs = [_pt(coords, idx)[0] for idx in tip_indices]
    features.append(max(tip_xs) - min(tip_xs))
    features.append(_angle(coords, THUMB_TIP, INDEX_MCP, INDEX_TIP))
    features.append(abs(_pt(coords, THUMB_TIP)[1] - _pt(coords, INDEX_TIP)[1]))

    # N1: 5 finger curl scores
    for mcp, pip, dip, tip in FINGERS:
        pip_a = _angle(coords, mcp, pip, dip)
        dip_a = _angle(coords, pip, dip, tip)
        features.append(1.0 - ((pip_a + dip_a) / 2.0) / np.pi)

    # N2: 5 binary extended flags
    extended_flags = []
    for ratio in ext_ratios:
        flag = 1.0 if ratio > 0.7 else 0.0
        extended_flags.append(flag)
        features.append(flag)

    # N3: thumb across palm
    features.append(_pt(coords, THUMB_TIP)[0] - _pt(coords, INDEX_MCP)[0])

    # N4: 3 adjacent finger attachment
    for i1, i2 in [(INDEX_TIP, MIDDLE_TIP), (MIDDLE_TIP, RING_TIP), (RING_TIP, PINKY_TIP)]:
        features.append(_dist(_pt(coords, i1), _pt(coords, i2)))

    # N5: number of extended fingers
    features.append(sum(extended_flags))

    # N6: fingers-together score
    adj = [_dist(_pt(coords, tip_indices[k]), _pt(coords, tip_indices[k+1])) for k in range(1, len(tip_indices)-1)]
    features.append(np.mean(adj) if adj else 0.0)

    # N7: palm orientation
    v1 = _pt(coords, INDEX_MCP) - _pt(coords, WRIST)
    v2 = _pt(coords, PINKY_MCP) - _pt(coords, WRIST)
    normal = np.cross(v1, v2)
    normal = normal / (np.linalg.norm(normal) + 1e-8)
    features.append(np.arcsin(np.clip(normal[2], -1, 1)))
    features.append(np.arctan2(normal[1], normal[0] + 1e-8))

    # N8: thumb-to-finger contact
    features.append(_dist(_pt(coords, THUMB_TIP), _pt(coords, INDEX_PIP)))
    features.append(_dist(_pt(coords, THUMB_TIP), _pt(coords, MIDDLE_PIP)))

    return np.array(features, dtype=np.float64)


# ── Prediction Smoother ─────────────────────────────────────────────────────
class PredictionSmoother:
    """Sliding-window probability averaging for stable predictions."""
    def __init__(self, window_size=5, n_classes=36):
        self.window = deque(maxlen=window_size)
        self.n_classes = n_classes

    def add(self, proba):
        self.window.append(proba.copy())

    def get_smoothed(self):
        if not self.window:
            return None, 0.0
        avg = np.mean(list(self.window), axis=0)
        idx = int(np.argmax(avg))
        return idx, float(avg[idx])

    def reset(self):
        self.window.clear()


# ── Detection Engine ────────────────────────────────────────────────────────
class DetectionEngine:
    """Full pipeline: image → hand detection → features → classification."""

    def __init__(self, models_dir=None):
        if models_dir is None:
            models_dir = Path(__file__).resolve().parent / "models"
        models_dir = Path(models_dir)

        # Load ONNX
        self.session = ort.InferenceSession(str(models_dir / "sign_classifier.onnx"))
        self.input_name = self.session.get_inputs()[0].name

        # Load scaler
        with open(models_dir / "scaler_params.json") as f:
            sd = json.load(f)
        self.scaler_mean = np.array(sd["mean"])
        self.scaler_scale = np.array(sd["scale"])
        self.n_features = sd["n_features"]

        # Load labels
        with open(models_dir / "label_classes.json") as f:
            self.labels = json.load(f)["classes"]

        # MediaPipe
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Smoother
        self.smoother = PredictionSmoother(window_size=5, n_classes=len(self.labels))

        # Sentence state
        self.sentence = []
        self.current_letter = None
        self.letter_start_time = None
        self.last_locked_time = 0
        self.last_hand_time = time.time()

        # Timing constants
        self.HOLD_TIME = 1.5       # seconds to hold for lock-in
        self.COOLDOWN_TIME = 0.5   # seconds after lock before next
        self.CONFIDENCE_THRESHOLD = 75.0  # ignore predictions below this %

        print(f"[DetectionEngine] Loaded: {self.n_features} features, "
              f"{len(self.labels)} classes")

    def process_frame(self, image_data):
        """
        Process a single frame (base64 or numpy array).
        Returns dict with prediction info.
        """
        now = time.time()

        # Decode image
        if isinstance(image_data, str):
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, np.ndarray):
            frame = image_data
        else:
            return {"hand_detected": False}

        if frame is None:
            return {"hand_detected": False}

        # Detect hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(rgb)

        if not result.multi_hand_landmarks:
            # No hand — reset tracking (no auto-space)
            self.current_letter = None
            self.letter_start_time = None
            return {
                "hand_detected": False,
                "sentence": "".join(self.sentence),
            }

        self.last_hand_time = now
        hand = result.multi_hand_landmarks[0]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

        # Landmark positions for client-side drawing (normalized x,y)
        landmarks = [{"x": round(float(lm.x), 4), "y": round(float(lm.y), 4)}
                     for lm in hand.landmark]

        # Extract features
        feat = extract_features(raw)
        if feat is None or len(feat) != self.n_features:
            return {"hand_detected": True, "sentence": "".join(self.sentence)}

        # Scale + predict
        scaled = ((feat - self.scaler_mean) / self.scaler_scale).astype(np.float32).reshape(1, -1)
        results = self.session.run(None, {self.input_name: scaled})
        proba = results[1][0].astype(np.float32)

        # Smooth
        self.smoother.add(proba)
        smooth_idx, smooth_conf = self.smoother.get_smoothed()

        # Raw prediction
        raw_idx = int(np.argmax(proba))
        raw_conf = float(proba[raw_idx])
        raw_letter = self.labels[raw_idx].upper()

        # Top 3
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [{"letter": self.labels[i].upper(), "confidence": round(float(proba[i]) * 100, 1)}
                for i in top3_idx]

        # Confidence gate: ignore low-confidence predictions
        smooth_conf_pct = smooth_conf * 100
        letter = self.labels[smooth_idx].upper() if (smooth_idx is not None and smooth_conf_pct >= self.CONFIDENCE_THRESHOLD) else ""

        # Hold-to-lock (only if confidence passes threshold)
        hold_progress = 0.0
        locked = False
        in_cooldown = (now - self.last_locked_time) < self.COOLDOWN_TIME

        if letter and not in_cooldown:
            if letter == self.current_letter and self.letter_start_time:
                elapsed = now - self.letter_start_time
                hold_progress = min(elapsed / self.HOLD_TIME, 1.5)
                if elapsed >= self.HOLD_TIME:
                    self.sentence.append(letter)
                    self.last_locked_time = now
                    self.letter_start_time = None
                    self.current_letter = None
                    locked = True
            else:
                self.current_letter = letter
                self.letter_start_time = now
        elif not letter:
            # Below threshold — reset hold tracking
            self.current_letter = None
            self.letter_start_time = None

        return {
            "hand_detected": True,
            "letter": raw_letter,
            "confidence": round(raw_conf * 100, 1),
            "smoothed_letter": letter,
            "smoothed_confidence": round(smooth_conf * 100, 1),
            "top3": top3,
            "hold_progress": round(hold_progress, 2),
            "locked": locked,
            "sentence": "".join(self.sentence),
            "landmarks": landmarks,
            "connections": HAND_CONNECTIONS,
        }

    def process_video(self, video_path):
        """
        Process a video file. Returns transcription with timestamps.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"error": "Cannot open video"}

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        # Process every Nth frame (~5 detections per second)
        sample_interval = max(1, int(fps / 5))

        # Use a fresh MediaPipe instance in static mode for video
        mp_static = mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.3,
        )

        detections = []
        frame_idx = 0
        prev_letter = None
        stability_count = 0
        stability_threshold = 3  # consecutive same predictions to lock
        last_locked_letter = None

        text_parts = []
        no_hand_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                timestamp = frame_idx / fps
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = mp_static.process(rgb)

                if result.multi_hand_landmarks:
                    no_hand_count = 0
                    hand = result.multi_hand_landmarks[0]
                    raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
                    feat = extract_features(raw)

                    if feat is not None and len(feat) == self.n_features:
                        scaled = ((feat - self.scaler_mean) / self.scaler_scale).astype(np.float32).reshape(1, -1)
                        res = self.session.run(None, {self.input_name: scaled})
                        proba = res[1][0]
                        pred_idx = int(np.argmax(proba))
                        letter = self.labels[pred_idx].upper()
                        conf = float(proba[pred_idx]) * 100

                        # Confidence gate for video too
                        if conf < self.CONFIDENCE_THRESHOLD:
                            prev_letter = None
                            stability_count = 0
                        elif letter == prev_letter:
                            stability_count += 1
                        else:
                            stability_count = 1
                            prev_letter = letter

                        if conf >= self.CONFIDENCE_THRESHOLD and stability_count >= stability_threshold and letter != last_locked_letter:
                            detections.append({
                                "time": round(timestamp, 2),
                                "letter": letter,
                                "confidence": round(conf, 1),
                            })
                            text_parts.append(letter)
                            last_locked_letter = letter
                else:
                    no_hand_count += 1
                    # No auto-space in video either
                    last_locked_letter = None
                    prev_letter = None
                    stability_count = 0

            frame_idx += 1

        cap.release()
        mp_static.close()

        return {
            "duration": round(duration, 2),
            "fps": round(fps, 1),
            "total_frames": total_frames,
            "frames_processed": frame_idx // sample_interval,
            "detections": detections,
            "transcription": "".join(text_parts).strip(),
        }

    def reset_sentence(self):
        self.sentence.clear()
        self.current_letter = None
        self.letter_start_time = None
        self.smoother.reset()

    def backspace(self):
        if self.sentence:
            self.sentence.pop()

    def add_space(self):
        if not self.sentence or self.sentence[-1] != " ":
            self.sentence.append(" ")

    def close(self):
        self.mp_hands.close()
