"""
ASL Sign Language Detection Engine.

Core inference pipeline for real-time ASL sign language recognition.
Handles MediaPipe hand detection, 170-feature extraction, ONNX model
inference, geometric disambiguation for confusable sign pairs,
invalid gesture rejection, and prediction smoothing.

This module's feature extraction is identical to the training code
in scripts/extract_landmarks.py, ensuring train–inference parity.
"""

import base64
import json
import time
import tempfile
import traceback
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
    """Get 3D point for landmark index from flattened normalised coords."""
    return coords[idx * 3: idx * 3 + 3]

def _dist(p1, p2):
    """Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))

def _angle(coords, i1, i2, i3):
    """Angle (radians) at point i2 formed by i1-i2-i3."""
    p1, p2, p3 = _pt(coords, i1), _pt(coords, i2), _pt(coords, i3)
    v1, v2 = p1 - p2, p3 - p2
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.arccos(np.clip(cos_a, -1.0, 1.0))

def _palm_center(coords):
    """Average position of wrist + all MCP joints."""
    indices = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    return np.array([_pt(coords, i) for i in indices]).mean(axis=0)

def _ext_ratio(coords, mcp, pip, dip, tip):
    """Finger extension ratio: direct tip–MCP distance / sum of bone segments."""
    direct = _dist(_pt(coords, tip), _pt(coords, mcp))
    bone = (_dist(_pt(coords, mcp), _pt(coords, pip)) +
            _dist(_pt(coords, pip), _pt(coords, dip)) +
            _dist(_pt(coords, dip), _pt(coords, tip)))
    return direct / (bone + 1e-8)


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

    @staticmethod
    def _normalize_coords(raw_landmarks):
        """Normalize landmarks: center on wrist, scale by max distance."""
        if raw_landmarks is None or len(raw_landmarks) != 63:
            return None
        coords = raw_landmarks.astype(np.float64).copy()
        wrist = coords[0:3].copy()
        for i in range(21):
            coords[i*3:i*3+3] -= wrist
        max_d = max(np.sqrt(np.sum(coords[i*3:i*3+3]**2)) for i in range(21))
        if max_d > 0:
            coords /= max_d
        return coords

    @staticmethod
    def _disambiguate(letter, raw_landmarks, proba, labels):
        """
        Multi-feature geometric disambiguation for commonly confused pairs.

        Uses weighted scoring across multiple hand geometry measurements
        rather than single-threshold checks, making it robust against
        noisy real-world webcam conditions (background clutter, hand
        movement, varying lighting).

        Covers:
          - O / 0 (circular vs wider circle)
          - O/0 vs Y (circular vs pinky+thumb extended)
          - O/0 vs J (circular vs pinky extended)
          - I vs J  (pinky up vs pinky J-hook)
          - X vs Z  (hooked index vs straight index)
        """
        if raw_landmarks is None or len(raw_landmarks) != 63:
            return letter

        coords = DetectionEngine._normalize_coords(raw_landmarks)
        if coords is None:
            return letter

        # Pre-compute commonly used values
        thumb_tip = _pt(coords, THUMB_TIP)
        index_tip = _pt(coords, INDEX_TIP)
        middle_tip = _pt(coords, MIDDLE_TIP)
        ring_tip = _pt(coords, RING_TIP)
        pinky_tip = _pt(coords, PINKY_TIP)
        palm_c = _palm_center(coords)

        # Extension ratios (reused across multiple rules)
        thumb_ext = _ext_ratio(coords, THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP)
        index_ext = _ext_ratio(coords, INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP)
        middle_ext = _ext_ratio(coords, MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP)
        ring_ext = _ext_ratio(coords, RING_MCP, RING_PIP, RING_DIP, RING_TIP)
        pinky_ext = _ext_ratio(coords, PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP)

        # ── O/0 vs Y disambiguation ──
        # 'Y' = thumb + pinky extended, middle/ring/index curled
        # 'O'/'0' = all fingers curled inward to form a circle
        # The key difference: Y has 2 clearly extended fingers (thumb+pinky)
        # while O/0 has 0 extended fingers (all curled toward thumb).
        if letter in ('Y', 'J') or (letter in ('0', 'O')):
            # Check if this is really an O/0 being misclassified as Y/J
            # O/0: all fingertips close together near palm, no finger extended
            # Y: pinky far from palm, thumb extended outward
            # J: pinky extended (similar to Y but without thumb extension)

            pinky_palm_dist = _dist(pinky_tip, palm_c)
            thumb_palm_dist = _dist(thumb_tip, palm_c)
            all_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
            avg_tip_palm = np.mean([_dist(t, palm_c) for t in all_tips])

            # Count clearly extended fingers (extension ratio > 0.70)
            ext_count = sum(1 for r in [thumb_ext, index_ext, middle_ext,
                                         ring_ext, pinky_ext] if r > 0.70)

            # Fingertip compactness: average pairwise distance among all tips
            pairwise = []
            for i in range(5):
                for j in range(i + 1, 5):
                    pairwise.append(_dist(all_tips[i], all_tips[j]))
            tip_compactness = np.mean(pairwise)

            if letter in ('Y', 'J'):
                # Model predicted Y or J — check if it's really O/0
                # O/0 signature: all fingers curled, tips clustered tight
                is_o_shape = (
                    ext_count <= 1 and         # At most 1 finger looks extended
                    pinky_palm_dist < 0.75 and  # Pinky is close to palm (curled)
                    tip_compactness < 0.50 and  # All fingertips clustered
                    avg_tip_palm < 0.65         # Tips generally near palm
                )
                if is_o_shape:
                    # Override to O — further disambiguate O vs 0 below
                    letter = 'O'

            # Now handle O vs 0 distinction (for original O/0 or overridden)
            if letter in ('0', 'O'):
                thumb_index_dist = _dist(thumb_tip, index_tip)
                outer_spread = _dist(ring_tip, palm_c) + _dist(pinky_tip, palm_c)
                avg_tip_spread = np.mean([_dist(all_tips[k], all_tips[k + 1])
                                          for k in range(4)])
                thumb_ring_dist = _dist(thumb_tip, ring_tip)
                thumb_pinky_dist = _dist(thumb_tip, pinky_tip)

                score_0 = 0.0
                score_0 += 2.0 if thumb_index_dist > 0.30 else -2.0
                score_0 += 1.5 if outer_spread > 1.20 else -1.5
                score_0 += 1.5 if avg_tip_spread > 0.30 else -1.5
                score_0 += 1.0 if (thumb_ring_dist + thumb_pinky_dist) > 0.55 else -1.0
                score_0 += 1.0 if (ring_ext + pinky_ext) > 1.30 else -1.0

                letter = '0' if score_0 > 0 else 'O'

        # ── I vs J disambiguation ──
        # Both have pinky extended, other fingers curled.
        # 'I' = pinky straight up, static pose
        # 'J' = pinky traces a J-hook (motion sign), in static capture the
        #       pinky tip tends to be lower/more lateral and the hand may
        #       be tilted (palm rotated outward).
        if letter in ('I', 'J'):
            # Pinky tip position relative to pinky MCP
            pinky_tip_y = pinky_tip[1] - _pt(coords, PINKY_MCP)[1]
            pinky_tip_x = abs(pinky_tip[0] - _pt(coords, PINKY_MCP)[0])

            # Palm orientation: compute palm normal
            v1 = _pt(coords, INDEX_MCP) - _pt(coords, WRIST)
            v2 = _pt(coords, PINKY_MCP) - _pt(coords, WRIST)
            normal = np.cross(v1, v2)
            normal = normal / (np.linalg.norm(normal) + 1e-8)
            palm_pitch = np.arcsin(np.clip(normal[2], -1, 1))

            # Pinky DIP angle (J-hook may have slight pinky curl)
            pinky_dip_angle = _angle(coords, PINKY_PIP, PINKY_DIP, PINKY_TIP)

            # Weighted scoring: positive = 'J', negative = 'I'
            score_j = 0.0
            # Pinky tip is lower relative to MCP → J (hooking downward)
            score_j += 2.0 if pinky_tip_y > -0.35 else -2.0
            # Pinky tip has lateral offset → J (sideways motion)
            score_j += 1.5 if pinky_tip_x > 0.12 else -1.5
            # Palm is tilted (larger pitch) → J (hand rotates during sign)
            score_j += 1.5 if abs(palm_pitch) > 0.4 else -1.5
            # Pinky DIP is slightly bent (hook shape) → J
            score_j += 1.0 if pinky_dip_angle < 2.7 else -1.0

            letter = 'J' if score_j > 0 else 'I'

        # ── X vs Z disambiguation ──
        # 'X' = index finger hooked/bent at DIP, fist closed
        # 'Z' = index finger extended straight, draws Z in air
        if letter in ('X', 'Z'):
            index_dip_angle = _angle(coords, INDEX_PIP, INDEX_DIP, INDEX_TIP)
            index_pip_angle = _angle(coords, INDEX_MCP, INDEX_PIP, INDEX_DIP)
            thumb_across = _pt(coords, THUMB_TIP)[0] - _pt(coords, INDEX_MCP)[0]
            index_tip_height = _pt(coords, INDEX_TIP)[1] - _pt(coords, INDEX_MCP)[1]

            score_x = 0.0
            score_x += 2.5 if index_dip_angle < 2.6 else -2.5
            score_x += 1.5 if index_pip_angle < 2.5 else -1.5
            score_x += 2.0 if index_ext < 0.85 else -2.0
            score_x += 1.0 if abs(thumb_across) > 0.15 else -1.0
            score_x += 1.0 if index_tip_height > -0.3 else -1.0

            letter = 'X' if score_x > 0 else 'Z'

        return letter

    @staticmethod
    def _is_valid_sign_pose(proba):
        """
        Reject random/non-ASL hand gestures using prediction quality checks.

        Returns True if the prediction looks like a genuine ASL sign,
        False if it looks like a random hand position.

        Uses two heuristics:
          1. Entropy: if the probability is spread across many classes
             (high entropy), the model is uncertain → likely not a real sign.
          2. Confidence gap: if top-1 and top-2 are very close, the model
             can't decide → likely not a clear sign.
        """
        top2_idx = np.argsort(proba)[-2:][::-1]
        top1_conf = float(proba[top2_idx[0]])
        top2_conf = float(proba[top2_idx[1]])

        # Reject if top-1 confidence is too low (< 35% raw)
        if top1_conf < 0.35:
            return False

        # Reject if confidence gap between #1 and #2 is too narrow
        # (model can't distinguish → likely nonsense gesture)
        gap = top1_conf - top2_conf
        if gap < 0.05 and top1_conf < 0.50:
            return False

        # Entropy check: high entropy = spread probability = uncertain
        # For 36 classes, max entropy ≈ 3.58. A confident sign should
        # have entropy well below 2.0.
        clipped = np.clip(proba, 1e-10, 1.0)
        entropy = -np.sum(clipped * np.log(clipped))
        if entropy > 2.5:
            return False

        return True

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

        # MediaPipe — tuned for real-world webcam conditions
        # Lower detection confidence to catch hands in noisy backgrounds;
        # higher tracking confidence to maintain stability once detected.
        self.mp_hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )

        # Smoother — 7-frame window for more stable predictions in noisy
        # real-world conditions (vs 5 for clean datasets)
        self.smoother = PredictionSmoother(window_size=7, n_classes=len(self.labels))

        # Sentence state
        self.sentence = []
        self.current_letter = None
        self.letter_start_time = None
        self.last_locked_time = 0
        self.last_hand_time = time.time()
        self.last_coords = None
        self._motion_ema = 0.0  # Exponential moving average of hand motion

        # Timing constants — tuned for real-world use
        self.HOLD_TIME = 1.2       # seconds to hold for lock-in (slightly faster)
        self.COOLDOWN_TIME = 0.4   # seconds after lock before next
        self.CONFIDENCE_THRESHOLD = 72.0  # lowered: real-world noise reduces conf
        self.MOTION_THRESHOLD = 0.035     # EMA motion threshold for stability
        self.MOTION_EMA_ALPHA = 0.4       # smoothing factor for motion EMA

        print(f"[DetectionEngine] Loaded: {self.n_features} features, "
              f"{len(self.labels)} classes")

    def process_frame(self, image_data, state=None):
        """
        Process a single frame (base64 or numpy array).
        Supports optional per-client state dict to avoid multi-user conflicts.
        Returns dict with prediction info.
        """
        now = time.time()
        is_dict = isinstance(state, dict)

        # Decode image
        if isinstance(image_data, str):
            img_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(image_data, np.ndarray):
            frame = image_data
        else:
            sentence = state["sentence"] if is_dict else self.sentence
            return {"hand_detected": False, "sentence": "".join(sentence)}

        if frame is None:
            sentence = state["sentence"] if is_dict else self.sentence
            return {"hand_detected": False, "sentence": "".join(sentence)}

        # Detect hand
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.mp_hands.process(rgb)

        if not result.multi_hand_landmarks:
            # No hand — reset tracking (no auto-space)
            if is_dict:
                state["current_letter"] = None
                state["letter_start_time"] = None
                state["last_coords"] = None
                sentence = state["sentence"]
            else:
                self.current_letter = None
                self.letter_start_time = None
                self.last_coords = None
                sentence = self.sentence
            return {
                "hand_detected": False,
                "sentence": "".join(sentence),
            }

        if is_dict:
            state["last_hand_time"] = now
        else:
            self.last_hand_time = now

        hand = result.multi_hand_landmarks[0]
        raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()

        # Track hand movement with exponential moving average (EMA)
        # for stable transition detection. Single-frame shift is too noisy
        # in real-world webcam conditions with background clutter.
        is_transitioning = False
        last_coords = state["last_coords"] if is_dict else self.last_coords
        motion_ema = state["motion_ema"] if is_dict else self._motion_ema

        if last_coords is not None:
            curr_pts = raw.reshape(21, 3)
            prev_pts = last_coords.reshape(21, 3)
            shift = np.mean(np.sqrt(np.sum((curr_pts - prev_pts) ** 2, axis=1)))
            # EMA smoothing: reduces false positives from webcam jitter/noise
            motion_ema = (self.MOTION_EMA_ALPHA * shift +
                          (1 - self.MOTION_EMA_ALPHA) * motion_ema)
            if motion_ema > self.MOTION_THRESHOLD:
                is_transitioning = True

        if is_dict:
            state["motion_ema"] = motion_ema
            state["last_coords"] = raw.copy()
        else:
            self._motion_ema = motion_ema
            self.last_coords = raw.copy()

        # Landmark positions for client-side drawing (normalized x,y)
        landmarks = [{"x": round(float(lm.x), 4), "y": round(float(lm.y), 4)}
                     for lm in hand.landmark]

        # Extract features
        feat = extract_features(raw)
        sentence_list = state["sentence"] if is_dict else self.sentence
        if feat is None or len(feat) != self.n_features:
            return {"hand_detected": True, "sentence": "".join(sentence_list)}

        # Scale + predict
        scaled = ((feat - self.scaler_mean) / self.scaler_scale).astype(np.float32).reshape(1, -1)
        results = self.session.run(None, {self.input_name: scaled})
        proba = results[1][0].astype(np.float32)

        # Smooth
        smoother = state["smoother"] if is_dict else self.smoother
        smoother.add(proba)
        smooth_idx, smooth_conf = smoother.get_smoothed()

        # Raw prediction
        raw_idx = int(np.argmax(proba))
        raw_conf = float(proba[raw_idx])
        raw_letter = self.labels[raw_idx].upper()

        # Top 3
        top3_idx = np.argsort(proba)[-3:][::-1]
        top3 = [{"letter": self.labels[i].upper(), "confidence": round(float(proba[i]) * 100, 1)}
                for i in top3_idx]

        # ── Invalid gesture rejection ──
        # Reject random hand positions that don't match any ASL sign well.
        # This prevents the UI from showing predictions on arbitrary gestures.
        if not self._is_valid_sign_pose(proba):
            sentence_list = state["sentence"] if is_dict else self.sentence
            return {
                "hand_detected": True,
                "letter": "",
                "confidence": 0.0,
                "smoothed_letter": "",
                "smoothed_confidence": 0.0,
                "top3": top3,
                "hold_progress": 0.0,
                "locked": False,
                "sentence": "".join(sentence_list),
                "landmarks": landmarks,
                "connections": HAND_CONNECTIONS,
            }

        # Confidence gate: ignore low-confidence predictions
        smooth_conf_pct = smooth_conf * 100
        # Lower threshold for motion signs ('J', 'Z') and circular signs ('O', '0')
        # to handle motion blur and right-hand variations.
        is_low_threshold_sign = (smooth_idx is not None and 
                                 self.labels[smooth_idx].upper() in ('O', '0', 'J', 'Z'))
        temp_threshold = 60.0 if is_low_threshold_sign else self.CONFIDENCE_THRESHOLD
        letter = self.labels[smooth_idx].upper() if (smooth_idx is not None and smooth_conf_pct >= temp_threshold) else ""

        # Apply disambiguation for confused pairs.
        # Also check if top-2 contains a confusable pair — if so,
        # run disambiguation even if the smoothed letter isn't one of them.
        if letter:
            letter = self._disambiguate(letter, raw, proba, np.array(self.labels))
        elif smooth_idx is not None and smooth_conf_pct >= (temp_threshold * 0.85):
            # Near-threshold: if top-2 are confusable pairs, still attempt
            # disambiguation — geometric features are more reliable than
            # the model's softmax in these edge cases
            tentative = self.labels[smooth_idx].upper()
            top2_letters = {self.labels[i].upper() for i in top3_idx[:2]}
            confusable_pairs = [{'0', 'O'}, {'X', 'Z'}, {'I', 'J'},
                                {'O', 'Y'}, {'0', 'Y'}, {'O', 'J'}, {'0', 'J'}]
            for pair in confusable_pairs:
                if top2_letters & pair and tentative in pair:
                    letter = self._disambiguate(tentative, raw, proba, np.array(self.labels))
                    break

        # Reset hold/prediction if hand is in active transition (moving)
        current_letter = state["current_letter"] if is_dict else self.current_letter
        letter_start_time = state["letter_start_time"] if is_dict else self.letter_start_time

        if is_transitioning:
            # Motion signs ('J', 'Z') require motion to be signed, so we bypass
            # the transition reset for them.
            if letter not in ('J', 'Z'):
                letter = ""
                current_letter = None
                letter_start_time = None
                if is_dict:
                    state["current_letter"] = None
                    state["letter_start_time"] = None
                else:
                    self.current_letter = None
                    self.letter_start_time = None

        # Hold-to-lock (only if confidence passes threshold)
        hold_progress = 0.0
        locked = False
        last_locked_time = state["last_locked_time"] if is_dict else self.last_locked_time
        in_cooldown = (now - last_locked_time) < self.COOLDOWN_TIME

        if letter and not in_cooldown:
            if letter == current_letter and letter_start_time:
                elapsed = now - letter_start_time
                # Motion signs require a shorter hold time since they are dynamic
                required_hold = 0.4 if letter in ('J', 'Z') else self.HOLD_TIME
                hold_progress = min(elapsed / required_hold, 1.5)
                if elapsed >= required_hold:
                    sentence_list.append(letter)
                    if is_dict:
                        state["last_locked_time"] = now
                        state["letter_start_time"] = None
                        state["current_letter"] = None
                    else:
                        self.last_locked_time = now
                        self.letter_start_time = None
                        self.current_letter = None
                    locked = True
            else:
                if is_dict:
                    state["current_letter"] = letter
                    state["letter_start_time"] = now
                else:
                    self.current_letter = letter
                    self.letter_start_time = now
        elif not letter:
            # Below threshold — reset hold tracking
            if is_dict:
                state["current_letter"] = None
                state["letter_start_time"] = None
            else:
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
            "sentence": "".join(sentence_list),
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

                        # Reject invalid/random gestures
                        if not self._is_valid_sign_pose(proba):
                            prev_letter = None
                            stability_count = 0
                            frame_idx += 1
                            continue

                        # Apply disambiguation for confused pairs
                        letter = self._disambiguate(letter, raw, proba, np.array(self.labels))

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
        try:
            mp_static.close()
        except Exception:
            pass  # Ignore close errors on some mediapipe versions

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
        self.last_coords = None

    def create_client_state(self):
        """Create a new state dictionary for a single client."""
        return {
            "sentence": [],
            "current_letter": None,
            "letter_start_time": None,
            "last_locked_time": 0.0,
            "last_hand_time": time.time(),
            "last_coords": None,
            "motion_ema": 0.0,
            "smoother": PredictionSmoother(window_size=7, n_classes=len(self.labels)),
        }

    def reset_client_state(self, state):
        """Reset the sentence and tracking state for a client."""
        state["sentence"].clear()
        state["current_letter"] = None
        state["letter_start_time"] = None
        state["last_locked_time"] = 0.0
        state["last_coords"] = None
        state["motion_ema"] = 0.0
        state["smoother"].reset()

    def backspace(self):
        if self.sentence:
            self.sentence.pop()

    def add_space(self):
        if not self.sentence or self.sentence[-1] != " ":
            self.sentence.append(" ")

    def close(self):
        self.mp_hands.close()
