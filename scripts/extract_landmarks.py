"""
Extract hand landmarks from ASL dataset images and compute enhanced feature vectors.

Processes all images in asl_dataset/ through MediaPipe Hands, extracts
21 hand landmarks, computes the enhanced 170-feature vector (original 150 +
20 new hand-state features), and saves numpy arrays for model training.

Data augmentation is applied (rotation, translation, noise) for robustness.

Usage:
    python scripts/extract_landmarks.py
"""

import os
import sys
import numpy as np
import cv2
import mediapipe as mp
from pathlib import Path
import time

# ---- Project paths ----
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "asl_dataset"
MODELS_DIR = PROJECT_DIR / "models"

# ---- MediaPipe landmark indices ----
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


# ===========================================================================
# Helper functions
# ===========================================================================

def get_point(coords, idx):
    """Get 3D point for landmark index from flattened coords."""
    return coords[idx * 3: idx * 3 + 3]


def euclidean_dist(p1, p2):
    """Euclidean distance between two 3D points."""
    return np.sqrt(np.sum((p1 - p2) ** 2))


def angle_between(coords, i1, i2, i3):
    """Angle at point i2 formed by i1-i2-i3, in radians."""
    p1, p2, p3 = get_point(coords, i1), get_point(coords, i2), get_point(coords, i3)
    v1 = p1 - p2
    v2 = p3 - p2
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)


def palm_center(coords):
    """Average of wrist + all MCPs."""
    indices = [WRIST, THUMB_CMC, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP]
    points = np.array([get_point(coords, i) for i in indices])
    return points.mean(axis=0)


# ===========================================================================
# Feature extraction — Enhanced 170-feature vector
# ===========================================================================

def extract_features(raw_landmarks):
    """
    Extract enhanced 170-feature vector from raw 21-landmark coordinates.
    
    Features 0-149:   Original 150 features (same as before)
    Features 150-169: 20 new hand-state features
    
    Args:
        raw_landmarks: array of shape (63,) — 21 landmarks × 3 coords each
    
    Returns:
        numpy array of 170 features, or None if input is invalid
    """
    if raw_landmarks is None or len(raw_landmarks) != 63:
        return None
    
    coords = raw_landmarks.astype(np.float64).copy()
    
    # ---- 1. Normalize: center on wrist ----
    wrist = coords[0:3].copy()
    for i in range(21):
        coords[i*3:i*3+3] -= wrist
    
    # ---- 2. Scale by max distance from wrist ----
    max_dist = 0
    for i in range(21):
        d = np.sqrt(np.sum(coords[i*3:i*3+3] ** 2))
        max_dist = max(max_dist, d)
    if max_dist > 0:
        coords /= max_dist
    
    features = []
    
    # ---- 3. All 63 normalized coordinates ----
    features.extend(coords.tolist())
    
    # ---- 4a. Key pairwise distances (20) ----
    for i1, i2 in KEY_DISTANCE_PAIRS:
        features.append(euclidean_dist(get_point(coords, i1), get_point(coords, i2)))
    
    # ---- 4b. Finger joint angles (15 = 3 per finger) ----
    for mcp, pip, dip, tip in FINGERS:
        features.append(angle_between(coords, WRIST, mcp, pip))
        features.append(angle_between(coords, mcp, pip, dip))
        features.append(angle_between(coords, pip, dip, tip))
    
    # ---- 4c. Fingertip-to-palm distances (5) ----
    pc = palm_center(coords)
    for tip_idx in FINGERTIPS:
        features.append(euclidean_dist(get_point(coords, tip_idx), pc))
    
    # ---- 4d. Finger extension ratios (5) ----
    ext_ratios = []
    for mcp, pip, dip, tip in FINGERS:
        p_mcp = get_point(coords, mcp)
        p_pip = get_point(coords, pip)
        p_dip = get_point(coords, dip)
        p_tip = get_point(coords, tip)
        direct = euclidean_dist(p_tip, p_mcp)
        bone_len = (euclidean_dist(p_mcp, p_pip) + 
                    euclidean_dist(p_pip, p_dip) + 
                    euclidean_dist(p_dip, p_tip))
        ratio = direct / (bone_len + 1e-8)
        ext_ratios.append(ratio)
        features.append(ratio)
    
    # ---- 4e. Bounding box aspect ratio (1) ----
    xs = [coords[i*3] for i in range(21)]
    ys = [coords[i*3+1] for i in range(21)]
    w_bbox = max(xs) - min(xs)
    h_bbox = max(ys) - min(ys)
    features.append(w_bbox / (h_bbox + 1e-8))
    
    # ---- 4f. Pairwise fingertip distances (10) ----
    for i in range(len(FINGERTIPS)):
        for j in range(i+1, len(FINGERTIPS)):
            features.append(euclidean_dist(
                get_point(coords, FINGERTIPS[i]),
                get_point(coords, FINGERTIPS[j])))
    
    # ---- 4g. Thumb cross-over features (4) ----
    thumb_tip = get_point(coords, THUMB_TIP)
    for k in range(1, len(FINGER_MCPS)):
        mcp_pt = get_point(coords, FINGER_MCPS[k])
        features.append(thumb_tip[0] - mcp_pt[0])  # Signed X-distance
    
    # ---- 4h. Finger curl indicators (5) ----
    wrist_pt = get_point(coords, WRIST)
    for tip_idx in FINGERTIPS:
        features.append(euclidean_dist(get_point(coords, tip_idx), wrist_pt))
    
    # ---- 4i. MCP splay angles (4) ----
    adjacent_mcps = [
        (THUMB_CMC, WRIST, INDEX_MCP),
        (INDEX_MCP, WRIST, MIDDLE_MCP),
        (MIDDLE_MCP, WRIST, RING_MCP),
        (RING_MCP, WRIST, PINKY_MCP),
    ]
    for a, b, c in adjacent_mcps:
        features.append(angle_between(coords, a, b, c))
    
    # ---- 4j. TIP splay angles (4) ----
    adjacent_tips = [
        (THUMB_TIP, WRIST, INDEX_TIP),
        (INDEX_TIP, WRIST, MIDDLE_TIP),
        (MIDDLE_TIP, WRIST, RING_TIP),
        (RING_TIP, WRIST, PINKY_TIP),
    ]
    for a, b, c in adjacent_tips:
        features.append(angle_between(coords, a, b, c))
    
    # ---- 4k. Thumb position features (2) ----
    thumb_tip_pt = get_point(coords, THUMB_TIP)
    index_mcp_pt = get_point(coords, INDEX_MCP)
    middle_mcp_pt = get_point(coords, MIDDLE_MCP)
    features.append(thumb_tip_pt[1] - index_mcp_pt[1])   # Y relative
    features.append(thumb_tip_pt[2] - middle_mcp_pt[2])   # Z relative
    
    # ---- 4l. Enhanced finger spread features (12) ----
    tip_indices = [THUMB_TIP, INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]
    
    # Adjacent fingertip X-axis separation (4)
    for k in range(len(tip_indices) - 1):
        p1 = get_point(coords, tip_indices[k])
        p2 = get_point(coords, tip_indices[k + 1])
        features.append(abs(p1[0] - p2[0]))
    
    # Adjacent DIP-level separation (4)
    dip_indices = [THUMB_IP, INDEX_DIP, MIDDLE_DIP, RING_DIP, PINKY_DIP]
    for k in range(len(dip_indices) - 1):
        p1 = get_point(coords, dip_indices[k])
        p2 = get_point(coords, dip_indices[k + 1])
        features.append(euclidean_dist(p1, p2))
    
    # Total finger spread score (1)
    total_spread = 0
    for k in range(len(tip_indices) - 1):
        p1 = get_point(coords, tip_indices[k])
        p2 = get_point(coords, tip_indices[k + 1])
        total_spread += euclidean_dist(p1, p2)
    features.append(total_spread)
    
    # Fingertip X-extent (1)
    tip_xs = [get_point(coords, idx)[0] for idx in tip_indices]
    features.append(max(tip_xs) - min(tip_xs))
    
    # Thumb abduction angle (1)
    features.append(angle_between(coords, THUMB_TIP, INDEX_MCP, INDEX_TIP))
    
    # Thumb-to-index tip Y separation (1)
    features.append(abs(
        get_point(coords, THUMB_TIP)[1] - get_point(coords, INDEX_TIP)[1]))
    
    # ================================================================
    # NEW FEATURES (20 new hand-state features, indices 150-169)
    # ================================================================
    
    # ---- N1. Finger curl scores (5) ----
    # Curl = how bent each finger is. Uses PIP and DIP angles.
    # Straight finger: PIP+DIP angles ~ π each → curl ~ 0
    # Curled finger: PIP+DIP angles ~ 0 → curl ~ 1
    for mcp, pip, dip, tip in FINGERS:
        pip_angle = angle_between(coords, mcp, pip, dip)
        dip_angle = angle_between(coords, pip, dip, tip)
        avg_angle = (pip_angle + dip_angle) / 2.0
        curl_score = 1.0 - (avg_angle / np.pi)
        features.append(curl_score)
    
    # ---- N2. Binary finger extended flags (5) ----
    # Is each finger extended? Based on extension ratio > 0.7
    EXTEND_THRESHOLD = 0.7
    extended_flags = []
    for ratio in ext_ratios:
        flag = 1.0 if ratio > EXTEND_THRESHOLD else 0.0
        extended_flags.append(flag)
        features.append(flag)
    
    # ---- N3. Thumb across palm (1) ----
    # Positive = thumb tip is past index MCP towards other fingers (folded)
    # For normalized coords, this depends on hand orientation
    thumb_tip_x = get_point(coords, THUMB_TIP)[0]
    index_mcp_x = get_point(coords, INDEX_MCP)[0]
    features.append(thumb_tip_x - index_mcp_x)
    
    # ---- N4. Adjacent finger attachment scores (3) ----
    # Distance between adjacent extended fingertip pairs (index-middle, middle-ring, ring-pinky)
    # Low = fingers together (B), high = fingers spread (5)
    adj_pairs = [(INDEX_TIP, MIDDLE_TIP), (MIDDLE_TIP, RING_TIP), (RING_TIP, PINKY_TIP)]
    for i1, i2 in adj_pairs:
        features.append(euclidean_dist(get_point(coords, i1), get_point(coords, i2)))
    
    # ---- N5. Number of extended fingers (1) ----
    num_extended = sum(extended_flags)
    features.append(num_extended)
    
    # ---- N6. Fingers-together score (1) ----
    # Average distance between adjacent extended fingertips (only index-pinky, skip thumb)
    adj_tip_dists = []
    for k in range(1, len(tip_indices) - 1):
        d = euclidean_dist(get_point(coords, tip_indices[k]), 
                           get_point(coords, tip_indices[k + 1]))
        adj_tip_dists.append(d)
    features.append(np.mean(adj_tip_dists) if adj_tip_dists else 0.0)
    
    # ---- N7. Palm orientation (2) — pitch and yaw ----
    # Compute palm normal from wrist, index_mcp, pinky_mcp
    p_wrist = get_point(coords, WRIST)
    p_index_mcp = get_point(coords, INDEX_MCP)
    p_pinky_mcp = get_point(coords, PINKY_MCP)
    v1 = p_index_mcp - p_wrist
    v2 = p_pinky_mcp - p_wrist
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal) + 1e-8
    normal = normal / norm_len
    # Pitch (angle from XY plane) and Yaw (angle in XY plane)
    pitch = np.arcsin(np.clip(normal[2], -1, 1))  # Z component
    yaw = np.arctan2(normal[1], normal[0] + 1e-8)  # Y/X
    features.append(pitch)
    features.append(yaw)
    
    # ---- N8. Thumb-to-finger contact distances (2) ----
    # Distance from thumb tip to index PIP and middle PIP
    # Small = thumb tucked under fingers (M, N, T signs)
    features.append(euclidean_dist(get_point(coords, THUMB_TIP), 
                                    get_point(coords, INDEX_PIP)))
    features.append(euclidean_dist(get_point(coords, THUMB_TIP), 
                                    get_point(coords, MIDDLE_PIP)))
    
    return np.array(features, dtype=np.float64)


# ===========================================================================
# Data augmentation (applied to raw landmarks before feature extraction)
# ===========================================================================

def augment_landmarks(raw_landmarks, num_augments=4):
    """
    Generate augmented versions of raw landmarks.
    
    Applies rotation, translation jitter, and Gaussian noise.
    Returns a list of augmented landmark arrays (including the original).
    """
    augmented = [raw_landmarks.copy()]
    
    for _ in range(num_augments):
        lm = raw_landmarks.copy().reshape(21, 3)
        
        # Random 2D rotation (±12 degrees)
        angle = np.random.uniform(-12, 12) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(21):
            x, y = lm[i, 0], lm[i, 1]
            lm[i, 0] = cos_a * x - sin_a * y
            lm[i, 1] = sin_a * x + cos_a * y
        
        # Translation jitter
        tx = np.random.uniform(-0.02, 0.02)
        ty = np.random.uniform(-0.02, 0.02)
        lm[:, 0] += tx
        lm[:, 1] += ty
        
        # Per-landmark Gaussian noise (small)
        noise = np.random.normal(0, 0.005, lm.shape)
        lm += noise
        
        augmented.append(lm.flatten())
    
    return augmented


# ===========================================================================
# Main extraction
# ===========================================================================

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.3,
    )
    
    all_features = []
    all_labels = []
    
    classes = sorted(os.listdir(DATASET_DIR))
    classes = [c for c in classes if os.path.isdir(DATASET_DIR / c)]
    
    print(f"Dataset directory: {DATASET_DIR}")
    print(f"Found {len(classes)} classes: {classes}")
    print()
    
    total_images = 0
    total_detected = 0
    start_time = time.time()
    
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = DATASET_DIR / cls_name
        images = [f for f in os.listdir(cls_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        detected = 0
        for img_name in images:
            img_path = str(cls_dir / img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue
            
            total_images += 1
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = mp_hands.process(rgb)
            
            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                raw = np.array([
                    [lm.x, lm.y, lm.z] 
                    for lm in landmarks.landmark
                ]).flatten()
                
                # Augment and extract features
                augmented = augment_landmarks(raw, num_augments=4)
                for aug_lm in augmented:
                    feat = extract_features(aug_lm)
                    if feat is not None and len(feat) > 0:
                        all_features.append(feat)
                        all_labels.append(cls_name)
                
                detected += 1
                total_detected += 1
        
        print(f"  [{cls_idx+1:2d}/{len(classes)}] Class '{cls_name}': "
              f"{detected}/{len(images)} hands detected "
              f"-> {detected * 5} samples (with augmentation)")
    
    elapsed = time.time() - start_time
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    print(f"\n{'='*50}")
    print(f"Extraction complete in {elapsed:.1f}s")
    print(f"  Images processed: {total_images}")
    print(f"  Hands detected: {total_detected} ({100*total_detected/max(total_images,1):.1f}%)")
    print(f"  Feature vector size: {X.shape[1]}")
    print(f"  Total samples (with augmentation): {X.shape[0]}")
    print(f"  Labels: {len(np.unique(y))} classes")
    
    np.save(MODELS_DIR / "landmarks_data.npy", X)
    np.save(MODELS_DIR / "landmarks_labels.npy", y)
    print(f"\nSaved to:")
    print(f"  {MODELS_DIR / 'landmarks_data.npy'}")
    print(f"  {MODELS_DIR / 'landmarks_labels.npy'}")
    
    mp_hands.close()


if __name__ == "__main__":
    main()
