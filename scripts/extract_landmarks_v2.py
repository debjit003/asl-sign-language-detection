"""
Enhanced landmark extraction with aggressive detection and balanced augmentation.

Key improvements over original:
1. Multiple MediaPipe passes with different configs to maximize hand detection
2. Image preprocessing (contrast/brightness) to help MediaPipe on hard signs
3. Dynamic augmentation - more augments for classes with fewer detections
4. Target: minimum 200 samples per class after augmentation
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

# Import extract_features from existing script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_landmarks import extract_features


def try_detect_hand(image, mp_hands_list):
    """Try detecting hand with multiple MediaPipe configs."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    for mp_hands in mp_hands_list:
        result = mp_hands.process(rgb)
        if result.multi_hand_landmarks:
            return result.multi_hand_landmarks[0]
    
    return None


def preprocess_variants(image):
    """Generate preprocessed versions to help MediaPipe detect tricky hand poses."""
    variants = [image]
    
    # Brighter
    bright = cv2.convertScaleAbs(image, alpha=1.3, beta=30)
    variants.append(bright)
    
    # Higher contrast
    contrast = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    variants.append(contrast)
    
    # CLAHE (adaptive histogram equalization)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    variants.append(enhanced)
    
    # Slightly resized (sometimes helps)
    h, w = image.shape[:2]
    padded = cv2.copyMakeBorder(image, 40, 40, 40, 40,
                                 cv2.BORDER_CONSTANT, value=(200, 200, 200))
    variants.append(padded)
    
    return variants


def augment_landmarks(raw_landmarks, num_augments=4):
    """Generate augmented landmark arrays with rotation, translation, noise, and scale."""
    augmented = [raw_landmarks.copy()]
    
    for _ in range(num_augments):
        lm = raw_landmarks.copy().reshape(21, 3)
        
        # Random 2D rotation (+/- 15 degrees)
        angle = np.random.uniform(-15, 15) * np.pi / 180
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        for i in range(21):
            x, y = lm[i, 0], lm[i, 1]
            lm[i, 0] = cos_a * x - sin_a * y
            lm[i, 1] = sin_a * x + cos_a * y
        
        # Translation jitter
        tx = np.random.uniform(-0.03, 0.03)
        ty = np.random.uniform(-0.03, 0.03)
        lm[:, 0] += tx
        lm[:, 1] += ty
        
        # Scale jitter (+/- 5%)
        scale = np.random.uniform(0.95, 1.05)
        center = lm.mean(axis=0)
        lm = center + (lm - center) * scale
        
        # Per-landmark Gaussian noise
        noise = np.random.normal(0, 0.006, lm.shape)
        lm += noise
        
        augmented.append(lm.flatten())
    
    return augmented


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Multiple MediaPipe configs: standard -> aggressive
    mp_configs = [
        mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1,
            min_detection_confidence=0.3),
        mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1,
            min_detection_confidence=0.1),  # Very aggressive
        mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=2,
            min_detection_confidence=0.1),  # Allow 2 hands
    ]
    
    classes = sorted(os.listdir(DATASET_DIR))
    classes = [c for c in classes if os.path.isdir(DATASET_DIR / c)]
    
    print(f"Dataset: {DATASET_DIR}")
    print(f"Classes: {len(classes)}")
    print()
    
    # ---- Phase 1: Extract all raw landmarks ----
    print("=" * 60)
    print("PHASE 1: Extract landmarks (aggressive detection)")
    print("=" * 60)
    
    class_raw_landmarks = {}  # class -> list of raw landmark arrays
    
    start_time = time.time()
    
    for cls_idx, cls_name in enumerate(classes):
        cls_dir = DATASET_DIR / cls_name
        images = [f for f in os.listdir(cls_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        raw_landmarks = []
        
        for img_name in images:
            img = cv2.imread(str(cls_dir / img_name))
            if img is None:
                continue
            
            # Try original image first
            hand = try_detect_hand(img, mp_configs)
            
            # If not detected, try preprocessed variants
            if hand is None:
                for variant in preprocess_variants(img)[1:]:  # Skip original
                    hand = try_detect_hand(variant, mp_configs)
                    if hand:
                        break
            
            # If still not detected, try flipped
            if hand is None:
                flipped = cv2.flip(img, 1)
                hand = try_detect_hand(flipped, mp_configs)
            
            if hand:
                raw = np.array([[lm.x, lm.y, lm.z]
                                for lm in hand.landmark]).flatten()
                raw_landmarks.append(raw)
        
        class_raw_landmarks[cls_name] = raw_landmarks
        
        detection_rate = len(raw_landmarks) / max(len(images), 1) * 100
        status = "LOW" if len(raw_landmarks) < 30 else "OK"
        print(f"  [{cls_idx+1:2d}/{len(classes)}] '{cls_name}': "
              f"{len(raw_landmarks)}/{len(images)} detected "
              f"({detection_rate:.0f}%) [{status}]")
    
    # ---- Phase 2: Balance with dynamic augmentation ----
    print(f"\n{'=' * 60}")
    print("PHASE 2: Dynamic augmentation (balancing)")
    print("=" * 60)
    
    TARGET_MIN_SAMPLES = 300  # Minimum samples per class after augmentation
    
    all_features = []
    all_labels = []
    
    for cls_name in classes:
        raw_list = class_raw_landmarks[cls_name]
        n_detected = len(raw_list)
        
        if n_detected == 0:
            print(f"  '{cls_name}': SKIPPED (no detections)")
            continue
        
        # Calculate augmentation factor to reach target
        # Each raw landmark produces (1 + num_augments) samples
        desired_augments = max(4, (TARGET_MIN_SAMPLES // n_detected) - 1)
        desired_augments = min(desired_augments, 30)  # Cap at 30x
        
        cls_features = []
        for raw in raw_list:
            augmented = augment_landmarks(raw, num_augments=desired_augments)
            for aug_lm in augmented:
                feat = extract_features(aug_lm)
                if feat is not None and len(feat) > 0:
                    cls_features.append(feat)
                    all_labels.append(cls_name)
        
        all_features.extend(cls_features)
        
        print(f"  '{cls_name}': {n_detected} detections x "
              f"{desired_augments+1} augments = {len(cls_features)} samples")
    
    X = np.array(all_features)
    y = np.array(all_labels)
    
    elapsed = time.time() - start_time
    
    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"EXTRACTION COMPLETE ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"  Feature vector: {X.shape[1]} features")
    print(f"  Total samples: {X.shape[0]}")
    print(f"  Classes: {len(np.unique(y))}")
    
    # Per-class counts
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n  Per-class sample counts:")
    for cls, cnt in zip(unique, counts):
        print(f"    '{cls}': {cnt}")
    
    print(f"\n  Min: {counts.min()} | Max: {counts.max()} | "
          f"Mean: {counts.mean():.0f}")
    
    np.save(MODELS_DIR / "landmarks_data.npy", X)
    np.save(MODELS_DIR / "landmarks_labels.npy", y)
    print(f"\n  Saved to {MODELS_DIR}")
    
    # Close MediaPipe
    for mp_h in mp_configs:
        mp_h.close()


if __name__ == "__main__":
    main()
