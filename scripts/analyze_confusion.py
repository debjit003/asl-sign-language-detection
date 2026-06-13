"""
Analyze model confusion on the dataset — focus on T, X, M/N issues.
Runs the ONNX model on each class and reports what it predicts.
"""

import os, sys, json
import numpy as np
import cv2
import mediapipe as mp
import onnxruntime as ort
from pathlib import Path
from collections import Counter, defaultdict

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_DIR / "asl_dataset"
MODELS_DIR = PROJECT_DIR / "webapp" / "models"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_landmarks import extract_features

def main():
    # Load model
    session = ort.InferenceSession(str(MODELS_DIR / "sign_classifier.onnx"))
    input_name = session.get_inputs()[0].name
    
    with open(MODELS_DIR / "scaler_params.json") as f:
        sd = json.load(f)
    mean = np.array(sd["mean"])
    scale = np.array(sd["scale"])
    
    with open(MODELS_DIR / "label_classes.json") as f:
        labels = json.load(f)["classes"]
    
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)
    
    # Focus on problematic classes
    focus_classes = ['t', 'x', 'm', 'n', 's', 'a', 'e']
    
    print("=" * 70)
    print("CONFUSION ANALYSIS — Problematic Classes")
    print("=" * 70)
    
    confusion_matrix = {}
    
    for cls_name in sorted(os.listdir(DATASET_DIR)):
        cls_dir = DATASET_DIR / cls_name
        if not cls_dir.is_dir():
            continue
        
        images = [f for f in os.listdir(cls_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        predictions = Counter()
        confidences = defaultdict(list)
        detected = 0
        
        for img_name in images[:50]:  # Test up to 50 per class
            img = cv2.imread(str(cls_dir / img_name))
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            result = mp_hands.process(rgb)
            
            if result.multi_hand_landmarks:
                hand = result.multi_hand_landmarks[0]
                raw = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
                feat = extract_features(raw)
                
                if feat is not None and len(feat) == len(mean):
                    scaled = ((feat - mean) / scale).astype(np.float32).reshape(1, -1)
                    res = session.run(None, {input_name: scaled})
                    proba = res[1][0]
                    pred_idx = int(np.argmax(proba))
                    pred_label = labels[pred_idx]
                    conf = float(proba[pred_idx]) * 100
                    
                    predictions[pred_label] += 1
                    confidences[pred_label].append(conf)
                    detected += 1
        
        if detected == 0:
            print(f"\n  Class '{cls_name}': NO HANDS DETECTED in any image!")
            continue
        
        correct = predictions.get(cls_name, 0)
        accuracy = correct / detected * 100
        
        # Show all classes, but focus details on problematic ones
        is_focus = cls_name in focus_classes or accuracy < 90
        
        if is_focus:
            print(f"\n  Class '{cls_name}': {detected} detected, "
                  f"accuracy={accuracy:.1f}%")
            for pred, count in predictions.most_common(5):
                avg_conf = np.mean(confidences[pred])
                marker = " [OK]" if pred == cls_name else " [WRONG]"
                print(f"    -> Predicted '{pred}': {count}/{detected} "
                      f"({count/detected*100:.0f}%) avg_conf={avg_conf:.1f}%{marker}")
        else:
            print(f"  Class '{cls_name}': {accuracy:.0f}% correct ({detected} samples)")
        
        confusion_matrix[cls_name] = {
            "total": detected,
            "correct": correct,
            "accuracy": accuracy,
            "predictions": dict(predictions),
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY — Problem Classes (accuracy < 95%)")
    print("=" * 70)
    for cls, data in sorted(confusion_matrix.items(), key=lambda x: x[1]["accuracy"]):
        if data["accuracy"] < 95:
            preds = data["predictions"]
            wrong = {k: v for k, v in preds.items() if k != cls}
            print(f"  '{cls}': {data['accuracy']:.1f}% — misclassified as: {wrong}")
    
    mp_hands.close()


if __name__ == "__main__":
    main()
