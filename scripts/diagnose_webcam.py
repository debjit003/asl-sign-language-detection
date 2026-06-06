"""
Diagnostic: Test the ONNX model directly with webcam in Python.
This uses the EXACT same feature extraction as training.
If this correctly identifies signs, the bug is in Java feature computation.
If this ALSO fails, the bug is in the model/data.

Usage: python scripts/diagnose_webcam.py
Press ESC to quit.
"""

import cv2
import numpy as np
import mediapipe as mp
import onnxruntime as ort
import json
from pathlib import Path
import sys

# Import feature extraction from our training script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_landmarks import extract_features

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "src" / "main" / "resources" / "models"


def main():
    # Load ONNX model
    onnx_path = str(MODELS_DIR / "sign_classifier.onnx")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    
    # Load scaler
    with open(MODELS_DIR / "scaler_params.json") as f:
        scaler_data = json.load(f)
    mean = np.array(scaler_data["mean"])
    scale = np.array(scaler_data["scale"])
    n_features = scaler_data["n_features"]
    
    # Load labels
    with open(MODELS_DIR / "label_classes.json") as f:
        labels = json.load(f)["classes"]
    
    print(f"Model loaded: {n_features} features, {len(labels)} classes")
    print(f"Labels: {labels}")
    
    # MediaPipe
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_draw = mp.solutions.drawing_utils
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam")
        return
    
    print("\nWebcam opened. Show signs to camera. Press ESC to quit.")
    print("=" * 60)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mp_hands.process(rgb)
        
        prediction = "No hand"
        confidence = 0.0
        top3_text = ""
        
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Extract raw landmarks (same as training)
            raw = np.array([
                [lm.x, lm.y, lm.z] for lm in hand.landmark
            ]).flatten()
            
            # Compute features (EXACT same code as training)
            features = extract_features(raw)
            
            if features is not None and len(features) == n_features:
                # Scale (same as training)
                scaled = ((features - mean) / scale).astype(np.float32)
                scaled = scaled.reshape(1, -1)
                
                # Predict
                results = session.run(None, {input_name: scaled})
                proba = results[1][0]  # probabilities
                
                pred_idx = np.argmax(proba)
                confidence = proba[pred_idx]
                prediction = labels[pred_idx].upper()
                
                # Top 3
                top3_indices = np.argsort(proba)[-3:][::-1]
                top3_items = [(labels[i].upper(), proba[i]) for i in top3_indices]
                top3_text = " | ".join([f"{l}:{p:.0%}" for l, p in top3_items])
                
                # Debug: print key features for B/5/4 diagnosis
                # Features 150-154: finger curl scores (thumb, index, middle, ring, pinky)
                # Features 155-159: binary extended flags
                # Feature 160: thumb across palm
                # Features 161-163: adjacent finger attachment
                # Feature 164: num extended fingers
                # Feature 165: fingers-together score
                curls = features[150:155]
                extended = features[155:160]
                thumb_across = features[160]
                attach = features[161:164]
                num_ext = features[164]
                together = features[165]
                
                print(f"Pred: {prediction} ({confidence:.0%}) | "
                      f"Top3: {top3_text} | "
                      f"Ext:{extended} NumExt:{num_ext:.0f} "
                      f"ThumbAcross:{thumb_across:.3f} "
                      f"Together:{together:.3f} "
                      f"Attach:{[f'{a:.3f}' for a in attach]}")
        
        # Draw prediction on frame
        color = (0, 255, 0) if confidence > 0.5 else (0, 165, 255)
        cv2.putText(frame, f"{prediction} ({confidence:.0%})", 
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
        cv2.putText(frame, top3_text, (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Diagnostic - Python ONNX", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()


if __name__ == "__main__":
    main()
