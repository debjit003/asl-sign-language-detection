"""
Export the best trained model + scaler to ONNX format for Java inference.

Converts the sklearn model to ONNX using skl2onnx, exports scaler parameters
and label classes as JSON files, and verifies the export by running test inference.

Output files go to: webapp/models/

Usage:
    python scripts/export_to_onnx.py
"""

import pickle
import json
import numpy as np
import os
from pathlib import Path

from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnxruntime as ort

# ---- Project paths ----
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
WEBAPP_MODELS_DIR = PROJECT_DIR / "webapp" / "models"


def main():
    os.makedirs(WEBAPP_MODELS_DIR, exist_ok=True)
    
    # ---- Load trained artifacts ----
    print("Loading trained model...")
    with open(MODELS_DIR / "best_model.pkl", "rb") as f:
        model = pickle.load(f)
    
    with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
        le = pickle.load(f)
    
    with open(MODELS_DIR / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    
    model_type = type(model).__name__
    n_features = scaler.n_features_in_
    n_classes = len(le.classes_)
    
    print(f"  Model type: {model_type}")
    print(f"  Classes ({n_classes}): {list(le.classes_)}")
    print(f"  Feature count: {n_features}")
    
    # ---- Convert model to ONNX ----
    print(f"\nConverting {model_type} to ONNX...")
    initial_types = [('float_input', FloatTensorType([None, n_features]))]
    
    # Options to get probabilities (disable zipmap for array output)
    options = {}
    if hasattr(model, 'predict_proba'):
        options[type(model)] = {'zipmap': False}
    
    onnx_model = convert_sklearn(model, initial_types=initial_types, options=options)
    
    onnx_path = WEBAPP_MODELS_DIR / "sign_classifier.onnx"
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    
    onnx_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  ONNX model saved: {onnx_path} ({onnx_size_mb:.1f} MB)")
    
    # ---- Export scaler parameters as JSON ----
    print("\nExporting scaler parameters...")
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features": int(scaler.n_features_in_),
    }
    
    scaler_path = WEBAPP_MODELS_DIR / "scaler_params.json"
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f, indent=2)
    print(f"  Scaler params saved: {scaler_path}")
    
    # ---- Export label classes as JSON ----
    print("\nExporting label classes...")
    labels_data = {
        "classes": list(le.classes_),
    }
    
    labels_path = WEBAPP_MODELS_DIR / "label_classes.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f, indent=2)
    print(f"  Label classes saved: {labels_path}")
    
    # ---- Verify ONNX model ----
    print("\nVerifying ONNX export...")
    session = ort.InferenceSession(str(onnx_path))
    
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()
    print(f"  Input: {input_info.name}, shape={input_info.shape}, type={input_info.type}")
    for out in output_info:
        print(f"  Output: {out.name}, shape={out.shape}")
    
    # Run test inference
    X_test = np.load(MODELS_DIR / "landmarks_data.npy")[:10]
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    input_name = session.get_inputs()[0].name
    results = session.run(None, {input_name: X_test_scaled})
    
    # Compare with sklearn predictions
    sklearn_preds = model.predict(scaler.transform(X_test[:10]))
    onnx_preds = results[0]
    
    print(f"\n  Verification — sklearn vs ONNX predictions:")
    match_count = 0
    for i in range(min(10, len(sklearn_preds))):
        sklearn_label = le.inverse_transform([sklearn_preds[i]])[0]
        onnx_label = le.inverse_transform([onnx_preds[i]])[0] if isinstance(onnx_preds[i], (int, np.integer)) else str(onnx_preds[i])
        match = "[OK]" if str(sklearn_label) == str(onnx_label) else "[MISMATCH]"
        if str(sklearn_label) == str(onnx_label):
            match_count += 1
        print(f"    Sample {i}: sklearn={sklearn_label} | onnx={onnx_label} {match}")
    
    print(f"\n  Match rate: {match_count}/{min(10, len(sklearn_preds))}")
    
    if len(results) > 1:
        proba = results[1]
        print(f"  Probability output shape: {proba.shape}")
        print(f"  Sample proba range: [{proba[0].min():.4f}, {proba[0].max():.4f}]")
    
    print(f"\n[OK] Export complete!")
    print(f"  ONNX model: {onnx_path}")
    print(f"  Scaler:     {scaler_path}")
    print(f"  Labels:     {labels_path}")
    print(f"  Features:   {n_features}")
    print(f"  Classes:    {n_classes}")


if __name__ == "__main__":
    main()
