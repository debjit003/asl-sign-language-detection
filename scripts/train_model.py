"""
Train multiple classification models on extracted hand landmark features
and select the best one.

Models trained:
  1. RandomForestClassifier
  2. HistGradientBoostingClassifier  
  3. ExtraTreesClassifier
  4. MLPClassifier (Neural Network)

Evaluates each model on a held-out test set with special attention to
commonly confused sign pairs (B/5/4, M/N, A/S/E).

Usage:
    python scripts/train_model.py
"""

import os
import numpy as np
import pickle
import json
import time
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, 
                             confusion_matrix)
from sklearn.ensemble import (RandomForestClassifier, 
                              ExtraTreesClassifier,
                              HistGradientBoostingClassifier)
from sklearn.neural_network import MLPClassifier

# ---- Project paths ----
PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"


def print_confusion_pairs(y_true, y_pred, le, pairs):
    """Print confusion matrix for specific sign pairs."""
    for group_name, signs in pairs:
        indices = [i for i, lbl in enumerate(y_true) 
                   if le.inverse_transform([lbl])[0] in signs]
        if not indices:
            continue
        
        sub_true = [y_true[i] for i in indices]
        sub_pred = [y_pred[i] for i in indices]
        
        sign_indices = [le.transform([s])[0] for s in signs if s in le.classes_]
        
        print(f"\n  {group_name} confusion ({'/'.join(signs)}):")
        for t_sign in signs:
            if t_sign not in le.classes_:
                continue
            t_idx = le.transform([t_sign])[0]
            total = sum(1 for x in sub_true if x == t_idx)
            if total == 0:
                continue
            for p_sign in signs:
                if p_sign not in le.classes_:
                    continue
                p_idx = le.transform([p_sign])[0]
                count = sum(1 for t, p in zip(sub_true, sub_pred) 
                           if t == t_idx and p == p_idx)
                marker = " [OK]" if t_sign == p_sign else " [WRONG]" if count > 0 else ""
                print(f"    True={t_sign.upper()} -> Pred={p_sign.upper()}: "
                      f"{count}/{total} ({100*count/total:.0f}%){marker}")


def main():
    # ---- Load data ----
    print("Loading extracted features...")
    X = np.load(MODELS_DIR / "landmarks_data.npy")
    y_raw = np.load(MODELS_DIR / "landmarks_labels.npy")
    
    print(f"  Data shape: {X.shape}")
    print(f"  Feature count: {X.shape[1]}")
    print(f"  Classes: {len(np.unique(y_raw))}")
    print(f"  Samples per class: ~{len(y_raw) // len(np.unique(y_raw))}")
    
    # ---- Encode labels ----
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Label classes: {list(le.classes_)}")
    
    # ---- Train/test split ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n  Train: {X_train.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    
    # ---- StandardScaler ----
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---- Define models ----
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=3,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
        ),
        "GradientBoosting": HistGradientBoostingClassifier(
            max_iter=500,
            max_depth=8,
            learning_rate=0.1,
            random_state=42,
        ),
        "NeuralNetwork (MLP)": MLPClassifier(
            hidden_layer_sizes=(512, 256, 128, 64),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
        ),
    }
    
    # ---- Confusion pair groups ----
    confusion_pairs = [
        ("B/5/4", ["b", "5", "4"]),
        ("M/N/T", ["m", "n", "t"]),
        ("A/S/E", ["a", "s", "e"]),
        ("U/V/R", ["u", "v", "r"]),
        ("1/D/I", ["1", "d", "i"]),
    ]
    
    # ---- Train and evaluate each model ----
    results = {}
    
    print("\n" + "=" * 60)
    print("TRAINING MODELS")
    print("=" * 60)
    
    for name, model in models.items():
        print(f"\n{'-' * 50}")
        print(f"Training: {name}")
        print(f"{'-' * 50}")
        
        start = time.time()
        model.fit(X_train_scaled, y_train)
        train_time = time.time() - start
        
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            "model": model,
            "accuracy": accuracy,
            "train_time": train_time,
            "y_pred": y_pred,
        }
        
        print(f"  Accuracy: {accuracy * 100:.2f}%")
        print(f"  Train time: {train_time:.1f}s")
        
        # Show confusion for problematic pairs
        print_confusion_pairs(y_test, y_pred, le, confusion_pairs)
    
    # ---- Compare results ----
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n  {'Model':<25} {'Accuracy':>10} {'Train Time':>12}")
    print(f"  {'---'*8:25} {'---'*3:10} {'---'*4:12}")
    
    for name, res in sorted(results.items(), key=lambda x: -x[1]["accuracy"]):
        marker = " << BEST" if res["accuracy"] == max(r["accuracy"] for r in results.values()) else ""
        print(f"  {name:<25} {res['accuracy']*100:>9.2f}% {res['train_time']:>10.1f}s{marker}")
    
    # ---- Select best model ----
    best_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_model = results[best_name]["model"]
    best_accuracy = results[best_name]["accuracy"]
    
    print(f"\n  >> Best model: {best_name} ({best_accuracy*100:.2f}%)")
    
    # ---- Detailed classification report for best model ----
    print(f"\n{'=' * 60}")
    print(f"DETAILED REPORT — {best_name}")
    print(f"{'=' * 60}")
    best_pred = results[best_name]["y_pred"]
    print(classification_report(
        y_test, best_pred,
        target_names=[c.upper() for c in le.classes_],
        digits=3
    ))
    
    # ---- Save ALL models individually (for fallback during ONNX export) ----
    for name, res in results.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        model_path = MODELS_DIR / f"model_{safe_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(res["model"], f)
        print(f"  Saved: {model_path}")

    # ---- Save best model + scaler + label encoder ----
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    print(f"  Saved: {MODELS_DIR / 'best_model.pkl'}")
    
    with open(MODELS_DIR / "scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Saved: {MODELS_DIR / 'scaler.pkl'}")
    
    with open(MODELS_DIR / "label_encoder.pkl", "wb") as f:
        pickle.dump(le, f)
    print(f"  Saved: {MODELS_DIR / 'label_encoder.pkl'}")
    
    # ---- Also save all model results for reference ----
    summary = {
        "best_model": best_name,
        "best_accuracy": float(best_accuracy),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(le.classes_)),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "results": {
            name: {
                "accuracy": float(res["accuracy"]),
                "train_time": float(res["train_time"]),
            }
            for name, res in results.items()
        },
    }
    
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {MODELS_DIR / 'training_summary.json'}")
    
    print(f"\n[OK] Training complete! Best model: {best_name} ({best_accuracy*100:.2f}%)")


if __name__ == "__main__":
    main()
