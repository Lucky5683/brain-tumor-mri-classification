"""Phase 3: Controlled Baseline Comparison on Identical Grouped Validation Split."""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

PROJECT_ROOT = Path("X:/Mini Project/brain-tumor-mri-classification")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.compat  # noqa: F401
from config import CLASS_NAMES, EXPERIMENTS_DIR, MODELS_DIR, REPORTS_DIR, TRAIN_DIR
from src.data.dataset import get_train_val_datasets

MANIFEST_PATH = EXPERIMENTS_DIR / "duplicate_aware_split" / "training_manifest.csv"

MODELS_TO_EVALUATE = [
    {
        "id": "baseline_cnn",
        "name": "Custom 4-Block CNN",
        "path": PROJECT_ROOT / "baseline_cnn.keras",
        "model_type": "cnn",
        "split_note": "Trained on random 80/20 split; evaluated on grouped val",
    },
    {
        "id": "efficientnet_b0_finetuned",
        "name": "EfficientNet-B0 (Two-Stage)",
        "path": MODELS_DIR / "efficientnet_b0_finetuned.keras",
        "model_type": "efficientnet",
        "split_note": "Trained on random 80/20 split; evaluated on grouped val",
    },
    {
        "id": "multiscale_fusion_best",
        "name": "Multi-Scale Fusion (Standard Split)",
        "path": MODELS_DIR / "multiscale_fusion_best.keras",
        "model_type": "efficientnet",
        "split_note": "Trained on random 80/20 split; evaluated on grouped val",
    },
    {
        "id": "multiscale_fusion_grouped_best",
        "name": "Multi-Scale Fusion (Grouped Split)",
        "path": MODELS_DIR / "multiscale_fusion_grouped_best.keras",
        "model_type": "efficientnet",
        "split_note": "Trained on grouped split; evaluated on grouped val",
    },
]


def run_comparison():
    print("=" * 75)
    print("PHASE 3: CONTROLLED BASELINE COMPARISON ON FIXED VALIDATION SET")
    print("=" * 75)

    comparison_results = []

    for item in MODELS_TO_EVALUATE:
        model_path = item["path"]
        print(f"\nEvaluating: {item['name']}")
        print(f"Path: {model_path}")

        if not model_path.is_file():
            print(f"WARNING: Model file missing: {model_path}. Skipping.")
            continue

        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        model = tf.keras.models.load_model(model_path, compile=False)
        param_count = model.count_params()
        print(f"Parameters: {param_count:,} | Size: {file_size_mb:.2f} MB")

        # Load dataset with appropriate normalizer
        _, val_ds, _, _, val_paths, val_labels = get_train_val_datasets(
            train_dir=TRAIN_DIR,
            model_type=item["model_type"],
            batch_size=32,
            use_augmentation=False,
            split_manifest=MANIFEST_PATH,
        )

        y_true = np.array(val_labels)

        # Measure inference latency
        t0 = time.perf_counter()
        probs = model.predict(val_ds, verbose=0)
        elapsed = time.perf_counter() - t0
        latency_ms_per_img = (elapsed / len(y_true)) * 1000

        preds = np.argmax(probs, axis=1)

        acc = float(accuracy_score(y_true, preds))
        macro_f1 = float(f1_score(y_true, preds, average="macro"))
        macro_prec = float(precision_score(y_true, preds, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, preds, average="macro", zero_division=0))

        per_f1 = f1_score(y_true, preds, average=None)
        per_rec = recall_score(y_true, preds, average=None)
        per_prec = precision_score(y_true, preds, average=None, zero_division=0)
        cm = confusion_matrix(y_true, preds).tolist()

        res = {
            "model_id": item["id"],
            "name": item["name"],
            "parameters": param_count,
            "size_mb": round(file_size_mb, 2),
            "latency_ms_per_img": round(latency_ms_per_img, 2),
            "split_note": item["split_note"],
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "per_class": {
                cls: {
                    "precision": round(float(per_prec[i]), 4),
                    "recall": round(float(per_rec[i]), 4),
                    "f1": round(float(per_f1[i]), 4),
                }
                for i, cls in enumerate(CLASS_NAMES)
            },
            "confusion_matrix": cm,
        }
        comparison_results.append(res)
        print(f"Acc: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f} | Glioma Rec: {per_rec[0]*100:.2f}% | Latency: {latency_ms_per_img:.1f}ms/img")

    # Save summary table
    out_json = REPORTS_DIR / "baseline_comparison.json"
    out_json.write_text(json.dumps(comparison_results, indent=2), encoding="utf-8")

    # Save CSV
    flat_rows = []
    for r in comparison_results:
        row = {
            "Model": r["name"],
            "Parameters": r["parameters"],
            "Size (MB)": r["size_mb"],
            "Accuracy": r["accuracy"],
            "Macro-F1": r["macro_f1"],
            "Macro-Precision": r["macro_precision"],
            "Macro-Recall": r["macro_recall"],
            "Glioma Recall": r["per_class"]["glioma"]["recall"],
            "Meningioma Recall": r["per_class"]["meningioma"]["recall"],
            "NoTumor Recall": r["per_class"]["notumor"]["recall"],
            "Pituitary Recall": r["per_class"]["pituitary"]["recall"],
            "Latency (ms/img)": r["latency_ms_per_img"],
            "Split Note": r["split_note"],
        }
        flat_rows.append(row)

    df = pd.DataFrame(flat_rows)
    out_csv = REPORTS_DIR / "baseline_comparison.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved baseline comparison to:\n  {out_json}\n  {out_csv}")
    print("\nSummary Table:")
    print(df[["Model", "Parameters", "Accuracy", "Macro-F1", "Glioma Recall", "Meningioma Recall"]].to_string(index=False))


if __name__ == "__main__":
    run_comparison()
