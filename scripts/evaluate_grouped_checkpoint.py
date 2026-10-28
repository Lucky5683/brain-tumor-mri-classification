"""Evaluation script for the similarity-grouped multi-scale fusion checkpoint."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
    ConfusionMatrixDisplay,
)
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.compat  # noqa: F401
from config import CLASS_NAMES, EXPERIMENTS_DIR, MODELS_DIR, TRAIN_DIR
from src.data.dataset import get_train_val_datasets

CHECKPOINT_PATH = MODELS_DIR / "multiscale_fusion_grouped_best.keras"
MANIFEST_PATH = EXPERIMENTS_DIR / "duplicate_aware_split" / "training_manifest.csv"
OUTPUT_DIR = EXPERIMENTS_DIR / "duplicate_aware_split"
SUMMARY_PATH = EXPERIMENTS_DIR / "multiscale_fusion_grouped_results.json"


def evaluate_checkpoint():
    print("=" * 70)
    print("EVALUATING GROUPED MULTI-SCALE FUSION CHECKPOINT")
    print("=" * 70)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)
    print("Model loaded successfully. Parameters:", model.count_params())

    print(f"Loading validation dataset from manifest: {MANIFEST_PATH}")
    _, val_ds, _, _, val_paths, val_labels = get_train_val_datasets(
        train_dir=TRAIN_DIR,
        model_type="efficientnet",
        batch_size=32,
        use_augmentation=False,
        split_manifest=MANIFEST_PATH,
    )

    val_count = len(val_paths)
    print(f"Validation images found: {val_count}")

    print("Running inference on validation dataset...")
    t0 = time.time()
    probabilities = model.predict(val_ds, verbose=1)
    inference_time = time.time() - t0
    print(f"Inference completed in {inference_time:.2f}s ({val_count / inference_time:.1f} img/s).")

    predictions = np.argmax(probabilities, axis=1)
    y_true = np.asarray(val_labels)
    confidences = np.max(probabilities, axis=1) * 100.0

    acc = float(accuracy_score(y_true, predictions))
    macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, predictions, average="weighted", zero_division=0))
    macro_p = float(precision_score(y_true, predictions, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, predictions, average="weighted", zero_division=0))
    macro_r = float(recall_score(y_true, predictions, average="macro", zero_division=0))
    weighted_r = float(recall_score(y_true, predictions, average="weighted", zero_division=0))

    print("\n" + "=" * 50)
    print("VALIDATION EVALUATION METRICS")
    print("=" * 50)
    print(f"Validation Accuracy:  {acc * 100:.2f}% ({acc:.6f})")
    print(f"Validation Macro-F1:  {macro_f1 * 100:.2f}% ({macro_f1:.6f})")
    print(f"Validation Macro-P:   {macro_p * 100:.2f}% ({macro_p:.6f})")
    print(f"Validation Macro-R:   {macro_r * 100:.2f}% ({macro_r:.6f})")
    print(f"Weighted-F1:          {weighted_f1 * 100:.2f}% ({weighted_f1:.6f})")

    # Classification report
    report_dict = classification_report(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report_dict).transpose()
    print("\nClassification Report:")
    print(report_df.round(4).to_string())

    report_csv_path = OUTPUT_DIR / "validation_classification_report.csv"
    report_df.to_csv(report_csv_path)
    print(f"\nSaved classification report to: {report_csv_path}")

    # Confusion matrix
    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print("\nConfusion Matrix:")
    print(cm_df.to_string())

    cm_csv_path = OUTPUT_DIR / "validation_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d", ax=ax, xticks_rotation=25)
    plt.title(f"Grouped Multi-Scale Fusion Validation Confusion Matrix\n(Accuracy: {acc * 100:.2f}%, Macro-F1: {macro_f1 * 100:.2f}%)")
    plt.tight_layout()
    cm_png_path = OUTPUT_DIR / "validation_confusion_matrix.png"
    plt.savefig(cm_png_path, dpi=200)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_png_path}")

    # Prediction records
    records = []
    for i, p in enumerate(val_paths):
        rel_path = str(p.relative_to(PROJECT_ROOT)).replace("\\", "/")
        act_class = CLASS_NAMES[y_true[i]]
        pred_class = CLASS_NAMES[predictions[i]]
        conf = round(float(confidences[i]), 2)
        records.append({
            "image_path": rel_path,
            "actual_label": act_class,
            "predicted_label": pred_class,
            "confidence_percent": conf,
            "correct": bool(act_class == pred_class),
        })
    pred_df = pd.DataFrame(records)
    pred_csv_path = OUTPUT_DIR / "validation_predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"Saved validation predictions to: {pred_csv_path}")

    # Compare with saved results
    if SUMMARY_PATH.is_file():
        saved_summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
        saved_best = saved_summary.get("best_validation_metrics", {})
        print("\n" + "=" * 50)
        print("COMPARISON WITH SAVED RESULTS SUMMARY")
        print("=" * 50)
        print(f"Metric              Reproduced       Saved Summary    Diff")
        print(f"----------------------------------------------------------")
        for key, name in [
            ("val_accuracy", "Accuracy"),
            ("val_macro_f1", "Macro-F1"),
            ("val_macro_precision", "Macro Precision"),
            ("val_macro_recall", "Macro Recall"),
        ]:
            val_saved = saved_best.get(key, None)
            val_rep = {"val_accuracy": acc, "val_macro_f1": macro_f1, "val_macro_precision": macro_p, "val_macro_recall": macro_r}[key]
            if val_saved is not None:
                diff = val_rep - val_saved
                print(f"{name:<18} {val_rep * 100:.4f}%        {val_saved * 100:.4f}%       {diff * 100:+.4f}%")
            else:
                print(f"{name:<18} {val_rep * 100:.4f}%        N/A")

    # Save detailed evaluation metrics json
    metrics_json = {
        "checkpoint": str(CHECKPOINT_PATH.resolve()),
        "manifest": str(MANIFEST_PATH.resolve()),
        "validation_samples": val_count,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "weighted_f1": weighted_f1,
        "weighted_precision": weighted_p,
        "weighted_recall": weighted_r,
        "per_class": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1_score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"]),
            }
            for cls in CLASS_NAMES
        },
        "confusion_matrix": cm.tolist(),
    }
    metrics_json_path = OUTPUT_DIR / "validation_metrics.json"
    metrics_json_path.write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
    print(f"Saved validation metrics JSON to: {metrics_json_path}")


if __name__ == "__main__":
    evaluate_checkpoint()
