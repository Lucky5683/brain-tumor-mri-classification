"""Reproduce validation baseline metrics on the 1,153 duplicate-aware validation cohort."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.compat  # noqa: F401
from config import CLASS_NAMES, EXPERIMENTS_DIR, MODELS_DIR, TRAIN_DIR
from src.data.dataset import get_train_val_datasets

MANIFEST_PATH = EXPERIMENTS_DIR / "duplicate_aware_split" / "training_manifest.csv"
OUTPUT_DIR = EXPERIMENTS_DIR / "pretraining_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_MODELS = [
    {
        "id": "multiscale_fusion_grouped_best",
        "name": "EfficientNet-B0 Multi-Scale Fusion (Grouped Checkpoint)",
        "path": MODELS_DIR / "multiscale_fusion_grouped_best.keras",
        "model_type": "efficientnet",
    },
    {
        "id": "efficientnet_b0_finetuned",
        "name": "EfficientNet-B0 (Two-Stage Fine-Tuned Baseline)",
        "path": MODELS_DIR / "efficientnet_b0_finetuned.keras",
        "model_type": "efficientnet",
    },
]


def evaluate_model_on_val(model_info, val_ds, val_paths, val_labels):
    model_path = model_info["path"]
    print(f"\n--- Evaluating: {model_info['name']} ---")
    print(f"Path: {model_path}")

    if not model_path.is_file():
        return {"error": f"File not found: {model_path}"}

    model = tf.keras.models.load_model(model_path, compile=False)
    t0 = time.time()
    probabilities = model.predict(val_ds, verbose=1)
    inference_sec = time.time() - t0

    predictions = np.argmax(probabilities, axis=1)
    y_true = np.asarray(val_labels)

    acc = float(accuracy_score(y_true, predictions))
    bal_acc = float(balanced_accuracy_score(y_true, predictions))
    macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, predictions, average="weighted", zero_division=0))
    macro_p = float(precision_score(y_true, predictions, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, predictions, average="weighted", zero_division=0))
    macro_r = float(recall_score(y_true, predictions, average="macro", zero_division=0))
    weighted_r = float(recall_score(y_true, predictions, average="weighted", zero_division=0))

    report_dict = classification_report(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, predictions, labels=list(range(len(CLASS_NAMES))))

    # Detailed glioma and meningioma error breakdown
    glioma_idx = CLASS_NAMES.index("glioma")
    meningioma_idx = CLASS_NAMES.index("meningioma")
    notumor_idx = CLASS_NAMES.index("notumor")
    pituitary_idx = CLASS_NAMES.index("pituitary")

    glioma_as_meningioma = int(cm[glioma_idx, meningioma_idx])
    glioma_as_notumor = int(cm[glioma_idx, notumor_idx])
    glioma_as_pituitary = int(cm[glioma_idx, pituitary_idx])
    meningioma_as_glioma = int(cm[meningioma_idx, glioma_idx])
    meningioma_as_pituitary = int(cm[meningioma_idx, pituitary_idx])
    meningioma_as_notumor = int(cm[meningioma_idx, notumor_idx])

    result = {
        "model_id": model_info["id"],
        "model_name": model_info["name"],
        "model_path": str(model_path.resolve()),
        "total_parameters": model.count_params(),
        "trainable_parameters": sum(int(tf.size(w)) for w in model.trainable_weights),
        "validation_samples": len(val_labels),
        "inference_seconds": round(inference_sec, 2),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_precision": macro_p,
            "weighted_precision": weighted_p,
            "macro_recall": macro_r,
            "weighted_recall": weighted_r,
        },
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
        "confusion_matrix_labels": list(CLASS_NAMES),
        "diagnostic_confusion_analysis": {
            "glioma_total_val": int(cm[glioma_idx].sum()),
            "glioma_correct": int(cm[glioma_idx, glioma_idx]),
            "glioma_misclassified_as_meningioma": glioma_as_meningioma,
            "glioma_misclassified_as_notumor": glioma_as_notumor,
            "glioma_misclassified_as_pituitary": glioma_as_pituitary,
            "meningioma_total_val": int(cm[meningioma_idx].sum()),
            "meningioma_correct": int(cm[meningioma_idx, meningioma_idx]),
            "meningioma_misclassified_as_glioma": meningioma_as_glioma,
            "meningioma_misclassified_as_pituitary": meningioma_as_pituitary,
            "meningioma_misclassified_as_notumor": meningioma_as_notumor,
        },
    }

    print(f"Accuracy:        {acc * 100:.2f}%")
    print(f"Balanced Acc:    {bal_acc * 100:.2f}%")
    print(f"Macro-F1:        {macro_f1:.4f}")
    print(f"Macro-Precision: {macro_p:.4f}")
    print(f"Macro-Recall:    {macro_r:.4f}")
    print("Confusion Matrix:")
    print(pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_string())

    return result


def main():
    print("=" * 70)
    print("PRE-TRAINING AUDIT: REPRODUCING VALIDATION BASELINES")
    print("=" * 70)
    print(f"Loading validation dataset from manifest: {MANIFEST_PATH}")

    # Load unaugmented validation dataset
    _, val_ds, _, _, val_paths, val_labels = get_train_val_datasets(
        train_dir=TRAIN_DIR,
        model_type="efficientnet",
        batch_size=32,
        use_augmentation=False,
        split_manifest=MANIFEST_PATH,
    )

    print(f"Loaded {len(val_paths)} validation images.")

    all_evaluations = {}
    for target in TARGET_MODELS:
        res = evaluate_model_on_val(target, val_ds, val_paths, val_labels)
        all_evaluations[target["id"]] = res

    # Historical comparison check for multiscale_fusion_grouped_best
    hist_results_path = EXPERIMENTS_DIR / "multiscale_fusion_grouped_results.json"
    historical_summary = {}
    if hist_results_path.is_file():
        historical_summary = json.loads(hist_results_path.read_text(encoding="utf-8"))

    baseline_output = {
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(MANIFEST_PATH.resolve()),
        "validation_samples": len(val_labels),
        "reproduced_models": all_evaluations,
        "historical_grouped_fusion_record": historical_summary.get("best_validation_metrics", {}),
        "discrepancy_check": {
            "reproduced_acc": all_evaluations["multiscale_fusion_grouped_best"]["metrics"]["accuracy"],
            "historical_acc": historical_summary.get("best_validation_metrics", {}).get("val_accuracy"),
            "reproduced_macro_f1": all_evaluations["multiscale_fusion_grouped_best"]["metrics"]["macro_f1"],
            "historical_macro_f1": historical_summary.get("best_validation_metrics", {}).get("val_macro_f1"),
        },
    }

    out_file = OUTPUT_DIR / "validation_baseline.json"
    out_file.write_text(json.dumps(baseline_output, indent=2), encoding="utf-8")
    print(f"\nSaved validation baseline output to: {out_file}")


if __name__ == "__main__":
    main()
