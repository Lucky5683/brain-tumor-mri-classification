"""
Evaluate models/efficientnet_b0_finetuned.keras on official test set (N=1,600).
Saves metrics and plots to reports/efficientnet_b0_finetuned/.
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    auc,
    ConfusionMatrixDisplay,
)
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.compat  # noqa: F401
from config import CLASS_NAMES, DISPLAY_NAMES, TEST_DIR
from src.data.dataset import get_test_dataset

CHECKPOINT = ROOT / "models" / "efficientnet_b0_finetuned.keras"
OUT_DIR = ROOT / "reports" / "efficientnet_b0_finetuned"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print(f"Loading {CHECKPOINT}...")
    model = tf.keras.models.load_model(CHECKPOINT, compile=False)
    test_ds, test_paths, test_labels = get_test_dataset(test_dir=TEST_DIR, model_type="efficientnet", batch_size=32)
    y_true = np.array(test_labels, dtype=np.int32)
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=len(CLASS_NAMES))

    start = time.time()
    probs = model.predict(test_ds, verbose=1)
    infer_time = time.time() - start
    preds = np.argmax(probs, axis=1)

    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_p = float(precision_score(y_true, preds, average="macro", zero_division=0))
    weighted_p = float(precision_score(y_true, preds, average="weighted", zero_division=0))
    macro_r = float(recall_score(y_true, preds, average="macro", zero_division=0))
    weighted_r = float(recall_score(y_true, preds, average="weighted", zero_division=0))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    roc_auc_macro = float(roc_auc_score(y_true_oh, probs, average="macro", multi_class="ovr"))
    roc_auc_weighted = float(roc_auc_score(y_true_oh, probs, average="weighted", multi_class="ovr"))

    rep = classification_report(y_true, preds, labels=list(range(4)), target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, preds, labels=list(range(4)))

    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        per_class[cname] = {
            "precision": float(rep[cname]["precision"]),
            "recall": float(rep[cname]["recall"]),
            "f1": float(rep[cname]["f1-score"]),
            "support": int(rep[cname]["support"]),
            "roc_auc_ovr": float(roc_auc_score(y_true_oh[:, idx], probs[:, idx])),
        }

    metrics_json = {
        "model_path": str(CHECKPOINT),
        "test_images_count": len(y_true),
        "inference_seconds": round(infer_time, 2),
        "ms_per_image": round((infer_time / len(y_true)) * 1000, 2),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_macro": macro_p,
            "precision_weighted": weighted_p,
            "recall_macro": macro_r,
            "recall_weighted": weighted_r,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1,
            "roc_auc_macro_ovr": roc_auc_macro,
            "roc_auc_weighted_ovr": roc_auc_weighted,
            "per_class": per_class,
        },
        "confusion_matrix": cm.tolist(),
    }

    with (OUT_DIR / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    pd.DataFrame(rep).transpose().to_csv(OUT_DIR / "classification_report.csv")

    # Plot Confusion Matrix
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[DISPLAY_NAMES[c] for c in CLASS_NAMES])
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
    ax.set_title("Confusion Matrix: EfficientNet-B0 V2 (Fine-Tuned)\n(Official Test Set, N=1,600)", fontsize=12, pad=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "confusion_matrix.png", dpi=300)
    plt.close(fig)

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]
    for idx, cname in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_true_oh[:, idx], probs[:, idx])
        ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f"{DISPLAY_NAMES[cname]} (AUC = {auc(fpr, tpr):.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
    ax.set_title(f"ROC Curves: EfficientNet-B0 V2 (Fine-Tuned)\nMacro ROC-AUC = {roc_auc_macro:.4f}", fontsize=12, fontweight="bold")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "roc_curve.png", dpi=300)
    plt.close(fig)

    # Plot PR curve
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    for idx, cname in enumerate(CLASS_NAMES):
        prec_c, rec_c, _ = precision_recall_curve(y_true_oh[:, idx], probs[:, idx])
        ap = average_precision_score(y_true_oh[:, idx], probs[:, idx])
        ax.plot(rec_c, prec_c, color=colors[idx], lw=2, label=f"{DISPLAY_NAMES[cname]} (AP = {ap:.4f})")
    ax.set_title("Precision-Recall Curves: EfficientNet-B0 V2 (Fine-Tuned)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.legend(loc="lower left")
    ax.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "precision_recall_curve.png", dpi=300)
    plt.close(fig)

    print(f"Done! Test Acc: {acc:.4f}, Macro-F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()
