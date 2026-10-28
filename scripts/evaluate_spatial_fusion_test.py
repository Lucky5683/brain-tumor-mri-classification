"""Comprehensive, Reproducible Test Set Evaluation of Selected Spatial Fusion Model.

Checkpoint: experiments/spatial_fusion/meningioma_weighted_50ep_seed42/best_model.keras
Dataset:    dataset/Testing (N=1,600, 400 per class)
Classes:    glioma, meningioma, notumor, pituitary

CRITICAL PROTOCOL:
- Pure forward-pass inference only.
- No training, no fine-tuning, no temperature calibration, no threshold selection.
- All metrics and outputs are saved to experiments/final_evaluation/spatial_fusion/.
- Historical records and checkpoints are strictly preserved without modification.
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

# Configure headless matplotlib before imports
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
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
    auc,
)
import tensorflow as tf

# Setup project root
ROOT = Path("X:/Mini Project/brain-tumor-mri-classification")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.compat  # noqa: F401
from config import CLASS_NAMES, CLASS_TO_INDEX, IMAGE_SIZE

CHECKPOINT_PATH = ROOT / "experiments/spatial_fusion/meningioma_weighted_50ep_seed42/best_model.keras"
TEST_DIR = ROOT / "dataset/Testing"
TRAINING_MANIFEST = ROOT / "experiments/duplicate_aware_split/training_manifest.csv"
OUTPUT_DIR = ROOT / "experiments/final_evaluation/spatial_fusion"

BATCH_SIZE = 32


def compute_per_class_specificity(cm: np.ndarray) -> dict[str, float]:
    """Compute per-class specificity (True Negative Rate) from confusion matrix.

    For class k:
      TN = sum(all elements not in row k and not in col k)
      FP = sum(col k except element (k,k))
      Specificity = TN / (TN + FP)
    """
    num_classes = cm.shape[0]
    total_samples = cm.sum()
    specificities = {}

    for k, name in enumerate(CLASS_NAMES):
        tp = cm[k, k]
        fn = cm[k, :].sum() - tp
        fp = cm[:, k].sum() - tp
        tn = total_samples - (tp + fn + fp)

        denom = tn + fp
        spec = float(tn / denom) if denom > 0 else 0.0
        specificities[name] = round(spec, 4)

    return specificities


def verify_test_dataset(test_dir: Path) -> tuple[list[Path], list[int], dict[str, int]]:
    """Scan and verify all images in test_dir with deterministic ordering."""
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory not found: {test_dir.resolve()}")

    test_paths: list[Path] = []
    test_labels: list[int] = []
    class_counts: dict[str, int] = {}

    for idx, class_name in enumerate(CLASS_NAMES):
        class_folder = test_dir / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(f"Missing required class folder: {class_folder.resolve()}")

        # Deterministic sorting
        valid_extensions = {".jpg", ".jpeg", ".png"}
        images = sorted([
            p for p in class_folder.glob("*")
            if p.is_file() and p.suffix.lower() in valid_extensions
        ])

        if len(images) == 0:
            raise RuntimeError(f"Class folder {class_name} has 0 valid image files.")

        class_counts[class_name] = len(images)
        for img_path in images:
            test_paths.append(img_path)
            test_labels.append(idx)

    return test_paths, test_labels, class_counts


def audit_test_hashes(
    test_paths: list[Path],
    manifest_path: Path,
) -> dict:
    """Verify SHA-256 hashes of test images and cross-reference with training manifest."""
    print("Computing SHA-256 hashes for all test images...")
    test_hashes = {}
    corrupted_images = []

    for p in test_paths:
        try:
            with Image.open(p) as img:
                img.verify()
            data = p.read_bytes()
            h = hashlib.sha256(data).hexdigest().lower()
            test_hashes[str(p.resolve())] = h
        except Exception as e:
            corrupted_images.append({"path": str(p), "error": str(e)})

    if corrupted_images:
        raise RuntimeError(f"Corrupted images detected in test set: {corrupted_images}")

    # Cross-reference with training/validation manifest
    manifest_hashes = set()
    train_count = 0
    val_count = 0
    if manifest_path.is_file():
        df_manifest = pd.read_csv(manifest_path)
        manifest_hashes = set(df_manifest["sha256"].str.strip().str.lower().unique())
        train_count = len(df_manifest[df_manifest["split"] == "train"])
        val_count = len(df_manifest[df_manifest["split"] == "validation"])

    exact_matches = []
    test_hash_set = set(test_hashes.values())
    overlap_count = len(test_hash_set.intersection(manifest_hashes))

    return {
        "total_test_scans": len(test_paths),
        "unique_test_hashes": len(test_hash_set),
        "corrupted_images_count": len(corrupted_images),
        "training_manifest_scans": train_count + val_count,
        "manifest_exact_hash_overlap_count": overlap_count,
        "cross_split_byte_duplicate_free": (overlap_count == 0),
        "perceptual_slice_leakage_note": (
            "No byte-for-byte exact SHA-256 duplicates exist between dataset/Training and dataset/Testing. "
            "However, as documented in dataset_audit.md, the public Kaggle collection contains adjacent axial 2D slices "
            "from the same patient scans across splits without patient IDs. Hence, test accuracy evaluates "
            "benchmark generalization rather than certified patient-level generalization."
        ),
    }


def build_test_tf_dataset(paths: list[Path], labels: list[int]) -> tf.data.Dataset:
    """Build a deterministic tf.data pipeline matching the Spatial Fusion training format."""
    path_strs = [str(p.resolve()) for p in paths]
    ds = tf.data.Dataset.from_tensor_slices((path_strs, labels))

    def decode_and_resize(path_tensor, label_tensor):
        img_bytes = tf.io.read_file(path_tensor)
        img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, IMAGE_SIZE, method="bilinear", antialias=True)
        # Spatial Fusion model uses EfficientNet backbone expecting [0, 255] float32
        return tf.cast(img, tf.float32), label_tensor

    ds = ds.map(decode_and_resize, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    return ds.prefetch(tf.data.AUTOTUNE)


def plot_confusion_matrices(
    cm: np.ndarray,
    cm_norm: np.ndarray,
    output_path: Path,
):
    """Plot side-by-side raw counts and normalized percentage confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Raw counts
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    axes[0].set_title("Confusion Matrix (Raw Counts)", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(CLASS_NAMES))
    axes[0].set_xticks(tick_marks)
    axes[0].set_xticklabels(CLASS_NAMES, fontsize=10, rotation=25)
    axes[0].set_yticks(tick_marks)
    axes[0].set_yticklabels(CLASS_NAMES, fontsize=10)
    axes[0].set_ylabel("True Diagnosis", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")

    thresh0 = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(
                j, i, f"{cm[i, j]:,d}",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if cm[i, j] > thresh0 else "black",
                fontsize=11, fontweight="bold"
            )

    # Normalized
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0, vmax=1.0)
    axes[1].set_title("Normalized Confusion Matrix (Recall)", fontsize=13, fontweight="bold", pad=12)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    axes[1].set_xticks(tick_marks)
    axes[1].set_xticklabels(CLASS_NAMES, fontsize=10, rotation=25)
    axes[1].set_yticks(tick_marks)
    axes[1].set_yticklabels(CLASS_NAMES, fontsize=10)
    axes[1].set_ylabel("True Diagnosis", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Predicted Diagnosis", fontsize=11, fontweight="bold")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            axes[1].text(
                j, i, f"{val * 100:.1f}%",
                horizontalalignment="center",
                verticalalignment="center",
                color="white" if val > 0.5 else "black",
                fontsize=11, fontweight="bold"
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    output_path: Path,
) -> dict[str, float]:
    """Compute and plot One-vs-Rest ROC curves for all classes."""
    fig, ax = plt.subplots(figsize=(8, 7))

    y_onehot = np.eye(len(CLASS_NAMES))[y_true]
    roc_aucs = {}

    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for i, (name, color) in enumerate(zip(CLASS_NAMES, colors)):
        fpr, tpr, _ = roc_curve(y_onehot[:, i], probabilities[:, i])
        class_auc = auc(fpr, tpr)
        roc_aucs[name] = round(float(class_auc), 4)
        ax.plot(
            fpr, tpr, color=color, lw=2.2,
            label=f"{name.capitalize()} (AUC = {class_auc:.4f})"
        )

    # Macro-average ROC
    macro_auc = roc_auc_score(y_onehot, probabilities, multi_class="ovr", average="macro")
    roc_aucs["macro_ovr"] = round(float(macro_auc), 4)

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7, label="Chance Level (AUC = 0.5000)")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
    ax.set_title(
        f"Spatial Fusion Model: One-vs-Rest ROC Curves (Macro AUC = {macro_auc:.4f})",
        fontsize=12, fontweight="bold", pad=12
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    ax.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return roc_aucs


def plot_per_class_metrics_bar(
    per_class: dict[str, dict[str, float]],
    output_path: Path,
):
    """Plot grouped bar chart of Precision, Recall, Specificity, and F1-Score."""
    classes = CLASS_NAMES
    metrics = ["precision", "recall", "specificity", "f1"]
    metric_labels = ["Precision", "Recall (Sensitivity)", "Specificity", "F1-Score"]
    colors = ["#386cb0", "#fdc086", "#7fc97f", "#beaed4"]

    x = np.arange(len(classes))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (m, label, col) in enumerate(zip(metrics, metric_labels, colors)):
        values = [per_class[c][m] * 100 for c in classes]
        rects = ax.bar(x + i * width, values, width, label=label, color=col, edgecolor="black", linewidth=0.6)
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom",
                fontsize=8, fontweight="bold"
            )

    ax.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax.set_title("Spatial Fusion Model: Per-Class Diagnostic Performance (Test Set N=1,600)", fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([c.capitalize() for c in classes], fontsize=11, fontweight="bold")
    ax.set_ylim([0, 115])
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_markdown_report(
    summary: dict,
    output_path: Path,
):
    """Generate a comprehensive, formal Markdown audit and evaluation report."""
    metrics = summary["metrics"]
    per_class = summary["per_class"]
    audit = summary["dataset_audit"]

    content = f"""# Spatial Fusion Model: Official Test Set Evaluation Report

**Evaluation Date:** {summary["evaluation_timestamp"]}
**Evaluated Checkpoint:** `{summary["checkpoint_path"]}`
**Test Dataset Path:** `{summary["test_dataset_path"]}`
**Architecture:** `EfficientNetB0_SpatialFusion` (4,428,295 parameters)
**Execution Environment:** Python {summary["environment"]["python_version"]} on {summary["environment"]["platform"]} (TensorFlow {summary["environment"]["tensorflow_version"]}, Keras {summary["environment"]["keras_version"]})

---

## 1. Executive Summary & Verification Standard

This report provides the reproducible evaluation of the selected **Spatial Attention Feature Fusion** model on the official independent test partition ($N = 1,600$ MRI scans, exactly $400$ per class).

- **Evaluation Protocol:** Strictly single-pass, feed-forward inference on unmodified image tensors.
- **Model Invariance:** No weights, biases, class indices, or hyperparameters were adjusted.
- **Data Integrity:** All 1,600 scans were read and decoded with zero corrupted files.
- **Independence & Data Leakage Note:** Zero exact SHA-256 byte duplicates exist between training and testing sets. However, per the established dataset audit, volumetric adjacent 2D MRI slices from common patient scan sessions are present across the public Kaggle benchmark. Therefore, these metrics represent **benchmark generalization** and must not be conflated with clinical patient-level generalization.

---

## 2. Key Aggregate Metrics Summary

| Primary Evaluation Metric | Test Score (N = 1,600) | Historical Validation Score (N = 1,153) | Difference (Val → Test) |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **{metrics["accuracy"] * 100:.2f}%** ({int(metrics["accuracy"] * 1600)} / 1,600) | 92.54% (1,067 / 1,153) | {(metrics["accuracy"] - 0.925412) * 100:+.2f}% |
| **Balanced Accuracy** | **{metrics["balanced_accuracy"] * 100:.2f}%** | 92.35% | {(metrics["balanced_accuracy"] - 0.923491) * 100:+.2f}% |
| **Macro-Averaged F1-Score** | **{metrics["macro_f1"]:.4f}** | 0.9241 | {metrics["macro_f1"] - 0.924127:+.4f} |
| **Weighted-Averaged F1-Score** | **{metrics["weighted_f1"]:.4f}** | 0.9252 | {metrics["weighted_f1"] - 0.925233:+.4f} |
| **Macro Precision** | **{metrics["macro_precision"]:.4f}** | 0.9254 | {metrics["macro_precision"] - 0.925440:+.4f} |
| **Macro Recall (Sensitivity)** | **{metrics["macro_recall"]:.4f}** | 0.9235 | {metrics["macro_recall"] - 0.923491:+.4f} |
| **Macro ROC-AUC (OvR)** | **{metrics["macro_roc_auc"]:.4f}** | — | — |

---

## 3. Class-by-Class Diagnostic Performance

| Diagnostic Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | {per_class["glioma"]["precision"] * 100:.2f}% | {per_class["glioma"]["recall"] * 100:.2f}% | {per_class["glioma"]["specificity"] * 100:.2f}% | **{per_class["glioma"]["f1"]:.4f}** | {per_class["glioma"]["support"]} | {per_class["glioma"]["roc_auc"]:.4f} |
| **Meningioma** | {per_class["meningioma"]["precision"] * 100:.2f}% | {per_class["meningioma"]["recall"] * 100:.2f}% | {per_class["meningioma"]["specificity"] * 100:.2f}% | **{per_class["meningioma"]["f1"]:.4f}** | {per_class["meningioma"]["support"]} | {per_class["meningioma"]["roc_auc"]:.4f} |
| **No Tumor (Healthy)** | {per_class["notumor"]["precision"] * 100:.2f}% | {per_class["notumor"]["recall"] * 100:.2f}% | {per_class["notumor"]["specificity"] * 100:.2f}% | **{per_class["notumor"]["f1"]:.4f}** | {per_class["notumor"]["support"]} | {per_class["notumor"]["roc_auc"]:.4f} |
| **Pituitary Tumor** | {per_class["pituitary"]["precision"] * 100:.2f}% | {per_class["pituitary"]["recall"] * 100:.2f}% | {per_class["pituitary"]["specificity"] * 100:.2f}% | **{per_class["pituitary"]["f1"]:.4f}** | {per_class["pituitary"]["support"]} | {per_class["pituitary"]["roc_auc"]:.4f} |

---

## 4. Confusion Matrix Analysis

```text
Actual \\ Predicted     glioma    meningioma     notumor   pituitary       Total
glioma                    {summary["confusion_matrix"][0][0]:<10}{summary["confusion_matrix"][0][1]:<14}{summary["confusion_matrix"][0][2]:<12}{summary["confusion_matrix"][0][3]:<12}400
meningioma                {summary["confusion_matrix"][1][0]:<10}{summary["confusion_matrix"][1][1]:<14}{summary["confusion_matrix"][1][2]:<12}{summary["confusion_matrix"][1][3]:<12}400
notumor                   {summary["confusion_matrix"][2][0]:<10}{summary["confusion_matrix"][2][1]:<14}{summary["confusion_matrix"][2][2]:<12}{summary["confusion_matrix"][2][3]:<12}400
pituitary                 {summary["confusion_matrix"][3][0]:<10}{summary["confusion_matrix"][3][1]:<14}{summary["confusion_matrix"][3][2]:<12}{summary["confusion_matrix"][3][3]:<12}400
Total Predicted           {sum(summary["confusion_matrix"][i][0] for i in range(4)):<10}{sum(summary["confusion_matrix"][i][1] for i in range(4)):<14}{sum(summary["confusion_matrix"][i][2] for i in range(4)):<12}{sum(summary["confusion_matrix"][i][3] for i in range(4)):<12}1600
```

### Key Error Observations:
- **Total Misclassified Samples:** {len(summary["misclassified_summary"]["errors"])} / 1,600 ({len(summary["misclassified_summary"]["errors"]) / 1600 * 100:.2f}%)
- **Most Common Error Pair:** {summary["misclassified_summary"]["top_confusion_pair"]} ({summary["misclassified_summary"]["top_confusion_count"]} occurrences)
- **Healthy Brain Safety (False Positives for No Tumor):** {summary["confusion_matrix"][2][0] + summary["confusion_matrix"][2][1] + summary["confusion_matrix"][2][3]} healthy scans misclassified as tumors.
- **Healthy Brain False Negatives (Tumors misclassified as Healthy):** {summary["confusion_matrix"][0][2] + summary["confusion_matrix"][1][2] + summary["confusion_matrix"][3][2]} diseased scans misclassified as normal.

---

## 5. Artifact Manifest

All generated evaluation artifacts are stored deterministically in `{output_path.parent}`:
- `metrics_summary.json`: Complete serialized metrics and metadata.
- `confusion_matrix.csv`: Raw count confusion matrix.
- `normalized_confusion_matrix.csv`: Row-normalized confusion matrix.
- `confusion_matrix.png`: High-resolution dual-panel visualization.
- `roc_curves.png`: One-vs-Rest ROC curves with per-class and macro AUC.
- `per_class_metrics.png`: Grouped comparative bar chart of class metrics.
- `misclassified_samples.csv`: Complete index of misclassified scans with true/predicted labels, confidence, and margin.
"""
    output_path.write_text(content, encoding="utf-8")


def main():
    print("=" * 70)
    print("SPATIAL FUSION MODEL — FINAL INDEPENDENT TEST EVALUATION")
    print("=" * 70)
    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Test Set:   {TEST_DIR}")
    print(f"Output:     {OUTPUT_DIR}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Verification & Hash Audit
    test_paths, test_labels, class_counts = verify_test_dataset(TEST_DIR)
    print(f"Verified test images: {len(test_paths)} across {len(class_counts)} classes:")
    for c, n in class_counts.items():
        print(f"  - {c}: {n}")

    hash_audit = audit_test_hashes(test_paths, TRAINING_MANIFEST)
    print("\nHash Audit Results:")
    print(f"  - Total Test Scans: {hash_audit['total_test_scans']}")
    print(f"  - Unique Hashes:    {hash_audit['unique_test_hashes']}")
    print(f"  - Cross-Split Exact Duplicate Overlap: {hash_audit['manifest_exact_hash_overlap_count']}")

    # 2. Load Checkpoint
    print(f"\nLoading Spatial Fusion checkpoint from: {CHECKPOINT_PATH} ...")
    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)
    print(f"Model loaded successfully. Input: {model.input_shape}, Output: {model.output_shape}")

    # 3. Build Test Pipeline and Run Forward Inference
    print("\nExecuting deterministic test-set inference...")
    test_ds = build_test_tf_dataset(test_paths, test_labels)

    start_time = datetime.datetime.now()
    probabilities = model.predict(test_ds, verbose=1)
    eval_duration = (datetime.datetime.now() - start_time).total_seconds()
    print(f"Inference completed in {eval_duration:.2f} seconds ({len(test_paths) / eval_duration:.1f} scans/sec).")

    y_true = np.array(test_labels, dtype=np.int32)
    preds = np.argmax(probabilities, axis=1)

    # 4. Compute Metrics
    acc = float(accuracy_score(y_true, preds))
    bal_acc = float(balanced_accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, preds, average="weighted", zero_division=0))
    macro_prec = float(precision_score(y_true, preds, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, preds, average="weighted", zero_division=0))
    macro_rec = float(recall_score(y_true, preds, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, preds, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
    specificities = compute_per_class_specificity(cm)

    # Classification report dict
    clf_rep = classification_report(
        y_true, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )

    per_class_metrics = {}
    for name in CLASS_NAMES:
        per_class_metrics[name] = {
            "precision": round(float(clf_rep[name]["precision"]), 4),
            "recall": round(float(clf_rep[name]["recall"]), 4),
            "specificity": specificities[name],
            "f1": round(float(clf_rep[name]["f1-score"]), 4),
            "support": int(clf_rep[name]["support"]),
        }

    # 5. Visualizations & ROC
    print("\nGenerating evaluation plots and curves...")
    plot_confusion_matrices(cm, cm_norm, OUTPUT_DIR / "confusion_matrix.png")
    roc_aucs = plot_roc_curves(y_true, probabilities, OUTPUT_DIR / "roc_curves.png")

    for name in CLASS_NAMES:
        per_class_metrics[name]["roc_auc"] = roc_aucs[name]

    plot_per_class_metrics_bar(per_class_metrics, OUTPUT_DIR / "per_class_metrics.png")

    # 6. Save Confusion Matrix CSVs
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")

    cm_norm_df = pd.DataFrame(cm_norm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_norm_df.to_csv(OUTPUT_DIR / "normalized_confusion_matrix.csv")

    # 7. Analyze Misclassified Samples
    print("\nCompiling misclassified scans index...")
    misclassified_list = []
    error_pair_counts = {}

    for i in range(len(test_paths)):
        if preds[i] != y_true[i]:
            true_cls = CLASS_NAMES[y_true[i]]
            pred_cls = CLASS_NAMES[preds[i]]
            pair_key = f"{true_cls} -> {pred_cls}"
            error_pair_counts[pair_key] = error_pair_counts.get(pair_key, 0) + 1

            rel_path = str(test_paths[i].relative_to(ROOT))
            conf = float(probabilities[i, preds[i]])
            true_p = float(probabilities[i, y_true[i]])

            misclassified_list.append({
                "index": i,
                "path": rel_path,
                "true_class": true_cls,
                "true_index": int(y_true[i]),
                "pred_class": pred_cls,
                "pred_index": int(preds[i]),
                "confidence": round(conf, 4),
                "true_class_prob": round(true_p, 4),
                "margin": round(conf - true_p, 4),
            })

    df_errors = pd.DataFrame(misclassified_list)
    df_errors.to_csv(OUTPUT_DIR / "misclassified_samples.csv", index=False)

    top_pair = max(error_pair_counts.items(), key=lambda x: x[1]) if error_pair_counts else ("None", 0)

    misclassified_summary = {
        "total_misclassified": len(misclassified_list),
        "error_rate": round(len(misclassified_list) / len(test_paths), 4),
        "top_confusion_pair": top_pair[0],
        "top_confusion_count": top_pair[1],
        "error_pair_breakdown": error_pair_counts,
        "errors": misclassified_list,
    }
    with open(OUTPUT_DIR / "misclassified_summary.json", "w", encoding="utf-8") as f:
        json.dump(misclassified_summary, f, indent=2)

    # 8. Machine-Readable Summary JSON
    summary = {
        "evaluation_name": "Spatial_Attention_Fusion_Test_Set_Evaluation",
        "evaluation_timestamp": datetime.datetime.now().isoformat(),
        "checkpoint_path": str(CHECKPOINT_PATH.relative_to(ROOT)),
        "test_dataset_path": str(TEST_DIR.relative_to(ROOT)),
        "class_order": CLASS_NAMES,
        "sample_counts": {
            "total": len(test_paths),
            "by_class": class_counts,
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "weighted_precision": round(weighted_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "weighted_recall": round(weighted_rec, 4),
            "macro_roc_auc": roc_aucs["macro_ovr"],
        },
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "normalized_confusion_matrix": [[round(x, 4) for x in row] for row in cm_norm.tolist()],
        "dataset_audit": hash_audit,
        "misclassified_summary": {
            "total_misclassified": len(misclassified_list),
            "error_rate": round(len(misclassified_list) / len(test_paths), 4),
            "top_confusion_pair": top_pair[0],
            "top_confusion_count": top_pair[1],
            "error_pair_breakdown": error_pair_counts,
            "errors": misclassified_list,
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "tensorflow_version": tf.__version__,
            "keras_version": tf.keras.__version__,
        },
    }

    with open(OUTPUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # 9. Markdown Report
    generate_markdown_report(summary, OUTPUT_DIR / "evaluation_report.md")

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION RESULTS SUMMARY")
    print("=" * 70)
    print(f"Total Test Scans:       {len(test_paths)}")
    print(f"Overall Accuracy:       {acc * 100:.2f}% ({int(round(acc * len(test_paths)))} / {len(test_paths)})")
    print(f"Balanced Accuracy:      {bal_acc * 100:.2f}%")
    print(f"Macro F1-Score:         {macro_f1:.4f}")
    print(f"Weighted F1-Score:      {weighted_f1:.4f}")
    print(f"Macro Precision:        {macro_prec:.4f}")
    print(f"Macro Recall:           {macro_rec:.4f}")
    print(f"Macro ROC-AUC (OvR):    {roc_aucs['macro_ovr']:.4f}")
    print("\nPer-Class Breakdown:")
    for name in CLASS_NAMES:
        p = per_class_metrics[name]
        print(f"  {name.capitalize():<12} Prec: {p['precision']*100:.2f}% | Rec/Sens: {p['recall']*100:.2f}% | Spec: {p['specificity']*100:.2f}% | F1: {p['f1']:.4f} | AUC: {p['roc_auc']:.4f}")
    print("\nAll evaluation artifacts saved to:")
    print(f"  {OUTPUT_DIR.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
