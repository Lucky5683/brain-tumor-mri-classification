"""
Final Evaluation Script: Selected Multi-Scale Fusion (Seed 42) vs Baseline EfficientNet-B0.

Evaluates both models on the official dataset/Testing split (N=1,600).
Generates comprehensive comparative metrics, classification reports,
confusion matrices, ROC curves, and detailed per-class breakdowns.

Note on research integrity:
An earlier checkpoint was previously evaluated on this test set during historical baseline logging.
This evaluation is executed to benchmark the validation-selected multi-seed model
against the original reference baseline, without model selection or hyperparameter tuning.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# CPU threading configuration
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
    auc,
    ConfusionMatrixDisplay,
)
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.compat  # noqa: F401
from config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    TEST_DIR,
)
from src.data.dataset import get_test_dataset

# Paths
SELECTED_CHECKPOINT = (
    ROOT / "experiments" / "multiseed_fusion" / "seed_42" / "best_model.keras"
)
BASELINE_CHECKPOINT = ROOT / "efficientnet_b0.keras"

OUTPUT_DIR = ROOT / "reports" / "final_multiseed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    y_true: np.ndarray,
    model_name: str,
) -> dict:
    """Evaluate model on test dataset and compute all metrics."""
    start_time = time.time()
    probabilities = model.predict(test_ds, verbose=1)
    inference_time = time.time() - start_time
    predictions = np.argmax(probabilities, axis=1)

    acc = float(accuracy_score(y_true, predictions))
    bal_acc = float(balanced_accuracy_score(y_true, predictions))

    macro_prec = float(precision_score(y_true, predictions, average="macro", zero_division=0))
    weighted_prec = float(precision_score(y_true, predictions, average="weighted", zero_division=0))

    macro_rec = float(recall_score(y_true, predictions, average="macro", zero_division=0))
    weighted_rec = float(recall_score(y_true, predictions, average="weighted", zero_division=0))

    macro_f1 = float(f1_score(y_true, predictions, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, predictions, average="weighted", zero_division=0))

    # ROC-AUC computation (One-vs-Rest)
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=len(CLASS_NAMES))
    try:
        roc_auc_macro = float(roc_auc_score(y_true_oh, probabilities, average="macro", multi_class="ovr"))
        roc_auc_weighted = float(roc_auc_score(y_true_oh, probabilities, average="weighted", multi_class="ovr"))
    except Exception as e:
        print(f"ROC-AUC warning for {model_name}: {e}")
        roc_auc_macro = None
        roc_auc_weighted = None

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    per_class = {}
    for idx, cname in enumerate(CLASS_NAMES):
        try:
            auc_class = float(roc_auc_score(y_true_oh[:, idx], probabilities[:, idx]))
        except Exception:
            auc_class = None

        per_class[cname] = {
            "display_name": DISPLAY_NAMES[cname],
            "precision": float(report[cname]["precision"]),
            "recall": float(report[cname]["recall"]),
            "f1_score": float(report[cname]["f1-score"]),
            "support": int(report[cname]["support"]),
            "roc_auc_ovr": auc_class,
        }

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )

    return {
        "model_name": model_name,
        "inference_seconds": round(inference_time, 2),
        "ms_per_image": round((inference_time / len(y_true)) * 1000, 2),
        "metrics": {
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_macro": macro_prec,
            "precision_weighted": weighted_prec,
            "recall_macro": macro_rec,
            "recall_weighted": weighted_rec,
            "f1_macro": macro_f1,
            "f1_weighted": weighted_f1,
            "roc_auc_macro": roc_auc_macro,
            "roc_auc_weighted": roc_auc_weighted,
        },
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "probabilities": probabilities,
        "predictions": predictions,
        "classification_report_dict": report,
    }


def plot_confusion_matrix(cm: np.ndarray, model_title: str, output_path: Path):
    """Plot and save confusion matrix with high visual quality."""
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    disp_labels = [DISPLAY_NAMES[c] for c in CLASS_NAMES]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=disp_labels)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=True)
    ax.set_title(f"Confusion Matrix: {model_title}\n(Official Test Set, N=1,600)", fontsize=12, pad=12, fontweight="bold")
    ax.set_xlabel("Predicted Class", fontsize=11, fontweight="bold")
    ax.set_ylabel("True Class", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion matrix plot: {output_path}")


def plot_combined_roc_curves(
    results_dict: dict[str, dict],
    y_true: np.ndarray,
    output_path: Path,
):
    """Plot multi-class ROC curves comparing both models."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=len(CLASS_NAMES))
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    for ax_idx, (m_id, res) in enumerate(results_dict.items()):
        ax = axes[ax_idx]
        probs = res["probabilities"]

        for idx, cname in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_true_oh[:, idx], probs[:, idx])
            roc_val = auc(fpr, tpr)
            ax.plot(
                fpr, tpr,
                color=colors[idx],
                lw=2,
                label=f"{DISPLAY_NAMES[cname]} (AUC = {roc_val:.4f})"
            )

        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6, label="Random Chance (0.50)")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11, fontweight="bold")
        macro_auc_str = f"{res['metrics']['roc_auc_macro']:.4f}" if res['metrics']['roc_auc_macro'] else "N/A"
        ax.set_title(
            f"{res['model_name']}\nMacro ROC-AUC = {macro_auc_str}",
            fontsize=12,
            fontweight="bold",
            pad=10
        )
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
        ax.grid(True, linestyle=":", alpha=0.6)

    plt.suptitle("Multi-Class One-vs-Rest ROC Curves Comparison (Official Test Set N=1,600)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved ROC curves plot: {output_path}")


def main():
    print("=" * 70)
    print("FINAL TEST EVALUATION: MULTI-SCALE FUSION (SEED 42) vs BASELINE")
    print("=" * 70)

    if not SELECTED_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Selected checkpoint not found: {SELECTED_CHECKPOINT}")
    if not BASELINE_CHECKPOINT.is_file():
        raise FileNotFoundError(f"Baseline checkpoint not found: {BASELINE_CHECKPOINT}")

    print(f"Selected Model: {SELECTED_CHECKPOINT.name} (Seed 42)")
    print(f"Baseline Model: {BASELINE_CHECKPOINT.name}")
    print(f"Test Directory: {TEST_DIR}")

    # Load official test set
    test_ds, test_paths, test_labels = get_test_dataset(
        test_dir=TEST_DIR,
        model_type="efficientnet",
        batch_size=32,
    )
    y_true = np.array(test_labels, dtype=np.int32)
    print(f"\nLoaded {len(test_paths)} official test images across {len(CLASS_NAMES)} classes.")
    print(f"Class support: {dict(pd.Series(y_true).value_counts().sort_index())}")

    # 1. Evaluate Selected Multi-Scale Fusion Model
    print("\n[1/2] Loading and evaluating Selected Multi-Scale Fusion Checkpoint...")
    fusion_model = tf.keras.models.load_model(SELECTED_CHECKPOINT, compile=False)
    fusion_results = evaluate_model(
        model=fusion_model,
        test_ds=test_ds,
        y_true=y_true,
        model_name="Multi-Scale Feature Fusion (Seed 42 Best)",
    )

    # 2. Evaluate Baseline EfficientNet-B0 Model
    print("\n[2/2] Loading and evaluating Baseline EfficientNet-B0 Checkpoint...")
    baseline_model = tf.keras.models.load_model(BASELINE_CHECKPOINT, compile=False)
    baseline_results = evaluate_model(
        model=baseline_model,
        test_ds=test_ds,
        y_true=y_true,
        model_name="EfficientNet-B0 (Reference Baseline)",
    )

    # Save Classification Reports
    pd.DataFrame(fusion_results["classification_report_dict"]).transpose().to_csv(
        OUTPUT_DIR / "selected_fusion_classification_report.csv"
    )
    with (OUTPUT_DIR / "selected_fusion_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(fusion_results["classification_report_dict"], f, indent=2)

    pd.DataFrame(baseline_results["classification_report_dict"]).transpose().to_csv(
        OUTPUT_DIR / "baseline_effnet_classification_report.csv"
    )
    with (OUTPUT_DIR / "baseline_effnet_classification_report.json").open("w", encoding="utf-8") as f:
        json.dump(baseline_results["classification_report_dict"], f, indent=2)

    # Save Confusion Matrices as CSV
    np.savetxt(
        OUTPUT_DIR / "selected_fusion_confusion_matrix.csv",
        np.array(fusion_results["confusion_matrix"]),
        delimiter=",",
        fmt="%d",
    )
    np.savetxt(
        OUTPUT_DIR / "baseline_effnet_confusion_matrix.csv",
        np.array(baseline_results["confusion_matrix"]),
        delimiter=",",
        fmt="%d",
    )

    # Plot Confusion Matrices
    plot_confusion_matrix(
        np.array(fusion_results["confusion_matrix"]),
        "Multi-Scale Feature Fusion (Seed 42)",
        OUTPUT_DIR / "selected_fusion_confusion_matrix.png",
    )
    plot_confusion_matrix(
        np.array(baseline_results["confusion_matrix"]),
        "EfficientNet-B0 Baseline",
        OUTPUT_DIR / "baseline_effnet_confusion_matrix.png",
    )

    # Plot Combined ROC Curves
    eval_dict = {
        "fusion": fusion_results,
        "baseline": baseline_results,
    }
    plot_combined_roc_curves(eval_dict, y_true, OUTPUT_DIR / "roc_curve_comparison.png")

    # Generate Model Comparison Table
    metrics_comp = [
        {
            "Metric": "Test Accuracy",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['metrics']['accuracy']:.4f} ({fusion_results['metrics']['accuracy']*100:.2f}%)",
            "EfficientNet-B0 Baseline": f"{baseline_results['metrics']['accuracy']:.4f} ({baseline_results['metrics']['accuracy']*100:.2f}%)",
            "Delta (Fusion - Baseline)": f"{(fusion_results['metrics']['accuracy'] - baseline_results['metrics']['accuracy'])*100:+.2f}%",
        },
        {
            "Metric": "Balanced Accuracy",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['metrics']['balanced_accuracy']:.4f}",
            "EfficientNet-B0 Baseline": f"{baseline_results['metrics']['balanced_accuracy']:.4f}",
            "Delta (Fusion - Baseline)": f"{(fusion_results['metrics']['balanced_accuracy'] - baseline_results['metrics']['balanced_accuracy'])*100:+.2f}%",
        },
        {
            "Metric": "Macro-F1 Score",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['metrics']['f1_macro']:.4f}",
            "EfficientNet-B0 Baseline": f"{baseline_results['metrics']['f1_macro']:.4f}",
            "Delta (Fusion - Baseline)": f"{fusion_results['metrics']['f1_macro'] - baseline_results['metrics']['f1_macro']:+.4f}",
        },
        {
            "Metric": "Weighted-F1 Score",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['metrics']['f1_weighted']:.4f}",
            "EfficientNet-B0 Baseline": f"{baseline_results['metrics']['f1_weighted']:.4f}",
            "Delta (Fusion - Baseline)": f"{fusion_results['metrics']['f1_weighted'] - baseline_results['metrics']['f1_weighted']:+.4f}",
        },
        {
            "Metric": "Macro Precision",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['metrics']['precision_macro']:.4f}",
            "EfficientNet-B0 Baseline": f"{baseline_results['metrics']['precision_macro']:.4f}",
            "Delta (Fusion - Baseline)": f"{fusion_results['metrics']['precision_macro'] - baseline_results['metrics']['precision_macro']:+.4f}",
        },
        {
            "Metric": "Macro Recall",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['metrics']['recall_macro']:.4f}",
            "EfficientNet-B0 Baseline": f"{baseline_results['metrics']['recall_macro']:.4f}",
            "Delta (Fusion - Baseline)": f"{fusion_results['metrics']['recall_macro'] - baseline_results['metrics']['recall_macro']:+.4f}",
        },
        {
            "Metric": "Macro ROC-AUC (OvR)",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['metrics']['roc_auc_macro']:.4f}" if fusion_results['metrics']['roc_auc_macro'] else "N/A",
            "EfficientNet-B0 Baseline": f"{baseline_results['metrics']['roc_auc_macro']:.4f}" if baseline_results['metrics']['roc_auc_macro'] else "N/A",
            "Delta (Fusion - Baseline)": f"{fusion_results['metrics']['roc_auc_macro'] - baseline_results['metrics']['roc_auc_macro']:+.4f}" if fusion_results['metrics']['roc_auc_macro'] and baseline_results['metrics']['roc_auc_macro'] else "N/A",
        },
        {
            "Metric": "Glioma Recall",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['per_class']['glioma']['recall']:.4f} ({fusion_results['per_class']['glioma']['recall']*100:.2f}%)",
            "EfficientNet-B0 Baseline": f"{baseline_results['per_class']['glioma']['recall']:.4f} ({baseline_results['per_class']['glioma']['recall']*100:.2f}%)",
            "Delta (Fusion - Baseline)": f"{(fusion_results['per_class']['glioma']['recall'] - baseline_results['per_class']['glioma']['recall'])*100:+.2f}%",
        },
        {
            "Metric": "Meningioma Recall",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['per_class']['meningioma']['recall']:.4f} ({fusion_results['per_class']['meningioma']['recall']*100:.2f}%)",
            "EfficientNet-B0 Baseline": f"{baseline_results['per_class']['meningioma']['recall']:.4f} ({baseline_results['per_class']['meningioma']['recall']*100:.2f}%)",
            "Delta (Fusion - Baseline)": f"{(fusion_results['per_class']['meningioma']['recall'] - baseline_results['per_class']['meningioma']['recall'])*100:+.2f}%",
        },
        {
            "Metric": "No Tumor Recall",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['per_class']['notumor']['recall']:.4f} ({fusion_results['per_class']['notumor']['recall']*100:.2f}%)",
            "EfficientNet-B0 Baseline": f"{baseline_results['per_class']['notumor']['recall']:.4f} ({baseline_results['per_class']['notumor']['recall']*100:.2f}%)",
            "Delta (Fusion - Baseline)": f"{(fusion_results['per_class']['notumor']['recall'] - baseline_results['per_class']['notumor']['recall'])*100:+.2f}%",
        },
        {
            "Metric": "Pituitary Recall",
            "Multi-Scale Fusion (Seed 42)": f"{fusion_results['per_class']['pituitary']['recall']:.4f} ({fusion_results['per_class']['pituitary']['recall']*100:.2f}%)",
            "EfficientNet-B0 Baseline": f"{baseline_results['per_class']['pituitary']['recall']:.4f} ({baseline_results['per_class']['pituitary']['recall']*100:.2f}%)",
            "Delta (Fusion - Baseline)": f"{(fusion_results['per_class']['pituitary']['recall'] - baseline_results['per_class']['pituitary']['recall'])*100:+.2f}%",
        },
    ]

    comp_df = pd.DataFrame(metrics_comp)
    comp_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

    # Save detailed comparison JSON
    summary_data = {
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "test_dataset_path": str(TEST_DIR),
        "total_test_samples": len(y_true),
        "selected_fusion_model": {
            "checkpoint": str(SELECTED_CHECKPOINT),
            "inference_seconds": fusion_results["inference_seconds"],
            "ms_per_image": fusion_results["ms_per_image"],
            "metrics": fusion_results["metrics"],
            "per_class": fusion_results["per_class"],
            "confusion_matrix": fusion_results["confusion_matrix"],
        },
        "baseline_model": {
            "checkpoint": str(BASELINE_CHECKPOINT),
            "inference_seconds": baseline_results["inference_seconds"],
            "ms_per_image": baseline_results["ms_per_image"],
            "metrics": baseline_results["metrics"],
            "per_class": baseline_results["per_class"],
            "confusion_matrix": baseline_results["confusion_matrix"],
        },
        "comparison_table": metrics_comp,
        "methodological_caveats": [
            "Baseline EfficientNet-B0 was historically evaluated on this official test set prior to this experiment.",
            "Multi-scale fusion model was selected strictly on validation Macro-F1 across 3 seeds on the duplicate-aware manifest.",
            "Perceptual hash grouping in training manifest reduces slice overlap but does not guarantee patient-level independence."
        ]
    }

    with (OUTPUT_DIR / "final_test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)

    # Save CSV of per-sample predictions for error analysis
    test_pred_df = pd.DataFrame({
        "image_path": [str(p) for p in test_paths],
        "true_label": [CLASS_NAMES[y] for y in y_true],
        "fusion_pred": [CLASS_NAMES[p] for p in fusion_results["predictions"]],
        "fusion_conf": [float(np.max(prob)) for prob in fusion_results["probabilities"]],
        "baseline_pred": [CLASS_NAMES[p] for p in baseline_results["predictions"]],
        "baseline_conf": [float(np.max(prob)) for prob in baseline_results["probabilities"]],
    })
    test_pred_df.to_csv(OUTPUT_DIR / "test_predictions_comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("FINAL TEST EVALUATION COMPLETE")
    print("=" * 70)
    print(comp_df.to_string(index=False))
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
