"""Phase 7: Final Locked Evaluation on Untouched Official Test Set (N=1,600).

CRITICAL RESEARCH INTEGRITY NOTICE:
This evaluation is executed exactly ONCE after all model architecture,
calibration temperatures, and selective prediction thresholds are completely locked.
No tuning, hyperparameter adjustments, or threshold selections are performed on test data.
"""
from __future__ import annotations

import json
import math
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
    roc_auc_score,
    ConfusionMatrixDisplay,
)
import tensorflow as tf

PROJECT_ROOT = Path("X:/Mini Project/brain-tumor-mri-classification")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.compat  # noqa: F401
from config import CLASS_NAMES, DISPLAY_NAMES, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TEST_DIR
from src.data.dataset import get_test_dataset
from src.evaluation.calibration import compute_ece_and_reliability

CHECKPOINT_PATH = MODELS_DIR / "multiscale_fusion_grouped_best.keras"
CALIBRATION_JSON = REPORTS_DIR / "final_calibration_results.json"
SELECTIVE_CSV = REPORTS_DIR / "selective_prediction_results.csv"


def extract_logits(model: tf.keras.Model, test_ds: tf.data.Dataset) -> np.ndarray:
    """Extract penultimate features and compute exact pre-softmax logits."""
    pred_layer = model.get_layer("fusion_predictions")
    weights, biases = pred_layer.get_weights()

    penultimate_submodel = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("classifier_dropout").output,
    )
    features = penultimate_submodel.predict(test_ds, verbose=0)
    logits = np.dot(features, weights) + biases
    return logits


def compute_bootstrap_intervals(
    y_true: np.ndarray,
    preds: np.ndarray,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> dict:
    """Compute 95% bootstrap confidence intervals."""
    rng = np.random.RandomState(seed)
    n = len(y_true)
    boot_acc = []
    boot_f1 = []
    boot_glioma_rec = []
    boot_meningioma_rec = []

    for _ in range(n_bootstraps):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        yp_sample = preds[idx]

        boot_acc.append(accuracy_score(yt_sample, yp_sample))
        boot_f1.append(f1_score(yt_sample, yp_sample, average="macro", zero_division=0))

        # Glioma (class 0) recall
        g_mask = (yt_sample == 0)
        if np.sum(g_mask) > 0:
            boot_glioma_rec.append(np.mean(yp_sample[g_mask] == 0))

        # Meningioma (class 1) recall
        m_mask = (yt_sample == 1)
        if np.sum(m_mask) > 0:
            boot_meningioma_rec.append(np.mean(yp_sample[m_mask] == 1))

    return {
        "n_bootstraps": n_bootstraps,
        "confidence_level": 0.95,
        "accuracy": {
            "mean": round(float(np.mean(boot_acc)), 4),
            "std": round(float(np.std(boot_acc)), 4),
            "ci_lower": round(float(np.percentile(boot_acc, 2.5)), 4),
            "ci_upper": round(float(np.percentile(boot_acc, 97.5)), 4),
        },
        "macro_f1": {
            "mean": round(float(np.mean(boot_f1)), 4),
            "std": round(float(np.std(boot_f1)), 4),
            "ci_lower": round(float(np.percentile(boot_f1, 2.5)), 4),
            "ci_upper": round(float(np.percentile(boot_f1, 97.5)), 4),
        },
        "glioma_recall": {
            "mean": round(float(np.mean(boot_glioma_rec)), 4),
            "std": round(float(np.std(boot_glioma_rec)), 4),
            "ci_lower": round(float(np.percentile(boot_glioma_rec, 2.5)), 4),
            "ci_upper": round(float(np.percentile(boot_glioma_rec, 97.5)), 4),
        },
        "meningioma_recall": {
            "mean": round(float(np.mean(boot_meningioma_rec)), 4),
            "std": round(float(np.std(boot_meningioma_rec)), 4),
            "ci_lower": round(float(np.percentile(boot_meningioma_rec, 2.5)), 4),
            "ci_upper": round(float(np.percentile(boot_meningioma_rec, 97.5)), 4),
        },
    }


def main():
    print("=" * 80)
    print("PHASE 7: FINAL LOCKED TEST EVALUATION (OFFICIAL TEST SET N=1,600)")
    print("=" * 80)

    # 1. Load locked calibration temperature and selective threshold from validation
    if not CALIBRATION_JSON.is_file():
        raise FileNotFoundError(f"Calibration results missing: {CALIBRATION_JSON}")
    with CALIBRATION_JSON.open("r") as f:
        calib_data = json.load(f)
    optimal_t = float(calib_data["optimal_temperature"])
    print(f"Validation-Locked Temperature T*: {optimal_t:.4f}")

    # Determine validation-locked selective prediction threshold (target: high accuracy, high coverage)
    locked_tau = 0.70  # Default operating threshold chosen strictly from validation analysis
    if SELECTIVE_CSV.is_file():
        df_sel = pd.read_csv(SELECTIVE_CSV)
        # Choose threshold that gave >= 90% accuracy on val with coverage >= 80%
        candidates = df_sel[(df_sel["selective_accuracy"] >= 0.90) & (df_sel["coverage"] >= 0.80)]
        if len(candidates) > 0:
            locked_tau = float(candidates.iloc[0]["threshold"])
    print(f"Validation-Locked Selective Threshold tau*: {locked_tau:.2f}")

    # 2. Load Model
    print(f"\nLoading locked checkpoint: {CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)
    print(f"Model parameters: {model.count_params():,}")

    # 3. Load Official Test Set
    print(f"\nLoading official test set from: {TEST_DIR}")
    test_ds, test_paths, test_labels = get_test_dataset(
        test_dir=TEST_DIR,
        model_type="efficientnet",
        batch_size=32,
    )
    y_true = np.array(test_labels)
    assert len(y_true) == 1600, f"Expected 1,600 test images, got {len(y_true)}"

    # 4. Predict
    t0 = time.perf_counter()
    logits = extract_logits(model, test_ds)
    inference_time = time.perf_counter() - t0
    latency_ms = (inference_time / len(y_true)) * 1000
    print(f"Inference completed in {inference_time:.2f}s ({latency_ms:.2f} ms/image)")

    # Probabilities
    raw_probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    raw_probs /= np.sum(raw_probs, axis=1, keepdims=True)

    # Scaled probabilities
    scaled_logits = logits / optimal_t
    calibrated_probs = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
    calibrated_probs /= np.sum(calibrated_probs, axis=1, keepdims=True)

    preds = np.argmax(calibrated_probs, axis=1)
    confidences = np.max(calibrated_probs, axis=1)

    # 5. Full Test Set Evaluation (100% Coverage)
    acc = float(accuracy_score(y_true, preds))
    macro_f1 = float(f1_score(y_true, preds, average="macro"))
    macro_prec = float(precision_score(y_true, preds, average="macro"))
    macro_rec = float(recall_score(y_true, preds, average="macro"))

    # OvR ROC-AUC
    y_one_hot = np.zeros_like(calibrated_probs)
    for i, t in enumerate(y_true):
        y_one_hot[i, t] = 1.0
    roc_auc = float(roc_auc_score(y_one_hot, calibrated_probs, multi_class="ovr", average="macro"))

    per_prec = precision_score(y_true, preds, average=None)
    per_rec = recall_score(y_true, preds, average=None)
    per_f1 = f1_score(y_true, preds, average=None)
    cm = confusion_matrix(y_true, preds)

    # Calibration metrics on test set
    uncal_calib = compute_ece_and_reliability(y_true, raw_probs)
    cal_calib = compute_ece_and_reliability(y_true, calibrated_probs)

    # Bootstrap intervals
    boot_stats = compute_bootstrap_intervals(y_true, preds)

    print("\n" + "=" * 50)
    print("OFFICIAL TEST RESULTS (100% COVERAGE):")
    print("=" * 50)
    print(f"Accuracy:         {acc * 100:.2f}% (95% CI: [{boot_stats['accuracy']['ci_lower']*100:.2f}%, {boot_stats['accuracy']['ci_upper']*100:.2f}%])")
    print(f"Macro-F1:         {macro_f1:.4f} (95% CI: [{boot_stats['macro_f1']['ci_lower']:.4f}, {boot_stats['macro_f1']['ci_upper']:.4f}])")
    print(f"Macro Precision:  {macro_prec:.4f}")
    print(f"Macro Recall:     {macro_rec:.4f}")
    print(f"Macro ROC-AUC:    {roc_auc:.4f}")
    print(f"Uncalibrated ECE: {uncal_calib['ece']:.4f}")
    print(f"Calibrated ECE:   {cal_calib['ece']:.4f} (Brier: {cal_calib['brier_score']:.4f})")

    print("\nPer-Class Performance on Official Test Set:")
    print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support'}")
    print("-" * 55)
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:<15} {per_prec[i]:<10.4f} {per_rec[i]:<10.4f} {per_f1[i]:<10.4f} {np.sum(y_true == i)}")

    print("\nConfusion Matrix (Official Test Set):")
    print(cm)

    # 6. Selective Prediction on Official Test Set
    accepted_idx = np.where(confidences >= locked_tau)[0]
    rejected_idx = np.where(confidences < locked_tau)[0]
    test_coverage = len(accepted_idx) / len(y_true)

    sel_acc = float(accuracy_score(y_true[accepted_idx], preds[accepted_idx]))
    sel_f1 = float(f1_score(y_true[accepted_idx], preds[accepted_idx], average="macro", zero_division=0))
    sel_prec = float(precision_score(y_true[accepted_idx], preds[accepted_idx], average="macro", zero_division=0))
    sel_rec = float(recall_score(y_true[accepted_idx], preds[accepted_idx], average="macro", zero_division=0))

    print("\n" + "=" * 50)
    print(f"CALIBRATED SELECTIVE PREDICTION (tau* = {locked_tau:.2f}):")
    print("=" * 50)
    print(f"Coverage:           {test_coverage * 100:.2f}% ({len(accepted_idx)} accepted, {len(rejected_idx)} referred to radiologist)")
    print(f"Selective Accuracy: {sel_acc * 100:.2f}%  (Improvement: +{(sel_acc - acc)*100:.2f}%)")
    print(f"Selective Macro-F1: {sel_f1:.4f}  (Improvement: +{(sel_f1 - macro_f1):.4f})")

    # 7. Save All Deliverables
    # Classification Report CSV
    clf_report_df = pd.DataFrame(classification_report(y_true, preds, target_names=CLASS_NAMES, output_dict=True)).T
    clf_report_path = REPORTS_DIR / "final_classification_report.csv"
    clf_report_df.to_csv(clf_report_path)

    # Confusion Matrix Image
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
    ax.set_title(f"Final Multi-Scale Fusion Test Confusion Matrix\nAccuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}", fontweight="bold")
    plt.tight_layout()
    cm_fig_path = REPORTS_DIR / "final_confusion_matrix.png"
    plt.savefig(cm_fig_path, dpi=300)
    plt.close()

    # Confusion Matrix CSV
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_csv_path = REPORTS_DIR / "final_confusion_matrix.csv"
    cm_df.to_csv(cm_csv_path)

    # Test Predictions CSV
    pred_records = []
    for i in range(len(y_true)):
        pred_records.append({
            "image_path": str(test_paths[i]),
            "image_name": test_paths[i].name,
            "true_label": CLASS_NAMES[y_true[i]],
            "predicted_label": CLASS_NAMES[preds[i]],
            "is_correct": bool(y_true[i] == preds[i]),
            "calibrated_confidence": round(float(confidences[i]), 4),
            "selective_status": "Accepted" if i in accepted_idx else "Referred_to_Radiologist",
            "prob_glioma": round(float(calibrated_probs[i, 0]), 4),
            "prob_meningioma": round(float(calibrated_probs[i, 1]), 4),
            "prob_notumor": round(float(calibrated_probs[i, 2]), 4),
            "prob_pituitary": round(float(calibrated_probs[i, 3]), 4),
        })
    df_preds = pd.DataFrame(pred_records)
    test_preds_path = REPORTS_DIR / "final_test_predictions.csv"
    df_preds.to_csv(test_preds_path, index=False)

    # Final Experiment Summary JSON
    summary_data = {
        "project_title": "BRAIN TUMOR MRI IMAGE CLASSIFICATION USING EFFICIENTNET-B0 WITH GRAD-CAM",
        "selected_architecture": "Multi-Scale Intermediate & Deep Feature Fusion (EfficientNet-B0 backbone)",
        "feature_layers": {
            "intermediate": "block6d_add (192-dim, 14x14 spatial)",
            "deep": "top_activation (1280-dim, 7x7 spatial)",
            "fused_vector_dim": 1472,
            "dense_head": "Dropout(0.35) -> Dense(256, relu) -> Dropout(0.35) -> Dense(4, softmax)",
        },
        "selected_checkpoint": str(CHECKPOINT_PATH),
        "dataset_split": "Duplicate-aware similarity grouping (pHash distance <= 4 within classes)",
        "training_samples": 4447,
        "validation_samples": 1153,
        "official_test_samples": 1600,
        "calibration": {
            "method": "Post-hoc Temperature Scaling strictly on validation logits",
            "optimal_temperature_T": optimal_t,
            "test_uncalibrated_ece": uncal_calib["ece"],
            "test_calibrated_ece": cal_calib["ece"],
            "test_calibrated_brier": cal_calib["brier_score"],
        },
        "official_test_metrics_full_coverage": {
            "accuracy": round(acc, 4),
            "macro_f1": round(macro_f1, 4),
            "macro_precision": round(macro_prec, 4),
            "macro_recall": round(macro_rec, 4),
            "macro_roc_auc": round(roc_auc, 4),
            "per_class": {
                cls: {
                    "precision": round(float(per_prec[i]), 4),
                    "recall": round(float(per_rec[i]), 4),
                    "f1": round(float(per_f1[i]), 4),
                    "support": int(np.sum(y_true == i)),
                }
                for i, cls in enumerate(CLASS_NAMES)
            },
            "confusion_matrix": cm.tolist(),
            "bootstrap_95ci": boot_stats,
        },
        "calibrated_selective_prediction": {
            "operating_threshold_tau": locked_tau,
            "test_coverage": round(test_coverage, 4),
            "accepted_samples": len(accepted_idx),
            "referred_samples": len(rejected_idx),
            "selective_accuracy": round(sel_acc, 4),
            "selective_macro_f1": round(sel_f1, 4),
            "accuracy_improvement": round((sel_acc - acc) * 100, 2),
        },
        "comparison_with_baseline_efficientnet_b0": {
            "baseline_test_accuracy": 0.8750,
            "baseline_test_macro_f1": 0.8699,
            "baseline_glioma_recall": 0.6575,
            "multiscale_test_accuracy": round(acc, 4),
            "multiscale_test_macro_f1": round(macro_f1, 4),
            "multiscale_glioma_recall": round(float(per_rec[0]), 4),
            "selective_accuracy": round(sel_acc, 4),
            "selective_macro_f1": round(sel_f1, 4),
        },
        "clinical_limitations": [
            "Educational research prototype; NOT approved for clinical diagnostics.",
            "Selective prediction refers ambiguous / low-confidence scans for neuroradiologist review.",
            "Grad-CAM heatmaps highlight model attribution, not anatomical lesion segmentation.",
            "Dataset lacks institutional patient IDs, meaning patient-level slice clustering remains an unavoidable empirical limitation.",
        ]
    }

    summary_path = REPORTS_DIR / "final_experiment_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary_data, f, indent=2)

    print(f"\nAll final deliverables successfully generated and saved:")
    print(f"  {summary_path}")
    print(f"  {clf_report_path}")
    print(f"  {cm_fig_path}")
    print(f"  {cm_csv_path}")
    print(f"  {test_preds_path}")


if __name__ == "__main__":
    main()
