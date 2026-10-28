"""Phase 5: Reliability-Aware Multi-Scale Brain Tumor MRI Classification.

Implements and evaluates:
1. Post-hoc Temperature Scaling strictly on validation logits.
2. Expected Calibration Error (ECE), MCE, and Brier Score before and after calibration.
3. Calibrated Selective Prediction (Abstention / Triage) sweeping coverage vs. accuracy.
4. Grad-CAM Explanation Stability under mild, label-preserving perturbations.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import pearsonr
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
from config import CLASS_NAMES, EXPERIMENTS_DIR, FIGURES_DIR, MODELS_DIR, REPORTS_DIR, TRAIN_DIR
from src.data.dataset import get_train_val_datasets
from src.evaluation.calibration import compute_ece_and_reliability, TemperatureScaler
from src.evaluation.robustness import (
    apply_brightness_perturbation,
    apply_contrast_perturbation,
    apply_gaussian_noise,
)

CHECKPOINT_PATH = MODELS_DIR / "multiscale_fusion_grouped_best.keras"
MANIFEST_PATH = EXPERIMENTS_DIR / "duplicate_aware_split" / "training_manifest.csv"


def extract_logits_and_features(model: tf.keras.Model, val_ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Extract penultimate features and compute exact pre-softmax logits."""
    # The output layer is 'fusion_predictions' (Dense with softmax)
    pred_layer = model.get_layer("fusion_predictions")
    weights, biases = pred_layer.get_weights()  # weights shape: (256, 4), biases shape: (4,)

    # Submodel up to classifier_dropout (the input to fusion_predictions)
    penultimate_submodel = tf.keras.Model(
        inputs=model.inputs,
        outputs=model.get_layer("classifier_dropout").output,
    )

    features = penultimate_submodel.predict(val_ds, verbose=0)
    logits = np.dot(features, weights) + biases
    return logits, features


def evaluate_selective_prediction(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    thresholds: list[float] | None = None,
) -> pd.DataFrame:
    """Evaluate accuracy, macro-F1, and class recalls at varying confidence thresholds."""
    if thresholds is None:
        thresholds = [0.0, 0.40, 0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

    confidences = np.max(calibrated_probs, axis=1)
    predictions = np.argmax(calibrated_probs, axis=1)

    records = []
    total_samples = len(y_true)

    for tau in thresholds:
        accepted_idx = np.where(confidences >= tau)[0]
        rejected_idx = np.where(confidences < tau)[0]

        coverage = len(accepted_idx) / total_samples

        if len(accepted_idx) == 0:
            continue

        acc = float(accuracy_score(y_true[accepted_idx], predictions[accepted_idx]))
        macro_f1 = float(f1_score(y_true[accepted_idx], predictions[accepted_idx], average="macro", zero_division=0))
        macro_prec = float(precision_score(y_true[accepted_idx], predictions[accepted_idx], average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true[accepted_idx], predictions[accepted_idx], average="macro", zero_division=0))

        # Per-class recall on accepted set
        per_rec = recall_score(y_true[accepted_idx], predictions[accepted_idx], average=None, zero_division=0, labels=[0, 1, 2, 3])

        # Rejected composition
        rejected_by_class = {cls: 0 for cls in CLASS_NAMES}
        if len(rejected_idx) > 0:
            for true_cls in y_true[rejected_idx]:
                rejected_by_class[CLASS_NAMES[true_cls]] += 1

        rec = {
            "threshold": tau,
            "coverage": round(coverage, 4),
            "accepted_count": len(accepted_idx),
            "rejected_count": len(rejected_idx),
            "selective_accuracy": round(acc, 4),
            "selective_macro_f1": round(macro_f1, 4),
            "selective_macro_precision": round(macro_prec, 4),
            "selective_macro_recall": round(macro_rec, 4),
            "glioma_recall": round(float(per_rec[0]), 4),
            "meningioma_recall": round(float(per_rec[1]), 4),
            "notumor_recall": round(float(per_rec[2]), 4),
            "pituitary_recall": round(float(per_rec[3]), 4),
            "rejected_glioma": rejected_by_class["glioma"],
            "rejected_meningioma": rejected_by_class["meningioma"],
            "rejected_notumor": rejected_by_class["notumor"],
            "rejected_pituitary": rejected_by_class["pituitary"],
        }
        records.append(rec)

    return pd.DataFrame(records)


def compute_gradcam_heatmap(
    model: tf.keras.Model,
    img_rgb: np.ndarray,
    target_class_idx: int,
) -> np.ndarray:
    """Compute Grad-CAM heatmap for the multiscale fusion model with respect to top_activation."""
    # In multiscale fusion, feature_extractor produces [block6d_add, top_activation]
    # We can inspect gradients of the model output w.r.t the feature extractor's top_activation
    feature_extractor = model.get_layer("multiscale_feature_extractor")

    # We create a gradient tape submodel from image input to top_activation and final prediction
    # Or run tape through the whole model watching the feature tensor:
    img_tensor = tf.convert_to_tensor(img_rgb[np.newaxis, ...], dtype=tf.float32)

    with tf.GradientTape() as tape:
        intermediate_features, final_features = feature_extractor(img_tensor)
        tape.watch(final_features)

        # Follow the remaining layers
        intermediate_vector = model.get_layer("intermediate_pool")(intermediate_features)
        final_vector = model.get_layer("final_pool")(final_features)
        fused = model.get_layer("multiscale_fusion")([intermediate_vector, final_vector])
        x = model.get_layer("fusion_dense")(fused)
        preds = model.get_layer("fusion_predictions")(x)
        class_score = preds[0, target_class_idx]

    grads = tape.gradient(class_score, final_features)
    if grads is None:
        return np.zeros((224, 224), dtype=np.float32)

    # Global average pooling of gradients
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight the feature maps
    final_features_val = final_features[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, final_features_val), axis=-1).numpy()

    # ReLU and normalization
    heatmap = np.maximum(heatmap, 0.0)
    max_val = np.max(heatmap)
    if max_val > 1e-8:
        heatmap /= max_val

    # Resize to original image size 224x224
    heatmap_resized = cv2.resize(heatmap, (224, 224), interpolation=cv2.INTER_LINEAR)
    return heatmap_resized.astype(np.float32)


def evaluate_explanation_stability(
    model: tf.keras.Model,
    image_paths: list[Path],
    ground_truths: list[int],
    calibrated_probs: np.ndarray,
    scaler: TemperatureScaler,
) -> tuple[pd.DataFrame, dict]:
    """Test explanation stability under mild perturbations for high-confidence vs uncertain samples."""
    results = []

    perturbation_fns = {
        "contrast_+15%": lambda img: apply_contrast_perturbation(img, 1.15),
        "brightness_+10%": lambda img: apply_brightness_perturbation(img, 0.10),
        "gaussian_noise_sigma10": lambda img: apply_gaussian_noise(img, sigma=10.0),
    }

    for idx, img_path in enumerate(image_paths):
        true_cls = ground_truths[idx]
        probs = calibrated_probs[idx]
        pred_cls = int(np.argmax(probs))
        conf = float(np.max(probs))
        is_correct = (pred_cls == true_cls)

        # Load clean image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_224 = cv2.resize(img_rgb, (224, 224), interpolation=cv2.INTER_AREA)

        # Baseline Grad-CAM
        clean_heatmap = compute_gradcam_heatmap(model, img_rgb_224, pred_cls)
        clean_mask = (clean_heatmap >= 0.50).astype(float)

        for pert_name, pert_fn in perturbation_fns.items():
            pert_img = pert_fn(img_rgb_224)
            pert_heatmap = compute_gradcam_heatmap(model, pert_img, pred_cls)
            pert_mask = (pert_heatmap >= 0.50).astype(float)

            # Pearson correlation of continuous heatmaps
            r_val, _ = pearsonr(clean_heatmap.flatten(), pert_heatmap.flatten())
            if math.isnan(r_val):
                r_val = 0.0

            # Spatial IoU of top activation mask
            intersection = np.sum((clean_mask > 0) & (pert_mask > 0))
            union = np.sum((clean_mask > 0) | (pert_mask > 0))
            iou = float(intersection / union) if union > 0 else 1.0

            results.append({
                "image_name": img_path.name,
                "true_label": CLASS_NAMES[true_cls],
                "pred_label": CLASS_NAMES[pred_cls],
                "confidence": round(conf, 4),
                "is_correct": is_correct,
                "confidence_group": "High (>=0.80)" if conf >= 0.80 else "Borderline (<0.80)",
                "perturbation": pert_name,
                "pearson_r": round(float(r_val), 4),
                "mask_iou": round(float(iou), 4),
            })

    df = pd.DataFrame(results)

    # Summary by confidence group
    summary = {}
    for group in ["High (>=0.80)", "Borderline (<0.80)"]:
        group_df = df[df["confidence_group"] == group]
        if len(group_df) > 0:
            summary[group] = {
                "mean_pearson_r": round(float(group_df["pearson_r"].mean()), 4),
                "std_pearson_r": round(float(group_df["pearson_r"].std()), 4),
                "mean_mask_iou": round(float(group_df["mask_iou"].mean()), 4),
                "std_mask_iou": round(float(group_df["mask_iou"].std()), 4),
                "count": len(group_df),
            }

    return df, summary


def main():
    print("=" * 75)
    print("PHASE 5: CALIBRATED SELECTIVE PREDICTION & EXPLANATION STABILITY")
    print("=" * 75)

    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model = tf.keras.models.load_model(CHECKPOINT_PATH, compile=False)

    print(f"Loading validation dataset from manifest: {MANIFEST_PATH}")
    _, val_ds, _, _, val_paths, val_labels = get_train_val_datasets(
        train_dir=TRAIN_DIR,
        model_type="efficientnet",
        batch_size=32,
        use_augmentation=False,
        split_manifest=MANIFEST_PATH,
    )
    y_true = np.array(val_labels)

    # 1. Extract logits & compute calibration
    print("\n--- Extracting Logits and Computing Calibration ---")
    val_logits, _ = extract_logits_and_features(model, val_ds)

    # Uncalibrated probabilities
    raw_probs = np.exp(val_logits - np.max(val_logits, axis=1, keepdims=True))
    raw_probs /= np.sum(raw_probs, axis=1, keepdims=True)

    uncal_calib = compute_ece_and_reliability(y_true, raw_probs)
    print(f"Uncalibrated Validation ECE: {uncal_calib['ece']:.4f} | MCE: {uncal_calib['mce']:.4f} | Brier: {uncal_calib['brier_score']:.4f}")

    # Fit temperature scaler strictly on validation logits
    scaler = TemperatureScaler()
    optimal_t = scaler.fit(val_logits, y_true)
    print(f"Fitted Optimal Temperature T*: {optimal_t:.4f}")

    calibrated_probs = scaler.scale_probabilities(val_logits)
    cal_calib = compute_ece_and_reliability(y_true, calibrated_probs)
    print(f"Calibrated Validation ECE:   {cal_calib['ece']:.4f} | MCE: {cal_calib['mce']:.4f} | Brier: {cal_calib['brier_score']:.4f}")

    # Save calibration results
    calib_summary = {
        "optimal_temperature": round(optimal_t, 4),
        "uncalibrated": {
            "ece": uncal_calib["ece"],
            "mce": uncal_calib["mce"],
            "brier_score": uncal_calib["brier_score"],
        },
        "calibrated": {
            "ece": cal_calib["ece"],
            "mce": cal_calib["mce"],
            "brier_score": cal_calib["brier_score"],
        },
        "ece_reduction_percent": round((uncal_calib["ece"] - cal_calib["ece"]) / uncal_calib["ece"] * 100, 2),
    }
    with (REPORTS_DIR / "final_calibration_results.json").open("w") as f:
        json.dump(calib_summary, f, indent=2)

    # Plot reliability diagram comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title in zip(axes, [uncal_calib, cal_calib], [f"Uncalibrated (T=1.00, ECE={uncal_calib['ece']:.4f})", f"Temperature Scaled (T={optimal_t:.2f}, ECE={cal_calib['ece']:.4f})"]):
        bins = data["bins"]
        confidences = [b["confidence"] for b in bins if b["confidence"] is not None]
        accuracies = [b["accuracy"] for b in bins if b["accuracy"] is not None]
        counts = [b["count"] for b in bins if b["confidence"] is not None]

        ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
        ax.bar(confidences, accuracies, width=0.08, alpha=0.6, color="#2E86AB", edgecolor="black", label="Outputs")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Empirical Accuracy")
        ax.set_title(title, fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="upper left")

    plt.tight_layout()
    fig_path = REPORTS_DIR / "figures" / "calibration_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved calibration figure to: {fig_path}")

    # 2. Calibrated Selective Prediction Analysis
    print("\n--- Evaluating Calibrated Selective Prediction (Abstention) ---")
    selective_df = evaluate_selective_prediction(y_true, calibrated_probs)
    selective_csv = REPORTS_DIR / "selective_prediction_results.csv"
    selective_df.to_csv(selective_csv, index=False)
    print(f"Saved selective prediction table to: {selective_csv}")
    print(selective_df[["threshold", "coverage", "selective_accuracy", "selective_macro_f1", "glioma_recall", "meningioma_recall"]].to_string(index=False))

    # Plot Risk-Coverage curve
    fig, ax1 = plt.subplots(figsize=(8, 5))
    color = "tab:blue"
    ax1.set_xlabel("Confidence Threshold (tau)", fontweight="bold")
    ax1.set_ylabel("Coverage (Accepted Ratio)", color=color, fontweight="bold")
    ax1.plot(selective_df["threshold"], selective_df["coverage"], color=color, marker="o", linewidth=2, label="Coverage")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2 = ax1.twinx()
    color = "tab:green"
    ax2.set_ylabel("Selective Accuracy", color=color, fontweight="bold")
    ax2.plot(selective_df["threshold"], selective_df["selective_accuracy"], color=color, marker="s", linewidth=2, label="Selective Accuracy")
    ax2.plot(selective_df["threshold"], selective_df["selective_macro_f1"], color="tab:orange", marker="^", linestyle="--", linewidth=2, label="Selective Macro-F1")
    ax2.tick_params(axis="y", labelcolor=color)

    plt.title("Selective Prediction: Accuracy vs. Coverage Trade-Off", fontweight="bold")
    fig.tight_layout()
    risk_cov_path = REPORTS_DIR / "figures" / "selective_prediction_risk_coverage.png"
    plt.savefig(risk_cov_path, dpi=300)
    plt.close()
    print(f"Saved risk-coverage plot to: {risk_cov_path}")

    # 3. Explanation Stability Experiment
    print("\n--- Running Explanation Stability Experiment ---")
    # Sample 8 images from validation set: 4 high confidence (>0.85) and 4 borderline (<0.70)
    confidences = np.max(calibrated_probs, axis=1)
    high_conf_indices = np.where(confidences >= 0.85)[0][:8]
    border_indices = np.where(confidences < 0.70)[0][:8]
    test_indices = np.concatenate([high_conf_indices, border_indices])

    sample_paths = [val_paths[i] for i in test_indices]
    sample_labels = [val_labels[i] for i in test_indices]

    stab_df, stab_summary = evaluate_explanation_stability(
        model, sample_paths, sample_labels, calibrated_probs[test_indices], scaler
    )

    stab_csv = REPORTS_DIR / "explanation_stability_results.csv"
    stab_df.to_csv(stab_csv, index=False)

    stab_json = REPORTS_DIR / "explanation_stability_summary.json"
    with stab_json.open("w") as f:
        json.dump(stab_summary, f, indent=2)

    print(f"Saved explanation stability results to:\n  {stab_csv}\n  {stab_json}")
    print("\nExplanation Stability Summary by Confidence Group:")
    for group, stats in stab_summary.items():
        print(f"  {group}: Mean Pearson r = {stats['mean_pearson_r']:.4f} +/- {stats['std_pearson_r']:.4f} | Mean IoU = {stats['mean_mask_iou']:.4f}")


if __name__ == "__main__":
    main()
