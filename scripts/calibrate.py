"""Model probability calibration evaluation, temperature scaling, and reliability diagrams."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from config import CLASS_NAMES, DATA_DIR, FIGURES_DIR, REPORTS_DIR, TEST_DIR, VALIDATION_FRACTION
import src.models.compat
from src.data.dataset import get_test_dataset, get_train_val_datasets
from src.evaluation.calibration import (
    TemperatureScaler,
    compute_ece_and_reliability,
    plot_reliability_diagram,
)


def run_calibration_evaluation(
    model_path: Path,
    data_dir: Path = DATA_DIR,
    model_type: str = "efficientnet",
    output_dir: Path = REPORTS_DIR,
) -> dict:
    """Evaluate calibration, fit temperature scaling on validation set, and evaluate on test set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from: {model_path.resolve()} ...")
    model = tf.keras.models.load_model(model_path)

    # 1. Load VALIDATION set to fit temperature scaling strictly on validation data
    print("Loading validation set to fit temperature scaling...")
    _, val_ds, _, _, _, val_labels = get_train_val_datasets(
        train_dir=data_dir / "Training",
        model_type=model_type,
        validation_fraction=VALIDATION_FRACTION,
        use_augmentation=False,
    )

    val_probs = model.predict(val_ds, verbose=0)
    # Approximate unscaled logits from softmax probabilities: log(p + eps)
    eps = 1e-12
    val_logits = np.log(np.clip(val_probs, eps, 1.0))
    val_labels_arr = np.array(val_labels)

    scaler = TemperatureScaler()
    optimal_temp = scaler.fit(val_logits, val_labels_arr)
    print(f"Learned optimal temperature on validation set: T = {optimal_temp:.4f}")

    # 2. Load untouched TEST set to evaluate uncalibrated vs calibrated probabilities
    print("Loading official test set to evaluate calibration performance...")
    test_ds, _, test_labels = get_test_dataset(test_dir=data_dir / "Testing", model_type=model_type)
    test_probs = model.predict(test_ds, verbose=0)
    test_labels_arr = np.array(test_labels)

    # Uncalibrated metrics on test set
    uncalibrated_stats = compute_ece_and_reliability(test_labels_arr, test_probs, n_bins=10)
    plot_reliability_diagram(
        uncalibrated_stats,
        output_path=fig_dir / "reliability_diagram.png",
        title="Uncalibrated Reliability Diagram",
    )
    plot_reliability_diagram(
        uncalibrated_stats,
        output_path=output_dir / "reliability_diagram.png",
        title="Uncalibrated Reliability Diagram",
    )

    # Calibrated metrics using learned temperature
    test_logits = np.log(np.clip(test_probs, eps, 1.0))
    calibrated_probs = scaler.scale_probabilities(test_logits)
    calibrated_stats = compute_ece_and_reliability(test_labels_arr, calibrated_probs, n_bins=10)
    plot_reliability_diagram(
        calibrated_stats,
        output_path=fig_dir / "reliability_diagram_temperature_scaled.png",
        title=f"Calibrated (T={optimal_temp:.2f}) Reliability Diagram",
    )
    plot_reliability_diagram(
        calibrated_stats,
        output_path=output_dir / "reliability_diagram_temperature_scaled.png",
        title=f"Calibrated (T={optimal_temp:.2f}) Reliability Diagram",
    )

    report = {
        "model_path": str(model_path.resolve()),
        "temperature_scaling": {
            "optimal_temperature": round(optimal_temp, 4),
            "fitted_split": "Validation set (never fitted on Test set)",
        },
        "uncalibrated_test_metrics": {
            "ece": uncalibrated_stats["ece"],
            "mce": uncalibrated_stats["mce"],
            "brier_score": uncalibrated_stats["brier_score"],
            "bins": uncalibrated_stats["bins"],
        },
        "calibrated_test_metrics": {
            "ece": calibrated_stats["ece"],
            "mce": calibrated_stats["mce"],
            "brier_score": calibrated_stats["brier_score"],
            "bins": calibrated_stats["bins"],
        },
        "clinical_statement": (
            "Model probability outputs represent statistical softmax confidence values. "
            "They do not signify verified biological or clinical disease certainty."
        ),
    }

    json_path = output_dir / "calibration_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("MODEL CALIBRATION SUMMARY")
    print("=" * 60)
    print(f"Optimal Temperature (T) : {optimal_temp:.4f}")
    print(f"Uncalibrated Test ECE   : {uncalibrated_stats['ece']:.4f} (Brier: {uncalibrated_stats['brier_score']:.4f})")
    print(f"Calibrated Test ECE     : {calibrated_stats['ece']:.4f} (Brier: {calibrated_stats['brier_score']:.4f})")
    print(f"Saved report to         : {json_path.resolve()}")
    print(f"Saved figures to        : {fig_dir.resolve()}")
    print("=" * 60)

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model calibration and temperature scaling.")
    parser.add_argument("--model-path", type=Path, default=Path("efficientnet_b0.keras"))
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--model-type", choices=["cnn", "efficientnet"], default="efficientnet")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    run_calibration_evaluation(
        model_path=args.model_path,
        data_dir=args.data_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )
