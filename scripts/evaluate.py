"""Official Test Set Evaluation Pipeline with Bootstrap Confidence Intervals for V2.

Evaluates candidate model checkpoints on the untouched official test set:
- Accuracy, Precision, Recall, Macro-F1, Weighted-F1
- One-vs-Rest ROC-AUC curves
- Per-class confusion matrix and classification report
- 95% Non-parametric bootstrap confidence intervals (1,000 iterations)
- Machine-readable outputs: reports/final_metrics.json, reports/classification_report.csv
- Figures: reports/figures/confusion_matrix.png, roc_curve.png, class_metrics.png
"""

from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf

from config import CLASS_NAMES, DISPLAY_NAMES, FIGURES_DIR, REPORTS_DIR, TEST_DIR
import src.models.compat  # ensure compatibility shim
from src.data.dataset import get_test_dataset
from src.evaluation.metrics import (
    compute_bootstrap_confidence_intervals,
    compute_comprehensive_metrics,
    plot_confusion_matrix,
    plot_multiclass_precision_recall_curves,
    plot_multiclass_roc_curves,
    plot_per_class_metrics_bar,
)


def evaluate_model_checkpoint(
    model_path: Path,
    test_dir: Path = TEST_DIR,
    model_type: str = "efficientnet",
    output_dir: Path = REPORTS_DIR,
    batch_size: int = 32,
    n_bootstraps: int = 1000,
) -> dict:
    """Evaluate saved model against untouched official test set."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model checkpoint: {model_path.resolve()} ...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading untouched test dataset from: {test_dir.resolve()} ...")
    test_ds, test_paths, test_labels = get_test_dataset(test_dir=test_dir, model_type=model_type, batch_size=batch_size)

    print("Running model inference on test set...")
    y_probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.array(test_labels)

    # 1. Comprehensive Metrics
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_probs, class_names=CLASS_NAMES)

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, class_names=CLASS_NAMES, output_path=fig_dir / "confusion_matrix.png")
    plot_confusion_matrix(cm, class_names=CLASS_NAMES, output_path=output_dir / "confusion_matrix.png")

    # 3. ROC Curves
    plot_multiclass_roc_curves(y_true, y_probs, class_names=CLASS_NAMES, output_path=fig_dir / "roc_curve.png")
    plot_multiclass_roc_curves(y_true, y_probs, class_names=CLASS_NAMES, output_path=output_dir / "roc_curve.png")

    # 4. Precision-Recall Curves
    plot_multiclass_precision_recall_curves(y_true, y_probs, class_names=CLASS_NAMES, output_path=fig_dir / "precision_recall_curve.png")
    plot_multiclass_precision_recall_curves(y_true, y_probs, class_names=CLASS_NAMES, output_path=output_dir / "precision_recall_curve.png")

    # 5. Per-Class Metrics Bar Chart
    plot_per_class_metrics_bar(metrics["per_class"], output_path=fig_dir / "class_metrics.png")
    plot_per_class_metrics_bar(metrics["per_class"], output_path=output_dir / "class_metrics.png")

    # 6. Bootstrap Confidence Intervals (95%)
    print(f"Computing 95% bootstrap confidence intervals ({n_bootstraps} iterations)...")
    boot_ci, boot_df = compute_bootstrap_confidence_intervals(y_true, y_pred, n_bootstraps=n_bootstraps)

    # Save bootstrap metrics CSV and CI JSON
    boot_csv_path = output_dir / "bootstrap_metrics.csv"
    boot_df.to_csv(boot_csv_path, index=False)
    ci_json_path = output_dir / "confidence_intervals.json"
    ci_json_path.write_text(json.dumps(boot_ci, indent=2), encoding="utf-8")

    # 7. Classification Report CSV
    clf_dict = classification_report(y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    clf_df = pd.DataFrame(clf_dict).transpose()
    clf_csv_path = output_dir / "classification_report.csv"
    clf_df.to_csv(clf_csv_path)

    # 8. Final Metrics JSON
    final_results = {
        "model_path": str(model_path.resolve()),
        "test_images_count": len(y_true),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "bootstrap_confidence_intervals": boot_ci,
        "evaluation_notes": (
            "Evaluation performed strictly on untouched official test set. "
            "Bootstrap intervals reflect test sampling distribution only."
        ),
    }

    json_path = output_dir / "final_metrics.json"
    json_path.write_text(json.dumps(final_results, indent=2), encoding="utf-8")

    # Console Summary
    print("\n" + "=" * 60)
    print("FINAL TEST EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model Checkpoint   : {model_path.name}")
    print(f"Total Test Images  : {len(y_true):,}")
    print(f"Test Accuracy      : {metrics['accuracy']*100:.2f}% (95% CI: [{boot_ci['accuracy']['ci_lower']*100:.2f}%, {boot_ci['accuracy']['ci_upper']*100:.2f}%])")
    print(f"Macro F1-Score     : {metrics['f1_macro']:.4f} (95% CI: [{boot_ci['macro_f1']['ci_lower']:.4f}, {boot_ci['macro_f1']['ci_upper']:.4f}])")
    print(f"Macro Recall       : {metrics['recall_macro']:.4f} (95% CI: [{boot_ci['macro_recall']['ci_lower']:.4f}, {boot_ci['macro_recall']['ci_upper']:.4f}])")
    print(f"Macro Precision    : {metrics['precision_macro']:.4f} (95% CI: [{boot_ci['macro_precision']['ci_lower']:.4f}, {boot_ci['macro_precision']['ci_upper']:.4f}])")
    if metrics["roc_auc_macro_ovr"] is not None:
        print(f"ROC-AUC (Macro OvR): {metrics['roc_auc_macro_ovr']:.4f}")

    gl_stats = metrics["per_class"]["glioma"]
    gl_ci = boot_ci.get("glioma_recall")
    ci_str = f" (95% CI: [{gl_ci['ci_lower']*100:.2f}%, {gl_ci['ci_upper']*100:.2f}%])" if gl_ci else ""
    print(f"Glioma Recall      : {gl_stats['recall']*100:.2f}%{ci_str} (Precision: {gl_stats['precision']*100:.2f}%, F1: {gl_stats['f1']*100:.2f}%)")

    print("\nPer-Class Breakdown:")
    for cname in CLASS_NAMES:
        stats = metrics["per_class"][cname]
        dname = DISPLAY_NAMES.get(cname, cname.title())
        print(f"  {dname:12s}: Precision = {stats['precision']*100:5.2f}%, Recall = {stats['recall']*100:5.2f}%, F1 = {stats['f1']*100:5.2f}%, Support = {stats['support']}")

    print("=" * 60)
    print(f"Saved artifacts to {output_dir.resolve()}:")
    print(f"  - final_metrics.json")
    print(f"  - confidence_intervals.json")
    print(f"  - bootstrap_metrics.csv")
    print(f"  - classification_report.csv")
    print(f"  - confusion_matrix.png")
    print(f"  - roc_curve.png")
    print(f"  - precision_recall_curve.png")
    print(f"  - class_metrics.png")
    print(f"  - figures/confusion_matrix.png")
    print(f"  - figures/roc_curve.png")
    print(f"  - figures/class_metrics.png")

    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on official test set.")
    parser.add_argument("--model-path", type=Path, default=Path("efficientnet_b0.keras"))
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--model-type", choices=["cnn", "efficientnet"], default="efficientnet")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    evaluate_model_checkpoint(
        model_path=args.model_path,
        test_dir=args.test_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
    )
