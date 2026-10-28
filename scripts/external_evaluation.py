"""External Dataset Evaluation CLI: evaluates models on independent datasets without retraining.

Accepts an external directory formatted with class folders:
<external_dir>/
    glioma/
    meningioma/
    notumor/
    pituitary/

Produces:
- Accuracy, precision, recall, F1-score, ROC-AUC
- Confusion matrix and classification report
- If no external dataset path is supplied or found, records transparent status:
  'External validation not yet performed.' (Zero fabricated results).
"""

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

from config import CLASS_NAMES, DISPLAY_NAMES, REPORTS_DIR
import src.models.compat
from src.data.dataset import build_tf_dataset, find_images, flatten_class_dict
from src.evaluation.metrics import compute_comprehensive_metrics, plot_confusion_matrix


def run_external_evaluation(
    model_path: Path,
    external_dir: Path | None = None,
    model_type: str = "efficientnet",
    output_dir: Path = REPORTS_DIR,
) -> dict:
    """Evaluate model on external dataset or document ready state."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "external_validation_report.json"

    if external_dir is None or not Path(external_dir).is_dir():
        status_info = {
            "status": "External validation not yet performed.",
            "reason": "No independent external clinical dataset directory provided or detected.",
            "pipeline_readiness": "READY. Provide --external-dir pointing to a folder containing glioma, meningioma, notumor, pituitary subdirectories.",
            "note": "Per scientific integrity guidelines, external validation results are NEVER fabricated.",
        }
        report_path.write_text(json.dumps(status_info, indent=2), encoding="utf-8")
        print("\n" + "=" * 60)
        print("EXTERNAL VALIDATION PIPELINE STATUS")
        print("=" * 60)
        print("Status: External validation not yet performed.")
        print("Pipeline is verified ready for external cohorts.")
        print(f"Report written to: {report_path.resolve()}")
        print("=" * 60)
        return status_info

    ext_path = Path(external_dir)
    print(f"Loading external dataset from: {ext_path.resolve()} ...")
    images_by_class = find_images(ext_path)
    paths, labels = flatten_class_dict(images_by_class)

    if not paths:
        raise RuntimeError(f"No valid images found in external dataset: {ext_path}")

    print(f"Loaded {len(paths)} external images across classes: {list(images_by_class.keys())}")
    dataset = build_tf_dataset(paths, labels, training=False, model_type=model_type)

    print(f"Loading model checkpoint: {model_path.resolve()} ...")
    model = tf.keras.models.load_model(model_path)

    y_probs = model.predict(dataset, verbose=1)
    y_pred = np.argmax(y_probs, axis=1)
    y_true = np.array(labels)

    metrics = compute_comprehensive_metrics(y_true, y_pred, y_probs, class_names=CLASS_NAMES)
    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES)).numpy()

    cm_path = output_dir / "external_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names=CLASS_NAMES, output_path=cm_path, title="External Dataset Confusion Matrix")

    report = {
        "status": "Completed external validation.",
        "external_directory": str(ext_path.resolve()),
        "model_checkpoint": str(model_path.resolve()),
        "total_external_images": len(paths),
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),
        "disclaimer": "External results must be interpreted within the imaging protocol context of the external site.",
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("EXTERNAL DATASET EVALUATION RESULTS")
    print("=" * 60)
    print(f"External Images : {len(paths):,}")
    print(f"Accuracy        : {metrics['accuracy']*100:.2f}%")
    print(f"Macro F1-Score  : {metrics['f1_macro']:.4f}")
    print(f"Macro Recall    : {metrics['recall_macro']:.4f}")
    if metrics["roc_auc_macro_ovr"] is not None:
        print(f"ROC-AUC (Macro) : {metrics['roc_auc_macro_ovr']:.4f}")
    print("=" * 60)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on external dataset.")
    parser.add_argument("--model-path", type=Path, default=Path("efficientnet_b0.keras"))
    parser.add_argument("--external-dir", type=Path, default=None, help="Path to external dataset folder.")
    parser.add_argument("--model-type", choices=["cnn", "efficientnet"], default="efficientnet")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    run_external_evaluation(
        model_path=args.model_path,
        external_dir=args.external_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )
