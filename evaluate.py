"""Evaluate saved model checkpoints (Baseline CNN or EfficientNetB0) on the official untouched test set.

This script:
1. Loads the saved model checkpoint (baseline_cnn.keras or efficientnet_b0.keras).
2. Preprocesses the test images without data augmentation using the appropriate normalization.
3. Computes accuracy, loss, precision, recall, F1-score, ROC-AUC, classification report, confusion matrix,
   and exact counts of correct and incorrect predictions.
4. Saves a readable report and a styled confusion matrix plot.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
import tensorflow as tf

from preprocess import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    build_dataset,
    find_images,
    flatten_with_labels,
)


CNN_MODEL_PATH = Path("baseline_cnn.keras")
CNN_REPORT_PATH = Path("baseline_evaluation.txt")
CNN_CM_PATH = Path("baseline_confusion_matrix.png")

EFFNET_MODEL_PATH = Path("efficientnet_b0.keras")
EFFNET_REPORT_PATH = Path("efficientnet_evaluation.txt")
EFFNET_CM_PATH = Path("efficientnet_confusion_matrix.png")

TEST_DIR = Path("dataset/Testing")


def load_test_dataset(
    test_dir: Path,
    model_type: str = "cnn",
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, list[Path], np.ndarray]:
    """Load and prepare untouched test dataset for model evaluation."""
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Test directory does not exist: {test_dir.resolve()}")

    images_by_class = find_images(test_dir)
    paths, labels = flatten_with_labels(images_by_class)
    labels_array = np.array(labels, dtype=np.int32)

    dataset = build_dataset(
        paths=paths,
        labels=labels,
        training=False,
        model_type=model_type,
        batch_size=batch_size,
    )
    return dataset, paths, labels_array


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: list[str],
    output_path: Path,
    title: str = "Test Set Confusion Matrix",
) -> None:
    """Render and save a clean, readable confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel="True Class",
        xlabel="Predicted Class",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=11,
                fontweight="bold",
            )

    fig.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def evaluate_model(
    model_path: Path,
    model_type: str = "cnn",
    test_dir: Path = TEST_DIR,
    batch_size: int = BATCH_SIZE,
    report_path: Path | None = None,
    cm_path: Path | None = None,
) -> dict:
    """Evaluate model on official untouched test dataset and save summary artifacts."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model not found: {model_path.resolve()}")

    if report_path is None:
        report_path = EFFNET_REPORT_PATH if model_type == "efficientnet" else CNN_REPORT_PATH
    if cm_path is None:
        cm_path = EFFNET_CM_PATH if model_type == "efficientnet" else CNN_CM_PATH

    model_display_name = "EfficientNetB0 (Transfer Learning)" if model_type == "efficientnet" else "Baseline Custom CNN"
    header_title = "BRAIN TUMOR MRI CLASSIFICATION - EFFICIENTNET-B0 EVALUATION" if model_type == "efficientnet" else "BRAIN TUMOR MRI CLASSIFICATION - BASELINE CNN EVALUATION"
    plot_title = f"{'EfficientNetB0' if model_type == 'efficientnet' else 'Baseline CNN'} - Test Set Confusion Matrix"

    print(f"Loading {model_display_name} from: {model_path}...")
    model = tf.keras.models.load_model(model_path)

    print(f"Loading test dataset from: {test_dir} with model_type='{model_type}'...")
    test_dataset, test_paths, y_true = load_test_dataset(test_dir, model_type=model_type, batch_size=batch_size)
    total_images = len(test_paths)
    print(f"Total test images: {total_images} across {len(CLASS_NAMES)} classes.")

    print("Evaluating model loss and accuracy...")
    eval_results = model.evaluate(test_dataset, verbose=0)
    test_loss = float(eval_results[0])
    keras_acc = float(eval_results[1])

    print("Generating predictions...")
    y_probs = model.predict(test_dataset, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # Multiclass ROC-AUC (One-vs-Rest)
    try:
        roc_auc_macro = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
        roc_auc_weighted = roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted")
        auc_str = f"ROC-AUC (Macro OvR)   : {roc_auc_macro:.4f}\nROC-AUC (Weighted OvR): {roc_auc_weighted:.4f}"
    except Exception as exc:
        roc_auc_macro = None
        roc_auc_weighted = None
        auc_str = f"ROC-AUC: N/A ({exc})"

    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    num_correct = int(np.sum(np.diag(cm)))
    num_incorrect = int(total_images - num_correct)

    # Save Confusion Matrix Plot
    save_confusion_matrix_plot(cm, CLASS_NAMES, cm_path, title=plot_title)
    print(f"Confusion matrix image saved to: {cm_path.resolve()}")

    # Prepare readable evaluation text
    results_text = f"""==================================================
{header_title}
==================================================
Model Architecture     : {model_display_name}
Model Checkpoint       : {model_path}
Test Directory         : {test_dir}
Total Test Images      : {total_images}
Correct Predictions    : {num_correct} / {total_images} ({num_correct / total_images * 100:.2f}%)
Incorrect Predictions  : {num_incorrect} / {total_images} ({num_incorrect / total_images * 100:.2f}%)
Image Dimensions       : {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB
Classes ({len(CLASS_NAMES)})       : {', '.join(CLASS_NAMES)}

--------------------------------------------------
OVERALL PERFORMANCE METRICS
--------------------------------------------------
Test Loss              : {test_loss:.4f}
Test Accuracy          : {acc * 100:.2f}% ({acc:.4f})
Precision (Macro)      : {prec_macro:.4f}
Precision (Weighted)   : {prec_weighted:.4f}
Recall (Macro)         : {rec_macro:.4f}
Recall (Weighted)      : {rec_weighted:.4f}
F1-Score (Macro)       : {f1_mac:.4f}
F1-Score (Weighted)    : {f1_wt:.4f}
{auc_str}

--------------------------------------------------
PER-CLASS CLASSIFICATION REPORT
--------------------------------------------------
{report}

--------------------------------------------------
CONFUSION MATRIX
--------------------------------------------------
Columns = Predicted, Rows = Actual
Classes: {CLASS_NAMES}

{cm}

Per-class breakdown:
"""
    for idx, class_name in enumerate(CLASS_NAMES):
        total_class = np.sum(cm[idx])
        correct_class = cm[idx, idx]
        class_acc = (correct_class / total_class * 100) if total_class > 0 else 0.0
        results_text += f"  - {class_name:12s}: {correct_class:4d}/{total_class:4d} correct ({class_acc:6.2f}% recall)\n"

    results_text += "==================================================\n"

    report_path.write_text(results_text, encoding="utf-8")
    print(f"Evaluation report saved to: {report_path.resolve()}")

    print("\n" + results_text)

    return {
        "test_loss": test_loss,
        "test_accuracy": acc,
        "precision_macro": prec_macro,
        "precision_weighted": prec_weighted,
        "recall_macro": rec_macro,
        "recall_weighted": rec_weighted,
        "f1_macro": f1_mac,
        "f1_weighted": f1_wt,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
        "num_correct": num_correct,
        "num_incorrect": num_incorrect,
        "classification_report": report,
        "confusion_matrix": cm,
    }


# Backwards compatibility alias
evaluate_baseline = evaluate_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained models on the test set.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["cnn", "efficientnet"],
        default="cnn",
        help="Type of model to evaluate ('cnn' or 'efficientnet').",
    )
    parser.add_argument("--model-path", type=Path, default=None, help="Path to saved model (.keras).")
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR, help="Path to untouched test folder.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size for evaluation.")
    parser.add_argument("--report-path", type=Path, default=None, help="Output evaluation text file.")
    parser.add_argument("--cm-path", type=Path, default=None, help="Output confusion matrix image.")
    arguments = parser.parse_args()

    model_path = arguments.model_path
    if model_path is None:
        model_path = EFFNET_MODEL_PATH if arguments.model_type == "efficientnet" else CNN_MODEL_PATH

    evaluate_model(
        model_path=model_path,
        model_type=arguments.model_type,
        test_dir=arguments.test_dir,
        batch_size=arguments.batch_size,
        report_path=arguments.report_path,
        cm_path=arguments.cm_path,
    )


if __name__ == "__main__":
    main()

