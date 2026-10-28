
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from config import (
    TEST_DIR,
    MODELS_DIR,
    FIGURES_DIR,
    IMAGE_SIZE,
    CLASS_NAMES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Update these paths if your saved model filenames differ.
MODEL_PATHS = {
    "Custom CNN": PROJECT_ROOT / "baseline_cnn.keras",
    "EfficientNet-B0": PROJECT_ROOT / "efficientnet_b0.keras",
}

def evaluate_model(model_name, model_path):
    if not model_path.exists():
        print(f"\nSkipping {model_name}: model not found at {model_path}")
        return None

    print(f"\nEvaluating {model_name}...")

    model = tf.keras.models.load_model(model_path)

    # Load the official test folder, keeping the configured class order.
    test_data = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMAGE_SIZE,
        batch_size=32,
        label_mode="int",
        class_names=CLASS_NAMES,
        shuffle=False,
    )

    y_true = np.concatenate([labels.numpy() for _, labels in test_data])

    # EfficientNet-B0 Keras applications usually include preprocessing
    # internally. For other models, match the preprocessing used in training.
    probabilities = model.predict(test_data, verbose=1)

    if probabilities.ndim != 2 or probabilities.shape[1] != len(CLASS_NAMES):
        raise ValueError(
            f"{model_name} returned shape {probabilities.shape}; "
            f"expected (number_of_images, {len(CLASS_NAMES)} classes)."
        )

    y_pred = np.argmax(probabilities, axis=1)

    results = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_f1": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "macro_roc_auc_ovr": None,
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        ),
        "test_images": int(len(y_true)),
    }

    try:
        y_true_one_hot = np.eye(len(CLASS_NAMES))[y_true]
        results["macro_roc_auc_ovr"] = float(
            roc_auc_score(
                y_true_one_hot,
                probabilities,
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError as exc:
        print("ROC-AUC could not be calculated:", exc)

    print(f"\n{model_name}")
    print("-" * 45)
    print(f"Test images : {results['test_images']}")
    print(f"Accuracy    : {results['accuracy']:.4%}")
    print(f"Precision   : {results['macro_precision']:.4f}")
    print(f"Recall      : {results['macro_recall']:.4f}")
    print(f"Macro F1    : {results['macro_f1']:.4f}")
    print(f"Macro AUC   : {results['macro_roc_auc_ovr']}")

    print("\nClass-wise results:")
    print(classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    # Save confusion matrix.
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASS_NAMES)))
    )
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    display.plot(cmap="Blues", xticks_rotation=45)
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()

    safe_name = model_name.lower().replace(" ", "_").replace("-", "_")
    plt.savefig(FIGURES_DIR / f"{safe_name}_confusion_matrix.png", dpi=150)
    plt.close()

    return results


def main():
    all_results = {}

    for model_name, model_path in MODEL_PATHS.items():
        try:
            result = evaluate_model(model_name, model_path)
            if result is not None:
                all_results[model_name] = result
        except Exception as exc:
            print(f"\nError evaluating {model_name}: {exc}")

    output_path = PROJECT_ROOT / "reports" / "final_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    print(f"\nResults saved to: {output_path}")
    print(f"Confusion matrices saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
