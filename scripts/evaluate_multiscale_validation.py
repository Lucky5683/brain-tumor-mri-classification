
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import CLASS_NAMES, TRAIN_DIR, RANDOM_SEED, VALIDATION_FRACTION
from src.data.dataset import get_train_val_datasets

MODEL_PATH = ROOT / "models" / "multiscale_fusion_best.keras"
OUTPUT_DIR = ROOT / "experiments" / "multiscale_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Re-create the exact deterministic split used during training.
    _, val_ds, _, _, val_paths, val_labels = get_train_val_datasets(
        train_dir=TRAIN_DIR,
        model_type="efficientnet",
        batch_size=32,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
        use_augmentation=False,
    )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = model.predict(val_ds, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    y_true = np.asarray(val_labels)

    print("\nValidation accuracy:", round(accuracy_score(y_true, predictions), 4))
    print("Validation macro-F1:", round(
        f1_score(y_true, predictions, average="macro", zero_division=0), 4
    ))

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_path = OUTPUT_DIR / "classification_report.csv"
    report_df.to_csv(report_path)
    print("\nClassification report:")
    print(report_df.round(4).to_string())
    print("\nSaved:", report_path)

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
    )
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    cm_path = OUTPUT_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)
    print("\nConfusion matrix (rows = actual, columns = predicted):")
    print(cm_df.to_string())
    print("Saved:", cm_path)

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=CLASS_NAMES,
    )
    display.plot(cmap="Blues", values_format="d", xticks_rotation=25)
    plt.title("Multi-Scale Fusion — Validation Confusion Matrix")
    plt.tight_layout()
    figure_path = OUTPUT_DIR / "confusion_matrix.png"
    plt.savefig(figure_path, dpi=200)
    plt.close()
    print("Saved:", figure_path)


if __name__ == "__main__":
    main()
