
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import classification_report, confusion_matrix


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MANIFEST = (
    ROOT
    / "experiments"
    / "duplicate_aware_split"
    / "training_manifest.csv"
)

# IMPORTANT: Use the 50-epoch model, not the older 12-epoch model.
MODEL_PATH = (
    ROOT
    / "experiments"
    / "spatial_fusion"
    / "meningioma_weighted_50ep_seed42"
    / "best_model.keras"
)

# Save diagnostic results alongside the 50-epoch checkpoint.
OUT_DIR = (
    ROOT
    / "experiments"
    / "spatial_fusion"
    / "meningioma_weighted_50ep_seed42"
)

CLASS_NAMES = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary",
]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# 2. CHECK REQUIRED FILES
# ============================================================

if not MANIFEST.is_file():
    raise FileNotFoundError(
        f"Validation manifest not found:\n{MANIFEST}"
    )

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )

print("=" * 65)
print("50-EPOCH MODEL VALIDATION DIAGNOSTICS")
print("=" * 65)
print(f"Project root : {ROOT}")
print(f"Manifest     : {MANIFEST}")
print(f"Checkpoint   : {MODEL_PATH}")
print(f"Output folder: {OUT_DIR}")


# ============================================================
# 3. LOAD VALIDATION MANIFEST
# ============================================================

df = pd.read_csv(MANIFEST)

required_columns = {"path", "label", "split"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Manifest is missing required columns: {missing_columns}"
    )

df = df[
    df["split"].astype(str).str.strip().str.lower()
    == "validation"
].copy()

if df.empty:
    raise ValueError(
        "No validation rows found in the manifest."
    )

df["actual_class"] = df["label"].astype(str).str.strip().str.lower()

unknown = set(df["actual_class"].unique()) - set(CLASS_NAMES)

if unknown:
    raise ValueError(
        f"Unexpected class labels in manifest: {unknown}"
    )

print(f"\nValidation images: {len(df)}")
print("\nValidation class counts:")
print(
    df["actual_class"]
    .value_counts()
    .reindex(CLASS_NAMES, fill_value=0)
    .to_string()
)


# ============================================================
# 4. RESOLVE IMAGE PATHS
# ============================================================

def resolve_path(value):
    """
    Resolve image paths stored in the manifest.

    Supports:
    - Absolute paths that still exist
    - Paths relative to the project root
    - Paths saved with an old Windows drive letter
    """

    p = Path(str(value))

    # Existing absolute path
    if p.is_absolute() and p.is_file():
        return p

    # Relative path from project root
    candidate = ROOT / p

    if candidate.is_file():
        return candidate

    # Recover a path beginning at the dataset directory.
    parts = p.parts

    for i, part in enumerate(parts):
        if part.lower() == "dataset":
            candidate = ROOT.joinpath(*parts[i:])

            if candidate.is_file():
                return candidate

    raise FileNotFoundError(
        f"Cannot locate validation image:\n{value}\n"
        f"Project root: {ROOT}"
    )


print("\nResolving validation image paths...")

df["resolved_path"] = df["path"].map(resolve_path)

print("All validation image paths resolved successfully.")


# ============================================================
# 5. LOAD AND PREPROCESS IMAGES
# ============================================================

def load_image(path):
    """
    Decode an image and resize it to 224 x 224 RGB.

    This uses the same EfficientNet preprocessing convention
    as the original diagnostic script.
    """

    image = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
        antialias=True,
    )

    image = tf.keras.applications.efficientnet.preprocess_input(
        tf.cast(image, tf.float32)
    )

    return image


# ============================================================
# 6. LOAD THE CORRECT CHECKPOINT
# ============================================================

print("\nLoading the 50-epoch model...")

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False,
)

print("Model loaded successfully.")

print(f"Model input shape : {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

if isinstance(model.output_shape, list):
    raise ValueError(
        "This script expects a model with a single output."
    )

if model.output_shape[-1] != len(CLASS_NAMES):
    raise ValueError(
        f"Expected {len(CLASS_NAMES)} output classes, "
        f"but model output shape is {model.output_shape}."
    )


# ============================================================
# 7. RUN VALIDATION PREDICTIONS
# ============================================================

print("\nRunning predictions on the validation set...")

rows = []

for start in range(0, len(df), BATCH_SIZE):

    batch_df = df.iloc[start:start + BATCH_SIZE]

    batch_images = tf.stack([
        load_image(path)
        for path in batch_df["resolved_path"]
    ])

    probabilities = model.predict(
        batch_images,
        verbose=0,
    )

    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(CLASS_NAMES)
    ):
        raise ValueError(
            f"Unexpected model output shape: {probabilities.shape}"
        )

    for (_, row), probability in zip(
        batch_df.iterrows(),
        probabilities,
    ):

        # The model is expected to return four class probabilities.
        # If outputs are not probabilities, investigate the model
        # architecture before interpreting confidence values.
        pred_idx = int(np.argmax(probability))

        actual = row["actual_class"]
        predicted = CLASS_NAMES[pred_idx]

        result = {
            "path": str(row["resolved_path"]),
            "actual_class": actual,
            "predicted_class": predicted,
            "confidence": float(probability[pred_idx]),
            "correct": actual == predicted,
        }

        for i, class_name in enumerate(CLASS_NAMES):
            result[f"prob_{class_name}"] = float(probability[i])

        rows.append(result)

    completed = min(start + len(batch_df), len(df))

    print(
        f"Processed {completed}/{len(df)} validation images",
        end="\r",
    )

print("\nPrediction process completed.")

predictions = pd.DataFrame(rows)


# ============================================================
# 8. SAVE ALL PREDICTIONS AND ERRORS
# ============================================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

pred_path = OUT_DIR / "weighted_validation_predictions.csv"

predictions.to_csv(
    pred_path,
    index=False,
)

errors = predictions[
    ~predictions["correct"]
].copy()

error_path = OUT_DIR / "weighted_validation_misclassified.csv"

errors.to_csv(
    error_path,
    index=False,
)

print(f"\nTotal validation images : {len(predictions)}")
print(f"Correct predictions    : {int(predictions['correct'].sum())}")
print(f"Incorrect predictions  : {len(errors)}")
print(
    f"Validation accuracy    : "
    f"{predictions['correct'].mean() * 100:.4f}%"
)


# ============================================================
# 9. CLASSIFICATION REPORT
# ============================================================

print("\n" + "=" * 65)
print("VALIDATION CLASSIFICATION REPORT")
print("=" * 65)

report = classification_report(
    predictions["actual_class"],
    predictions["predicted_class"],
    labels=CLASS_NAMES,
    target_names=CLASS_NAMES,
    zero_division=0,
    digits=4,
    output_dict=True,
)

print(
    classification_report(
        predictions["actual_class"],
        predictions["predicted_class"],
        labels=CLASS_NAMES,
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )
)

report_df = pd.DataFrame(report).transpose()

report_path = OUT_DIR / "diagnostic_classification_report.csv"

report_df.to_csv(report_path)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

print("\n" + "=" * 65)
print("CONFUSION MATRIX")
print("=" * 65)

cm = confusion_matrix(
    predictions["actual_class"],
    predictions["predicted_class"],
    labels=CLASS_NAMES,
)

cm_df = pd.DataFrame(
    cm,
    index=CLASS_NAMES,
    columns=CLASS_NAMES,
)

print(cm_df.to_string())

cm_path = OUT_DIR / "diagnostic_confusion_matrix.csv"

cm_df.to_csv(cm_path)


# ============================================================
# 11. MENINGIOMA PREDICTED AS PITUITARY
# ============================================================

print("\n" + "=" * 65)
print("MENINGIOMA PREDICTED AS PITUITARY")
print("=" * 65)

subset = errors[
    (errors["actual_class"] == "meningioma")
    & (errors["predicted_class"] == "pituitary")
].copy()

if subset.empty:

    print("No meningioma-to-pituitary errors found.")

else:

    print(f"Count: {len(subset)}")

    display_columns = [
        "path",
        "confidence",
        "prob_meningioma",
        "prob_pituitary",
    ]

    print(
        subset[display_columns]
        .sort_values("confidence", ascending=False)
        .head(30)
        .to_string(index=False)
    )


# ============================================================
# 12. SUMMARY AND OUTPUT FILES
# ============================================================

print("\n" + "=" * 65)
print("DIAGNOSTIC SUMMARY")
print("=" * 65)

print(f"Checkpoint: {MODEL_PATH}")
print(f"Validation images: {len(predictions)}")
print(f"Accuracy: {report['accuracy'] * 100:.4f}%")
print(f"Macro F1: {report['macro avg']['f1-score'] * 100:.4f}%")
print(
    f"Balanced accuracy: "
    f"{report['macro avg']['recall'] * 100:.4f}%"
)

print("\nSaved files:")

print(f"1. Predictions       : {pred_path}")
print(f"2. Misclassified     : {error_path}")
print(f"3. Classification    : {report_path}")
print(f"4. Confusion matrix  : {cm_path}")

print("\nDiagnostics completed successfully.")
