
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "experiments/spatial_fusion/meningioma_weighted_seed42/best_model.keras"
)

MANIFEST = (
    ROOT
    / "experiments/duplicate_aware_split/training_manifest.csv"
)

PREDICTIONS = (
    ROOT
    / "experiments/spatial_fusion/meningioma_weighted_seed42/"
    "weighted_validation_predictions.csv"
)

GRADCAM_RESULTS = (
    ROOT
    / "experiments/spatial_fusion/meningioma_weighted_seed42/"
    "gradcam_error_investigation/gradcam_results.csv"
)

TARGET = "Tr-me_1101.jpg"
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)


# ============================================================
# VALIDATE FILES
# ============================================================

for file_path in [MODEL_PATH, MANIFEST]:
    if not file_path.exists():
        raise FileNotFoundError(f"Required file not found: {file_path}")


# ============================================================
# RESOLVE IMAGE PATHS
# ============================================================

def resolve_path(value):
    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidate = ROOT / path

    if candidate.exists():
        return candidate

    # Support paths saved relative to the dataset directory.
    parts = list(path.parts)

    for index, part in enumerate(parts):
        if part.lower() == "dataset":
            candidate = ROOT.joinpath(*parts[index:])

            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Cannot locate image: {value}")


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def load_image(path):
    image_bytes = tf.io.read_file(str(path))

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False
    )

    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        IMAGE_SIZE,
        method="bilinear",
        antialias=True
    )

    image = tf.keras.applications.efficientnet.preprocess_input(
        tf.cast(image, tf.float32)
    )

    return image


# ============================================================
# LOAD VALIDATION MANIFEST
# ============================================================

manifest = pd.read_csv(MANIFEST)

normalized_paths = (
    manifest["path"]
    .astype(str)
    .str.replace("\\", "/", regex=False)
)

mask = (
    manifest["split"].astype(str).str.lower().eq("validation")
    & normalized_paths.str.endswith(TARGET)
)

target_rows = manifest.loc[mask].copy()

if target_rows.empty:
    raise RuntimeError(
        f"{TARGET} was not found in the validation manifest."
    )

target_rows["resolved_path"] = target_rows["path"].map(resolve_path)

print("\n" + "=" * 65)
print("1. VALIDATION MANIFEST")
print("=" * 65)

for _, row in target_rows.iterrows():
    image_path = row["resolved_path"]

    sha256 = hashlib.sha256(image_path.read_bytes()).hexdigest()

    print(f"Image: {image_path.name}")
    print(f"Full path: {image_path}")
    print(f"Actual label: {row['label']}")
    print(f"SHA-256: {sha256}")


# ============================================================
# INSPECT SAVED VALIDATION PREDICTION
# ============================================================

print("\n" + "=" * 65)
print("2. SAVED VALIDATION PREDICTION")
print("=" * 65)

if PREDICTIONS.exists():
    saved = pd.read_csv(PREDICTIONS)

    saved_paths = (
        saved["path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    saved_matches = saved.loc[saved_paths.str.endswith(TARGET)]

    if saved_matches.empty:
        print("No matching image in saved validation CSV.")
    else:
        print(saved_matches.to_string(index=False))
else:
    print(f"Saved predictions file not found: {PREDICTIONS}")


# ============================================================
# INSPECT GRAD-CAM RESULTS
# ============================================================

print("\n" + "=" * 65)
print("3. GRAD-CAM RESULTS")
print("=" * 65)

if GRADCAM_RESULTS.exists():
    gradcam = pd.read_csv(GRADCAM_RESULTS)

    # Produce a Boolean mask for each row.
    # This avoids the previous pandas Series axis error.
    row_mask = gradcam.astype(str).apply(
        lambda column: column.str.contains(
            TARGET,
            regex=False,
            na=False
        ),
        axis=0
    ).any(axis=1)

    gradcam_matches = gradcam.loc[row_mask]

    print("Available columns:", gradcam.columns.tolist())

    if gradcam_matches.empty:
        print(f"No row containing {TARGET} was found.")
    else:
        print(gradcam_matches.to_string(index=False))
else:
    print(f"Grad-CAM results file not found: {GRADCAM_RESULTS}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\n" + "=" * 65)
print("4. LOAD MODEL")
print("=" * 65)

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False
)

print(f"Checkpoint: {MODEL_PATH}")
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")


# ============================================================
# RUN FRESH INFERENCE
# ============================================================

print("\n" + "=" * 65)
print("5. FRESH MODEL PREDICTION")
print("=" * 65)

fresh_results = []

for _, row in target_rows.iterrows():
    image_path = row["resolved_path"]

    image_tensor = load_image(image_path)
    batch = tf.expand_dims(image_tensor, axis=0)

    probabilities = model(batch, training=False).numpy()[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} outputs, "
            f"received {len(probabilities)}."
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    result = {
        "filename": image_path.name,
        "actual_class": str(row["label"]),
        "predicted_class": predicted_class,
        "confidence": confidence,
    }

    for index, class_name in enumerate(CLASS_NAMES):
        result[f"prob_{class_name}"] = float(probabilities[index])

    fresh_results.append(result)

    print(f"\nImage: {image_path.name}")
    print(f"Actual class: {row['label']}")
    print(f"Fresh predicted class: {predicted_class}")
    print(f"Fresh confidence: {confidence:.6%}")

    print("\nClass probabilities:")

    for index, class_name in enumerate(CLASS_NAMES):
        print(f"  {class_name}: {probabilities[index]:.6%}")


# ============================================================
# COMPARE SAVED AND FRESH CONFIDENCE
# ============================================================

print("\n" + "=" * 65)
print("6. CONFIDENCE COMPARISON")
print("=" * 65)

if PREDICTIONS.exists():
    saved = pd.read_csv(PREDICTIONS)

    saved_paths = (
        saved["path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    saved_matches = saved.loc[saved_paths.str.endswith(TARGET)]

    for fresh in fresh_results:
        print(f"\nImage: {fresh['filename']}")

        if saved_matches.empty:
            print("No saved prediction available for comparison.")
            continue

        for _, saved_row in saved_matches.iterrows():
            saved_class = str(saved_row["predicted_class"])
            saved_confidence = float(saved_row["confidence"])

            print(f"Saved predicted class: {saved_class}")
            print(f"Fresh predicted class: {fresh['predicted_class']}")
            print(f"Saved confidence: {saved_confidence:.6%}")
            print(f"Fresh confidence: {fresh['confidence']:.6%}")

            difference = abs(
                saved_confidence - fresh["confidence"]
            )

            print(f"Absolute confidence difference: {difference:.6%}")

            if saved_class != fresh["predicted_class"]:
                print("WARNING: Predicted classes do not match.")
            elif difference > 0.01:
                print(
                    "WARNING: Confidence differs by more than "
                    "1 percentage point."
                )
            else:
                print("Saved and fresh confidence are close.")


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DIAGNOSTIC COMPLETE")
print("=" * 65)

print("No model weights or evaluation files were modified.")
print("Review the comparisons above before changing the model.")
