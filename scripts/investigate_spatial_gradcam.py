
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    ROOT
    / "experiments/spatial_fusion/meningioma_weighted_seed42/best_model.keras"
)

MANIFEST_PATH = (
    ROOT / "experiments/duplicate_aware_split/training_manifest.csv"
)

PREDICTIONS_PATH = (
    ROOT
    / "experiments/spatial_fusion/meningioma_weighted_seed42/"
    "weighted_validation_predictions.csv"
)

OUTPUT_DIR = (
    ROOT
    / "experiments/spatial_fusion/meningioma_weighted_seed42/"
    "gradcam_verified"
)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)

# Four errors and two correct controls.
TARGET_FILES = [
    "Tr-me_886.jpg",
    "Tr-me_309.jpg",
    "Tr-me_307.jpg",
    "Tr-me_1101.jpg",
    "Tr-me_1196.jpg",
    "Tr-me_311.jpg",
]

TARGET_LAYER_NAME = "fusion_conv"


# ============================================================
# PATH AND IMAGE HELPERS
# ============================================================

def resolve_path(value):
    path = Path(str(value))

    if path.is_absolute() and path.exists():
        return path

    candidate = ROOT / path
    if candidate.exists():
        return candidate

    parts = list(path.parts)

    for index, part in enumerate(parts):
        if part.lower() == "dataset":
            candidate = ROOT.joinpath(*parts[index:])
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"Cannot resolve image path: {value}")


def load_image(path):
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


def load_original_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB").resize(
            IMAGE_SIZE,
            Image.Resampling.BILINEAR,
        )


def safe_filename(name):
    return Path(name).stem.replace(" ", "_")


# ============================================================
# VALIDATE INPUT FILES
# ============================================================

for required in [MODEL_PATH, MANIFEST_PATH, PREDICTIONS_PATH]:
    if not required.exists():
        raise FileNotFoundError(f"Required file not found: {required}")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD VALIDATION MANIFEST AND SAVED PREDICTIONS
# ============================================================

manifest = pd.read_csv(MANIFEST_PATH)
manifest = manifest[
    manifest["split"].astype(str).str.lower().eq("validation")
].copy()

manifest["filename"] = manifest["path"].map(
    lambda value: Path(str(value).replace("\\", "/")).name
)

predictions_df = pd.read_csv(PREDICTIONS_PATH)
predictions_df["filename"] = predictions_df["path"].map(
    lambda value: Path(str(value).replace("\\", "/")).name
)

selected_rows = []

for filename in TARGET_FILES:
    matches = manifest[manifest["filename"].eq(filename)]

    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one validation manifest row for {filename}; "
            f"found {len(matches)}."
        )

    saved_matches = predictions_df[
        predictions_df["filename"].eq(filename)
    ]

    if len(saved_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one saved prediction for {filename}; "
            f"found {len(saved_matches)}."
        )

    row = matches.iloc[0].copy()
    row["resolved_path"] = resolve_path(row["path"])

    if str(row["label"]) != str(saved_matches.iloc[0]["actual_class"]):
        raise RuntimeError(
            f"Actual-label mismatch for {filename}: "
            f"manifest={row['label']}, "
            f"saved CSV={saved_matches.iloc[0]['actual_class']}"
        )

    row["saved_prediction"] = saved_matches.iloc[0]
    selected_rows.append(row)

print(f"Selected validation images: {len(selected_rows)}")


# ============================================================
# LOAD MODEL AND BUILD GRAD-CAM GRAPH
# ============================================================

model = tf.keras.models.load_model(
    str(MODEL_PATH),
    compile=False,
)

target_layer = model.get_layer(TARGET_LAYER_NAME)

grad_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=[target_layer.output, model.output],
)

print(f"Loaded checkpoint: {MODEL_PATH}")
print(f"Grad-CAM target layer: {TARGET_LAYER_NAME}")
print(f"Feature tensor: {target_layer.output.shape}")


# ============================================================
# COMPUTE GRAD-CAM USING THE ACTUAL MODEL GRAPH
# ============================================================

def compute_gradcam(image_tensor, class_index):
    batch = tf.expand_dims(image_tensor, axis=0)

    with tf.GradientTape() as tape:
        feature_maps, probabilities = grad_model(
            batch,
            training=False,
        )

        class_score = probabilities[:, class_index]

    gradients = tape.gradient(class_score, feature_maps)

    if gradients is None:
        raise RuntimeError(
            "Gradients are unavailable. Check the selected target layer."
        )

    feature_maps = feature_maps[0]
    gradients = gradients[0]

    weights = tf.reduce_mean(
        gradients,
        axis=(0, 1),
    )

    heatmap = tf.reduce_sum(
        feature_maps * weights,
        axis=-1,
    )

    heatmap = tf.maximum(heatmap, 0)
    maximum = tf.reduce_max(heatmap)

    heatmap = tf.math.divide_no_nan(heatmap, maximum)

    heatmap = tf.image.resize(
        heatmap[..., tf.newaxis],
        IMAGE_SIZE,
        method="bilinear",
    )[..., 0]

    return heatmap.numpy(), float(
        tf.reduce_mean(tf.abs(gradients)).numpy()
    )


# ============================================================
# RUN INFERENCE, VERIFY CONFIDENCE, SAVE HEATMAPS
# ============================================================

results = []

for row in selected_rows:
    image_path = row["resolved_path"]
    filename = image_path.name
    true_class = str(row["label"])
    saved = row["saved_prediction"]

    image_tensor = load_image(image_path)

    # Use the full model graph for predictions.
    probabilities = model(
        tf.expand_dims(image_tensor, axis=0),
        training=False,
    ).numpy()[0]

    if probabilities.shape != (len(CLASS_NAMES),):
        raise RuntimeError(
            f"Unexpected output shape for {filename}: {probabilities.shape}"
        )

    predicted_index = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index])

    saved_class = str(saved["predicted_class"])
    saved_confidence = float(saved["confidence"])
    confidence_difference = abs(confidence - saved_confidence)

    # Save an audit row even if verification fails.
    row_result = {
        "filename": filename,
        "true_class": true_class,
        "saved_predicted_class": saved_class,
        "fresh_predicted_class": predicted_class,
        "saved_confidence": saved_confidence,
        "fresh_confidence": confidence,
        "confidence_difference": confidence_difference,
        "class_match": predicted_class == saved_class,
        "confidence_match_within_1e-4": confidence_difference <= 1e-4,
        "target_layer": TARGET_LAYER_NAME,
        "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
    }

    for index, class_name in enumerate(CLASS_NAMES):
        row_result[f"prob_{class_name}"] = float(probabilities[index])

    print(
        f"\n{filename}: actual={true_class}, "
        f"saved={saved_class} ({saved_confidence:.6f}), "
        f"fresh={predicted_class} ({confidence:.6f}), "
        f"difference={confidence_difference:.6f}"
    )

    # Refuse to interpret Grad-CAM until prediction checks pass.
    if predicted_class != saved_class or confidence_difference > 1e-4:
        row_result["gradcam_generated"] = False
        row_result["status"] = "PREDICTION_VERIFICATION_FAILED"
        results.append(row_result)

        print(
            "  WARNING: Prediction verification failed. "
            "Heatmap was not generated for this image."
        )
        continue

    heatmap, gradient_abs_mean = compute_gradcam(
        image_tensor,
        predicted_index,
    )

    rgb_image = load_original_rgb(image_path)
    heatmap_uint8 = np.uint8(np.clip(heatmap, 0, 1) * 255)

    stem = safe_filename(filename)

    original_path = OUTPUT_DIR / f"{stem}_original.png"
    heatmap_path = OUTPUT_DIR / f"{stem}_heatmap.png"
    overlay_path = OUTPUT_DIR / f"{stem}_overlay.png"
    comparison_path = OUTPUT_DIR / f"{stem}_comparison.png"

    rgb_image.save(original_path)

    plt.figure(figsize=(5, 5))
    plt.imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(heatmap_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Visual overlay only; not a segmentation mask.
    plt.figure(figsize=(5, 5))
    plt.imshow(rgb_image)
    plt.imshow(heatmap, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(overlay_path, dpi=160, bbox_inches="tight", pad_inches=0)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(rgb_image)
    axes[0].set_title("Original MRI")

    axes[1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM")

    axes[2].imshow(rgb_image)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.40, vmin=0, vmax=1)
    axes[2].set_title("Overlay")

    for axis in axes:
        axis.axis("off")

    fig.suptitle(
        f"{filename}\nActual: {true_class} | Predicted: {predicted_class} "
        f"({confidence:.2%})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(comparison_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    row_result.update({
        "gradcam_generated": True,
        "status": "VERIFIED",
        "gradient_abs_mean": gradient_abs_mean,
        "heatmap_min": float(np.min(heatmap)),
        "heatmap_max": float(np.max(heatmap)),
        "heatmap_mean": float(np.mean(heatmap)),
        "original_path": str(original_path),
        "heatmap_path": str(heatmap_path),
        "overlay_path": str(overlay_path),
        "comparison_path": str(comparison_path),
    })

    results.append(row_result)


# ============================================================
# SAVE AUDIT RESULTS
# ============================================================

results_df = pd.DataFrame(results)

csv_path = OUTPUT_DIR / "gradcam_verified_results.csv"
json_path = OUTPUT_DIR / "run_summary.json"

results_df.to_csv(csv_path, index=False)

summary = {
    "checkpoint": str(MODEL_PATH),
    "manifest": str(MANIFEST_PATH),
    "saved_predictions": str(PREDICTIONS_PATH),
    "target_layer": TARGET_LAYER_NAME,
    "selected_images": len(results_df),
    "prediction_checks_passed": int(
        (
            results_df["class_match"]
            & results_df["confidence_match_within_1e-4"]
        ).sum()
    ),
    "gradcam_generated": int(results_df["gradcam_generated"].sum()),
    "failed_verification": int(
        (~results_df["class_match"]
         | ~results_df["confidence_match_within_1e-4"]).sum()
    ),
    "results_csv": str(csv_path),
}

json_path.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\n" + "=" * 65)
print("GRAD-CAM VERIFICATION SUMMARY")
print("=" * 65)
print(json.dumps(summary, indent=2))
print("\nNo model weights or original validation results were modified.")
