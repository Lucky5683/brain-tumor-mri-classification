from pathlib import Path
import numpy as np
import tensorflow as tf

from src.explainability.gradcam import (
    extract_gradcam_submodels,
    compute_gradcam_heatmap,
)

checkpoint = Path(
    r"experiments\spatial_fusion\meningioma_weighted_50ep_seed42\best_model.keras"
)

if not checkpoint.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint.resolve()}")

print("Loading final spatial-fusion model...")
model = tf.keras.models.load_model(checkpoint, compile=False)

feature_extractor, head_layers = extract_gradcam_submodels(model)

print("Model:", model.name)
print("Input shape:", model.input_shape)
print("Output shape:", model.output_shape)
print("Feature extractor:", feature_extractor.name)
print("Head layers:", [layer.name for layer in head_layers])

# Synthetic input for a technical smoke test only.
sample = np.zeros((1, 224, 224, 3), dtype=np.float32)

print("\nComputing Grad-CAM...")
heatmap = compute_gradcam_heatmap(
    feature_extractor,
    head_layers,
    sample,
    target_class_idx=0,
)

print("Heatmap shape:", heatmap.shape)
print("Heatmap finite:", bool(np.isfinite(heatmap).all()))
print("Heatmap minimum:", float(heatmap.min()))
print("Heatmap maximum:", float(heatmap.max()))

assert np.isfinite(heatmap).all(), "Heatmap contains invalid values."
assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, (
    "Heatmap values fall outside [0, 1]."
)

print("\nGRAD-CAM SMOKE TEST PASSED")
