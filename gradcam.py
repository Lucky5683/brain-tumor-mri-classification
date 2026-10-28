"""Grad-CAM for EfficientNet-B0 (Root compatibility wrapper for V2)."""

from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from config import CLASS_NAMES, DISPLAY_NAMES, IMAGE_SIZE
import src.models.compat
from src.data.preprocessing import preprocess_single_image
from src.explainability.gradcam import (
    compute_gradcam_heatmap,
    extract_gradcam_submodels,
    generate_gradcam_figure,
    get_target_conv_layer,
    overlay_heatmap_on_image,
)

DEFAULT_MODEL_PATH = Path("efficientnet_b0.keras")
TARGET_CONV_LAYER = "top_activation"


def load_and_prepare_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    conv_layer_name: str = TARGET_CONV_LAYER,
) -> tuple[tf.keras.Model, tf.keras.Model, list[tf.keras.layers.Layer]]:
    """Load model and build feature extractor and head pipeline."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found at: {model_path.resolve()}")
    model = tf.keras.models.load_model(model_path)
    feat_extractor, head_layers = extract_gradcam_submodels(model)
    return model, feat_extractor, head_layers


def preprocess_image(image_input: Path | str | Image.Image) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Standardized preprocessing matching V1 signature."""
    return preprocess_single_image(image_input, model_type="efficientnet")


def compute_gradcam(
    feature_extractor: tf.keras.Model,
    head_layers: list[tf.keras.layers.Layer],
    preprocessed_image: np.ndarray,
    target_class_idx: int,
) -> np.ndarray:
    """Compute Grad-CAM heatmap."""
    return compute_gradcam_heatmap(feature_extractor, head_layers, preprocessed_image, target_class_idx)


def overlay_heatmap(
    heatmap_2d: np.ndarray,
    orig_image_rgb: np.ndarray,
    alpha: float = 0.40,
    colormap_name: str = "jet",
) -> tuple[np.ndarray, np.ndarray]:
    """Overlay heatmap on RGB image."""
    return overlay_heatmap_on_image(heatmap_2d, orig_image_rgb, alpha=alpha, colormap_name=colormap_name.lower())


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM visualization CLI.")
    parser.add_argument("--image", type=Path, default=None, help="Path to MRI image.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, default=Path("gradcam_output.png"))
    args = parser.parse_args()

    if args.image is None:
        print("Please supply an image path using --image <path>.")
        return

    full_model, feature_extractor, head_layers = load_and_prepare_model(args.model)
    input_tensor, orig_rgb, _ = preprocess_image(args.image)

    probs = full_model.predict(input_tensor, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_name = CLASS_NAMES[pred_idx]
    conf = float(probs[pred_idx])

    heatmap_2d = compute_gradcam(feature_extractor, head_layers, input_tensor, pred_idx)
    heatmap_rgb, overlay_rgb = overlay_heatmap(heatmap_2d, orig_rgb, alpha=0.40)

    generate_gradcam_figure(orig_rgb, heatmap_rgb, overlay_rgb, pred_name, conf, output_path=args.output)
    print(f"Prediction: {DISPLAY_NAMES.get(pred_name, pred_name.title())} ({conf*100:.2f}%)")
    print(f"Grad-CAM figure saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
