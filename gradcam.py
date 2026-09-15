"""Grad-CAM (Gradient-weighted Class Activation Mapping) for EfficientNetB0.

This script:
1. Loads the trained EfficientNetB0 model (efficientnet_b0.keras).
2. Extracts the final convolutional feature layer ('top_activation' in the EfficientNetB0 backbone).
3. Computes gradients of the target class score with respect to feature maps using tf.GradientTape.
4. Generates a normalized class activation heatmap.
5. Resizes and overlays the heatmap onto the original MRI image.
6. Saves a publication-quality 3-panel visualization:
   [Original MRI] | [Grad-CAM Heatmap] | [Grad-CAM Overlay]
   with predicted class and prediction confidence.

Usage:
  python gradcam.py --image "dataset/Testing/glioma/Te-gl_1.jpg"
  python gradcam.py --examples
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from preprocess import CLASS_NAMES, IMAGE_SIZE


DEFAULT_MODEL_PATH = Path("efficientnet_b0.keras")
TARGET_CONV_LAYER = "top_activation"


def load_and_prepare_model(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    conv_layer_name: str = TARGET_CONV_LAYER,
) -> tuple[tf.keras.Model, tf.keras.Model, list[tf.keras.layers.Layer]]:
    """Load the EfficientNetB0 model and construct the feature extractor and head pipeline.

    Returns:
        full_model: The loaded end-to-end Keras model.
        feature_extractor: Sub-model mapping input image -> final conv feature maps.
        head_layers: Sequential list of classification head layers.
    """
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path.resolve()}")

    full_model = tf.keras.models.load_model(model_path)

    # EfficientNetB0 backbone is the 'efficientnetb0' layer
    backbone = full_model.get_layer("efficientnetb0")

    # Locate the target convolutional/activation layer in the backbone
    try:
        conv_layer = backbone.get_layer(conv_layer_name)
    except ValueError:
        # Fallback to 'top_conv' if 'top_activation' is not found
        conv_layer_name = "top_conv"
        conv_layer = backbone.get_layer(conv_layer_name)

    feature_extractor = tf.keras.Model(
        inputs=backbone.input,
        outputs=conv_layer.output,
        name="backbone_feature_extractor",
    )

    head_layers = [
        full_model.get_layer("global_avg_pool"),
        full_model.get_layer("head_dropout_1"),
        full_model.get_layer("head_dense"),
        full_model.get_layer("head_dropout_2"),
        full_model.get_layer("predictions"),
    ]

    return full_model, feature_extractor, head_layers


def preprocess_image(image_input: Path | str | Image.Image) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Load and preprocess an MRI image preserving the original image for display.

    Args:
        image_input: File path (str/Path) or an existing PIL.Image instance.

    Returns:
        preprocessed_tensor: Float32 array shaped (1, 224, 224, 3) ready for EfficientNet.
        orig_image_rgb: Original uint8 RGB numpy array (H, W, 3).
        orig_size: Tuple (width, height) of the original image.
    """
    if isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found: {path.resolve()}")
        pil_img = Image.open(path).convert("RGB")

    orig_size = pil_img.size  # (width, height)
    orig_image_rgb = np.array(pil_img, dtype=np.uint8)

    # Resize to 224x224 using bilinear interpolation
    resized_pil = pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized_pil, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)

    # Official EfficientNet preprocessing
    preprocessed_tensor = tf.keras.applications.efficientnet.preprocess_input(img_array)
    return preprocessed_tensor, orig_image_rgb, orig_size



def compute_gradcam(
    feature_extractor: tf.keras.Model,
    head_layers: list[tf.keras.layers.Layer],
    preprocessed_image: np.ndarray,
    target_class_idx: int,
) -> np.ndarray:
    """Compute Grad-CAM activation heatmap for the specified class index.

    Args:
        feature_extractor: Sub-model outputting backbone feature maps (7, 7, 1280).
        head_layers: Head layers mapping feature maps -> class predictions.
        preprocessed_image: Input tensor (1, 224, 224, 3).
        target_class_idx: Integer index of the class to explain.

    Returns:
        heatmap: 2D numpy array (7, 7) normalized between 0.0 and 1.0.
    """
    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(preprocessed_image, training=False)
        tape.watch(conv_outputs)

        # Forward pass through classification head
        x = conv_outputs
        for layer in head_layers:
            x = layer(x, training=False)
        predictions = x
        loss = predictions[:, target_class_idx]

    # Gradients of target class score with respect to feature maps
    grads = tape.gradient(loss, conv_outputs)
    if grads is None:
        raise RuntimeError("Grad-CAM gradient calculation returned None.")

    # Global average pooling of gradients: alpha_k = (1/Z) * sum(grad)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Weight feature map channels by gradient importance: sum_k (alpha_k * A_k)
    cam = conv_outputs[0] @ pooled_grads[..., tf.newaxis]
    cam = tf.squeeze(cam)

    # Apply ReLU: only features having positive influence on the class
    cam = tf.maximum(cam, 0.0)

    # Normalize to [0.0, 1.0]
    cam_max = tf.math.reduce_max(cam)
    if float(cam_max) > 0.0:
        cam = cam / cam_max
    else:
        cam = tf.zeros_like(cam)

    return cam.numpy()


def overlay_heatmap(
    heatmap_2d: np.ndarray,
    orig_image_rgb: np.ndarray,
    alpha: float = 0.40,
    colormap_name: str = "jet",
) -> tuple[np.ndarray, np.ndarray]:
    """Resize heatmap to original MRI dimensions, apply colormap, and blend with MRI.

    Args:
        heatmap_2d: 2D array (7, 7) with values in [0, 1].
        orig_image_rgb: Original uint8 RGB image array (H, W, 3).
        alpha: Heatmap blend opacity (0.0 = only MRI, 1.0 = only heatmap).
        colormap_name: Matplotlib colormap name ('jet', 'viridis', etc.).

    Returns:
        heatmap_rgb: Colorized heatmap as uint8 array (H, W, 3).
        blended: Overlay image as uint8 array (H, W, 3).
    """
    orig_h, orig_w = orig_image_rgb.shape[:2]

    # Resize 7x7 heatmap up to original image dimensions
    heatmap_pil = Image.fromarray(np.uint8(255 * heatmap_2d))
    heatmap_resized = heatmap_pil.resize((orig_w, orig_h), Image.Resampling.BICUBIC)
    heatmap_norm = np.array(heatmap_resized, dtype=np.float32) / 255.0

    # Colorize heatmap using colormap
    colormap = plt.get_cmap(colormap_name)
    heatmap_colored = colormap(heatmap_norm)[:, :, :3]  # Drop alpha channel, values in [0, 1]
    heatmap_rgb = np.uint8(255 * heatmap_colored)

    # Blend with original MRI
    orig_float = orig_image_rgb.astype(np.float32) / 255.0
    overlay_float = (1.0 - alpha) * orig_float + alpha * heatmap_colored
    overlay_float = np.clip(overlay_float, 0.0, 1.0)
    blended = np.uint8(255 * overlay_float)

    return heatmap_rgb, blended


def generate_gradcam_visualization(
    image_path: Path | str,
    output_path: Path | str | None = None,
    model_path: Path | str = DEFAULT_MODEL_PATH,
    target_class: str | None = None,
    alpha: float = 0.40,
    true_label: str | None = None,
) -> dict:
    """Run end-to-end Grad-CAM on a single MRI image and save the 3-panel figure."""
    image_path = Path(image_path)
    if output_path is None:
        output_path = image_path.parent / f"{image_path.stem}_gradcam.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load model and prepare feature extractor
    full_model, feature_extractor, head_layers = load_and_prepare_model(model_path)

    # 2. Preprocess image
    preprocessed_img, orig_image_rgb, orig_size = preprocess_image(image_path)

    # 3. Model prediction
    predictions = full_model.predict(preprocessed_img, verbose=0)[0]
    pred_idx = int(np.argmax(predictions))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(predictions[pred_idx])

    # 4. Determine target class for Grad-CAM
    if target_class is not None:
        if target_class.lower() not in CLASS_NAMES:
            raise ValueError(f"Unknown target class '{target_class}'. Must be one of {CLASS_NAMES}")
        cam_class_idx = CLASS_NAMES.index(target_class.lower())
        cam_class_name = target_class.lower()
    else:
        cam_class_idx = pred_idx
        cam_class_name = pred_class

    # 5. Compute Grad-CAM heatmap
    heatmap_2d = compute_gradcam(
        feature_extractor=feature_extractor,
        head_layers=head_layers,
        preprocessed_image=preprocessed_img,
        target_class_idx=cam_class_idx,
    )

    # 6. Overlay heatmap on original image
    heatmap_rgb, overlay_rgb = overlay_heatmap(
        heatmap_2d=heatmap_2d,
        orig_image_rgb=orig_image_rgb,
        alpha=alpha,
        colormap_name="jet",
    )

    # 7. Render publication-style 3-panel figure
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    # Panel 1: Original MRI
    axes[0].imshow(orig_image_rgb)
    panel1_title = "Original MRI"
    if true_label:
        panel1_title += f" (True: {true_label})"
    axes[0].set_title(panel1_title, fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Panel 2: Grad-CAM Heatmap
    im1 = axes[1].imshow(heatmap_rgb)
    axes[1].set_title(f"Grad-CAM Heatmap\n(Layer: {TARGET_CONV_LAYER})", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    # Panel 3: Grad-CAM Overlay
    axes[2].imshow(overlay_rgb)
    axes[2].set_title(f"Grad-CAM Overlay\n({cam_class_name.upper()}, alpha={alpha:.2f})", fontsize=12, fontweight="bold")
    axes[2].axis("off")

    # Main Super-title with prediction and confidence
    main_title = f"EfficientNetB0 Prediction: {pred_class.upper()} ({confidence * 100:.2f}%)"
    fig.suptitle(main_title, fontsize=14, fontweight="bold", y=0.98)

    # Subtext explaining all class probabilities
    prob_str = " | ".join(f"{c}: {predictions[i]*100:.1f}%" for i, c in enumerate(CLASS_NAMES))
    fig.text(0.5, 0.03, f"Class Probabilities: {prob_str}", ha="center", fontsize=10, style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 0.94])
    plt.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    print(f"\nGrad-CAM generated successfully!")
    print(f"  Input image        : {image_path}")
    print(f"  Predicted class    : {pred_class} ({confidence * 100:.2f}%)")
    if true_label:
        print(f"  True class         : {true_label}")
    print(f"  Explaining class   : {cam_class_name}")
    print(f"  Feature conv layer : {TARGET_CONV_LAYER}")
    print(f"  Output saved to    : {output_path.resolve()}")

    return {
        "image_path": str(image_path),
        "predicted_class": pred_class,
        "confidence": confidence,
        "target_class": cam_class_name,
        "probabilities": {name: float(predictions[i]) for i, name in enumerate(CLASS_NAMES)},
        "output_path": str(output_path.resolve()),
        "conv_layer_used": TARGET_CONV_LAYER,
    }


def run_sample_examples(model_path: Path | str = DEFAULT_MODEL_PATH) -> list[dict]:
    """Run Grad-CAM on one representative sample from each of the four classes."""
    output_dir = Path("gradcam_samples")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Use 1 representative image from each class in dataset/Testing
    sample_targets = [
        ("dataset/Testing/glioma/Te-gl_1.jpg", "glioma"),
        ("dataset/Testing/meningioma/Te-aug-me_1.jpg", "meningioma"),
        ("dataset/Testing/notumor/Te-no_1.jpg", "notumor"),
        ("dataset/Testing/pituitary/Te-pi_1.jpg", "pituitary"),
    ]

    results = []
    print("\n==================================================")
    print("RUNNING GRAD-CAM ON REPRESENTATIVE CLASS EXAMPLES")
    print("==================================================")
    for img_path_str, expected_class in sample_targets:
        img_path = Path(img_path_str)
        out_path = output_dir / f"gradcam_{expected_class}_{img_path.stem}.png"
        res = generate_gradcam_visualization(
            image_path=img_path,
            output_path=out_path,
            model_path=model_path,
            true_label=expected_class,
        )
        results.append(res)
    print("\n==================================================")
    print(f"All {len(results)} sample visualizations saved in: {output_dir.resolve()}")
    print("==================================================")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM explainability for EfficientNetB0 Brain Tumor MRI classification.")
    parser.add_argument("--image", type=str, default=None, help="Path to brain MRI image to explain.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH, help="Path to saved model checkpoint.")
    parser.add_argument("--output", type=str, default=None, help="Output path for visualization image.")
    parser.add_argument("--target-class", type=str, default=None, help="Optional class to explain (default: predicted class).")
    parser.add_argument("--alpha", type=float, default=0.40, help="Overlay opacity for heatmap (0.0-1.0).")
    parser.add_argument("--examples", action="store_true", help="Generate example Grad-CAM visualizations for all 4 classes.")
    args = parser.parse_args()

    if args.examples:
        run_sample_examples(model_path=args.model_path)
    elif args.image:
        generate_gradcam_visualization(
            image_path=args.image,
            output_path=args.output,
            model_path=args.model_path,
            target_class=args.target_class,
            alpha=args.alpha,
        )
    else:
        # Default behavior: run examples if no arguments provided
        print("No image specified. Running representative examples across all 4 classes...")
        run_sample_examples(model_path=args.model_path)


if __name__ == "__main__":
    main()
