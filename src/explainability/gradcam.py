
"""Grad-CAM implementation for EfficientNet-B0, spatial-fusion,
multi-scale fusion, and compatible CNN models.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from config import IMAGE_SIZE
import src.models.compat  # Ensure compatibility shim is loaded


def get_target_conv_layer(
    model: tf.keras.Model,
    candidate_names: list[str] | None = None,
) -> tuple[tf.keras.layers.Layer, str]:
    """Find a suitable convolutional or activation layer."""

    if candidate_names is None:
        candidate_names = [
            "fusion_swish",
            "top_activation",
            "top_conv",
            "block7a_project_conv",
        ]

    # Prefer the spatial-fusion feature map when present.
    for name in candidate_names:
        try:
            layer = model.get_layer(name)
            return layer, name
        except ValueError:
            pass

    # Search nested EfficientNet backbones.
    for backbone_name in ("efficientnetb0", "efficientnet-b0"):
        try:
            backbone = model.get_layer(backbone_name)
        except ValueError:
            continue

        for name in candidate_names:
            try:
                layer = backbone.get_layer(name)
                return layer, name
            except ValueError:
                continue

        for layer in reversed(backbone.layers):
            if isinstance(
                layer,
                (tf.keras.layers.Conv2D, tf.keras.layers.Activation),
            ):
                return layer, layer.name

    # Fall back to the final suitable top-level layer.
    for layer in reversed(model.layers):
        if isinstance(
            layer,
            (tf.keras.layers.Conv2D, tf.keras.layers.Activation),
        ):
            return layer, layer.name

    raise ValueError(
        "Could not locate a suitable convolutional or activation "
        "layer for Grad-CAM."
    )


def extract_gradcam_submodels(
    full_model: tf.keras.Model,
) -> tuple[tf.keras.Model, list[tf.keras.layers.Layer]]:
    """Create a feature extractor and its corresponding classifier head."""

    layer_names = {layer.name for layer in full_model.layers}

    # Spatial-fusion architecture:
    # image -> spatial feature extractor -> fusion head -> predictions.
    if "fusion_swish" in layer_names and "spatial_predictions" in layer_names:
        target_layer = full_model.get_layer("fusion_swish")

        feature_extractor = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=target_layer.output,
            name="spatial_fusion_gradcam_extractor",
        )

        head_layer_names = [
            "fusion_pool",
            "fusion_dropout",
            "fusion_dense",
            "classifier_dropout",
            "spatial_predictions",
        ]

        head_layers = [
            full_model.get_layer(name)
            for name in head_layer_names
        ]

        return feature_extractor, head_layers

    # Multi-scale fusion is handled by its dedicated Grad-CAM function.
    if "multiscale_feature_extractor" in layer_names:
        return full_model, []

    # Standard EfficientNet architecture.
    try:
        backbone = full_model.get_layer("efficientnetb0")
        target_conv_layer, _ = get_target_conv_layer(full_model)

        feature_extractor = tf.keras.Model(
            inputs=backbone.input,
            outputs=target_conv_layer.output,
            name="gradcam_feature_extractor",
        )

        head_layer_names = [
            "global_avg_pool",
            "head_dropout_1",
            "head_dense",
            "head_dropout_2",
            "predictions",
        ]

        head_layers = [
            full_model.get_layer(name)
            for name in head_layer_names
            if name in layer_names
        ]

        return feature_extractor, head_layers

    except ValueError:
        pass

    # Direct Functional or Sequential models, including compatible CNNs.
    target_conv_layer, _ = get_target_conv_layer(full_model)

    matching_indices = [
        index
        for index, layer in enumerate(full_model.layers)
        if layer.name == target_conv_layer.name
    ]

    if not matching_indices:
        raise ValueError(
            f"Could not locate target layer {target_conv_layer.name!r} "
            "in the model."
        )

    conv_index = matching_indices[0]

    try:
        feature_extractor = tf.keras.Model(
            inputs=full_model.inputs,
            outputs=target_conv_layer.output,
            name="gradcam_feature_extractor",
        )
    except (AttributeError, ValueError):
        # Compatibility path for Sequential models in Keras 3.
        input_tensor = tf.keras.Input(
            shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)
        )
        x = input_tensor

        for layer in full_model.layers[: conv_index + 1]:
            x = layer(x)

        feature_extractor = tf.keras.Model(
            inputs=input_tensor,
            outputs=x,
            name="gradcam_feature_extractor",
        )

    head_layers = full_model.layers[conv_index + 1:]

    return feature_extractor, head_layers


def compute_multiscale_fusion_gradcam(
    model: tf.keras.Model,
    preprocessed_tensor: np.ndarray,
    target_class_idx: int,
) -> np.ndarray:
    """Compute Grad-CAM for the existing multi-scale fusion model."""

    sub_extractor = model.get_layer("multiscale_feature_extractor")
    img_tensor = tf.convert_to_tensor(
        preprocessed_tensor,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        intermediate_features, final_features = sub_extractor(
            img_tensor,
            training=False,
        )
        tape.watch(final_features)

        intermediate_vector = model.get_layer("intermediate_pool")(
            intermediate_features
        )
        final_vector = model.get_layer("final_pool")(final_features)

        fused = model.get_layer("multiscale_fusion")(
            [intermediate_vector, final_vector]
        )
        x = model.get_layer("fusion_dense")(fused)
        predictions = model.get_layer("fusion_predictions")(x)

        class_score = predictions[:, target_class_idx]

    gradients = tape.gradient(class_score, final_features)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable for the multi-scale model."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = final_features[0]

    heatmap = tf.reduce_sum(
        pooled_gradients * feature_maps,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0.0)

    heatmap_max = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, heatmap_max)

    return heatmap.numpy()


def compute_gradcam_heatmap(
    feature_extractor: tf.keras.Model,
    head_layers: list[tf.keras.layers.Layer],
    preprocessed_tensor: np.ndarray,
    target_class_idx: int,
) -> np.ndarray:
    """Compute a first-order Grad-CAM heatmap in the range [0, 1]."""

    # Preserve the existing multi-scale implementation.
    if any(
        layer.name == "multiscale_feature_extractor"
        for layer in feature_extractor.layers
    ):
        return compute_multiscale_fusion_gradcam(
            feature_extractor,
            preprocessed_tensor,
            target_class_idx,
        )

    img_tensor = tf.convert_to_tensor(
        preprocessed_tensor,
        dtype=tf.float32,
    )

    with tf.GradientTape() as tape:
        conv_outputs = feature_extractor(
            img_tensor,
            training=False,
        )
        tape.watch(conv_outputs)

        x = conv_outputs

        # For spatial fusion this applies:
        # fusion_pool -> fusion_dropout -> fusion_dense
        # -> classifier_dropout -> spatial_predictions.
        # Other supported models retain their configured head layers.
        for layer in head_layers:
            x = layer(x, training=False)

        if x.shape.rank != 2:
            raise ValueError(
                "The classifier head must produce a two-dimensional "
                f"(batch, classes) output; got shape {x.shape}."
            )

        if (
            x.shape[-1] is not None
            and not 0 <= target_class_idx < x.shape[-1]
        ):
            raise ValueError(
                f"Class index {target_class_idx} is out of range "
                f"for {x.shape[-1]} output classes."
            )

        class_score = x[:, target_class_idx]

    gradients = tape.gradient(class_score, conv_outputs)

    if gradients is None:
        raise RuntimeError(
            "Grad-CAM gradients are unavailable. Verify that the "
            "feature extractor output is connected to the selected "
            "classification head."
        )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2),
    )

    feature_maps = conv_outputs[0]

    heatmap = tf.reduce_sum(
        feature_maps * pooled_gradients,
        axis=-1,
    )
    heatmap = tf.maximum(heatmap, 0.0)

    heatmap_max = tf.reduce_max(heatmap)
    heatmap = tf.math.divide_no_nan(heatmap, heatmap_max)

    return heatmap.numpy()


def overlay_heatmap_on_image(
    heatmap_2d: np.ndarray,
    original_rgb: np.ndarray,
    alpha: float = 0.40,
    colormap_name: str = "jet",
) -> tuple[np.ndarray, np.ndarray]:
    """Resize and colorize a heatmap, then blend it with an RGB image.

    Returns:
        colored_heatmap_rgb: Colorized heatmap, shape (H, W, 3).
        blended_overlay_rgb: Blended image, shape (H, W, 3).
    """

    if original_rgb.ndim != 3 or original_rgb.shape[-1] != 3:
        raise ValueError("original_rgb must have shape (height, width, 3).")

    if heatmap_2d.ndim != 2:
        raise ValueError("heatmap_2d must be a two-dimensional array.")

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0.0 and 1.0.")

    height, width = original_rgb.shape[:2]

    normalized_heatmap = np.nan_to_num(
        heatmap_2d,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )
    normalized_heatmap = np.clip(normalized_heatmap, 0.0, 1.0)

    heatmap_image = Image.fromarray(
        (normalized_heatmap * 255).astype(np.uint8)
    )
    resized_image = heatmap_image.resize(
        (width, height),
        Image.Resampling.BILINEAR,
    )

    resized_heatmap = (
        np.asarray(resized_image, dtype=np.float32) / 255.0
    )

    colormap = plt.colormaps.get_cmap(colormap_name.lower())
    colored_rgba = colormap(resized_heatmap)

    colored_heatmap_rgb = (
        colored_rgba[..., :3] * 255
    ).astype(np.uint8)

    original_float = np.clip(
        original_rgb,
        0,
        255,
    ).astype(np.float32)

    blended = (
        (1.0 - alpha) * original_float
        + alpha * colored_heatmap_rgb.astype(np.float32)
    )

    blended_overlay_rgb = np.clip(
        blended,
        0,
        255,
    ).astype(np.uint8)

    return colored_heatmap_rgb, blended_overlay_rgb
