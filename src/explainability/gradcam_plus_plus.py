"""Grad-CAM++ (Generalized Gradient-Weighted Class Activation Mapping) implementation.

Grad-CAM++ (Chattopadhay et al., 2018) computes higher-order gradient weights for finer
and more complete saliency localization over standard Grad-CAM.
"""

from __future__ import annotations
import numpy as np
import tensorflow as tf

from src.explainability.gradcam import extract_gradcam_submodels, overlay_heatmap_on_image


def compute_gradcam_plus_plus_heatmap(
    feature_extractor: tf.keras.Model,
    head_layers: list[tf.keras.layers.Layer],
    preprocessed_tensor: np.ndarray,
    target_class_idx: int,
) -> np.ndarray:
    """Compute Grad-CAM++ activation heatmap using second- and third-order gradients."""
    if any(l.name == "multiscale_feature_extractor" for l in feature_extractor.layers):
        from src.explainability.gradcam import compute_multiscale_fusion_gradcam
        return compute_multiscale_fusion_gradcam(feature_extractor, preprocessed_tensor, target_class_idx)

    img_tensor = tf.convert_to_tensor(preprocessed_tensor, dtype=tf.float32)

    with tf.GradientTape() as tape_3:
        with tf.GradientTape() as tape_2:
            with tf.GradientTape() as tape_1:
                tape_1.watch(img_tensor)
                conv_outputs = feature_extractor(img_tensor, training=False)
                tape_1.watch(conv_outputs)
                tape_2.watch(conv_outputs)
                tape_3.watch(conv_outputs)

                x = conv_outputs
                for layer in head_layers:
                    x = layer(x, training=False)
                class_score = tf.exp(x[:, target_class_idx])

            first_deriv = tape_1.gradient(class_score, conv_outputs)
        second_deriv = tape_2.gradient(first_deriv, conv_outputs)
    third_deriv = tape_3.gradient(second_deriv, conv_outputs)

    if first_deriv is None or second_deriv is None or third_deriv is None:
        raise RuntimeError("Failed to compute higher order gradients for Grad-CAM++.")

    conv_val = conv_outputs[0]
    grad_1 = first_deriv[0]
    grad_2 = second_deriv[0]
    grad_3 = third_deriv[0]

    # Alpha weights formula for Grad-CAM++
    global_sum = tf.reduce_sum(conv_val * grad_3, axis=(0, 1), keepdims=True)
    denom = 2.0 * grad_2 + global_sum
    denom = tf.where(denom != 0.0, denom, tf.ones_like(denom))
    alphas = grad_2 / denom

    alphas = tf.maximum(alphas, 0.0)
    alphas_normalized = alphas / (tf.reduce_sum(alphas, axis=(0, 1), keepdims=True) + 1e-7)

    weights = tf.reduce_sum(alphas_normalized * tf.maximum(grad_1, 0.0), axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(weights, conv_val), axis=-1)

    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val

    return heatmap.numpy()
