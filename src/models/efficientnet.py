"""EfficientNet-B0 architecture and two-stage transfer learning setup."""

from __future__ import annotations
import tensorflow as tf
from config import DROPOUT_RATE, IMAGE_SIZE, NUM_CLASSES
import src.models.compat  # ensure compatibility shim is loaded


def build_efficientnet_b0_stage1(
    num_classes: int = NUM_CLASSES,
    learning_rate: float = 1e-3,
    dropout_rate: float = DROPOUT_RATE,
) -> tf.keras.Model:
    """Build Stage 1 transfer learning model with completely frozen ImageNet backbone."""
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    )
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), name="input_image")
    # training=False keeps BatchNorm layers strictly in inference mode
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(dropout_rate, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_stage1")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def prepare_stage2_finetuning(
    stage1_model: tf.keras.Model,
    unfreeze_layers: int = 20,
    learning_rate: float = 1e-5,
) -> tf.keras.Model:
    """Unfreeze the top portion of the backbone and recompile with a small learning rate.

    Keeps BatchNormalization layers frozen to maintain stable running statistics.
    """
    backbone = stage1_model.get_layer("efficientnetb0")
    backbone.trainable = True

    # Freeze all except the last `unfreeze_layers`
    for layer in backbone.layers[:-unfreeze_layers]:
        layer.trainable = False

    # Freeze BatchNormalization layers across the entire backbone
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    stage1_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return stage1_model


def count_parameters(model: tf.keras.Model) -> dict[str, int]:
    """Return dictionary of parameter counts."""
    import numpy as np
    total = model.count_params()
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    frozen = total - trainable
    return {"total": total, "trainable": trainable, "frozen": frozen}
