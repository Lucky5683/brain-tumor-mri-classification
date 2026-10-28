"""Modular Baseline Custom CNN architecture for Brain Tumor MRI Classification."""

from __future__ import annotations
import tensorflow as tf
from config import IMAGE_SIZE, NUM_CLASSES, RANDOM_SEED


def build_custom_cnn(
    input_shape: tuple[int, int, int] = (IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    num_classes: int = NUM_CLASSES,
    dropout_1: float = 0.30,
    dropout_2: float = 0.40,
    learning_rate: float = 1e-3,
) -> tf.keras.Model:
    """Build the baseline 4-block convolutional network."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape, name="input_image"),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu", name="conv2d_1"),
            tf.keras.layers.MaxPooling2D(name="maxpool_1"),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu", name="conv2d_2"),
            tf.keras.layers.MaxPooling2D(name="maxpool_2"),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu", name="conv2d_3"),
            tf.keras.layers.MaxPooling2D(name="maxpool_3"),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu", name="conv2d_4"),
            tf.keras.layers.MaxPooling2D(name="maxpool_4"),
            tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool"),
            tf.keras.layers.Dropout(dropout_1, name="dropout_gap"),
            tf.keras.layers.Dense(128, activation="relu", name="dense_head"),
            tf.keras.layers.Dropout(dropout_2, name="dropout_head"),
            tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions"),
        ],
        name="baseline_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model
