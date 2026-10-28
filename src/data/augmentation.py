"""Medically controlled data augmentation pipeline for Brain Tumor MRI Classification.

Transformations are strictly conservative and medically plausible for neuroimaging:
1. Horizontal flip: Brain axial slices possess rough bilateral symmetry across the sagittal plane.
2. Vertical flip: PROHIBITED. Inverting along the anterior-posterior axis creates unphysical
   anatomical representations that degrade clinical feature learning.
3. Modest rotation (+/- 5%, ~18 degrees): Simulates realistic head tilt inside MRI coils.
4. Modest zoom/translation (+/- 5-8%): Accommodates variations in patient positioning and FOV.
5. Modest brightness/contrast (+/- 10%): Simulates slight scanner gain/T1/T2 acquisition variances.

Augmentation is strictly applied to the training set only; validation and test sets remain pristine.
"""

from __future__ import annotations
import tensorflow as tf
from config import AUGMENTATION_CONFIG, RANDOM_SEED


def build_training_augmentation(
    config: dict = AUGMENTATION_CONFIG,
    seed: int = RANDOM_SEED,
) -> tf.keras.Sequential:
    """Construct Keras sequential augmentation layer based on configuration."""
    layers = []

    if config.get("horizontal_flip", True):
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))

    rot = config.get("rotation_factor", 0.05)
    if rot > 0:
        layers.append(tf.keras.layers.RandomRotation(rot, seed=seed, fill_mode="constant", fill_value=0.0))

    zoom = config.get("zoom_height_factor", 0.08)
    if zoom > 0:
        layers.append(tf.keras.layers.RandomZoom(zoom, seed=seed, fill_mode="constant", fill_value=0.0))

    trans = config.get("translation_factor", 0.05)
    if trans > 0:
        layers.append(tf.keras.layers.RandomTranslation(trans, trans, seed=seed, fill_mode="constant", fill_value=0.0))

    bright = config.get("brightness_factor", 0.10)
    if bright > 0:
        layers.append(tf.keras.layers.RandomBrightness(bright, seed=seed, value_range=(0.0, 255.0)))

    contrast = config.get("contrast_factor", 0.10)
    if contrast > 0:
        layers.append(tf.keras.layers.RandomContrast(contrast, seed=seed))

    return tf.keras.Sequential(layers, name="medical_training_augmentation")
