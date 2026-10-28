"""Image loading and normalization pipelines for Brain Tumor MRI Classification V2."""

from __future__ import annotations
from pathlib import Path
from PIL import Image
import numpy as np
import tensorflow as tf

from config import IMAGE_SIZE, CLASS_NAMES, CLASS_TO_INDEX


def load_image_and_label(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode JPEG/PNG MRI image, cast to 3 RGB channels, and resize to standard 224x224."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    image = tf.cast(image, tf.float32)
    return image, label


def normalize_for_cnn(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Scale pixel intensity values to [0, 1] for baseline CNN."""
    return images / 255.0, labels


def normalize_for_efficientnet(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply standard EfficientNet preprocessing (expects [0, 255] float input)."""
    return tf.keras.applications.efficientnet.preprocess_input(images), labels


def preprocess_single_image(
    image_input: str | Path | Image.Image,
    model_type: str = "efficientnet",
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Standardized preprocessing for inference on a single image.

    Args:
        image_input: File path (str/Path) or PIL Image instance.
        model_type: 'efficientnet' or 'cnn'.

    Returns:
        tensor_batch: NumPy array of shape (1, 224, 224, 3) ready for model.predict.
        original_rgb: Original uint8 RGB image array (H, W, 3) for display/Grad-CAM.
        orig_size: (width, height) tuple of original input image.
    """
    if isinstance(image_input, (str, Path)):
        path = Path(image_input)
        if not path.is_file():
            raise FileNotFoundError(f"Image not found at: {path.resolve()}")
        pil_img = Image.open(path).convert("RGB")
    elif isinstance(image_input, Image.Image):
        pil_img = image_input.convert("RGB")
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    orig_size = pil_img.size  # (W, H)
    original_rgb = np.array(pil_img, dtype=np.uint8)

    # Standardized 224x224 bilinear resize
    resized_pil = pil_img.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    img_array = np.array(resized_pil, dtype=np.float32)
    tensor_batch = np.expand_dims(img_array, axis=0)

    if model_type == "cnn":
        tensor_batch = tensor_batch / 255.0
    elif model_type == "efficientnet":
        tensor_batch = np.asarray(tf.keras.applications.efficientnet.preprocess_input(tensor_batch))
    else:
        raise ValueError(f"Unknown model_type: '{model_type}'. Expected 'cnn' or 'efficientnet'.")

    return tensor_batch, original_rgb, orig_size
