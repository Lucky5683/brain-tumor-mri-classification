"""Load and preprocess the Brain Tumor MRI dataset for later training.

This script does not write resized copies of the images. It resizes images while
they are loaded, which keeps the project small and leaves the original dataset
unchanged.

The official Testing folder is never used to make the validation split.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

import tensorflow as tf


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Keep this order fixed. The folder named "notumor" represents the no-tumor class.
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]


# These transformations are deliberately conservative for MRI images. They are
# used only while training and do not alter the files on disk.
TRAIN_AUGMENTATION = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        tf.keras.layers.RandomRotation(0.05, seed=RANDOM_SEED),
        tf.keras.layers.RandomZoom(0.10, seed=RANDOM_SEED),
    ],
    name="training_augmentation",
)


def find_images(split_dir: Path) -> dict[str, list[Path]]:
    """Return supported JPEG and PNG images, grouped by the four classes."""
    images_by_class: dict[str, list[Path]] = {}
    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Expected class folder not found: {class_dir}")
        images = sorted(
            path for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"No supported image files found in: {class_dir}")
        images_by_class[class_name] = images
    return images_by_class


def make_stratified_split(
    images_by_class: dict[str, list[Path]],
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Create a reproducible validation split with the same class ratio."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1.")

    train_paths, train_labels, validation_paths, validation_labels = [], [], [], []
    for label, class_name in enumerate(CLASS_NAMES):
        class_paths = images_by_class[class_name].copy()
        random.Random(seed + label).shuffle(class_paths)
        validation_count = round(len(class_paths) * validation_fraction)

        validation_paths.extend(class_paths[:validation_count])
        validation_labels.extend([label] * validation_count)
        train_paths.extend(class_paths[validation_count:])
        train_labels.extend([label] * (len(class_paths) - validation_count))

    return train_paths, train_labels, validation_paths, validation_labels


def flatten_with_labels(images_by_class: dict[str, list[Path]]) -> tuple[list[Path], list[int]]:
    """Convert class-grouped paths to paths and integer labels."""
    paths, labels = [], []
    for label, class_name in enumerate(CLASS_NAMES):
        paths.extend(images_by_class[class_name])
        labels.extend([label] * len(images_by_class[class_name]))
    return paths, labels


def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Decode JPEG/PNG input, convert it to RGB, and resize to 224x224."""
    image_bytes = tf.io.read_file(path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, antialias=True)
    image = tf.cast(image, tf.float32)
    return image, label


def normalize_for_cnn(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Scale RGB pixel values from 0-255 to 0-1 for the custom CNN."""
    return images / 255.0, labels


def normalize_for_efficientnet(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Apply EfficientNetB0's official input preprocessing.

    Current Keras EfficientNet models include their own rescaling layer, so this
    function keeps the expected 0-255 floating-point input range.
    """
    return tf.keras.applications.efficientnet.preprocess_input(images), labels


def build_dataset(
    paths: list[Path],
    labels: list[int],
    *,
    training: bool,
    model_type: str,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Build one train, validation, or test dataset without changing source files."""
    dataset = tf.data.Dataset.from_tensor_slices(([str(path) for path in paths], labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    if training:
        dataset = dataset.map(
            lambda images, labels: (TRAIN_AUGMENTATION(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if model_type == "cnn":
        normalizer = normalize_for_cnn
    elif model_type == "efficientnet":
        normalizer = normalize_for_efficientnet
    else:
        raise ValueError("model_type must be 'cnn' or 'efficientnet'.")

    return dataset.map(normalizer, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)


def get_datasets(
    data_dir: str | Path = "dataset",
    model_type: str = "cnn",
    batch_size: int = BATCH_SIZE,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return train, validation, and untouched official-test datasets.

    Use model_type="cnn" for values normalized to 0-1. Use
    model_type="efficientnet" for EfficientNetB0's official preprocessing.
    """
    data_dir = Path(data_dir)
    training_images = find_images(data_dir / "Training")
    testing_images = find_images(data_dir / "Testing")

    train_paths, train_labels, validation_paths, validation_labels = make_stratified_split(
        training_images, validation_fraction, seed
    )
    test_paths, test_labels = flatten_with_labels(testing_images)

    train_dataset = build_dataset(
        train_paths, train_labels, training=True, model_type=model_type, batch_size=batch_size, seed=seed
    )
    validation_dataset = build_dataset(
        validation_paths, validation_labels, training=False, model_type=model_type, batch_size=batch_size, seed=seed
    )
    test_dataset = build_dataset(
        test_paths, test_labels, training=False, model_type=model_type, batch_size=batch_size, seed=seed
    )
    return train_dataset, validation_dataset, test_dataset


def get_train_validation_datasets(
    data_dir: str | Path = "dataset",
    model_type: str = "cnn",
    batch_size: int = BATCH_SIZE,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Return only train and validation datasets without accessing Testing/."""
    data_dir = Path(data_dir)
    training_images = find_images(data_dir / "Training")
    train_paths, train_labels, validation_paths, validation_labels = make_stratified_split(
        training_images, validation_fraction, seed
    )

    train_dataset = build_dataset(
        train_paths, train_labels, training=True, model_type=model_type, batch_size=batch_size, seed=seed
    )
    validation_dataset = build_dataset(
        validation_paths, validation_labels, training=False, model_type=model_type, batch_size=batch_size, seed=seed
    )
    return train_dataset, validation_dataset


def print_split_summary(data_dir: Path, validation_fraction: float, seed: int) -> None:
    """Print counts before any model is trained."""
    training_images = find_images(data_dir / "Training")
    testing_images = find_images(data_dir / "Testing")
    train_paths, train_labels, validation_paths, validation_labels = make_stratified_split(
        training_images, validation_fraction, seed
    )
    test_paths, test_labels = flatten_with_labels(testing_images)

    print(f"Final image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB")
    print(f"Random seed: {seed}")
    print(f"Train / validation / official test: {len(train_paths)} / {len(validation_paths)} / {len(test_paths)}")
    print("Class counts (train / validation / official test):")
    for label, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name}: "
            f"{Counter(train_labels)[label]} / {Counter(validation_labels)[label]} / {Counter(test_labels)[label]}"
        )
    print("Augmentation: horizontal flip, small rotation (5%), small zoom (10%) — training only.")
    print("Normalization: custom CNN uses 0-1 scaling; EfficientNetB0 uses its official preprocessing.")
    print("The official Testing folder is not augmented and is not used for validation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the Brain Tumor MRI datasets without training a model.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-type", choices=["cnn", "efficientnet"], default="cnn")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()

    print_split_summary(arguments.data_dir, VALIDATION_FRACTION, RANDOM_SEED)
    # Build the lazy TensorFlow input pipelines. This does not train a model.
    get_datasets(arguments.data_dir, arguments.model_type, arguments.batch_size)


if __name__ == "__main__":
    main()
