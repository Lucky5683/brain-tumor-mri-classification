
"""Reproducible dataset loading, duplicate-aware splitting, and tf.data pipelines."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path

import tensorflow as tf

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    DATA_DIR,
    IMAGE_EXTENSIONS,
    RANDOM_SEED,
    REPORTS_DIR,
    TEST_DIR,
    TRAIN_DIR,
    VALIDATION_FRACTION,
)
from src.data.augmentation import build_training_augmentation
from src.data.preprocessing import (
    load_image_and_label,
    normalize_for_cnn,
    normalize_for_efficientnet,
)


def find_images(split_dir: Path) -> dict[str, list[Path]]:
    """Scan a dataset directory and group image paths by class."""
    split_dir = Path(split_dir)
    images_by_class: dict[str, list[Path]] = {}

    for class_name in CLASS_NAMES:
        class_dir = split_dir / class_name

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Required class directory not found: {class_dir.resolve()}"
            )

        images = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise RuntimeError(
                f"No valid images found in: {class_dir.resolve()}"
            )

        images_by_class[class_name] = images

    return images_by_class


def make_stratified_split(
    images_by_class: dict[str, list[Path]],
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Generate a deterministic, class-stratified train/validation split."""
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction must be strictly between 0 and 1."
        )

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        paths = images_by_class[class_name].copy()

        rng = random.Random(seed + label_idx)
        rng.shuffle(paths)

        val_count = round(len(paths) * validation_fraction)

        val_paths.extend(paths[:val_count])
        val_labels.extend([label_idx] * val_count)

        train_paths.extend(paths[val_count:])
        train_labels.extend([label_idx] * (len(paths) - val_count))

    return train_paths, train_labels, val_paths, val_labels


def flatten_class_dict(
    images_by_class: dict[str, list[Path]],
) -> tuple[list[Path], list[int]]:
    """Flatten class-grouped image paths into paths and integer labels."""
    paths, labels = [], []

    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_paths = images_by_class[class_name]
        paths.extend(class_paths)
        labels.extend([label_idx] * len(class_paths))

    return paths, labels


def load_manifest_split(
    manifest_path: Path,
    images_by_class: dict[str, list[Path]],
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """
    Load a fixed train/validation manifest.

    Validates dataset coverage, class labels, file hashes, and ensures
    that identical SHA-256 image contents cannot cross the two splits.
    """
    manifest_path = Path(manifest_path)

    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Split manifest not found: {manifest_path.resolve()}"
        )

    expected_paths = {
        path.resolve()
        for paths in images_by_class.values()
        for path in paths
    }

    train_paths, train_labels = [], []
    val_paths, val_labels = [], []
    seen_paths = set()
    declared_group_splits = {}

    with manifest_path.open("r", newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)

        required_columns = {
            "path",
            "label",
            "split",
            "sha256",
            "group_id",
        }

        if (
            reader.fieldnames is None
            or not required_columns.issubset(reader.fieldnames)
        ):
            raise ValueError(
                "Manifest must contain columns: "
                f"{sorted(required_columns)}"
            )

        for row in reader:
            path = Path(row["path"]).resolve()
            label = row["label"].strip()
            split = row["split"].strip().lower()
            declared_hash = row["sha256"].strip().lower()
            group_id = row["group_id"].strip()

            if path in seen_paths:
                raise ValueError(
                    f"Duplicate path in manifest: {path}"
                )
            seen_paths.add(path)

            if label not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class label {label!r}: {path}"
                )

            if split not in {"train", "validation"}:
                raise ValueError(
                    f"Invalid split {split!r}: {path}"
                )

            if path not in expected_paths:
                raise ValueError(
                    f"Manifest contains an image outside the current "
                    f"training dataset: {path}"
                )

            if not path.is_file():
                raise FileNotFoundError(
                    f"Manifest image is missing: {path}"
                )

            # Check that the image is inside its declared class folder.
            if path.parent.name != label:
                raise ValueError(
                    f"Label/path mismatch: label={label}, path={path}"
                )

            if not declared_hash or not group_id:
                raise ValueError(
                    f"Missing SHA-256 hash or group ID: {path}"
                )

            # Verify that the file content still matches the manifest.
            digest = hashlib.sha256(path.read_bytes()).hexdigest()

            if digest != declared_hash:
                raise ValueError(
                    f"SHA-256 mismatch for {path}. "
                    "The file may have changed since the manifest "
                    "was generated. Regenerate the manifest."
                )

            # The same group ID must never be assigned to both splits.
            previous_split = declared_group_splits.setdefault(
                group_id, split
            )

            if previous_split != split:
                raise ValueError(
                    "An exact-duplicate group crosses train/validation: "
                    f"{group_id}"
                )

            label_idx = CLASS_NAMES.index(label)

            if split == "train":
                train_paths.append(path)
                train_labels.append(label_idx)
            else:
                val_paths.append(path)
                val_labels.append(label_idx)

    # Ensure the manifest covers every image in the training directory.
    missing_paths = expected_paths - seen_paths
    if missing_paths:
        examples = [str(path) for path in list(missing_paths)[:5]]
        raise ValueError(
            f"Manifest is missing {len(missing_paths)} dataset images. "
            f"Examples: {examples}"
        )

    if not train_paths or not val_paths:
        raise ValueError(
            "Manifest must contain both training and validation images."
        )

    # Independently check actual file hashes across both splits.
    actual_hash_splits = {}

    for path in train_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_hash_splits.setdefault(digest, set()).add("train")

    for path in val_paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        actual_hash_splits.setdefault(digest, set()).add("validation")

    crossing_hashes = [
        digest
        for digest, splits in actual_hash_splits.items()
        if len(splits) > 1
    ]

    if crossing_hashes:
        raise ValueError(
            f"Found {len(crossing_hashes)} exact SHA-256 duplicate "
            "groups crossing train/validation."
        )

    print(f"Loaded duplicate-aware manifest: {manifest_path}")
    print(f"Verified training images: {len(train_paths)}")
    print(f"Verified validation images: {len(val_paths)}")
    print("Exact SHA-256 duplicates crossing splits: 0")

    return train_paths, train_labels, val_paths, val_labels


def save_split_metadata(
    train_paths: list[Path],
    val_paths: list[Path],
    test_paths: list[Path] | None = None,
    output_path: Path = REPORTS_DIR / "split_metadata.json",
) -> None:
    """Save explicit split paths and counts for reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def relative_paths(paths: list[Path]) -> list[str]:
        result = []

        for path in paths:
            try:
                result.append(str(path.relative_to(DATA_DIR)))
            except ValueError:
                result.append(str(path))

        return result

    metadata = {
        "random_seed": RANDOM_SEED,
        "validation_fraction": VALIDATION_FRACTION,
        "counts": {
            "train": len(train_paths),
            "validation": len(val_paths),
            "test": len(test_paths) if test_paths is not None else None,
        },
        "train_samples": relative_paths(train_paths),
        "validation_samples": relative_paths(val_paths),
    }

    if test_paths is not None:
        metadata["test_samples"] = relative_paths(test_paths)

    output_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def build_tf_dataset(
    paths: list[Path],
    labels: list[int],
    *,
    training: bool,
    model_type: str = "efficientnet",
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
    augmentation: tf.keras.Sequential | None = None,
) -> tf.data.Dataset:
    """Build the TensorFlow input pipeline."""
    if len(paths) != len(labels):
        raise ValueError(
            f"Paths/labels length mismatch: {len(paths)} vs {len(labels)}"
        )

    if not paths:
        raise ValueError("Cannot build a dataset from zero images.")

    path_strings = [str(path) for path in paths]
    dataset = tf.data.Dataset.from_tensor_slices(
        (path_strings, labels)
    )

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        load_image_and_label,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    dataset = dataset.batch(
        batch_size,
        drop_remainder=False,
    )

    # Apply augmentation only to training data.
    if training and augmentation is not None:
        dataset = dataset.map(
            lambda x, y: (augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    if model_type == "cnn":
        normalizer = normalize_for_cnn
    elif model_type == "efficientnet":
        normalizer = normalize_for_efficientnet
    else:
        raise ValueError(
            f"Unsupported model_type: {model_type!r}"
        )

    dataset = dataset.map(
        normalizer,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.prefetch(tf.data.AUTOTUNE)


def get_train_val_datasets(
    train_dir: Path = TRAIN_DIR,
    model_type: str = "efficientnet",
    batch_size: int = BATCH_SIZE,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
    use_augmentation: bool = True,
    split_manifest: Path | None = None,
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    list[Path],
    list[int],
    list[Path],
    list[int],
]:
    """
    Return training and validation datasets.

    If split_manifest is provided, use its fixed split. Otherwise,
    retain the existing deterministic stratified split behavior.
    Never reads the official Testing directory.
    """
    images_by_class = find_images(train_dir)

    if split_manifest is not None:
        train_p, train_l, val_p, val_l = load_manifest_split(
            split_manifest,
            images_by_class,
        )
    else:
        train_p, train_l, val_p, val_l = make_stratified_split(
            images_by_class,
            validation_fraction=validation_fraction,
            seed=seed,
        )

    augmentation = (
        build_training_augmentation(seed=seed)
        if use_augmentation
        else None
    )

    train_ds = build_tf_dataset(
        train_p,
        train_l,
        training=True,
        model_type=model_type,
        batch_size=batch_size,
        seed=seed,
        augmentation=augmentation,
    )

    val_ds = build_tf_dataset(
        val_p,
        val_l,
        training=False,
        model_type=model_type,
        batch_size=batch_size,
        seed=seed,
        augmentation=None,
    )

    save_split_metadata(train_p, val_p)

    return train_ds, val_ds, train_p, train_l, val_p, val_l


def get_test_dataset(
    test_dir: Path = TEST_DIR,
    model_type: str = "efficientnet",
    batch_size: int = BATCH_SIZE,
) -> tuple[tf.data.Dataset, list[Path], list[int]]:
    """Load the official test set. Call only after model selection."""
    images_by_class = find_images(test_dir)
    test_p, test_l = flatten_class_dict(images_by_class)

    test_ds = build_tf_dataset(
        test_p,
        test_l,
        training=False,
        model_type=model_type,
        batch_size=batch_size,
        seed=RANDOM_SEED,
        augmentation=None,
    )

    return test_ds, test_p, test_l
