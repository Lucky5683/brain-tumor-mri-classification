"""Load and preprocess the Brain Tumor MRI dataset for later training (V2 Compatibility Wrapper).

Maintains complete backward compatibility with V1 function signatures while leveraging
the robust V2 data pipeline and configurations in src.data and config.py.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import tensorflow as tf

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_EXTENSIONS,
    IMAGE_SIZE,
    RANDOM_SEED,
    VALIDATION_FRACTION,
)
from src.data.augmentation import build_training_augmentation
from src.data.dataset import (
    build_tf_dataset,
    find_images,
    flatten_class_dict as flatten_with_labels,
    get_test_dataset,
    get_train_val_datasets,
    make_stratified_split,
)
from src.data.preprocessing import (
    load_image_and_label as load_image,
    normalize_for_cnn,
    normalize_for_efficientnet,
)

# Standard V1 augmentation reference for backward compatibility
TRAIN_AUGMENTATION = build_training_augmentation()


def build_dataset(
    paths: list[Path],
    labels: list[int],
    *,
    training: bool,
    model_type: str,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> tf.data.Dataset:
    """Build train, validation, or test dataset using V2 pipeline."""
    aug = TRAIN_AUGMENTATION if training else None
    return build_tf_dataset(
        paths=paths,
        labels=labels,
        training=training,
        model_type=model_type,
        batch_size=batch_size,
        seed=seed,
        augmentation=aug,
    )


def get_datasets(
    data_dir: str | Path = "dataset",
    model_type: str = "cnn",
    batch_size: int = BATCH_SIZE,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """Return train, validation, and untouched official-test datasets."""
    data_path = Path(data_dir)
    train_ds, val_ds, _, _, _, _ = get_train_val_datasets(
        train_dir=data_path / "Training",
        model_type=model_type,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    test_ds, _, _ = get_test_dataset(
        test_dir=data_path / "Testing",
        model_type=model_type,
        batch_size=batch_size,
    )
    return train_ds, val_ds, test_ds


def get_train_validation_datasets(
    data_dir: str | Path = "dataset",
    model_type: str = "cnn",
    batch_size: int = BATCH_SIZE,
    validation_fraction: float = VALIDATION_FRACTION,
    seed: int = RANDOM_SEED,
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    """Return only train and validation datasets without accessing Testing/."""
    data_path = Path(data_dir)
    train_ds, val_ds, _, _, _, _ = get_train_val_datasets(
        train_dir=data_path / "Training",
        model_type=model_type,
        batch_size=batch_size,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    return train_ds, val_ds


def print_split_summary(data_dir: Path, validation_fraction: float, seed: int) -> None:
    """Print split counts and configuration."""
    from collections import Counter
    training_images = find_images(data_dir / "Training")
    testing_images = find_images(data_dir / "Testing")
    train_p, train_l, val_p, val_l = make_stratified_split(
        training_images, validation_fraction, seed
    )
    test_p, test_l = flatten_with_labels(testing_images)

    print(f"Final image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB")
    print(f"Random seed: {seed}")
    print(f"Train / validation / official test: {len(train_p)} / {len(val_p)} / {len(test_p)}")
    print("Class counts (train / validation / official test):")
    for label, class_name in enumerate(CLASS_NAMES):
        print(
            f"  {class_name}: "
            f"{Counter(train_l)[label]} / {Counter(val_l)[label]} / {Counter(test_l)[label]}"
        )
    print("Augmentation: horizontal flip, small rotation (5%), small zoom (8%) — training only.")
    print("The official Testing folder is untouched and never used for validation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Brain Tumor MRI datasets without training.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--model-type", choices=["cnn", "efficientnet"], default="cnn")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    print_split_summary(args.data_dir, VALIDATION_FRACTION, RANDOM_SEED)
    get_datasets(args.data_dir, args.model_type, args.batch_size)


if __name__ == "__main__":
    main()
