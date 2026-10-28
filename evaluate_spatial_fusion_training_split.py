
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
ROOT = Path(__file__).resolve().parent

MANIFEST = (
    ROOT
    / "experiments"
    / "duplicate_aware_split"
    / "training_manifest.csv"
)

CHECKPOINT = (
    ROOT
    / "experiments"
    / "spatial_fusion"
    / "meningioma_weighted_50ep_seed42"
    / "best_model.keras"
)

OUTPUT = (
    ROOT
    / "experiments"
    / "spatial_fusion"
    / "meningioma_weighted_50ep_seed42"
    / "training_split_checkpoint_evaluation.json"
)

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}

IMAGE_SIZE = 224
BATCH_SIZE = 32

CLASS_WEIGHTS = {
    0: 1.0,
    1: 1.50,
    2: 1.0,
    3: 1.0,
}


def resolve_path(value):
    path = Path(str(value))

    candidates = (
        [path]
        if path.is_absolute()
        else [ROOT / path, ROOT / "dataset" / path]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Training image not found: {value}")


def decode_image(path, label):
    image = tf.io.decode_image(
        tf.io.read_file(path),
        channels=3,
        expand_animations=False,
    )
    image.set_shape([None, None, 3])

    image = tf.image.resize(
        image,
        [IMAGE_SIZE, IMAGE_SIZE],
        method="bilinear",
        antialias=True,
    )

    # Match the original experiment's 0–255 input range.
    image = tf.cast(image, tf.float32)
    return image, label


def main():
    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    df = pd.read_csv(MANIFEST)

    required_columns = {"path", "label", "split"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["label"] = df["label"].astype(str).str.strip().str.lower()

    # Strictly select the training split.
    train_df = df[df["split"].isin(["train", "training"])].copy()

    if train_df.empty:
        raise ValueError("No training images were found in the manifest.")

    if not set(train_df["label"]).issubset(set(CLASSES)):
        raise ValueError("Unexpected class labels in the training split.")

    if set(train_df["label"]) != set(CLASSES):
        raise ValueError("One or more expected classes are missing.")

    train_df["resolved_path"] = train_df["path"].map(resolve_path)

    labels = train_df["label"].map(CLASS_TO_ID).to_numpy(dtype=np.int32)
    paths = train_df["resolved_path"].map(str).to_numpy()

    # Load compatibility definitions before deserializing the model.
    import src.models.compat  # noqa: F401

    model = tf.keras.models.load_model(CHECKPOINT, compile=False)

    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    dataset = dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    all_probabilities = []

    print(f"Checkpoint: {CHECKPOINT}")
    print(f"Training images: {len(train_df)}")
    print("Evaluating the saved checkpoint; no training will occur.")

    for images, _ in dataset:
        probabilities = model(images, training=False)
        all_probabilities.append(np.asarray(probabilities))

    probabilities = np.concatenate(all_probabilities, axis=0)

    if probabilities.shape != (len(labels), len(CLASSES)):
        raise ValueError(
            f"Unexpected model output shape: {probabilities.shape}"
        )

    if not np.isfinite(probabilities).all():
        raise ValueError("Model predictions contain non-finite values.")

    predictions = np.argmax(probabilities, axis=1)

    # Unweighted classification accuracy, matching the intent of
    # the original compiled accuracy metric.
    accuracy = accuracy_score(labels, predictions)

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    balanced_acc = balanced_accuracy_score(labels, predictions)

    # Per-example sparse categorical cross-entropy.
    # Apply the same class weights used in the original fit() call.
    probabilities_tensor = tf.convert_to_tensor(
        probabilities, dtype=tf.float32
    )
    labels_tensor = tf.convert_to_tensor(labels, dtype=tf.int32)

    per_example_loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels_tensor,
        probabilities_tensor,
        from_logits=False,
    ).numpy()

    sample_weights = np.array(
        [CLASS_WEIGHTS[int(label)] for label in labels],
        dtype=np.float64,
    )

    unweighted_loss = float(np.mean(per_example_loss))
    class_weighted_loss = float(
        np.sum(per_example_loss * sample_weights) / len(labels)
    )

    result = {
        "evaluation_type": "saved_checkpoint_training_split_evaluation",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "split": "train",
        "number_of_images": int(len(labels)),
        "classes": CLASSES,
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "balanced_accuracy": float(balanced_acc),
        "unweighted_sparse_categorical_crossentropy": unweighted_loss,
        "class_weighted_sparse_categorical_crossentropy": (
            class_weighted_loss
        ),
        "class_weights": {
            CLASSES[index]: weight
            for index, weight in CLASS_WEIGHTS.items()
        },
        "note": (
            "Metrics were computed by evaluating the saved checkpoint on "
            "the unaugmented training split. They are not historical "
            "epoch-50 training logs and do not reproduce training-mode "
            "augmentation or dropout."
        ),
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("\nTRAINING-SPLIT CHECKPOINT EVALUATION")
    print("-" * 45)
    print(f"Images evaluated: {len(labels)}")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Balanced accuracy: {balanced_acc * 100:.2f}%")
    print(f"Unweighted loss: {unweighted_loss:.4f}")
    print(f"Class-weighted loss: {class_weighted_loss:.4f}")
    print(f"\nResults saved to:\n{OUTPUT}")


if __name__ == "__main__":
    main()
