
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import src.models.compat  # noqa: F401
from src.models.spatial_fusion import build_spatial_fusion_model

MANIFEST = ROOT / "experiments/duplicate_aware_split/training_manifest.csv"
BASE_CHECKPOINT = ROOT / "models/efficientnet_b0_finetuned.keras"

# A NEW output folder protects earlier experiments.
OUT = ROOT / "experiments/spatial_fusion/meningioma_weighted_50ep_seed42"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}

CLASS_WEIGHTS = {
    0: 1.0,
    1: 1.50,  # Meningioma
    2: 1.0,
    3: 1.0,
}

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32

MAX_EPOCHS = 50
MIN_EPOCHS = 12
PATIENCE = 7

INITIAL_LR = 1e-5
MIN_LR = 1e-7
LR_PATIENCE = 2

OUT.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


print("=" * 65)
print("SPATIAL FUSION — SAFE 50-EPOCH-MAX TRAINING")
print("=" * 65)
print("Project:", ROOT)
print("TensorFlow:", tf.__version__)
print("Output:", OUT)

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

if not BASE_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {BASE_CHECKPOINT}")


# ============================================================
# 2. LOAD AND VALIDATE THE DUPLICATE-AWARE MANIFEST
# ============================================================

df = pd.read_csv(MANIFEST)

required = {"path", "label", "split", "sha256", "group_id"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing manifest columns: {sorted(missing)}")

df["split"] = df["split"].astype(str).str.strip().str.lower()
df["label"] = df["label"].astype(str).str.strip().str.lower()

train_df = df[df["split"].isin(["train", "training"])].copy()
val_df = df[df["split"].isin(["val", "validation"])].copy()

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation split is empty.")

if not set(df["label"].unique()).issubset(set(CLASSES)):
    raise ValueError(f"Unexpected labels: {df['label'].unique().tolist()}")


def resolve_path(value):
    p = Path(str(value))
    candidates = [p] if p.is_absolute() else [ROOT / p, ROOT / "dataset" / p]

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Image not found: {value}")


df["resolved_path"] = df["path"].map(resolve_path)

print("\nVerifying image hashes...")
hashes = {}

for _, row in df.iterrows():
    p = row["resolved_path"]
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    expected = str(row["sha256"]).strip().lower()

    if actual != expected:
        raise ValueError(f"SHA-256 mismatch: {p}")

    hashes[str(p)] = actual

train_df = df.loc[train_df.index].copy()
val_df = df.loc[val_df.index].copy()

train_hashes = {hashes[str(p)] for p in train_df["resolved_path"]}
val_hashes = {hashes[str(p)] for p in val_df["resolved_path"]}

if train_hashes & val_hashes:
    raise ValueError("Exact duplicates cross train/validation.")

train_groups = set(train_df["group_id"].astype(str))
val_groups = set(val_df["group_id"].astype(str))

if train_groups & val_groups:
    raise ValueError("Similarity groups cross train/validation.")

test_root = (ROOT / "dataset" / "Testing").resolve()

for p in df["resolved_path"]:
    if p == test_root or test_root in p.parents:
        raise ValueError(f"Official test image in manifest: {p}")

for name, split_df in [("train", train_df), ("validation", val_df)]:
    missing_classes = set(CLASSES) - set(split_df["label"].unique())
    if missing_classes:
        raise ValueError(f"{name} missing classes: {sorted(missing_classes)}")

print("Manifest checks passed.")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Exact duplicate overlap: 0")
print("Similarity group overlap: 0")
print("Official test images used: 0")

print("\nTraining class counts:")
print(train_df["label"].value_counts().reindex(CLASSES, fill_value=0))

print("\nValidation class counts:")
print(val_df["label"].value_counts().reindex(CLASSES, fill_value=0))


# ============================================================
# 3. IMAGE PIPELINE
# ============================================================

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

    # Keep the 0–255 range used by the existing model.
    return tf.cast(image, tf.float32), label


def make_datasets():
    train_paths = train_df["resolved_path"].map(str).to_numpy()
    train_labels = train_df["label"].map(LABEL_TO_ID).to_numpy(
        dtype=np.int32
    )

    val_paths = val_df["resolved_path"].map(str).to_numpy()
    val_labels = val_df["label"].map(LABEL_TO_ID).to_numpy(
        dtype=np.int32
    )

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.05, seed=SEED + 1),
            tf.keras.layers.RandomZoom(0.08, seed=SEED + 2),
            tf.keras.layers.RandomTranslation(
                0.04, 0.04, seed=SEED + 3
            ),
        ],
        name="spatial_train_augmentation_50ep",
    )

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels)
    )

    train_ds = train_ds.shuffle(
        len(train_df),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    train_ds = train_ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.map(
        lambda image, label: (
            augmentation(image, training=True),
            label,
        ),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    train_ds = train_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_labels)
    )
    val_ds = val_ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, val_labels


# ============================================================
# 4. VALIDATION METRICS AND CHECKPOINT TRACKER
# ============================================================

class ValidationTracker(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, y_true):
        super().__init__()
        self.val_ds = val_ds
        self.y_true = y_true
        self.best_f1 = -1.0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self.stale = 0
        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        probabilities = self.model.predict(self.val_ds, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        accuracy = float(accuracy_score(self.y_true, predictions))
        macro_f1 = float(
            f1_score(
                self.y_true,
                predictions,
                average="macro",
                zero_division=0,
            )
        )
        balanced_acc = float(
            balanced_accuracy_score(self.y_true, predictions)
        )
        val_loss = float(logs.get("val_loss", float("inf")))

        report = classification_report(
            self.y_true,
            predictions,
            labels=list(range(len(CLASSES))),
            target_names=CLASSES,
            output_dict=True,
            zero_division=0,
        )

        per_class = {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASSES
        }

        cm = confusion_matrix(
            self.y_true,
            predictions,
            labels=list(range(len(CLASSES))),
        )

        result = {
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": accuracy,
            "val_macro_f1": macro_f1,
            "val_balanced_accuracy": balanced_acc,
            "per_class": per_class,
            "confusion_matrix": cm.tolist(),
        }

        self.history.append(result)

        print(
            f"\nEpoch {epoch + 1}: "
            f"accuracy={accuracy:.4f}, "
            f"macro_F1={macro_f1:.4f}, "
            f"balanced_accuracy={balanced_acc:.4f}, "
            f"val_loss={val_loss:.4f}"
        )

        print("Per-class recall:", {
            name: round(per_class[name]["recall"], 4)
            for name in CLASSES
        })

        improved = (
            macro_f1 > self.best_f1 + 1e-8
            or (
                abs(macro_f1 - self.best_f1) <= 1e-8
                and val_loss < self.best_loss
            )
        )

        if improved:
            self.best_f1 = macro_f1
            self.best_loss = val_loss
            self.best_epoch = epoch + 1
            self.stale = 0

            self.model.save(OUT / "best_model.keras")

            pd.DataFrame(report).transpose().to_csv(
                OUT / "best_classification_report.csv"
            )

            np.savetxt(
                OUT / "best_confusion_matrix.csv",
                cm,
                delimiter=",",
                fmt="%d",
            )

            save_json(OUT / "best_metrics.json", result)
            print("Saved new best checkpoint.")
        else:
            self.stale += 1

        save_json(OUT / "history.json", self.history)

        save_json(
            OUT / "results.json",
            {
                "experiment": "meningioma_weighted_spatial_fusion_50ep",
                "class_weights": {
                    CLASSES[i]: weight
                    for i, weight in CLASS_WEIGHTS.items()
                },
                "max_epochs": MAX_EPOCHS,
                "epochs_completed": len(self.history),
                "best_epoch": self.best_epoch,
                "best_validation_macro_f1": self.best_f1,
                "best_validation_loss": self.best_loss,
                "official_test_evaluated": False,
                "history": self.history,
            },
        )


class ReduceLROnMacroF1Plateau(tf.keras.callbacks.Callback):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.previous_best = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        current_best = self.tracker.best_f1

        if current_best > self.previous_best + 1e-8:
            self.previous_best = current_best
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= LR_PATIENCE:
            optimizer = self.model.optimizer
            old_lr = float(
                tf.keras.backend.get_value(optimizer.learning_rate)
            )
            new_lr = max(old_lr * 0.5, MIN_LR)

            if new_lr < old_lr:
                optimizer.learning_rate.assign(new_lr)
                print(
                    f"Learning rate reduced: "
                    f"{old_lr:.2e} -> {new_lr:.2e}"
                )

            self.wait = 0


class StopOnValidationPlateau(tf.keras.callbacks.Callback):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def on_epoch_end(self, epoch, logs=None):
        if (
            epoch + 1 >= MIN_EPOCHS
            and self.tracker.stale >= PATIENCE
        ):
            print(
                f"Early stopping. Best epoch: "
                f"{self.tracker.best_epoch}"
            )
            self.model.stop_training = True


# ============================================================
# 5. BUILD THE SPATIAL FUSION MODEL
# ============================================================

train_ds, val_ds, y_val = make_datasets()

print("\nBuilding spatial fusion model...")
model = build_spatial_fusion_model(str(BASE_CHECKPOINT))

try:
    extractor = model.get_layer("spatial_feature_extractor")
except ValueError as exc:
    raise ValueError(
        "Expected layer 'spatial_feature_extractor' not found. "
        "Check src/models/spatial_fusion.py."
    ) from exc

# Freeze backbone, then fine-tune the last 20 eligible layers.
extractor.trainable = True

for layer in extractor.layers:
    layer.trainable = False

eligible_layers = [
    layer for layer in extractor.layers
    if not isinstance(
        layer,
        (tf.keras.layers.BatchNormalization,
         tf.keras.layers.InputLayer),
    )
]

for layer in eligible_layers[-20:]:
    layer.trainable = True

# Keep BatchNormalization layers frozen.
for layer in extractor.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Train fusion and classification layers.
for layer in model.layers:
    if layer.name != "spatial_feature_extractor":
        layer.trainable = True

trainable_params = sum(
    int(np.prod(weight.shape))
    for weight in model.trainable_weights
)

print("Backbone layers:", len(extractor.layers))
print("Fine-tuned backbone layers:", min(20, len(eligible_layers)))
print(f"Trainable parameters: {trainable_params:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=INITIAL_LR
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

tracker = ValidationTracker(val_ds, y_val)

print("\nStarting training...")
print("Class weights:", CLASS_WEIGHTS)
print("Maximum epochs:", MAX_EPOCHS)
print("Early stopping patience:", PATIENCE)
print("Official test set will NOT be evaluated.")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=[
        tracker,
        ReduceLROnMacroF1Plateau(tracker),
        StopOnValidationPlateau(tracker),
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    verbose=1,
)

if not (OUT / "best_model.keras").is_file():
    raise RuntimeError("No best checkpoint was saved.")

print("\n" + "=" * 65)
print("SPATIAL FUSION EXPERIMENT FINISHED")
print("=" * 65)
print("Epochs completed:", len(tracker.history))
print("Best epoch:", tracker.best_epoch)
print(f"Best validation macro-F1: {tracker.best_f1:.4f}")
print("Best validation loss:", tracker.best_loss)
print("Best model:", OUT / "best_model.keras")
print("Results:", OUT / "results.json")
print("Classification report:", OUT / "best_classification_report.csv")
print("Confusion matrix:", OUT / "best_confusion_matrix.csv")
print("Official test set was not evaluated.")
