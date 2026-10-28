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

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

import src.models.compat  # noqa: F401
from src.models.spatial_fusion import build_spatial_fusion_model

MANIFEST = ROOT / "experiments/duplicate_aware_split/training_manifest.csv"
BASE_CHECKPOINT = ROOT / "models/efficientnet_b0_finetuned.keras"
OUT = ROOT / "experiments/spatial_fusion/meningioma_weighted_seed42"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}

# Targeted experiment: prioritize meningioma errors.
CLASS_WEIGHTS = {0: 1.0, 1: 1.25, 2: 1.0, 3: 1.0}

SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 32
MAX_EPOCHS = 12
MIN_EPOCHS = 8
PATIENCE = 3
INITIAL_LR = 1e-5
MIN_LR = 1e-7

OUT.mkdir(parents=True, exist_ok=True)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

print("Project:", ROOT)
print("TensorFlow:", tf.__version__)
print("Devices:", tf.config.list_physical_devices())

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

if not BASE_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Base checkpoint not found: {BASE_CHECKPOINT}")


# ---------------------------------------------------------
# MANIFEST VALIDATION
# ---------------------------------------------------------

df = pd.read_csv(MANIFEST)

required = {"path", "label", "split", "sha256", "group_id"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Manifest is missing columns: {missing}")

df["split"] = df["split"].astype(str).str.strip().str.lower()

train_df = df[df["split"].isin(["train", "training"])].copy()
val_df = df[df["split"].isin(["val", "validation"])].copy()

if train_df.empty or val_df.empty:
    raise ValueError(
        f"Invalid train/validation splits: {df['split'].unique().tolist()}"
    )

if not set(df["label"].unique()).issubset(set(CLASSES)):
    raise ValueError("Unexpected class label found in manifest.")

def resolve_path(value):
    p = Path(str(value))

    candidates = (
        [p] if p.is_absolute()
        else [ROOT / p, ROOT / "dataset" / p]
    )

    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()

    raise FileNotFoundError(f"Manifest image not found: {value}")

df["resolved_path"] = df["path"].map(resolve_path)

actual_hashes = {}

for _, row in df.iterrows():
    path = row["resolved_path"]
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    expected_hash = str(row["sha256"]).strip().lower()

    if actual_hash.lower() != expected_hash:
        raise ValueError(f"SHA-256 mismatch: {path}")

    actual_hashes[str(path)] = actual_hash

train_df = df.loc[train_df.index].copy()
val_df = df.loc[val_df.index].copy()

train_hashes = {
    actual_hashes[str(p)] for p in train_df["resolved_path"]
}
val_hashes = {
    actual_hashes[str(p)] for p in val_df["resolved_path"]
}

if train_hashes & val_hashes:
    raise ValueError("Exact duplicate images cross train/validation.")

if set(train_df["group_id"]) & set(val_df["group_id"]):
    raise ValueError("Similarity groups cross train/validation.")

test_root = (ROOT / "dataset" / "Testing").resolve()

for path in df["resolved_path"]:
    if test_root in path.parents:
        raise ValueError(f"Official test image found in manifest: {path}")

print("\nManifest checks passed.")
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Exact hash overlap: 0")
print("Group overlap: 0")
print("Official test images used: 0")

print("\nTraining class counts:")
print(train_df["label"].value_counts().reindex(CLASSES, fill_value=0))

print("\nValidation class counts:")
print(val_df["label"].value_counts().reindex(CLASSES, fill_value=0))


# ---------------------------------------------------------
# DATA PIPELINE
# Keep pixel values in [0, 255] for EfficientNet.
# Augmentation is applied to training images only.
# ---------------------------------------------------------

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

    return tf.cast(image, tf.float32), label


def make_datasets():
    train_paths = train_df["resolved_path"].map(str).to_numpy()
    train_labels = train_df["label"].map(LABEL_TO_ID).to_numpy(np.int32)

    val_paths = val_df["resolved_path"].map(str).to_numpy()
    val_labels = val_df["label"].map(LABEL_TO_ID).to_numpy(np.int32)

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=SEED),
            tf.keras.layers.RandomRotation(0.05, seed=SEED + 1),
            tf.keras.layers.RandomZoom(0.08, seed=SEED + 2),
            tf.keras.layers.RandomTranslation(
                0.04, 0.04, seed=SEED + 3
            ),
        ],
        name="spatial_train_augmentation",
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


# ---------------------------------------------------------
# VALIDATION METRICS AND CHECKPOINTS
# ---------------------------------------------------------

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
        logs = logs if logs is not None else {}

        probabilities = self.model.predict(self.val_ds, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        macro_f1 = f1_score(
            self.y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        accuracy = accuracy_score(self.y_true, predictions)
        balanced_acc = balanced_accuracy_score(
            self.y_true, predictions
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

        result = {
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": float(accuracy),
            "val_macro_f1": float(macro_f1),
            "val_balanced_accuracy": float(balanced_acc),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(
                self.y_true,
                predictions,
                labels=list(range(len(CLASSES))),
            ).tolist(),
        }

        self.history.append(result)

        print(
            f"\nEpoch {epoch + 1}: "
            f"val_accuracy={accuracy:.4f}, "
            f"val_macro_F1={macro_f1:.4f}, "
            f"val_balanced_accuracy={balanced_acc:.4f}"
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
            self.best_f1 = float(macro_f1)
            self.best_loss = val_loss
            self.best_epoch = epoch + 1
            self.stale = 0

            self.model.save(OUT / "best_model.keras")

            pd.DataFrame(report).transpose().to_csv(
                OUT / "best_classification_report.csv"
            )

            np.savetxt(
                OUT / "best_confusion_matrix.csv",
                np.asarray(result["confusion_matrix"]),
                delimiter=",",
                fmt="%d",
            )

            print("Saved best checkpoint.")
        else:
            self.stale += 1

        with (OUT / "history.json").open("w", encoding="utf-8") as file:
            json.dump(self.history, file, indent=2)

        with (OUT / "results.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "best_epoch": self.best_epoch,
                    "best_validation_macro_f1": self.best_f1,
                    "best_validation_loss": self.best_loss,
                    "epochs_completed": len(self.history),
                    "history": self.history,
                    "official_test_evaluated": False,
                },
                file,
                indent=2,
            )


class MacroF1ReduceLR(tf.keras.callbacks.Callback):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker
        self.best = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        score = self.tracker.best_f1

        if score > self.best + 1e-8:
            self.best = score
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= 2:
            optimizer = self.model.optimizer
            old_lr = float(
                tf.keras.backend.get_value(optimizer.learning_rate)
            )
            new_lr = max(old_lr * 0.5, MIN_LR)

            if new_lr < old_lr:
                optimizer.learning_rate.assign(new_lr)
                print(f"Learning rate reduced: {old_lr:.2e} -> {new_lr:.2e}")

            self.wait = 0


class StopOnPlateau(tf.keras.callbacks.Callback):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def on_epoch_end(self, epoch, logs=None):
        if (
            epoch + 1 >= MIN_EPOCHS
            and self.tracker.stale >= PATIENCE
        ):
            print("Early stopping: validation macro-F1 plateau.")
            self.model.stop_training = True


# ---------------------------------------------------------
# BUILD MODEL AND CONFIGURE FINE-TUNING
# ---------------------------------------------------------

train_ds, val_ds, y_val = make_datasets()

print("\nBuilding spatial fusion model...")
model = build_spatial_fusion_model(str(BASE_CHECKPOINT))

# The extractor directly contains the EfficientNet layers.
extractor = model.get_layer("spatial_feature_extractor")
extractor.trainable = True

for layer in extractor.layers:
    layer.trainable = False

eligible_layers = [
    layer
    for layer in extractor.layers
    if not isinstance(layer, tf.keras.layers.BatchNormalization)
    and not isinstance(layer, tf.keras.layers.InputLayer)
]

for layer in eligible_layers[-20:]:
    layer.trainable = True

# Freeze all BatchNormalization layers for stable fine-tuning.
for layer in extractor.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

# Train the new fusion and classification layers.
for layer in model.layers:
    if layer.name != "spatial_feature_extractor":
        layer.trainable = True

trainable_params = sum(
    int(np.prod(weight.shape))
    for weight in model.trainable_weights
)

print("Backbone layers:", len(extractor.layers))
print("Selected fine-tuning layers:", min(20, len(eligible_layers)))
print(f"Trainable parameters: {trainable_params:,}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

tracker = ValidationTracker(val_ds, y_val)

print("\nStarting training...")
model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=MAX_EPOCHS,
    class_weight=CLASS_WEIGHTS,
    callbacks=[
        tracker,
        MacroF1ReduceLR(tracker),
        StopOnPlateau(tracker),
        tf.keras.callbacks.TerminateOnNaN(),
    ],
    verbose=1,
)

if not (OUT / "best_model.keras").is_file():
    raise RuntimeError("Training finished without saving a best checkpoint.")

print("\nTraining finished.")
print("Best epoch:", tracker.best_epoch)
print(f"Best validation macro-F1: {tracker.best_f1:.4f}")
print("Best validation loss:", tracker.best_loss)
print("Best checkpoint:", OUT / "best_model.keras")
print("Metrics directory:", OUT)
print("Official test set was not evaluated.")
