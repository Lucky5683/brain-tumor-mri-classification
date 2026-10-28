
"""
Multi-seed fine-tuning of the existing grouped multi-scale fusion model.

- Continues from the existing checkpoint for each seed.
- Uses the duplicate-aware manifest.
- Runs 3 seeds sequentially.
- Trains for at least 8 epochs and at most 12 per seed,
  unless training becomes unstable or fails.
- Selects checkpoints by validation macro-F1.
- Never evaluates the official test set.
"""

import os

# Configure CPU threads before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import sys
import json
import time
import random
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
    precision_score,
    recall_score,
)

# ---------------------------------------------------------
# PROJECT CONFIGURATION
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

# Compatibility shim for existing project checkpoints.
sys.path.insert(0, str(ROOT))
import src.models.compat  # noqa: F401

MANIFEST = (
    ROOT / "experiments" / "duplicate_aware_split"
    / "training_manifest.csv"
)

BASE_CHECKPOINT = ROOT / "models" / "multiscale_fusion_grouped_best.keras"

OUTPUT_ROOT = ROOT / "experiments" / "multiseed_fusion"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGE_SIZE = 224
BATCH_SIZE = 32

SEEDS = [42, 123, 2026]
MIN_EPOCHS = 8
MAX_EPOCHS = 12
PATIENCE_AFTER_MIN = 3

INITIAL_LR = 1e-5
MIN_LR = 1e-7
LR_FACTOR = 0.5
LR_PATIENCE = 2

UNFREEZE_LAST_LAYERS = 20

# ---------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

print("Project:", ROOT)
print("TensorFlow:", tf.__version__)
print("GPU devices:", tf.config.list_physical_devices("GPU"))
print("Training device:", "GPU" if tf.config.list_physical_devices("GPU") else "CPU")


# ---------------------------------------------------------
# MANIFEST VALIDATION
# ---------------------------------------------------------

if not MANIFEST.is_file():
    raise FileNotFoundError(f"Manifest missing: {MANIFEST}")

if not BASE_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Fusion checkpoint missing: {BASE_CHECKPOINT}")

df = pd.read_csv(MANIFEST)

required_columns = {"path", "label", "split", "sha256", "group_id"}
missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(f"Missing manifest columns: {sorted(missing_columns)}")

if len(df) != 5600:
    raise ValueError(f"Expected 5,600 manifest rows; found {len(df)}")

if set(df["label"].unique()) != set(CLASS_NAMES):
    raise ValueError(f"Unexpected class labels: {df['label'].unique()}")

if set(df["split"].unique()) != {"train", "validation"}:
    raise ValueError(f"Unexpected splits: {df['split'].unique()}")

train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
val_df = df[df["split"] == "validation"].copy().reset_index(drop=True)

if len(train_df) != 4447 or len(val_df) != 1153:
    raise ValueError(
        f"Unexpected split counts: train={len(train_df)}, "
        f"validation={len(val_df)}"
    )


def resolve_path(value):
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p


train_df["resolved_path"] = train_df["path"].map(resolve_path)
val_df["resolved_path"] = val_df["path"].map(resolve_path)

all_paths = pd.concat(
    [train_df["resolved_path"], val_df["resolved_path"]]
)

for p in all_paths:
    if not p.is_file():
        raise FileNotFoundError(f"Missing image: {p}")

# Check manifest hashes and split leakage before expensive training.
print("\nVerifying image hashes and split groups...")
actual_hashes = {}

for _, row in df.iterrows():
    p = resolve_path(row["path"])
    actual_hash = hashlib.sha256(p.read_bytes()).hexdigest()

    if actual_hash != row["sha256"]:
        raise ValueError(f"Manifest hash mismatch: {p}")

    actual_hashes[str(p.resolve())] = actual_hash

train_hashes = set(
    train_df["resolved_path"].map(lambda p: actual_hashes[str(p.resolve())])
)
val_hashes = set(
    val_df["resolved_path"].map(lambda p: actual_hashes[str(p.resolve())])
)

if train_hashes & val_hashes:
    raise ValueError("Exact duplicate hashes cross train/validation.")

train_groups = set(train_df["group_id"])
val_groups = set(val_df["group_id"])

if train_groups & val_groups:
    raise ValueError("Similarity groups cross train/validation.")

# Verify no official test image appears in the manifest.
test_root = (ROOT / "dataset" / "Testing").resolve()

for p in all_paths:
    if test_root in p.resolve().parents:
        raise ValueError(f"Official test image found in manifest: {p}")

print("Manifest checks passed.")
print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print("Train/validation hash overlap: 0")
print("Train/validation group overlap: 0")
print("Official test images in manifest: 0")

print("\nTraining distribution:")
print(train_df["label"].value_counts().reindex(CLASS_NAMES))
print("\nValidation distribution:")
print(val_df["label"].value_counts().reindex(CLASS_NAMES))


# ---------------------------------------------------------
# IMAGE PIPELINE
# EfficientNet-B0 uses its internal input rescaling.
# Keep input pixel values in the 0-255 range.
# ---------------------------------------------------------

def decode_image(path, label):
    image_bytes = tf.io.read_file(path)

    image = tf.io.decode_image(
        image_bytes,
        channels=3,
        expand_animations=False,
    )

    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    image = tf.cast(image, tf.float32)

    return image, label


def build_datasets(seed):
    train_paths = train_df["resolved_path"].map(str).to_numpy()
    train_labels = train_df["label"].map(LABEL_TO_ID).to_numpy(np.int32)

    val_paths = val_df["resolved_path"].map(str).to_numpy()
    val_labels = val_df["label"].map(LABEL_TO_ID).to_numpy(np.int32)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_paths, train_labels)
    )

    train_ds = train_ds.shuffle(
        buffer_size=len(train_df),
        seed=seed,
        reshuffle_each_iteration=True,
    )

    train_ds = train_ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    # Fresh augmentation layers for each seed.
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomRotation(0.05, seed=seed + 1),
            tf.keras.layers.RandomZoom(0.08, seed=seed + 2),
            tf.keras.layers.RandomTranslation(
                0.04, 0.04, seed=seed + 3
            ),
        ],
        name=f"train_augmentation_{seed}",
    )

    train_ds = train_ds.map(
        lambda x, y: (augmentation(x, training=True), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    val_ds = tf.data.Dataset.from_tensor_slices(
        (val_paths, val_labels)
    )

    val_ds = val_ds.map(
        decode_image,
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True

    train_ds = (
        train_ds.with_options(options)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        val_ds.with_options(options)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, val_labels


# ---------------------------------------------------------
# METRIC CALLBACK, BEST CHECKPOINT AND RECOVERY
# ---------------------------------------------------------

class ValidationMetrics(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, y_true, output_dir):
        super().__init__()
        self.val_ds = val_ds
        self.y_true = y_true
        self.output_dir = output_dir
        self.best_f1 = -1.0
        self.best_loss = float("inf")
        self.best_epoch = None
        self.stale_epochs = 0
        self.history = []

        self.best_path = output_dir / "best_model.keras"
        self.last_path = output_dir / "last_completed_epoch.keras"

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

        macro_precision = precision_score(
            self.y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        macro_recall = recall_score(
            self.y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        report = classification_report(
            self.y_true,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        val_loss = float(logs.get("val_loss", float("inf")))

        logs["val_macro_f1"] = float(macro_f1)
        logs["val_balanced_accuracy"] = float(balanced_acc)

        per_class = {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            }
            for name in CLASS_NAMES
        }

        epoch_result = {
            "epoch": epoch + 1,
            "val_loss": val_loss,
            "val_accuracy": float(accuracy),
            "val_macro_f1": float(macro_f1),
            "val_macro_precision": float(macro_precision),
            "val_macro_recall": float(macro_recall),
            "val_balanced_accuracy": float(balanced_acc),
            "per_class": per_class,
            "confusion_matrix": confusion_matrix(
                self.y_true,
                predictions,
                labels=list(range(len(CLASS_NAMES))),
            ).tolist(),
        }

        self.history.append(epoch_result)

        print(
            f"\nValidation epoch {epoch + 1}: "
            f"accuracy={accuracy:.4f}, "
            f"macro-F1={macro_f1:.4f}, "
            f"balanced-accuracy={balanced_acc:.4f}"
        )

        print(
            "Recall:",
            {
                name: round(per_class[name]["recall"], 4)
                for name in CLASS_NAMES
            },
        )

        # Save the last completed epoch for recovery.
        self.model.save(self.last_path)

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
            self.stale_epochs = 0

            self.model.save(self.best_path)

            pd.DataFrame(report).transpose().to_csv(
                self.output_dir / "best_classification_report.csv"
            )

            np.savetxt(
                self.output_dir / "best_confusion_matrix.csv",
                np.array(epoch_result["confusion_matrix"]),
                delimiter=",",
                fmt="%d",
            )

            print(f"Saved new best checkpoint: {self.best_path}")
        else:
            self.stale_epochs += 1

        pd.DataFrame(self.history).to_json(
            self.output_dir / "history.json",
            orient="records",
            indent=2,
        )

        with (self.output_dir / "results.json").open(
            "w", encoding="utf-8"
        ) as f:
            json.dump(
                {
                    "best_epoch": self.best_epoch,
                    "best_macro_f1": self.best_f1,
                    "best_validation_loss": self.best_loss,
                    "epochs_completed": len(self.history),
                    "history": self.history,
                },
                f,
                indent=2,
            )


class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, metrics_callback):
        super().__init__()
        self.metrics_callback = metrics_callback

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 < MIN_EPOCHS:
            return

        if self.metrics_callback.stale_epochs >= PATIENCE_AFTER_MIN:
            print(
                f"\nEarly stopping after epoch {epoch + 1}: "
                f"validation macro-F1 did not improve for "
                f"{PATIENCE_AFTER_MIN} consecutive epochs."
            )
            self.model.stop_training = True


class MacroF1ReduceLR(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best = -1.0
        self.wait = 0

    def on_epoch_end(self, epoch, logs=None):
        score = (logs or {}).get("val_macro_f1")

        if score is None:
            return

        if score > self.best + 1e-8:
            self.best = score
            self.wait = 0
            return

        self.wait += 1

        if self.wait >= LR_PATIENCE:
            current_lr = float(
                tf.keras.backend.get_value(
                    self.model.optimizer.learning_rate
                )
            )

            new_lr = max(current_lr * LR_FACTOR, MIN_LR)

            if new_lr < current_lr:
                tf.keras.backend.set_value(
                    self.model.optimizer.learning_rate,
                    new_lr,
                )
                print(
                    f"\nReducing learning rate: "
                    f"{current_lr:.2e} -> {new_lr:.2e}"
                )

            self.wait = 0


# ---------------------------------------------------------
# MODEL PREPARATION
# ---------------------------------------------------------

def prepare_fine_tuning(model):
    # Find the actual nested EfficientNet model.
    backbones = [
        layer
        for layer in model.layers
        if isinstance(layer, tf.keras.Model)
        and "efficientnet" in layer.name.lower()
    ]

    if len(backbones) != 1:
        raise RuntimeError(
            "Could not uniquely identify the nested EfficientNet "
            f"backbone. Found: {[b.name for b in backbones]}"
        )

    backbone = backbones[0]
    backbone.trainable = True

    # Freeze all backbone layers first.
    for layer in backbone.layers:
        layer.trainable = False

    # Select the last 20 non-BatchNorm layers.
    eligible = [
        layer for layer in backbone.layers
        if not isinstance(
            layer, tf.keras.layers.BatchNormalization
        )
    ]

    selected = eligible[-UNFREEZE_LAST_LAYERS:]

    for layer in selected:
        layer.trainable = True

    # Ensure all BatchNorm layers remain frozen.
    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Train the fusion head.
    for layer in model.layers:
        if layer is not backbone:
            layer.trainable = True

    trainable_params = sum(
        int(np.prod(w.shape))
        for w in model.trainable_weights
    )

    print("Backbone:", backbone.name)
    print("Backbone layers:", len(backbone.layers))
    print("Selected backbone layers:", len(selected))
    print("Trainable parameters:", f"{trainable_params:,}")

    if trainable_params <= 0:
        raise RuntimeError("No trainable parameters found.")

    return model


# ---------------------------------------------------------
# TRAIN EACH SEED FROM THE SAME ORIGINAL CHECKPOINT
# ---------------------------------------------------------

all_seed_results = []
start_all = time.time()

for seed in SEEDS:
    print("\n" + "=" * 70)
    print(f"STARTING SEED {seed}")
    print("=" * 70)

    seed_dir = OUTPUT_ROOT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    train_ds, val_ds, y_val = build_datasets(seed)

    # Always reload the original fusion checkpoint for a fair seed run.
    model = tf.keras.models.load_model(
        BASE_CHECKPOINT,
        compile=False,
    )

    model = prepare_fine_tuning(model)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=INITIAL_LR
        ),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    metrics_callback = ValidationMetrics(
        val_ds=val_ds,
        y_true=y_val,
        output_dir=seed_dir,
    )

    callbacks = [
        metrics_callback,
        MacroF1ReduceLR(),
        MacroF1EarlyStopping(metrics_callback),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    start = time.time()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=MAX_EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - start

    # Reload and independently verify the best saved checkpoint.
    if not metrics_callback.best_path.exists():
        raise RuntimeError(
            f"No best checkpoint saved for seed {seed}."
        )

    best_model = tf.keras.models.load_model(
        metrics_callback.best_path,
        compile=False,
    )

    probabilities = best_model.predict(val_ds, verbose=0)
    predictions = np.argmax(probabilities, axis=1)

    report = classification_report(
        y_val,
        predictions,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    seed_result = {
        "seed": seed,
        "epochs_completed": len(metrics_callback.history),
        "best_epoch": metrics_callback.best_epoch,
        "training_seconds": round(elapsed, 2),
        "best_checkpoint": str(metrics_callback.best_path),
        "validation_accuracy": float(
            accuracy_score(y_val, predictions)
        ),
        "validation_macro_f1": float(
            f1_score(
                y_val, predictions,
                average="macro",
                zero_division=0,
            )
        ),
        "validation_balanced_accuracy": float(
            balanced_accuracy_score(y_val, predictions)
        ),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
            }
            for name in CLASS_NAMES
        },
    }

    with (seed_dir / "final_seed_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(seed_result, f, indent=2)

    all_seed_results.append(seed_result)

    with (OUTPUT_ROOT / "seed_comparison.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(all_seed_results, f, indent=2)

    print("\nCompleted seed:", seed)
    print(json.dumps(seed_result, indent=2))

    # Free memory before the next seed.
    del model, best_model, train_ds, val_ds
    tf.keras.backend.clear_session()


# ---------------------------------------------------------
# FINAL SEED COMPARISON
# ---------------------------------------------------------

all_seed_results.sort(
    key=lambda result: result["validation_macro_f1"],
    reverse=True,
)

summary = {
    "baseline_checkpoint": str(BASE_CHECKPOINT),
    "manifest": str(MANIFEST),
    "official_test_evaluated": False,
    "seeds": all_seed_results,
    "best_seed": all_seed_results[0]["seed"],
    "best_checkpoint": all_seed_results[0]["best_checkpoint"],
    "best_validation_macro_f1": all_seed_results[0][
        "validation_macro_f1"
    ],
    "mean_validation_accuracy": float(
        np.mean([r["validation_accuracy"] for r in all_seed_results])
    ),
    "std_validation_accuracy": float(
        np.std([r["validation_accuracy"] for r in all_seed_results])
    ),
    "mean_validation_macro_f1": float(
        np.mean([r["validation_macro_f1"] for r in all_seed_results])
    ),
    "std_validation_macro_f1": float(
        np.std([r["validation_macro_f1"] for r in all_seed_results])
    ),
    "total_training_seconds": round(time.time() - start_all, 2),
    "caveat": (
        "Perceptual grouping does not prove patient-level independence. "
        "Seed results share the same validation set and are not "
        "independent test results."
    ),
}

with (OUTPUT_ROOT / "final_summary.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70)
print("ALL SEEDS COMPLETED")
print("=" * 70)
print(json.dumps(summary, indent=2))
print("\nOfficial test set was not evaluated.")
