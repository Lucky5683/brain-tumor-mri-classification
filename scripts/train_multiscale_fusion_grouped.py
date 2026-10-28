"""Train EfficientNet-B0 multi-scale fusion using a similarity-grouped split."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.models.compat  # noqa: F401

from config import (
    CLASS_NAMES,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_DIR,
)
from src.data.dataset import get_train_val_datasets
from src.models.multiscale_fusion import build_multiscale_fusion_model

# ---------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------
BATCH_SIZE = 32
STAGE1_EPOCHS = 5
STAGE2_EPOCHS = 4
STAGE1_LR = 1e-3
STAGE2_LR = 1e-5
UNFREEZE_LAYERS = 20

MANIFEST_PATH = (
    EXPERIMENTS_DIR
    / "duplicate_aware_split"
    / "training_manifest.csv"
)

# Keep this experiment separate from earlier checkpoints.
STAGE1_CHECKPOINT = MODELS_DIR / "multiscale_fusion_grouped_stage1_best.keras"
FINAL_CHECKPOINT = MODELS_DIR / "multiscale_fusion_grouped_best.keras"
SUMMARY_PATH = EXPERIMENTS_DIR / "multiscale_fusion_grouped_results.json"

EXPECTED_TRAIN_IMAGES = 4447
EXPECTED_VAL_IMAGES = 1153


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Save the best model based on validation macro-F1."""

    def __init__(self, val_ds, val_labels, checkpoint_path, patience=3):
        super().__init__()
        self.val_ds = val_ds
        self.val_labels = np.asarray(val_labels)
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.best_f1 = -1.0
        self.wait = 0
        self.best_metrics = {}

    def on_epoch_end(self, epoch, logs=None):
        probabilities = self.model.predict(self.val_ds, verbose=0)
        predictions = np.argmax(probabilities, axis=1)

        metrics = {
            "epoch": int(epoch + 1),
            "val_accuracy": float(
                accuracy_score(self.val_labels, predictions)
            ),
            "val_macro_f1": float(
                f1_score(
                    self.val_labels, predictions,
                    average="macro", zero_division=0
                )
            ),
            "val_macro_precision": float(
                precision_score(
                    self.val_labels, predictions,
                    average="macro", zero_division=0
                )
            ),
            "val_macro_recall": float(
                recall_score(
                    self.val_labels, predictions,
                    average="macro", zero_division=0
                )
            ),
        }

        print(
            f"\nEpoch {epoch + 1}: "
            f"val_accuracy={metrics['val_accuracy']:.4f}, "
            f"val_macro_f1={metrics['val_macro_f1']:.4f}, "
            f"val_macro_precision={metrics['val_macro_precision']:.4f}, "
            f"val_macro_recall={metrics['val_macro_recall']:.4f}"
        )

        if metrics["val_macro_f1"] > self.best_f1:
            self.best_f1 = metrics["val_macro_f1"]
            self.best_metrics = metrics.copy()
            self.wait = 0
            self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save(self.checkpoint_path)
            print(f"Saved best checkpoint: {self.checkpoint_path}")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print("Early stopping: validation macro-F1 did not improve.")
                self.model.stop_training = True


def configure_finetuning(model):
    """Unfreeze selected backbone layers while keeping BatchNorm frozen."""
    extractor = model.get_layer("multiscale_feature_extractor")
    extractor.trainable = True

    try:
        backbone = extractor.get_layer("efficientnetb0")
    except ValueError:
        backbone = extractor

    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = False

    candidates = [
        layer for layer in backbone.layers
        if not isinstance(layer, tf.keras.layers.BatchNormalization)
    ]

    for layer in candidates[-UNFREEZE_LAYERS:]:
        layer.trainable = True

    for layer in backbone.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    return model


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    print("=" * 65)
    print("MULTI-SCALE EFFICIENTNET-B0 — SIMILARITY-GROUPED EXPERIMENT")
    print("=" * 65)
    print("TensorFlow:", tf.__version__)
    print("GPU devices:", tf.config.list_physical_devices("GPU"))
    print("Classes:", CLASS_NAMES)
    print("Manifest:", MANIFEST_PATH)

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH.resolve()}\n"
            "Run: python scripts/prepare_training_manifest.py"
        )

    train_ds, val_ds, train_paths, train_labels, val_paths, val_labels = (
        get_train_val_datasets(
            train_dir=TRAIN_DIR,
            model_type="efficientnet",
            batch_size=BATCH_SIZE,
            seed=RANDOM_SEED,
            use_augmentation=True,
            split_manifest=MANIFEST_PATH,
        )
    )

    if (
        len(train_paths) != EXPECTED_TRAIN_IMAGES
        or len(val_paths) != EXPECTED_VAL_IMAGES
    ):
        raise RuntimeError(
            f"Unexpected split sizes: train={len(train_paths)}, "
            f"validation={len(val_paths)}. Expected "
            f"{EXPECTED_TRAIN_IMAGES} and {EXPECTED_VAL_IMAGES}."
        )

    # Confirm class labels align with the paths loaded by the manifest.
    if len(train_labels) != len(train_paths) or len(val_labels) != len(val_paths):
        raise RuntimeError("Path/label count mismatch.")

    print("\nSimilarity-grouped manifest loaded and validated.")
    print(f"Training images: {len(train_paths)}")
    print(f"Validation images: {len(val_paths)}")
    print("The official test set will not be used during training.")
    print("Note: grouping is based on perceptual similarity, not patient IDs.")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    # Stage 1: train the fusion/classification head.
    model = build_multiscale_fusion_model()
    extractor = model.get_layer("multiscale_feature_extractor")
    extractor.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(STAGE1_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stage1_monitor = MacroF1Checkpoint(
        val_ds, val_labels, STAGE1_CHECKPOINT, patience=3
    )

    stage1_start = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE1_EPOCHS,
        callbacks=[
            stage1_monitor,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-5,
                verbose=1,
            ),
        ],
        verbose=1,
    )
    stage1_seconds = time.time() - stage1_start

    if not STAGE1_CHECKPOINT.exists():
        raise RuntimeError(
            "Stage 1 did not save a checkpoint. Review training logs."
        )

    # Stage 2: fine-tune selected backbone layers.
    model = tf.keras.models.load_model(STAGE1_CHECKPOINT, compile=False)
    model = configure_finetuning(model)

    trainable_parameters = sum(
        int(np.prod(weight.shape)) for weight in model.trainable_weights
    )
    print("Stage 2 trainable parameters:", trainable_parameters)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(STAGE2_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    stage2_monitor = MacroF1Checkpoint(
        val_ds, val_labels, FINAL_CHECKPOINT, patience=3
    )

    stage2_start = time.time()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=STAGE2_EPOCHS,
        callbacks=[
            stage2_monitor,
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=2,
                min_lr=1e-7,
                verbose=1,
            ),
        ],
        verbose=1,
    )
    stage2_seconds = time.time() - stage2_start

    if stage2_monitor.best_metrics:
        selected_path = FINAL_CHECKPOINT
        selected_metrics = stage2_monitor.best_metrics
        selected_stage = 2
    else:
        selected_path = STAGE1_CHECKPOINT
        selected_metrics = stage1_monitor.best_metrics
        selected_stage = 1

    summary = {
        "experiment": "EfficientNet-B0 Multi-Scale Fusion",
        "split_type": "perceptual_similarity_grouped",
        "manifest_path": str(MANIFEST_PATH),
        "class_names": list(CLASS_NAMES),
        "training_images": len(train_paths),
        "validation_images": len(val_paths),
        "training_seed": int(RANDOM_SEED),
        "batch_size": BATCH_SIZE,
        "stage1_epochs_configured": STAGE1_EPOCHS,
        "stage2_epochs_configured": STAGE2_EPOCHS,
        "stage1_seconds": round(stage1_seconds, 1),
        "stage2_seconds": round(stage2_seconds, 1),
        "selected_stage": selected_stage,
        "selected_checkpoint": str(selected_path),
        "best_validation_metrics": selected_metrics,
        "test_set_evaluated": False,
        "caveat": (
            "Similarity grouping uses within-class perceptual hashes; "
            "it does not establish patient-level independence."
        ),
    }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 65)
    print("SIMILARITY-GROUPED MULTI-SCALE TRAINING COMPLETE")
    print("=" * 65)
    print("Selected checkpoint:", selected_path)
    print("Best validation metrics:")
    print(json.dumps(selected_metrics, indent=2))
    print("Experiment summary:", SUMMARY_PATH)
    print("Official test set evaluated: False")


if __name__ == "__main__":
    main()
