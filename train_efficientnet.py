"""Train an EfficientNetB0 transfer learning baseline with frozen ImageNet backbone.

This script:
1. Loads train and validation splits (with untouched dataset/Testing).
2. Builds an EfficientNetB0 model with frozen pretrained ImageNet weights.
3. Adds a custom classification head (GAP + Dropout + Dense + Dropout + Softmax).
4. Trains the model using EarlyStopping and ModelCheckpoint.
5. Saves the best model to efficientnet_b0.keras.
6. Saves training logs to efficientnet_history.json.
7. Plots training/validation loss and accuracy curves to efficientnet_training_curves.png.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from preprocess import (
    BATCH_SIZE,
    CLASS_NAMES,
    RANDOM_SEED,
    get_train_validation_datasets,
)


MODEL_PATH = Path("efficientnet_b0.keras")
HISTORY_PATH = Path("efficientnet_history.json")
CURVES_PATH = Path("efficientnet_training_curves.png")


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Set practical random seeds for repeatable training runs."""
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, tf.errors.UnimplementedError):
        pass


def build_efficientnet_b0(num_classes: int = len(CLASS_NAMES)) -> tf.keras.Model:
    """Build transfer learning model with frozen ImageNet EfficientNetB0 backbone."""
    # Pretrained ImageNet backbone
    base_model = tf.keras.applications.EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
    )
    # Freeze the pretrained backbone weights
    base_model.trainable = False

    inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_image")
    # training=False ensures BatchNorm layers stay in inference mode
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = tf.keras.layers.Dropout(0.30, name="head_dropout_1")(x)
    x = tf.keras.layers.Dense(128, activation="relu", name="head_dense")(x)
    x = tf.keras.layers.Dropout(0.30, name="head_dropout_2")(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnet_b0_transfer")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def plot_training_curves(history: dict, output_path: Path) -> None:
    """Plot and save training and validation accuracy/loss curves."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss subplot
    ax1.plot(epochs, history["loss"], "o-", label="Train Loss", color="#1f77b4")
    ax1.plot(epochs, history["val_loss"], "s--", label="Val Loss", color="#ff7f0e")
    ax1.set_title("EfficientNetB0 - Training & Validation Loss", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.legend(loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Accuracy subplot
    ax2.plot(epochs, [a * 100 for a in history["accuracy"]], "o-", label="Train Accuracy", color="#2ca02c")
    ax2.plot(epochs, [a * 100 for a in history["val_accuracy"]], "s--", label="Val Accuracy", color="#d62728")
    ax2.set_title("EfficientNetB0 - Training & Validation Accuracy", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Accuracy (%)", fontsize=11)
    ax2.legend(loc="lower right")
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def count_model_parameters(model: tf.keras.Model) -> tuple[int, int, int]:
    """Return total, trainable, and frozen (non-trainable) parameters count."""
    total_params = model.count_params()
    trainable_params = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    frozen_params = total_params - trainable_params
    return total_params, trainable_params, frozen_params


def train_model(
    data_dir: Path = Path("dataset"),
    epochs: int = 20,
    batch_size: int = BATCH_SIZE,
    model_path: Path = MODEL_PATH,
    history_path: Path = HISTORY_PATH,
    curves_path: Path = CURVES_PATH,
) -> dict:
    """Train EfficientNetB0 transfer learning model and log artifacts."""
    set_reproducible_seed()

    print("Loading train & validation datasets (Testing/ directory is NOT accessed)...")
    train_dataset, validation_dataset = get_train_validation_datasets(
        data_dir=data_dir,
        model_type="efficientnet",
        batch_size=batch_size,
        seed=RANDOM_SEED,
    )

    print("Building EfficientNetB0 transfer learning model...")
    model = build_efficientnet_b0()
    model.summary()

    total_params, trainable_params, frozen_params = count_model_parameters(model)
    print(f"\nParameter Summary:")
    print(f"  Total parameters    : {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters   : {frozen_params:,}")

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    print(f"\nStarting training for up to {epochs} epochs...")
    start_time = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed_seconds = time.perf_counter() - start_time

    # Identify best epoch based on lowest validation loss
    best_epoch_index = int(np.argmin(history.history["val_loss"]))
    best_epoch = best_epoch_index + 1

    train_acc_best = float(history.history["accuracy"][best_epoch_index])
    val_acc_best = float(history.history["val_accuracy"][best_epoch_index])
    train_loss_best = float(history.history["loss"][best_epoch_index])
    val_loss_best = float(history.history["val_loss"][best_epoch_index])

    results = {
        "architecture": "EfficientNetB0 (ImageNet pretrained, frozen backbone) + Custom GAP Head",
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "frozen_parameters": frozen_params,
        "training_time_seconds": round(elapsed_seconds, 2),
        "epochs_completed": len(history.history["loss"]),
        "best_epoch": best_epoch,
        "train_accuracy_at_best_epoch": round(train_acc_best, 4),
        "validation_accuracy_at_best_epoch": round(val_acc_best, 4),
        "train_loss_at_best_epoch": round(train_loss_best, 4),
        "validation_loss_at_best_epoch": round(val_loss_best, 4),
        "history": {key: [float(val) for val in values] for key, values in history.history.items()},
    }

    history_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved training history to: {history_path.resolve()}")

    plot_training_curves(results["history"], curves_path)
    print(f"Saved training curves to: {curves_path.resolve()}")
    print(f"Saved best model checkpoint to: {model_path.resolve()}")

    print("\n==================================================")
    print("EFFICIENTNET-B0 TRANSFER LEARNING SUMMARY")
    print("==================================================")
    print(f"Architecture         : {results['architecture']}")
    print(f"Total Parameters     : {results['total_parameters']:,}")
    print(f"Trainable Parameters : {results['trainable_parameters']:,}")
    print(f"Frozen Parameters    : {results['frozen_parameters']:,}")
    print(f"Training Time        : {results['training_time_seconds']:.2f} seconds ({results['training_time_seconds'] / 60:.1f} mins)")
    print(f"Epochs Completed     : {results['epochs_completed']}")
    print(f"Best Epoch           : {results['best_epoch']}")
    print(f"Train Accuracy (best): {results['train_accuracy_at_best_epoch'] * 100:.2f}% ({results['train_accuracy_at_best_epoch']:.4f})")
    print(f"Val Accuracy (best)  : {results['validation_accuracy_at_best_epoch'] * 100:.2f}% ({results['validation_accuracy_at_best_epoch']:.4f})")
    print(f"Train Loss (best)    : {results['train_loss_at_best_epoch']:.4f}")
    print(f"Val Loss (best)      : {results['validation_loss_at_best_epoch']:.4f}")
    print("==================================================")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 transfer learning model.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--history-path", type=Path, default=HISTORY_PATH)
    parser.add_argument("--curves-path", type=Path, default=CURVES_PATH)
    arguments = parser.parse_args()

    train_model(
        data_dir=arguments.data_dir,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        model_path=arguments.model_path,
        history_path=arguments.history_path,
        curves_path=arguments.curves_path,
    )


if __name__ == "__main__":
    main()
