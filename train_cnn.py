"""Train a simple CNN baseline without using the official test set.

Run from the project folder after installing requirements:
    python train_cnn.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import tensorflow as tf

from preprocess import BATCH_SIZE, CLASS_NAMES, RANDOM_SEED, get_train_validation_datasets


MODEL_PATH = Path("baseline_cnn.keras")
HISTORY_PATH = Path("cnn_history.json")


def set_reproducible_seed(seed: int = RANDOM_SEED) -> None:
    """Set practical random seeds for repeatable training runs."""
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, tf.errors.UnimplementedError):
        # Some TensorFlow operations or hardware configurations do not support it.
        pass


def build_cnn(num_classes: int = len(CLASS_NAMES)) -> tf.keras.Model:
    """Create a compact CNN suitable as a baseline model."""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(256, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.40),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ],
        name="baseline_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the baseline CNN on the training/validation split.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    arguments = parser.parse_args()

    set_reproducible_seed()
    train_dataset, validation_dataset = get_train_validation_datasets(
        data_dir=arguments.data_dir,
        model_type="cnn",
        batch_size=arguments.batch_size,
        seed=RANDOM_SEED,
    )

    model = build_cnn()
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=7, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor="val_loss", save_best_only=True, verbose=1
        ),
    ]

    start_time = time.perf_counter()
    history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=arguments.epochs,
        callbacks=callbacks,
        verbose=2,
    )
    elapsed_seconds = time.perf_counter() - start_time

    best_epoch_index = min(range(len(history.history["val_loss"])), key=history.history["val_loss"].__getitem__)
    best_epoch = best_epoch_index + 1
    results = {
        "best_epoch": best_epoch,
        "training_time_seconds": round(elapsed_seconds, 2),
        "train_loss_at_best_epoch": float(history.history["loss"][best_epoch_index]),
        "train_accuracy_at_best_epoch": float(history.history["accuracy"][best_epoch_index]),
        "validation_loss_at_best_epoch": float(history.history["val_loss"][best_epoch_index]),
        "validation_accuracy_at_best_epoch": float(history.history["val_accuracy"][best_epoch_index]),
        "parameters": model.count_params(),
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    HISTORY_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\nBaseline CNN training complete (official test set was not used).")
    print(f"Best epoch: {results['best_epoch']}")
    print(f"Training time: {results['training_time_seconds']:.2f} seconds")
    print(f"Parameters: {results['parameters']:,}")
    print(f"Train accuracy at best epoch: {results['train_accuracy_at_best_epoch']:.4f}")
    print(f"Validation accuracy at best epoch: {results['validation_accuracy_at_best_epoch']:.4f}")
    print(f"Train loss at best epoch: {results['train_loss_at_best_epoch']:.4f}")
    print(f"Validation loss at best epoch: {results['validation_loss_at_best_epoch']:.4f}")
    print(f"Saved best model: {MODEL_PATH.resolve()}")
    print(f"Saved history: {HISTORY_PATH.resolve()}")


if __name__ == "__main__":
    main()
