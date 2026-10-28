"""Train baseline custom CNN without accessing official test set."""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import tensorflow as tf

from config import BATCH_SIZE, DATA_DIR, MODELS_DIR, RANDOM_SEED, VALIDATION_FRACTION
import src.models.compat
from src.data.dataset import get_train_val_datasets
from src.models.custom_cnn import build_custom_cnn


def train_baseline_cnn(
    data_dir: Path = DATA_DIR,
    epochs: int = 30,
    batch_size: int = BATCH_SIZE,
    seed: int = RANDOM_SEED,
) -> dict:
    """Train baseline CNN strictly on training and validation splits."""
    tf.keras.utils.set_random_seed(seed)
    train_ds, val_ds, _, _, _, _ = get_train_val_datasets(
        train_dir=data_dir / "Training",
        model_type="cnn",
        batch_size=batch_size,
        validation_fraction=VALIDATION_FRACTION,
        seed=seed,
        use_augmentation=True,
    )

    model = build_custom_cnn()
    checkpoint_path = MODELS_DIR / "baseline_cnn.keras"
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ModelCheckpoint(filepath=str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=1)
    duration = time.time() - t0

    history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
    out_json = MODELS_DIR / "baseline_cnn_history.json"
    out_json.write_text(json.dumps({"training_time_sec": duration, "history": history_dict}, indent=2))
    print(f"Baseline CNN training finished in {duration:.1f}s. Checkpoint: {checkpoint_path}")
    return history_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--epochs", type=int, default=30)
    args = parser.parse_args()
    train_baseline_cnn(data_dir=args.data_dir, epochs=args.epochs)
