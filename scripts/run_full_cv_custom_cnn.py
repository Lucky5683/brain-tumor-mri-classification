
from pathlib import Path
import argparse
import json
import random
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from config import CLASS_NAMES
from src.data.preprocessing import load_image_and_label, normalize_for_cnn
from src.models.custom_cnn import build_custom_cnn

ROOT = PROJECT_ROOT / "experiments" / "five_fold_cv"
MANIFEST = ROOT / "fold_manifest.csv"
OUTPUT_ROOT = ROOT / "full_runs" / "custom_cnn_final_5ep"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    output_dir = OUTPUT_ROOT / f"fold_{args.fold + 1}"
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output: {output_dir}"
        )

    if not MANIFEST.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST)
    required = {"path", "label", "fold", "group_id", "sha256"}
    if not required.issubset(df.columns):
        raise ValueError(f"Manifest must contain columns: {sorted(required)}")

    if sorted(df["fold"].unique().tolist()) != list(range(5)):
        raise ValueError("Expected fold IDs 0, 1, 2, 3, 4.")

    train_df = df[df["fold"] != args.fold].copy()
    val_df = df[df["fold"] == args.fold].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Training or validation split is empty.")

    if set(train_df["group_id"]) & set(val_df["group_id"]):
        raise ValueError("Group leakage detected between training and validation.")

    if set(train_df["sha256"]) & set(val_df["sha256"]):
        raise ValueError("Exact duplicate image leakage detected.")

    for name, frame in [("training", train_df), ("validation", val_df)]:
        missing = [
            p for p in frame["path"].astype(str)
            if not Path(p).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} {name} paths missing. Example: {missing[0]}"
            )

    class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}

    def encode_labels(frame):
        labels = frame["label"].astype(str).str.strip()
        if not set(labels).issubset(class_to_index):
            raise ValueError(f"Unexpected labels: {sorted(set(labels))}")
        return labels.map(class_to_index).to_numpy(dtype=np.int32)

    train_paths = train_df["path"].astype(str).to_numpy()
    train_labels = encode_labels(train_df)
    val_paths = val_df["path"].astype(str).to_numpy()
    val_labels = encode_labels(val_df)

    def make_dataset(paths, labels, training):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(
                min(len(paths), 2000),
                seed=seed,
                reshuffle_each_iteration=True,
            )
        ds = ds.map(
            load_image_and_label,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            normalize_for_cnn,
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        return ds.batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_dataset(train_paths, train_labels, True)
    val_ds = make_dataset(val_paths, val_labels, False)

    # Create output only after input validation succeeds.
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint = output_dir / "best_model.keras"

    model = build_custom_cnn()
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(
            str(output_dir / "training_history.csv")
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]

    print(f"Validation fold: {args.fold}")
    print(f"Training images: {len(train_df)}")
    print(f"Validation images: {len(val_df)}")
    print(f"Epoch limit: {args.epochs}")
    print(f"Output directory: {output_dir}")

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    if not checkpoint.is_file():
        raise RuntimeError("No best checkpoint was saved.")

    best_model = tf.keras.models.load_model(checkpoint)
    probabilities = best_model.predict(val_ds, verbose=1)
    predictions = probabilities.argmax(axis=1)

    metrics = {
        "model": "custom_cnn",
        "fold_index": args.fold,
        "fold_number": args.fold + 1,
        "smoke_test": False,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history["loss"]),
        "training_images": len(train_df),
        "validation_images": len(val_df),
        "accuracy": float(accuracy_score(val_labels, predictions)),
        "macro_precision": float(
            precision_score(
                val_labels, predictions, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                val_labels, predictions, average="macro", zero_division=0
            )
        ),
        "macro_f1": float(
            f1_score(
                val_labels, predictions, average="macro", zero_division=0
            )
        ),
        "class_names": list(CLASS_NAMES),
        "confusion_matrix": confusion_matrix(
            val_labels, predictions,
            labels=list(range(len(CLASS_NAMES))),
        ).tolist(),
        "classification_report": classification_report(
            val_labels,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=list(CLASS_NAMES),
            output_dict=True,
            zero_division=0,
        ),
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print("\nFOLD RESULTS")
    for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
        print(f"{key}: {metrics[key]:.4f}")
    print(f"Saved metrics: {output_dir / 'metrics.json'}")
    print("This is one fold, not the final five-fold summary.")


if __name__ == "__main__":
    main()
