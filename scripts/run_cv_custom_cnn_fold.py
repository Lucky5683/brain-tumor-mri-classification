from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from config import CLASS_NAMES, IMAGE_SIZE
from src.data.preprocessing import load_image_and_label, normalize_for_cnn
from src.models.custom_cnn import build_custom_cnn

SEED = 42
FOLD = 0
BATCH_SIZE = 16
EPOCHS = 10

ROOT = PROJECT_ROOT / "experiments" / "five_fold_cv"
MANIFEST = ROOT / "fold_manifest.csv"
OUT = ROOT / "custom_cnn" / f"fold_{FOLD + 1}"
OUT.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

df = pd.read_csv(MANIFEST)
if "fold" not in df.columns:
    raise ValueError("fold_manifest.csv has no fold column")

train_df = df[df["fold"] != FOLD].copy()
val_df = df[df["fold"] == FOLD].copy()

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation split is empty")

if set(train_df["group_id"]) & set(val_df["group_id"]):
    raise ValueError("Duplicate-group leakage detected")

for part_name, part in [("train", train_df), ("validation", val_df)]:
    missing = [p for p in part["path"].astype(str) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} {part_name} paths are missing. Example: {missing[:3]}"
        )

class_to_index = {name: i for i, name in enumerate(CLASS_NAMES)}

def encode_labels(frame):
    labels = frame["label"].astype(str)
    if set(labels).issubset(class_to_index):
        return labels.map(class_to_index).to_numpy(dtype=np.int32)
    return pd.to_numeric(labels, errors="raise").to_numpy(dtype=np.int32)

train_paths = train_df["path"].astype(str).to_numpy()
train_labels = encode_labels(train_df)
val_paths = val_df["path"].astype(str).to_numpy()
val_labels = encode_labels(val_df)

def make_dataset(paths, labels, training):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(
            min(len(paths), 2000),
            seed=SEED,
            reshuffle_each_iteration=True,
        )
    ds = ds.map(load_image_and_label, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(normalize_for_cnn, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(train_paths, train_labels, True)
val_ds = make_dataset(val_paths, val_labels, False)

model = build_custom_cnn()
checkpoint = OUT / "best_model.keras"

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        str(checkpoint), monitor="val_loss",
        save_best_only=True, verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.CSVLogger(str(OUT / "training_history.csv")),
]

print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Classes:", CLASS_NAMES)
print("Output directory:", OUT)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

best_model = tf.keras.models.load_model(checkpoint)
probabilities = best_model.predict(val_ds, verbose=1)
predictions = probabilities.argmax(axis=1)

metrics = {
    "model": "custom_cnn",
    "fold": FOLD + 1,
    "training_images": int(len(train_df)),
    "validation_images": int(len(val_df)),
    "accuracy": float(accuracy_score(val_labels, predictions)),
    "macro_precision": float(
        precision_score(val_labels, predictions, average="macro", zero_division=0)
    ),
    "macro_recall": float(
        recall_score(val_labels, predictions, average="macro", zero_division=0)
    ),
    "macro_f1": float(
        f1_score(val_labels, predictions, average="macro", zero_division=0)
    ),
    "class_names": list(CLASS_NAMES),
    "confusion_matrix": confusion_matrix(
        val_labels, predictions, labels=list(range(len(CLASS_NAMES)))
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

with open(OUT / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("\nFOLD 1 RESULTS")
for key in ("accuracy", "macro_precision", "macro_recall", "macro_f1"):
    print(f"{key}: {metrics[key]:.4f}")

print("Saved checkpoint:", checkpoint)
print("Saved metrics:", OUT / "metrics.json")
print("Smoke test complete; this is not the full five-fold experiment.")
