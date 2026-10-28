import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MANIFEST = ROOT / "experiments" / "duplicate_aware_split" / "training_manifest.csv"
MODEL_PATH = ROOT / "experiments" / "spatial_fusion" / "seed_42" / "best_model.keras"
OUT_DIR = ROOT / "experiments" / "spatial_fusion" / "seed_42"

CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

df = pd.read_csv(MANIFEST)
df = df[df["split"].astype(str).str.lower().eq("validation")].copy()

if df.empty:
    raise ValueError("No validation rows found in the manifest.")

# Resolve image paths without touching the official test split.
def resolve_path(value):
    p = Path(str(value))
    if p.is_absolute() and p.exists():
        return p
    candidates = [ROOT / p, ROOT / "dataset" / p, ROOT / "data" / p]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve validation image: {value}")

df["resolved_path"] = df["path"].map(resolve_path)

# Use the same TensorFlow resize and EfficientNet preprocessing convention.
def load_image(path):
    raw = tf.io.read_file(str(path))
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, IMAGE_SIZE, method="bilinear", antialias=True)
    image = tf.keras.applications.efficientnet.preprocess_input(image)
    return image

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

images = []
for path in df["resolved_path"]:
    images.append(load_image(path))

dataset = tf.data.Dataset.from_tensor_slices(
    tf.stack(images)
).batch(BATCH_SIZE)

probabilities = model.predict(dataset, verbose=1)
predicted_ids = np.argmax(probabilities, axis=1)

def class_id(label):
    label = str(label).strip().lower()
    if label not in CLASSES:
        raise ValueError(f"Unknown class label in manifest: {label}")
    return CLASSES.index(label)

actual_ids = df["label"].map(class_id).to_numpy()
df["actual_class"] = [CLASSES[i] for i in actual_ids]
df["predicted_class"] = [CLASSES[i] for i in predicted_ids]
df["confidence"] = probabilities.max(axis=1)
df["correct"] = actual_ids == predicted_ids

# Include each class probability for error analysis.
for i, name in enumerate(CLASSES):
    df[f"prob_{name}"] = probabilities[:, i]

out_path = OUT_DIR / "validation_predictions.csv"
df.drop(columns=["resolved_path"]).to_csv(out_path, index=False)

errors = df[~df["correct"]].copy()
errors.to_csv(OUT_DIR / "validation_misclassified.csv", index=False)

print("\nValidation confusion matrix (rows=actual, columns=predicted):")
print(pd.DataFrame(
    confusion_matrix(actual_ids, predicted_ids, labels=range(len(CLASSES))),
    index=CLASSES,
    columns=CLASSES
))

print("\nMost frequent validation errors:")
print(
    errors.groupby(["actual_class", "predicted_class"])
    .size()
    .sort_values(ascending=False)
    .head(15)
    .to_string()
)

print(f"\nSaved predictions: {OUT_DIR / 'validation_predictions.csv'}")
print(f"Saved misclassified images: {OUT_DIR / 'validation_misclassified.csv'}")
