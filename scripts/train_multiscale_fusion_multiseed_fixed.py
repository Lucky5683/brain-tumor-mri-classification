"""Multi-seed fine-tuning for the grouped multi-scale fusion model.

Run from the project root:
    python -u scripts/train_multiscale_fusion_multiseed.py

The official Testing set is intentionally never loaded or evaluated.
"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import sys, json, time, random, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
)

# --------------------------- Configuration ---------------------------
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
import src.models.compat  # noqa: F401

MANIFEST = ROOT / "experiments" / "duplicate_aware_split" / "training_manifest.csv"
BASE_CHECKPOINT = ROOT / "models" / "multiscale_fusion_grouped_best.keras"
OUTPUT_ROOT = ROOT / "experiments" / "multiseed_fusion"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
LABEL_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 224
BATCH_SIZE = 32
SEEDS = [42, 123, 2026]
MIN_EPOCHS, MAX_EPOCHS = 8, 12
PATIENCE_AFTER_MIN = 3
INITIAL_LR, MIN_LR = 1e-5, 1e-7
LR_FACTOR, LR_PATIENCE = 0.5, 2
UNFREEZE_LAST_LAYERS = 20

try:
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except RuntimeError:
    pass

print("Project:", ROOT, flush=True)
print("TensorFlow:", tf.__version__, flush=True)
print("GPU devices:", tf.config.list_physical_devices("GPU"), flush=True)
print("Training device:", "GPU" if tf.config.list_physical_devices("GPU") else "CPU", flush=True)

# --------------------------- Validate manifest -----------------------
if not MANIFEST.is_file():
    raise FileNotFoundError(f"Manifest missing: {MANIFEST}")
if not BASE_CHECKPOINT.is_file():
    raise FileNotFoundError(f"Checkpoint missing: {BASE_CHECKPOINT}")

df = pd.read_csv(MANIFEST)
required = {"path", "label", "split", "sha256", "group_id"}
if required - set(df.columns):
    raise ValueError(f"Missing manifest columns: {sorted(required - set(df.columns))}")
if len(df) != 5600:
    raise ValueError(f"Expected 5600 rows, found {len(df)}")
if set(df["label"].unique()) != set(CLASS_NAMES):
    raise ValueError(f"Unexpected labels: {df['label'].unique()}")
if set(df["split"].unique()) != {"train", "validation"}:
    raise ValueError(f"Unexpected splits: {df['split'].unique()}")

train_df = df[df.split == "train"].copy().reset_index(drop=True)
val_df = df[df.split == "validation"].copy().reset_index(drop=True)
if len(train_df) != 4447 or len(val_df) != 1153:
    raise ValueError(f"Unexpected split sizes: train={len(train_df)}, val={len(val_df)}")

def resolve_path(value):
    p = Path(str(value))
    return p if p.is_absolute() else ROOT / p

train_df["resolved_path"] = train_df.path.map(resolve_path)
val_df["resolved_path"] = val_df.path.map(resolve_path)
all_paths = pd.concat([train_df.resolved_path, val_df.resolved_path], ignore_index=True)

print("Checking paths, SHA256 hashes, and group separation...", flush=True)
hashes = {}
for _, row in df.iterrows():
    p = resolve_path(row["path"])
    if not p.is_file():
        raise FileNotFoundError(f"Missing image: {p}")
    actual = hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != row["sha256"]:
        raise ValueError(f"Manifest hash mismatch: {p}")
    hashes[str(p.resolve())] = actual

train_hashes = {hashes[str(p.resolve())] for p in train_df.resolved_path}
val_hashes = {hashes[str(p.resolve())] for p in val_df.resolved_path}
if train_hashes & val_hashes:
    raise ValueError("Exact SHA256 duplicates cross train/validation.")
if set(train_df.group_id) & set(val_df.group_id):
    raise ValueError("Similarity groups cross train/validation.")

test_root = (ROOT / "dataset" / "Testing").resolve()
for p in all_paths:
    p = p.resolve()
    if p == test_root or test_root in p.parents:
        raise ValueError(f"Official test image found in manifest: {p}")

print("Manifest checks passed.", flush=True)
print(f"Training: {len(train_df)} | Validation: {len(val_df)}", flush=True)
print("No exact-hash or similarity-group crossing. Test set untouched.", flush=True)
print("Train class counts:\n", train_df.label.value_counts().reindex(CLASS_NAMES), flush=True)
print("Validation class counts:\n", val_df.label.value_counts().reindex(CLASS_NAMES), flush=True)

# --------------------------- Input pipeline --------------------------
# The saved EfficientNet has internal rescaling/normalization; inputs stay 0-255.
def decode_image(path, label):
    image = tf.io.decode_image(
        tf.io.read_file(path), channels=3, expand_animations=False
    )
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
    return tf.cast(image, tf.float32), label

def build_datasets(seed):
    x_train = train_df.resolved_path.map(str).to_numpy()
    y_train = train_df.label.map(LABEL_TO_ID).to_numpy(np.int32)
    x_val = val_df.resolved_path.map(str).to_numpy()
    y_val = val_df.label.map(LABEL_TO_ID).to_numpy(np.int32)

    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_ds = train_ds.shuffle(len(train_df), seed=seed, reshuffle_each_iteration=True)
    train_ds = train_ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(0.05, seed=seed + 1),
        tf.keras.layers.RandomZoom(0.08, seed=seed + 2),
        tf.keras.layers.RandomTranslation(0.04, 0.04, seed=seed + 3),
    ], name=f"augmentation_seed_{seed}")
    train_ds = train_ds.map(
        lambda image, label: (augmentation(image, training=True), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_ds = val_ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    train_ds = train_ds.with_options(opts).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.with_options(opts).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, val_ds, y_val

# --------------------------- Metrics/checkpoints ---------------------
class ValidationMetrics(tf.keras.callbacks.Callback):
    def __init__(self, val_ds, y_true, output_dir):
        super().__init__()
        self.val_ds, self.y_true, self.output_dir = val_ds, y_true, output_dir
        self.best_f1, self.best_loss = -1.0, float("inf")
        self.best_epoch, self.stale_epochs, self.history = None, 0, []
        self.best_path = output_dir / "best_model.keras"
        self.last_path = output_dir / "last_completed_epoch.keras"

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        probs = self.model.predict(self.val_ds, verbose=0)
        pred = np.argmax(probs, axis=1)
        macro_f1 = f1_score(self.y_true, pred, average="macro", zero_division=0)
        acc = accuracy_score(self.y_true, pred)
        bal_acc = balanced_accuracy_score(self.y_true, pred)
        macro_p = precision_score(self.y_true, pred, average="macro", zero_division=0)
        macro_r = recall_score(self.y_true, pred, average="macro", zero_division=0)
        report = classification_report(
            self.y_true, pred, labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES, output_dict=True, zero_division=0,
        )
        val_loss = float(logs.get("val_loss", float("inf")))
        logs["val_macro_f1"] = float(macro_f1)
        logs["val_balanced_accuracy"] = float(bal_acc)

        per_class = {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            } for name in CLASS_NAMES
        }
        cm = confusion_matrix(self.y_true, pred, labels=list(range(len(CLASS_NAMES))))
        row = {
            "epoch": epoch + 1, "val_loss": val_loss, "val_accuracy": float(acc),
            "val_macro_f1": float(macro_f1), "val_macro_precision": float(macro_p),
            "val_macro_recall": float(macro_r), "val_balanced_accuracy": float(bal_acc),
            "per_class": per_class, "confusion_matrix": cm.tolist(),
        }
        self.history.append(row)
        print(
            f"\nValidation epoch {epoch+1}: accuracy={acc:.4f}, "
            f"macro-F1={macro_f1:.4f}, balanced-accuracy={bal_acc:.4f}",
            flush=True,
        )
        print("Recall:", {n: round(per_class[n]["recall"], 4) for n in CLASS_NAMES}, flush=True)

        # Recovery checkpoint after every completed epoch; best checkpoint separately.
        self.model.save(self.last_path)
        improved = macro_f1 > self.best_f1 + 1e-8 or (
            abs(macro_f1 - self.best_f1) <= 1e-8 and val_loss < self.best_loss
        )
        if improved:
            self.best_f1, self.best_loss = float(macro_f1), val_loss
            self.best_epoch, self.stale_epochs = epoch + 1, 0
            self.model.save(self.best_path)
            pd.DataFrame(report).transpose().to_csv(
                self.output_dir / "best_classification_report.csv"
            )
            np.savetxt(self.output_dir / "best_confusion_matrix.csv", cm, delimiter=",", fmt="%d")
            print("Saved best checkpoint:", self.best_path, flush=True)
        else:
            self.stale_epochs += 1

        pd.DataFrame(self.history).to_csv(self.output_dir / "history.csv", index=False)
        with (self.output_dir / "results.json").open("w", encoding="utf-8") as f:
            json.dump({
                "best_epoch": self.best_epoch, "best_macro_f1": self.best_f1,
                "best_validation_loss": self.best_loss,
                "epochs_completed": len(self.history), "history": self.history,
            }, f, indent=2)

class MacroF1EarlyStopping(tf.keras.callbacks.Callback):
    def __init__(self, metrics_callback):
        super().__init__()
        self.metrics_callback = metrics_callback

    def on_epoch_end(self, epoch, logs=None):
        if epoch + 1 >= MIN_EPOCHS and self.metrics_callback.stale_epochs >= PATIENCE_AFTER_MIN:
            print(
                f"\nEarly stopping at epoch {epoch+1}: validation macro-F1 "
                f"did not improve for {PATIENCE_AFTER_MIN} epochs.",
                flush=True,
            )
            self.model.stop_training = True

class MacroF1ReduceLR(tf.keras.callbacks.Callback):
    def __init__(self):
        super().__init__()
        self.best, self.wait = -1.0, 0

    def on_epoch_end(self, epoch, logs=None):
        score = (logs or {}).get("val_macro_f1")
        if score is None:
            return
        if score > self.best + 1e-8:
            self.best, self.wait = score, 0
            return
        self.wait += 1
        if self.wait >= LR_PATIENCE:
            lr = self.model.optimizer.learning_rate
            old = float(tf.keras.backend.get_value(lr))
            new = max(old * LR_FACTOR, MIN_LR)
            if new < old:
                if hasattr(lr, "assign"):
                    lr.assign(new)
                else:
                    tf.keras.backend.set_value(lr, new)
                print(f"\nLearning rate: {old:.2e} -> {new:.2e}", flush=True)
            self.wait = 0

# --------------------------- Correct model structure -----------------
# Actual saved model has a top-level Functional named
# "multiscale_feature_extractor", not a layer named "efficientnet_b0".
def prepare_fine_tuning(model):
    try:
        extractor = model.get_layer("multiscale_feature_extractor")
    except ValueError as exc:
        raise RuntimeError(
            "Layer 'multiscale_feature_extractor' not found. Top-level layers: "
            + str([layer.name for layer in model.layers])
        ) from exc
    if not isinstance(extractor, tf.keras.Model):
        raise RuntimeError(f"{extractor.name} is not a nested Keras Model.")

    extractor.trainable = True
    for layer in extractor.layers:
        layer.trainable = False

    eligible = [
        layer for layer in extractor.layers
        if not isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.InputLayer))
    ]
    selected = eligible[-UNFREEZE_LAST_LAYERS:]
    for layer in selected:
        layer.trainable = True
    for layer in extractor.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Enable the fusion/classification head outside the feature extractor.
    for layer in model.layers:
        if layer is not extractor:
            layer.trainable = True

    trainable_params = sum(int(np.prod(w.shape)) for w in model.trainable_weights)
    bn_trainable = [
        layer.name for layer in extractor.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization) and layer.trainable
    ]
    print("Feature extractor:", extractor.name, flush=True)
    print("Fine-tuned layers:", [layer.name for layer in selected], flush=True)
    print(f"Trainable parameters: {trainable_params:,}", flush=True)
    print("Trainable BatchNormalization layers:", bn_trainable, flush=True)
    if trainable_params <= 0:
        raise RuntimeError("No trainable parameters.")
    if bn_trainable:
        raise RuntimeError(f"BatchNormalization layers must be frozen: {bn_trainable}")
    return model

# --------------------------- Train each seed -------------------------
all_results = []
overall_start = time.time()

for seed in SEEDS:
    print("\n" + "=" * 70, flush=True)
    print(f"STARTING SEED {seed}", flush=True)
    print("=" * 70, flush=True)

    seed_dir = OUTPUT_ROOT / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)

    train_ds, val_ds, y_val = build_datasets(seed)

    # Each seed starts from the same original grouped checkpoint.
    model = tf.keras.models.load_model(BASE_CHECKPOINT, compile=False)
    model = prepare_fine_tuning(model)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    metric_cb = ValidationMetrics(val_ds, y_val, seed_dir)
    callbacks = [
        metric_cb, MacroF1ReduceLR(), MacroF1EarlyStopping(metric_cb),
        tf.keras.callbacks.TerminateOnNaN(),
    ]
    started = time.time()
    model.fit(
        train_ds, validation_data=val_ds, epochs=MAX_EPOCHS,
        callbacks=callbacks, verbose=1,
    )
    elapsed = time.time() - started

    if not metric_cb.best_path.exists():
        raise RuntimeError(f"No best checkpoint saved for seed {seed}.")
    best_model = tf.keras.models.load_model(metric_cb.best_path, compile=False)
    pred = np.argmax(best_model.predict(val_ds, verbose=0), axis=1)
    report = classification_report(
        y_val, pred, labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES, output_dict=True, zero_division=0,
    )
    result = {
        "seed": seed, "epochs_completed": len(metric_cb.history),
        "best_epoch": metric_cb.best_epoch, "training_seconds": round(elapsed, 2),
        "best_checkpoint": str(metric_cb.best_path),
        "validation_accuracy": float(accuracy_score(y_val, pred)),
        "validation_macro_f1": float(f1_score(y_val, pred, average="macro", zero_division=0)),
        "validation_balanced_accuracy": float(balanced_accuracy_score(y_val, pred)),
        "per_class": {
            name: {
                "precision": float(report[name]["precision"]),
                "recall": float(report[name]["recall"]),
                "f1": float(report[name]["f1-score"]),
                "support": int(report[name]["support"]),
            } for name in CLASS_NAMES
        },
    }
    with (seed_dir / "final_seed_summary.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    all_results.append(result)
    with (OUTPUT_ROOT / "seed_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nCompleted seed {seed}:", flush=True)
    print(json.dumps(result, indent=2), flush=True)
    del model, best_model, train_ds, val_ds
    tf.keras.backend.clear_session()

# --------------------------- Final validation-only summary -----------
all_results.sort(key=lambda item: item["validation_macro_f1"], reverse=True)
summary = {
    "baseline_checkpoint": str(BASE_CHECKPOINT),
    "manifest": str(MANIFEST),
    "official_test_evaluated": False,
    "seeds": all_results,
    "best_seed": all_results[0]["seed"],
    "best_checkpoint": all_results[0]["best_checkpoint"],
    "best_validation_macro_f1": all_results[0]["validation_macro_f1"],
    "mean_validation_accuracy": float(np.mean([r["validation_accuracy"] for r in all_results])),
    "std_validation_accuracy": float(np.std([r["validation_accuracy"] for r in all_results])),
    "mean_validation_macro_f1": float(np.mean([r["validation_macro_f1"] for r in all_results])),
    "std_validation_macro_f1": float(np.std([r["validation_macro_f1"] for r in all_results])),
    "total_training_seconds": round(time.time() - overall_start, 2),
    "caveat": (
        "Perceptual grouping does not prove patient-level independence. "
        "All seeds share the same validation set and are not independent test results."
    ),
}
with (OUTPUT_ROOT / "final_summary.json").open("w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 70, flush=True)
print("ALL SEEDS COMPLETED", flush=True)
print(json.dumps(summary, indent=2), flush=True)
print("\nOfficial test set was not evaluated.", flush=True)
