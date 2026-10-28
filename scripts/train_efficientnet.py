"""Two-Stage EfficientNet-B0 transfer learning and fine-tuning pipeline.

Executes controlled experiments:
V2-E1: Two-Stage Fine-Tuning (Unweighted)
V2-E2: Two-Stage Fine-Tuning with Training-Derived Class Weighting

Monitors Validation Macro F1 as the primary model selection metric.
Test set is NEVER evaluated during training or hyperparameter tuning.
"""

from __future__ import annotations
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf

from config import (
    CLASS_NAMES,
    DATA_DIR,
    EXPERIMENTS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    TRAIN_DIR,
    VALIDATION_FRACTION,
)
import src.models.compat
from src.data.dataset import get_train_val_datasets
from src.models.efficientnet import (
    build_efficientnet_b0_stage1,
    count_parameters,
    prepare_stage2_finetuning,
)


def log_hardware_status() -> dict:
    """Detect and display TensorFlow compute device (GPU vs CPU fallback)."""
    tf_version = tf.__version__
    gpus = tf.config.list_physical_devices("GPU")
    has_gpu = bool(gpus)
    gpu_name = gpus[0].name if has_gpu else "None (CPU Fallback)"

    # Optimize CPU multi-threading
    try:
        import os
        cores = os.cpu_count() or 4
        tf.config.threading.set_intra_op_parallelism_threads(cores)
        tf.config.threading.set_inter_op_parallelism_threads(cores)
    except Exception:
        pass

    print("=" * 60)
    print("COMPUTE HARDWARE AUDIT")
    print("=" * 60)
    print(f"TensorFlow Version : {tf_version}")
    print(f"GPU Available      : {has_gpu}")
    print(f"Device Name        : {gpu_name}")
    if not has_gpu:
        print(f"Notice: Native Windows TF >= 2.11 uses CPU fallback ({cores} cores detected).")
    print("=" * 60 + "\n")

    return {"tf_version": tf_version, "gpu_available": has_gpu, "gpu_name": gpu_name}


class MacroF1Checkpoint(tf.keras.callbacks.Callback):
    """Callback evaluating Validation Macro-F1 at epoch end and saving best model."""

    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        val_labels: list[int],
        checkpoint_path: Path,
        patience: int = 4,
    ):
        super().__init__()
        self.val_dataset = val_dataset
        self.val_labels = np.array(val_labels)
        self.checkpoint_path = Path(checkpoint_path)
        self.patience = patience
        self.best_macro_f1 = -1.0
        self.best_metrics = {}
        self.wait = 0

    def on_epoch_end(self, epoch: int, logs: dict | None = None):
        logs = logs or {}
        val_probs = self.model.predict(self.val_dataset, verbose=0)
        val_preds = np.argmax(val_probs, axis=1)

        macro_f1 = float(f1_score(self.val_labels, val_preds, average="macro", zero_division=0))
        macro_rec = float(recall_score(self.val_labels, val_preds, average="macro", zero_division=0))
        macro_prec = float(precision_score(self.val_labels, val_preds, average="macro", zero_division=0))
        acc = float(np.mean(val_preds == self.val_labels))

        glioma_idx = CLASS_NAMES.index("glioma")
        gl_mask = self.val_labels == glioma_idx
        glioma_rec = float(np.mean(val_preds[gl_mask] == glioma_idx))

        logs["val_macro_f1"] = macro_f1
        logs["val_macro_recall"] = macro_rec
        logs["val_macro_precision"] = macro_prec
        logs["val_glioma_recall"] = glioma_rec

        print(
            f"\n[Validation Eval @ Epoch {epoch+1:02d}] "
            f"Macro-F1: {macro_f1:.4f} | Macro-Rec: {macro_rec:.4f} | "
            f"Glioma-Rec: {glioma_rec*100:.2f}% | Acc: {acc*100:.2f}% | "
            f"Val-Loss: {logs.get('val_loss', 0.0):.4f}"
        )

        if macro_f1 > self.best_macro_f1:
            self.best_macro_f1 = macro_f1
            self.wait = 0
            self.best_metrics = {
                "epoch": epoch + 1,
                "val_accuracy": round(acc, 4),
                "val_macro_f1": round(macro_f1, 4),
                "val_macro_recall": round(macro_rec, 4),
                "val_macro_precision": round(macro_prec, 4),
                "val_glioma_recall": round(glioma_rec, 4),
                "val_loss": round(float(logs.get("val_loss", 0.0)), 4),
            }
            self.model.save(self.checkpoint_path)
            print(f"  >>> Checkpoint Saved: {self.checkpoint_path.name} (Val Macro-F1 = {macro_f1:.4f})")
        else:
            self.wait += 1
            if self.wait >= self.patience:
                print(f"  >>> Early stopping triggered: no Val Macro-F1 improvement for {self.patience} epochs.")
                self.model.stop_training = True


def calculate_training_derived_class_weights(
    train_labels: list[int],
    unweighted_model: tf.keras.Model | None = None,
    train_ds: tf.data.Dataset | None = None,
) -> dict[int, float]:
    """Derive balanced class weights directly from training set.

    If an unweighted model is provided, weights are computed inversely proportional to
    per-class recall on the training set: W_k = (1 - Rec_k) / sum(1 - Rec_j) * C,
    amplifying weight on classes with higher misclassification rates (e.g. glioma).
    Otherwise, standard balanced class weights are used.
    """
    labels_arr = np.array(train_labels)
    unique_classes = np.unique(labels_arr)

    if unweighted_model is not None and train_ds is not None:
        print("Calculating error-based class weights from empirical training misclassification rates...")
        train_probs = unweighted_model.predict(train_ds, verbose=0)
        train_preds = np.argmax(train_probs, axis=1)

        weights = {}
        error_rates = []
        for c in unique_classes:
            mask = labels_arr == c
            rec = np.mean(train_preds[mask] == c) if np.any(mask) else 1.0
            err = max(0.05, 1.0 - rec)  # floor error rate at 5%
            error_rates.append(err)

        total_err = sum(error_rates)
        n_classes = len(unique_classes)
        for i, c in enumerate(unique_classes):
            # Normalize so mean weight is 1.0
            weights[int(c)] = float(round((error_rates[i] / total_err) * n_classes, 4))
        return weights

    weights = compute_class_weight("balanced", classes=unique_classes, y=labels_arr)
    return {int(c): float(round(w, 4)) for c, w in zip(unique_classes, weights)}


def append_to_experiment_registry(record: dict) -> None:
    """Append completed experiment record to experiments/results.csv."""
    csv_path = EXPERIMENTS_DIR / "results.csv"
    file_exists = csv_path.is_file()

    headers = [
        "experiment_id",
        "date",
        "model",
        "stage",
        "learning_rate",
        "batch_size",
        "epochs",
        "augmentation",
        "class_weighting",
        "best_val_accuracy",
        "val_macro_precision",
        "val_macro_recall",
        "val_macro_f1",
        "val_glioma_recall",
        "notes",
    ]

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)
    print(f"Logged experiment {record['experiment_id']} to {csv_path.resolve()}")


def run_experiment_e1(
    data_dir: Path = DATA_DIR,
    stage1_epochs: int = 5,
    stage2_epochs: int = 4,
    batch_size: int = 32,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-5,
    unfreeze_layers: int = 20,
    seed: int = RANDOM_SEED,
) -> tuple[dict, tf.keras.Model]:
    """Execute Experiment V2-E1: Two-Stage Fine-Tuning (Unweighted)."""
    print("\n" + "#" * 60)
    print("STARTING EXPERIMENT V2-E1: TWO-STAGE FINE-TUNING (UNWEIGHTED)")
    print("#" * 60)

    train_ds, val_ds, train_p, train_l, val_p, val_l = get_train_val_datasets(
        train_dir=data_dir / "Training",
        model_type="efficientnet",
        batch_size=batch_size,
        validation_fraction=VALIDATION_FRACTION,
        seed=seed,
        use_augmentation=True,
    )

    # STAGE 1: Head Warmup
    print("\n--- STAGE 1: Classification Head Warmup (Backbone Frozen) ---")
    model = build_efficientnet_b0_stage1(num_classes=len(CLASS_NAMES), learning_rate=stage1_lr)
    stage1_ckpt = MODELS_DIR / "efficientnet_b0_stage1.keras"

    s1_monitor = MacroF1Checkpoint(val_ds, val_l, stage1_ckpt, patience=4)
    s1_lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)

    t0 = time.time()
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=[s1_monitor, s1_lr_decay],
        verbose=1,
    )
    s1_time = time.time() - t0

    # Load best Stage 1 model before Stage 2
    if stage1_ckpt.is_file():
        model = tf.keras.models.load_model(stage1_ckpt)

    # STAGE 2: Fine-Tuning Top Backbone Layers
    print(f"\n--- STAGE 2: Fine-Tuning Top {unfreeze_layers} Layers (lr={stage2_lr}) ---")
    model = prepare_stage2_finetuning(model, unfreeze_layers=unfreeze_layers, learning_rate=stage2_lr)
    finetuned_ckpt = MODELS_DIR / "efficientnet_b0_finetuned.keras"

    s2_monitor = MacroF1Checkpoint(val_ds, val_l, finetuned_ckpt, patience=3)
    s2_lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)

    t1 = time.time()
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=[s2_monitor, s2_lr_decay],
        verbose=1,
    )
    s2_time = time.time() - t1

    # Load best fine-tuned checkpoint
    if finetuned_ckpt.is_file():
        model = tf.keras.models.load_model(finetuned_ckpt)

    best_metrics = s2_monitor.best_metrics or s1_monitor.best_metrics
    best_metrics["total_time_sec"] = round(s1_time + s2_time, 1)

    # Save training history
    history_out = MODELS_DIR / "history_v2_e1.json"
    history_out.write_text(json.dumps({
        "stage1": {k: [float(v) for v in vals] for k, vals in hist1.history.items()},
        "stage2": {k: [float(v) for v in vals] for k, vals in hist2.history.items()},
        "best_metrics": best_metrics,
    }, indent=2))

    # Log to experiments registry
    append_to_experiment_registry({
        "experiment_id": "V2-E1",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": "EfficientNet-B0 (Two-Stage)",
        "stage": "Stage 2 Fine-Tuned",
        "learning_rate": f"{stage1_lr}->{stage2_lr}",
        "batch_size": batch_size,
        "epochs": f"{len(hist1.history['loss'])}+{len(hist2.history['loss'])}",
        "augmentation": "Conservative (hflip, rot5%, zoom8%, trans5%, bright10%, contrast10%)",
        "class_weighting": "None",
        "best_val_accuracy": best_metrics.get("val_accuracy"),
        "val_macro_precision": best_metrics.get("val_macro_precision"),
        "val_macro_recall": best_metrics.get("val_macro_recall"),
        "val_macro_f1": best_metrics.get("val_macro_f1"),
        "val_glioma_recall": best_metrics.get("val_glioma_recall"),
        "notes": f"V2-E1 unweighted fine-tuning. Best Val Macro-F1={best_metrics.get('val_macro_f1')}",
    })

    return best_metrics, model


def run_experiment_e2(
    unweighted_model: tf.keras.Model,
    data_dir: Path = DATA_DIR,
    stage1_epochs: int = 5,
    stage2_epochs: int = 4,
    batch_size: int = 32,
    stage1_lr: float = 1e-3,
    stage2_lr: float = 1e-5,
    unfreeze_layers: int = 20,
    seed: int = RANDOM_SEED,
) -> tuple[dict, tf.keras.Model]:
    """Execute Experiment V2-E2: Two-Stage Training with Training-Derived Class Weights."""
    print("\n" + "#" * 60)
    print("STARTING EXPERIMENT V2-E2: TWO-STAGE TRAINING (CLASS-WEIGHTED)")
    print("#" * 60)

    train_ds, val_ds, train_p, train_l, val_p, val_l = get_train_val_datasets(
        train_dir=data_dir / "Training",
        model_type="efficientnet",
        batch_size=batch_size,
        validation_fraction=VALIDATION_FRACTION,
        seed=seed,
        use_augmentation=True,
    )

    # Derive class weights strictly from the training set
    eval_train_ds, _, _, _, _, _ = get_train_val_datasets(
        train_dir=data_dir / "Training",
        model_type="efficientnet",
        batch_size=batch_size,
        validation_fraction=VALIDATION_FRACTION,
        seed=seed,
        use_augmentation=False,
    )
    class_weights = calculate_training_derived_class_weights(train_l, unweighted_model, eval_train_ds)
    print(f"Calculated Training Set Class Weights: {class_weights}")
    for idx, cname in enumerate(CLASS_NAMES):
        print(f"  {cname:12s} (label {idx}): weight = {class_weights[idx]:.4f}")

    # STAGE 1: Head Warmup with Class Weights
    print("\n--- STAGE 1: Classification Head Warmup (Backbone Frozen, Class-Weighted) ---")
    model = build_efficientnet_b0_stage1(num_classes=len(CLASS_NAMES), learning_rate=stage1_lr)
    stage1_ckpt = MODELS_DIR / "efficientnet_b0_stage1_weighted.keras"

    s1_monitor = MacroF1Checkpoint(val_ds, val_l, stage1_ckpt, patience=4)
    s1_lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5, verbose=1)

    t0 = time.time()
    hist1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage1_epochs,
        callbacks=[s1_monitor, s1_lr_decay],
        class_weight=class_weights,
        verbose=1,
    )
    s1_time = time.time() - t0

    if stage1_ckpt.is_file():
        model = tf.keras.models.load_model(stage1_ckpt)

    # STAGE 2: Fine-Tuning with Class Weights
    print(f"\n--- STAGE 2: Fine-Tuning Top {unfreeze_layers} Layers (Class-Weighted, lr={stage2_lr}) ---")
    model = prepare_stage2_finetuning(model, unfreeze_layers=unfreeze_layers, learning_rate=stage2_lr)
    weighted_ckpt = MODELS_DIR / "efficientnet_b0_classweighted.keras"

    s2_monitor = MacroF1Checkpoint(val_ds, val_l, weighted_ckpt, patience=3)
    s2_lr_decay = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7, verbose=1)

    t1 = time.time()
    hist2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=stage2_epochs,
        callbacks=[s2_monitor, s2_lr_decay],
        class_weight=class_weights,
        verbose=1,
    )
    s2_time = time.time() - t1

    if weighted_ckpt.is_file():
        model = tf.keras.models.load_model(weighted_ckpt)

    best_metrics = s2_monitor.best_metrics or s1_monitor.best_metrics
    best_metrics["total_time_sec"] = round(s1_time + s2_time, 1)
    best_metrics["class_weights"] = class_weights

    # Save training history
    history_out = MODELS_DIR / "history_v2_e2.json"
    history_out.write_text(json.dumps({
        "stage1": {k: [float(v) for v in vals] for k, vals in hist1.history.items()},
        "stage2": {k: [float(v) for v in vals] for k, vals in hist2.history.items()},
        "class_weights": class_weights,
        "best_metrics": best_metrics,
    }, indent=2))

    # Log to experiments registry
    append_to_experiment_registry({
        "experiment_id": "V2-E2",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "model": "EfficientNet-B0 (Class-Weighted Two-Stage)",
        "stage": "Stage 2 Fine-Tuned",
        "learning_rate": f"{stage1_lr}->{stage2_lr}",
        "batch_size": batch_size,
        "epochs": f"{len(hist1.history['loss'])}+{len(hist2.history['loss'])}",
        "augmentation": "Conservative (hflip, rot5%, zoom8%, trans5%, bright10%, contrast10%)",
        "class_weighting": str(class_weights),
        "best_val_accuracy": best_metrics.get("val_accuracy"),
        "val_macro_precision": best_metrics.get("val_macro_precision"),
        "val_macro_recall": best_metrics.get("val_macro_recall"),
        "val_macro_f1": best_metrics.get("val_macro_f1"),
        "val_glioma_recall": best_metrics.get("val_glioma_recall"),
        "notes": f"V2-E2 class-weighted fine-tuning. Best Val Macro-F1={best_metrics.get('val_macro_f1')}",
    })

    return best_metrics, model


def run_all_experiments():
    """Execute hardware audit, V2-E1, and V2-E2 sequentially."""
    hw = log_hardware_status()

    # V2-E1
    e1_metrics, e1_model = run_experiment_e1(stage1_epochs=5, stage2_epochs=4)

    # V2-E2 (passes trained e1_model to compute empirical training misclassification rates)
    e2_metrics, e2_model = run_experiment_e2(unweighted_model=e1_model, stage1_epochs=5, stage2_epochs=4)

    print("\n" + "=" * 60)
    print("CONTROLLED EXPERIMENTS COMPLETE (VALIDATION EVALUATION)")
    print("=" * 60)
    print("Experiment V2-E1 (Unweighted):")
    print(f"  Val Macro-F1      : {e1_metrics['val_macro_f1']:.4f}")
    print(f"  Val Macro-Recall  : {e1_metrics['val_macro_recall']:.4f}")
    print(f"  Val Glioma-Recall : {e1_metrics['val_glioma_recall']*100:.2f}%")
    print(f"  Val Accuracy      : {e1_metrics['val_accuracy']*100:.2f}%")
    print("\nExperiment V2-E2 (Class-Weighted):")
    print(f"  Val Macro-F1      : {e2_metrics['val_macro_f1']:.4f}")
    print(f"  Val Macro-Recall  : {e2_metrics['val_macro_recall']:.4f}")
    print(f"  Val Glioma-Recall : {e2_metrics['val_glioma_recall']*100:.2f}%")
    print(f"  Val Accuracy      : {e2_metrics['val_accuracy']*100:.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    run_all_experiments()
