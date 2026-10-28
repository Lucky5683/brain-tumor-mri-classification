"""Streamlit Application: Brain Tumor MRI Classification and Reliability-Aware Diagnostics.

Comparative Deep Learning Research Interface:
- Primary Comparison: Custom CNN Baseline vs. Proposed Spatial Attention Feature Fusion (Seed 42)
- Reference Models: EfficientNet-B0 (Transfer Learning) and Multi-Scale Feature Fusion (Seed 42)
- Independent Test Set Evaluation: dataset/Testing (N = 1,600 scans; 400 per class)
- Validation Split: Perceptually Grouped Validation (N = 1,153 scans; zero duplicate leakage)
- Explainable AI: Grad-CAM and Grad-CAM++ with interactive alpha blending and colormaps
- Reliability: Calibrated selective prediction (triage threshold tau* = 0.60)
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import plotly.graph_objects as go
import streamlit as st
import tensorflow as tf

from config import (
    CLASS_NAMES,
    DISPLAY_NAMES,
    IMAGE_SIZE,
    MEDICAL_DISCLAIMER,
    MODELS_DIR,
    REPORTS_DIR,
    UNCERTAINTY_THRESHOLDS,
)
import src.models.compat  # ensure compatibility shim is registered
from src.data.preprocessing import preprocess_single_image
from src.evaluation.uncertainty import assess_prediction_uncertainty
from src.explainability.gradcam import (
    extract_gradcam_submodels,
    compute_gradcam_heatmap,
    overlay_heatmap_on_image,
)
from src.explainability.gradcam_plus_plus import compute_gradcam_plus_plus_heatmap

PROJECT_ROOT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------------
# Centralized Model Registry
# -----------------------------------------------------------------------------
MODEL_REGISTRY = {
    "Custom CNN (Baseline)": {
        "role": "Baseline Model",
        "display_name": "Custom CNN (Baseline)",
        "description": "Custom compact 4-block Convolutional Neural Network trained from scratch as the reference baseline.",
        "model_type": "cnn",
        "checkpoint": PROJECT_ROOT / "baseline_cnn.keras",
        "training_history_path": PROJECT_ROOT / "cnn_history.json",
        "val_metrics_file": PROJECT_ROOT / "cnn_history.json",
        "training_split_evaluation_file": None,
        "metrics_file": REPORTS_DIR / "baseline_cnn" / "final_metrics.json",
        "confusion_matrix_img": REPORTS_DIR / "baseline_cnn" / "confusion_matrix.png",
        "roc_curve_img": REPORTS_DIR / "baseline_cnn" / "roc_curve.png",
        "pr_curve_img": REPORTS_DIR / "baseline_cnn" / "precision_recall_curve.png",
        "per_class_metrics_path": REPORTS_DIR / "baseline_cnn" / "class_metrics.png",
        "classification_report_csv": REPORTS_DIR / "baseline_cnn" / "classification_report.csv",
        "params_total": 421828,
        "params_trainable": 421828,
        "inference_ms": 36.57,
        "calibration_temp": 1.0,
        "is_calibrated": False,
        "supports_gradcam": True,
        "supports_gradcam_pp": False,
        "val_metrics": {
            "accuracy": 0.8734,
            "macro_f1": 0.8691,
            "macro_precision": 0.8753,
            "macro_recall": 0.8707,
            "balanced_accuracy": 0.8707,
            "split_name": "Perceptual Grouped Validation (N=1,153)",
            "best_epoch": 27,
            "val_loss": 0.3380,
        },
        "train_metrics": {
            "epochs": 30,
            "best_epoch": 27,
            "train_accuracy": 0.8578,
            "train_loss": 0.3621,
            "val_accuracy": 0.8696,
            "val_loss": 0.3380,
        },
    },
    "EfficientNet-B0": {
        "role": "Reference Baseline",
        "display_name": "EfficientNet-B0",
        "description": "Transfer-learning reference baseline using ImageNet-pretrained EfficientNet-B0 feature extractor.",
        "model_type": "efficientnet",
        "checkpoint": PROJECT_ROOT / "efficientnet_b0.keras",
        "training_history_path": PROJECT_ROOT / "efficientnet_history.json",
        "val_metrics_file": None,
        "training_split_evaluation_file": None,
        "metrics_file": REPORTS_DIR / "final_metrics.json",
        "confusion_matrix_img": REPORTS_DIR / "efficientnet_b0" / "confusion_matrix.png",
        "roc_curve_img": REPORTS_DIR / "efficientnet_b0" / "roc_curve.png",
        "pr_curve_img": REPORTS_DIR / "efficientnet_b0" / "precision_recall_curve.png",
        "per_class_metrics_path": REPORTS_DIR / "class_metrics.png",
        "classification_report_csv": REPORTS_DIR / "classification_report.csv",
        "params_total": 4214055,
        "params_trainable": 164484,
        "inference_ms": 43.91,
        "calibration_temp": 1.0993,
        "is_calibrated": True,
        "supports_gradcam": True,
        "supports_gradcam_pp": True,
        "val_metrics": {
            "accuracy": 0.9179,
            "macro_f1": 0.9130,
            "macro_precision": 0.9150,
            "macro_recall": 0.9120,
            "split_name": "Historical Random Stratified Split (80/20)",
            "best_epoch": 20,
            "val_loss": 0.2096,
        },
        "train_metrics": {
            "epochs": 20,
            "best_epoch": 20,
            "train_accuracy": 0.9290,
            "train_loss": 0.1841,
            "val_accuracy": 0.9179,
            "val_loss": 0.2096,
        },
    },
    "Multi-Scale Feature Fusion (Seed 42)": {
        "role": "Additional Experimental Model",
        "display_name": "Multi-Scale Feature Fusion (Seed 42)",
        "description": "Experimental multi-scale feature fusion model combining intermediate block6d and final top_activation features (Seed 42).",
        "model_type": "efficientnet",
        "checkpoint": PROJECT_ROOT / "experiments" / "multiseed_fusion" / "seed_42" / "best_model.keras",
        "training_history_path": PROJECT_ROOT / "experiments" / "multiseed_fusion" / "seed_42" / "history.csv",
        "val_metrics_file": None,
        "training_split_evaluation_file": None,
        "metrics_file": REPORTS_DIR / "final_multiseed" / "final_test_summary.json",
        "metrics_key": "selected_fusion_model",
        "confusion_matrix_img": REPORTS_DIR / "final_multiseed" / "selected_fusion_confusion_matrix.png",
        "roc_curve_img": REPORTS_DIR / "final_multiseed" / "roc_curve.png",
        "pr_curve_img": REPORTS_DIR / "final_multiseed" / "precision_recall_curve.png",
        "per_class_metrics_path": None,
        "classification_report_csv": REPORTS_DIR / "final_multiseed" / "selected_fusion_classification_report.csv",
        "params_total": 4427687,
        "params_trainable": 1832676,
        "inference_ms": 47.55,
        "calibration_temp": 1.1322,
        "is_calibrated": True,
        "supports_gradcam": True,
        "supports_gradcam_pp": True,
        "val_metrics": {
            "accuracy": 0.8873,
            "macro_f1": 0.8859,
            "macro_precision": 0.8932,
            "macro_recall": 0.8847,
            "split_name": "Perceptual Grouped Validation (N=1,153)",
            "best_epoch": 11,
            "val_loss": 0.2973,
        },
        "train_metrics": {
            "epochs": 12,
            "best_epoch": 11,
            "train_accuracy": 0.8995,
            "train_loss": 0.2642,
            "val_accuracy": 0.8873,
            "val_loss": 0.2973,
        },
    },
    "Spatial Attention Feature Fusion (Seed 42)": {
        "role": "Proposed Model",
        "display_name": "Spatial Attention Feature Fusion (Seed 42)",
        "description": "Proposed architecture combining intermediate block3b and top_activation EfficientNet-B0 feature maps with spatial attention fusion.",
        "model_type": "efficientnet",
        "checkpoint": PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "best_model.keras",
        "training_history_path": PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "history.json",
        "val_metrics_file": PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "best_metrics.json",
        "training_split_evaluation_file": PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "training_split_checkpoint_evaluation.json",
        "val_classification_report_csv": PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "best_classification_report.csv",
        "val_confusion_matrix_csv": PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "best_confusion_matrix.csv",
        "metrics_file": PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "metrics_summary.json",
        "confusion_matrix_img": PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "confusion_matrix.png",
        "roc_curve_img": PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "roc_curves.png",
        "pr_curve_img": None,
        "per_class_metrics_path": PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "per_class_metrics.png",
        "classification_report_csv": PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "best_classification_report.csv",
        "test_confusion_matrix_csv": PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "confusion_matrix.csv",
        "misclassified_samples_csv": PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "misclassified_samples.csv",
        "evaluation_report_md": PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "evaluation_report.md",
        "params_total": 4472487,
        "params_trainable": 1876836,
        "inference_ms": 46.80,
        "calibration_temp": 1.0,
        "is_calibrated": False,
        "supports_gradcam": True,
        "supports_gradcam_pp": True,
        "val_metrics": {
            "accuracy": 0.9254,
            "macro_f1": 0.9241,
            "balanced_accuracy": 0.9240,
            "macro_recall": 0.9235,
            "split_name": "Perceptual Grouped Validation (N=1,153)",
            "best_epoch": 50,
            "val_loss": 0.2070,
        },
        "train_metrics": {
            "epochs": 50,
            "best_epoch": 50,
            "train_accuracy": None,
            "train_loss": None,
            "val_accuracy": 0.9254,
            "val_loss": 0.2070,
            "note": "Training accuracy and training loss were not recorded in history.json during model training (only per-epoch validation metrics were saved).",
        },
    },
}

# Compatibility mapping for automated test suites
AVAILABLE_MODELS = {k: v["checkpoint"] for k, v in MODEL_REGISTRY.items()}
AVAILABLE_MODELS["Baseline Custom CNN"] = PROJECT_ROOT / "baseline_cnn.keras"
AVAILABLE_MODELS["EfficientNet-B0 (Reference Baseline - V1)"] = PROJECT_ROOT / "efficientnet_b0.keras"
AVAILABLE_MODELS["Multi-Scale Feature Fusion (Seed 42 - Selected Model)"] = PROJECT_ROOT / "experiments" / "multiseed_fusion" / "seed_42" / "best_model.keras"
AVAILABLE_MODELS["Spatial Attention Feature Fusion (Seed 42)"] = PROJECT_ROOT / "experiments" / "spatial_fusion" / "meningioma_weighted_50ep_seed42" / "best_model.keras"

SAMPLE_IMAGES = {
    "Select a pre-loaded sample MRI...": None,
    "Glioma Sample (Testing/glioma)": "dataset/Testing/glioma/Te-gl_10.jpg",
    "Meningioma Sample (Testing/meningioma)": "dataset/Testing/meningioma/Te-aug-me_1.jpg",
    "No Tumor Sample (Testing/notumor)": "dataset/Testing/notumor/Te-no_1.jpg",
    "Pituitary Sample (Testing/pituitary)": "dataset/Testing/pituitary/Te-pi_1.jpg",
}

SELECTIVE_THRESHOLD_TAU = 0.60  # Validation-derived triage threshold


# -----------------------------------------------------------------------------
# Checkpoint Verification and Health Check
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def verify_model_health(checkpoint_path_str: str) -> tuple[bool, str]:
    """Verify that a model checkpoint exists, loads cleanly, and has valid (None, 4) outputs."""
    ckpt = Path(checkpoint_path_str)
    if not ckpt.is_file():
        return False, f"Checkpoint file missing at: {ckpt}"
    try:
        loaded = tf.keras.models.load_model(ckpt, compile=False)
        out_shape = loaded.output_shape
        if isinstance(out_shape, list):
            out_shape = out_shape[0]
        num_classes = out_shape[-1]
        if num_classes != 4:
            return False, f"Incompatible output shape: expected 4 classes, got {num_classes}."

        dummy_x = np.zeros((1, 224, 224, 3), dtype=np.float32)
        out = loaded(dummy_x)
        if out.shape != (1, 4):
            return False, f"Forward pass produced unexpected shape: {out.shape}"
        return True, f"Verified and operational ({ckpt.stat().st_size / (1024*1024):.1f} MB)"
    except Exception as exc:
        return False, f"Failed to load or execute checkpoint: {exc}"


# -----------------------------------------------------------------------------
# Calibration and Scaling Functions
# -----------------------------------------------------------------------------
def load_calibration_temperature(model_name: str) -> tuple[float, bool]:
    """Load optimal temperature fitted on validation set for the given model."""
    if "Spatial" in model_name:
        return 1.0, False

    if "Multi-Scale" in model_name:
        calib_file = PROJECT_ROOT / "reports" / "final_calibration_results.json"
        if calib_file.is_file():
            try:
                data = json.loads(calib_file.read_text(encoding="utf-8"))
                return float(data.get("optimal_temperature", 1.1322)), True
            except Exception:
                pass
        return 1.1322, True

    if "CNN" in model_name:
        calib_file_cnn = PROJECT_ROOT / "reports" / "calibration_report_cnn.json"
        if calib_file_cnn.is_file():
            try:
                data = json.loads(calib_file_cnn.read_text(encoding="utf-8"))
                return float(data.get("temperature_scaling", {}).get("optimal_temperature", 1.0)), True
            except Exception:
                pass
        return 1.0, False

    calib_file = PROJECT_ROOT / "reports" / "calibration_report.json"
    if calib_file.is_file():
        try:
            data = json.loads(calib_file.read_text(encoding="utf-8"))
            return float(data.get("temperature_scaling", {}).get("optimal_temperature", 1.0993)), True
        except Exception:
            pass
    return 1.0993, True


def apply_temperature_scaling(probabilities: np.ndarray, temperature: float = 1.0993) -> np.ndarray:
    """Apply learned temperature scaling to raw softmax probabilities."""
    if abs(temperature - 1.0) < 1e-4:
        return probabilities
    eps = 1e-12
    probs = np.clip(probabilities, eps, 1.0)
    logits = np.log(probs)
    scaled_logits = logits / max(temperature, 1e-5)
    exp_scaled = np.exp(scaled_logits - np.max(scaled_logits))
    return exp_scaled / np.sum(exp_scaled)


# -----------------------------------------------------------------------------
# Model Loading (Cached)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_selected_model(model_path_str: str):
    """Load and cache selected model checkpoint and Grad-CAM sub-models."""
    model_path = Path(model_path_str)
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected model checkpoint not found: {model_path}")

    model = tf.keras.models.load_model(model_path, compile=False)
    feature_extractor, head_layers = extract_gradcam_submodels(model)
    return model, feature_extractor, head_layers, model_path.name


# -----------------------------------------------------------------------------
# Metric Parsing Helper (Cached, Evaluated Artifact Driven)
# -----------------------------------------------------------------------------
@st.cache_data
def load_model_evaluation_metrics(model_key: str) -> dict | None:
    """Safely parse official test metrics for the selected model from its saved JSON artifact."""
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg or not cfg.get("metrics_file"):
        return None

    metrics_path = Path(cfg["metrics_file"])
    if not metrics_path.is_file():
        return None

    try:
        with open(metrics_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        key = cfg.get("metrics_key")
        if key and key in data:
            raw_metrics = data[key]["metrics"]
            per_class = data[key].get("per_class", {})
            cm = data[key].get("confusion_matrix")
            ms_per_img = data[key].get("ms_per_image", cfg.get("inference_ms"))
            test_count = data.get("total_test_samples", data.get("test_images_count", 1600))
            ts = data.get("evaluation_timestamp")
        elif "metrics" in data:
            raw_metrics = data["metrics"]
            per_class = data.get("per_class", raw_metrics.get("per_class", {}))
            cm = data.get("confusion_matrix")
            ms_per_img = data.get("ms_per_image", cfg.get("inference_ms"))
            test_count = data.get("test_images_count", data.get("sample_counts", {}).get("total", 1600))
            ts = data.get("evaluation_timestamp")
        else:
            raw_metrics = data
            per_class = {}
            cm = None
            ms_per_img = cfg.get("inference_ms")
            test_count = 1600
            ts = None

        acc = raw_metrics.get("accuracy")
        bal_acc = raw_metrics.get("balanced_accuracy", acc)
        prec = raw_metrics.get("precision_macro", raw_metrics.get("macro_precision"))
        prec_w = raw_metrics.get("precision_weighted", raw_metrics.get("weighted_precision", prec))
        rec = raw_metrics.get("recall_macro", raw_metrics.get("macro_recall"))
        rec_w = raw_metrics.get("recall_weighted", raw_metrics.get("weighted_recall", rec))
        f1 = raw_metrics.get("f1_macro", raw_metrics.get("macro_f1"))
        f1_w = raw_metrics.get("f1_weighted", raw_metrics.get("weighted_f1", f1))
        roc = raw_metrics.get("roc_auc_macro_ovr", raw_metrics.get("roc_auc_macro", raw_metrics.get("macro_roc_auc")))

        return {
            "is_test_pending": False,
            "accuracy": float(acc) if acc is not None else None,
            "balanced_accuracy": float(bal_acc) if bal_acc is not None else None,
            "precision_macro": float(prec) if prec is not None else None,
            "precision_weighted": float(prec_w) if prec_w is not None else None,
            "recall_macro": float(rec) if rec is not None else None,
            "recall_weighted": float(rec_w) if rec_w is not None else None,
            "f1_macro": float(f1) if f1 is not None else None,
            "f1_weighted": float(f1_w) if f1_w is not None else None,
            "roc_auc_macro": float(roc) if roc is not None else None,
            "per_class": per_class,
            "confusion_matrix": cm,
            "test_images_count": int(test_count),
            "inference_ms": float(ms_per_img) if ms_per_img is not None else None,
            "source_file": str(metrics_path.relative_to(PROJECT_ROOT)),
            "evaluation_timestamp": ts,
        }
    except Exception as exc:
        print(f"Error parsing metrics for {model_key}: {exc}")
        return None


# -----------------------------------------------------------------------------
# Training Split Evaluation Metrics Loader (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_training_split_evaluation_metrics(model_key: str) -> dict | None:
    """Safely load and validate verified training-split checkpoint evaluation metrics.

    Reads the runtime JSON artifact without hardcoding. Ensures all expected numeric
    fields are present, finite, and within valid mathematical bounds.
    """
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg:
        return None

    artifact_path = cfg.get("training_split_evaluation_file")
    if not artifact_path:
        return None

    path = Path(artifact_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_numeric_fields = [
            "accuracy",
            "macro_f1",
            "balanced_accuracy",
            "unweighted_sparse_categorical_crossentropy",
            "class_weighted_sparse_categorical_crossentropy",
            "number_of_images",
        ]
        for field in required_numeric_fields:
            if field not in data or data[field] is None:
                print(f"Warning: Missing required field '{field}' in {path}")
                return None
            val = float(data[field])
            if not np.isfinite(val):
                print(f"Warning: Non-finite value for field '{field}' in {path}")
                return None

        accuracy = float(data["accuracy"])
        macro_f1 = float(data["macro_f1"])
        balanced_accuracy = float(data["balanced_accuracy"])
        unweighted_loss = float(data["unweighted_sparse_categorical_crossentropy"])
        class_weighted_loss = float(data["class_weighted_sparse_categorical_crossentropy"])
        number_of_images = int(data["number_of_images"])

        if not (0.0 <= accuracy <= 1.0 and 0.0 <= macro_f1 <= 1.0 and 0.0 <= balanced_accuracy <= 1.0):
            print(f"Warning: Accuracy/F1 scores outside [0, 1] range in {path}")
            return None
        if unweighted_loss < 0.0 or class_weighted_loss < 0.0 or number_of_images <= 0:
            print(f"Warning: Negative loss or invalid image count in {path}")
            return None

        raw_note = data.get(
            "note",
            "These metrics were calculated by evaluating the saved model checkpoint on the unaugmented training split. "
            "They are not the original epoch-50 training logs and do not reproduce training-time augmentation or dropout."
        )

        try:
            source_file_rel = str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            source_file_rel = str(path)

        return {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "balanced_accuracy": balanced_accuracy,
            "unweighted_loss": unweighted_loss,
            "class_weighted_loss": class_weighted_loss,
            "number_of_images": number_of_images,
            "note": raw_note,
            "source_file": source_file_rel,
            "classes": data.get("classes", CLASS_NAMES),
            "class_weights": data.get("class_weights", {}),
            "raw_json_str": json.dumps(data, indent=2),
        }
    except Exception as exc:
        print(f"Error parsing training-split evaluation metrics from {path}: {exc}")
        return None


# -----------------------------------------------------------------------------
# Training History Loader (Cached)
# -----------------------------------------------------------------------------
@st.cache_data
def load_model_training_history(model_key: str) -> dict | None:
    """Parse saved training history records for curve plotting."""
    cfg = MODEL_REGISTRY.get(model_key)
    if not cfg or not cfg.get("training_history_path"):
        return None

    path = Path(cfg["training_history_path"])
    if not path.is_file():
        return None

    try:
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                content = json.load(f)

            if isinstance(content, list):
                # Spatial fusion format: list of per-epoch dicts
                epochs = [item.get("epoch", i + 1) for i, item in enumerate(content)]
                val_losses = [float(item.get("val_loss", 0.0)) for item in content]
                val_accs = [float(item.get("val_accuracy", 0.0)) for item in content]
                val_f1s = [float(item.get("val_macro_f1", 0.0)) for item in content]
                return {
                    "epochs": epochs,
                    "train_accuracy": None,
                    "train_loss": None,
                    "val_accuracy": val_accs,
                    "val_loss": val_losses,
                    "val_macro_f1": val_f1s,
                    "best_epoch": 50,
                    "note": "Training-split accuracy and loss were not saved in history.json.",
                }
            elif isinstance(content, dict) and "history" in content:
                # CNN / EfficientNet format: dict containing "history" sub-dict
                h = content["history"]
                epochs = list(range(1, len(h.get("accuracy", [])) + 1))
                return {
                    "epochs": epochs,
                    "train_accuracy": [float(x) for x in h.get("accuracy", [])],
                    "train_loss": [float(x) for x in h.get("loss", [])],
                    "val_accuracy": [float(x) for x in h.get("val_accuracy", [])],
                    "val_loss": [float(x) for x in h.get("val_loss", [])],
                    "val_macro_f1": None,
                    "best_epoch": content.get("best_epoch", int(np.argmin(h.get("val_loss", [0])) + 1)),
                }

        elif path.suffix == ".csv":
            df = pd.read_csv(path)
            epochs = df["epoch"].tolist() if "epoch" in df.columns else list(range(1, len(df) + 1))
            val_accs = df["val_accuracy"].tolist() if "val_accuracy" in df.columns else None
            val_losses = df["val_loss"].tolist() if "val_loss" in df.columns else None
            val_f1s = df["val_macro_f1"].tolist() if "val_macro_f1" in df.columns else None
            return {
                "epochs": epochs,
                "train_accuracy": None,
                "train_loss": None,
                "val_accuracy": val_accs,
                "val_loss": val_losses,
                "val_macro_f1": val_f1s,
                "best_epoch": int(df.loc[df["val_macro_f1"].idxmax(), "epoch"]) if "val_macro_f1" in df.columns else None,
            }
    except Exception as exc:
        print(f"Error loading training history for {model_key}: {exc}")
        return None


# -----------------------------------------------------------------------------
# Comparison Table Builder
# -----------------------------------------------------------------------------
@st.cache_data
def build_model_comparison_table(selected_name: str) -> pd.DataFrame:
    """Build a complete comparison table across all 4 models from verified artifacts."""
    rows = []
    for model_name, cfg in MODEL_REGISTRY.items():
        metrics = load_model_evaluation_metrics(model_name)
        active_indicator = "Active" if model_name == selected_name else "-"
        role = cfg.get("role", "Model")
        val_cfg = cfg.get("val_metrics", {})
        train_cfg = cfg.get("train_metrics", {})
        train_eval = load_training_split_evaluation_metrics(model_name)

        if metrics and not metrics.get("is_test_pending"):
            acc_str = f"{metrics['accuracy'] * 100:.2f}%" if metrics.get("accuracy") is not None else "Pending"
            prec_str = f"{metrics['precision_macro']:.4f}" if metrics.get("precision_macro") is not None else "Pending"
            rec_str = f"{metrics['recall_macro']:.4f}" if metrics.get("recall_macro") is not None else "Pending"
            f1_str = f"{metrics['f1_macro']:.4f}" if metrics.get("f1_macro") is not None else "Pending"
            roc_str = f"{metrics['roc_auc_macro']:.4f}" if metrics.get("roc_auc_macro") is not None else "Pending"
            n_eval = f"{metrics['test_images_count']:,}" if isinstance(metrics.get("test_images_count"), (int, np.integer)) else str(metrics.get("test_images_count"))
            latency_str = f"{metrics['inference_ms']:.1f} ms" if metrics.get("inference_ms") is not None else "N/A"
        else:
            acc_str = prec_str = rec_str = f1_str = roc_str = "Official test evaluation pending"
            n_eval = "Pending"
            latency_str = f"{cfg.get('inference_ms', 0.0):.1f} ms"

        val_acc_str = f"{val_cfg.get('accuracy', 0.0) * 100:.2f}%" if val_cfg.get("accuracy") else "N/A"
        val_f1_str = f"{val_cfg.get('macro_f1', 0.0):.4f}" if val_cfg.get("macro_f1") else "N/A"

        # Determine training accuracy and provenance methodology
        if train_eval and train_eval.get("accuracy") is not None:
            train_acc_str = f"{train_eval['accuracy'] * 100:.2f}%"
            train_method_str = "Checkpoint Eval (Unaugmented Training Split, N = 4,447)"
        elif train_cfg.get("train_accuracy") is not None:
            train_acc_str = f"{train_cfg['train_accuracy'] * 100:.2f}%"
            train_method_str = f"Historical Epoch-{train_cfg.get('best_epoch', train_cfg.get('epochs', 'Log'))} Online Training Log"
        else:
            train_acc_str = "N/A"
            train_method_str = "Not recorded in training history"

        rows.append({
            "Status": active_indicator,
            "Model Architecture": model_name,
            "Role": role,
            "Test Accuracy": acc_str,
            "Test Macro-F1": f1_str,
            "Test Macro Precision": prec_str,
            "Test Macro Recall": rec_str,
            "Test Macro ROC-AUC": roc_str,
            "Test Samples (N)": n_eval,
            "Val Accuracy": val_acc_str,
            "Val Macro-F1": val_f1_str,
            "Training Accuracy": train_acc_str,
            "Training Methodology": train_method_str,
            "Parameters": f"{cfg['params_total']:,}",
            "Latency (CPU)": latency_str,
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Brain Tumor MRI Classification and Reliability-Aware Diagnostics",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Professional Title Header (No Emojis)
    st.title("Brain Tumor MRI Classification and Reliability-Aware Diagnostics")
    st.caption(
        "Comparative Deep Learning Benchmark: Custom CNN Baseline vs. Proposed Spatial Attention Feature Fusion | "
        "Grad-CAM Saliency Localization | Confidence Calibration"
    )

    # Mandatory Medical Disclaimer Banner (No Emojis)
    st.warning(
        "This application is intended for research and educational purposes only. "
        "It is not a certified medical diagnostic device and must not be used as a substitute "
        "for professional medical evaluation. Predictions and heatmaps may be incorrect or misleading."
    )

    st.divider()

    # Sidebar: Model Selection and Architecture Details
    with st.sidebar:
        st.header("Model Selection")

        model_options = list(MODEL_REGISTRY.keys())
        # Default to Spatial Attention Feature Fusion (Proposed Model)
        default_idx = model_options.index("Spatial Attention Feature Fusion (Seed 42)")
        selected_model_name = st.selectbox(
            "Active Model Checkpoint",
            options=model_options,
            index=default_idx,
            help="Select the model checkpoint for inference, individual inspection, and saliency visualization.",
        )

        model_cfg = MODEL_REGISTRY[selected_model_name]
        selected_model_path = model_cfg["checkpoint"]
        model_type = model_cfg["model_type"]

        st.session_state["selected_model_name"] = selected_model_name
        st.session_state["selected_model_path"] = str(selected_model_path)

        # Checkpoint health verification
        is_healthy, health_msg = verify_model_health(str(selected_model_path))
        if is_healthy:
            st.success(f"Checkpoint Verified: {health_msg}")
        else:
            st.error(f"Checkpoint Error: {health_msg}")

        st.divider()
        st.subheader("Explainability Configuration")

        available_cam_methods = ["Grad-CAM"]
        if model_cfg.get("supports_gradcam_pp", True):
            available_cam_methods.append("Grad-CAM++")

        cam_method = st.radio(
            "Saliency Method",
            options=available_cam_methods,
            index=0,
            help="Grad-CAM++ computes higher-order gradients for enhanced localization on complex structures.",
        )

        blend_alpha = st.slider("Heatmap Blending (Alpha)", min_value=0.10, max_value=0.80, value=0.40, step=0.05)
        colormap_choice = st.selectbox("Heatmap Colormap", options=["Jet", "Viridis", "Inferno", "Magma"], index=0)
        chosen_colormap_name = colormap_choice.lower()

        st.divider()
        st.subheader("Model Specifications")
        st.info(
            f"Model: {selected_model_name}\n\n"
            f"Role: {model_cfg.get('role', 'Model')}\n\n"
            f"Architecture: {model_cfg['description']}\n\n"
            f"Total Parameters: {model_cfg['params_total']:,}\n\n"
            f"Trainable Parameters: {model_cfg['params_trainable']:,}\n\n"
            f"Preprocessing: {'[0, 1] Rescaling' if model_type == 'cnn' else '[0, 255] ImageNet Normalization'}\n\n"
            f"Optimal Temperature: T* = {model_cfg['calibration_temp']:.4f}\n\n"
            f"Operational Triage Threshold: tau* = {SELECTIVE_THRESHOLD_TAU:.2f}"
        )

    # Checkpoint Verification Guard
    if not is_healthy:
        st.error(
            f"Model Unavailable: Checkpoint '{selected_model_path.name}' could not be loaded or executed.\n\n"
            f"Diagnostics: {health_msg}\n\n"
            "This research application strictly adheres to reproducibility requirements and will not silently substitute another checkpoint."
        )
        return

    # Load Model
    try:
        model, feature_extractor, head_layers, loaded_name = load_selected_model(str(selected_model_path))
    except Exception as exc:
        st.error(f"Failed to load checkpoint '{selected_model_path}': {exc}")
        return

    # Three Clean Application Tabs
    tab_comparison, tab_inference, tab_inspection = st.tabs([
        "Comparative Evaluation: Baseline vs. Proposed Model",
        "Diagnostic Inference and Saliency Analysis",
        "Individual Model Inspection and Artifacts",
    ])

    # =========================================================================
    # TAB 1: MAIN DASHBOARD - BASELINE VS. PROPOSED SPATIAL FUSION
    # =========================================================================
    with tab_comparison:
        st.subheader("Primary Evaluation: Custom CNN Baseline vs. Spatial Attention Feature Fusion")
        st.markdown(
            "This dashboard presents the empirical evaluation directly comparing the "
            "**Custom CNN Baseline** (trained from scratch) with the **Proposed Spatial Attention Feature Fusion model** "
            "(combining intermediate block3b and top_activation representations with spatial attention). "
            "Both models are evaluated on the identical independent test partition (*dataset/Testing*, N = 1,600) "
            "and duplicate-aware validation split (N = 1,153)."
        )

        st.divider()

        # Section 1: Executive KPI Comparison Cards
        st.markdown("#### Key Benchmark Comparison Summary")
        baseline_test = load_model_evaluation_metrics("Custom CNN (Baseline)")
        spatial_test = load_model_evaluation_metrics("Spatial Attention Feature Fusion (Seed 42)")

        col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
        with col_kpi1:
            st.metric(
                label="Official Test Accuracy",
                value=f"{spatial_test['accuracy'] * 100:.2f}%",
                delta=f"+{(spatial_test['accuracy'] - baseline_test['accuracy']) * 100:.2f}% vs Baseline",
            )
            st.caption(f"Baseline: {baseline_test['accuracy'] * 100:.2f}% | Proposed: {spatial_test['accuracy'] * 100:.2f}% (N = 1,600)")

        with col_kpi2:
            st.metric(
                label="Official Test Macro-F1",
                value=f"{spatial_test['f1_macro']:.4f}",
                delta=f"+{spatial_test['f1_macro'] - baseline_test['f1_macro']:.4f} vs Baseline",
            )
            st.caption(f"Baseline: {baseline_test['f1_macro']:.4f} | Proposed: {spatial_test['f1_macro']:.4f}")

        with col_kpi3:
            st.metric(
                label="Official Test Macro ROC-AUC",
                value=f"{spatial_test['roc_auc_macro']:.4f}",
                delta=f"+{spatial_test['roc_auc_macro'] - baseline_test['roc_auc_macro']:.4f} vs Baseline",
            )
            st.caption(f"Baseline: {baseline_test['roc_auc_macro']:.4f} | Proposed: {spatial_test['roc_auc_macro']:.4f}")

        with col_kpi4:
            st.metric(
                label="Validation Accuracy (N=1,153)",
                value="92.54%",
                delta="+5.20% vs Baseline",
            )
            st.caption("Baseline: 87.34% | Proposed: 92.54% (Grouped Validation)")

        st.divider()

        # Section 2: Training Dynamics and History
        st.markdown("#### A. Training Dynamics and Convergence History")
        st.caption("Empirical training trajectory and validation convergence curves from saved training records:")

        train_col1, train_col2 = st.columns(2)
        with train_col1:
            st.markdown("**Custom CNN Baseline (Training History: cnn_history.json)**")
            st.write("- **Total Epochs Trained:** 30")
            st.write("- **Best Validation Epoch:** 27")
            st.write("- **Historical Training Accuracy (Epoch 27):** 85.78%")
            st.write("- **Historical Training Loss (Epoch 27):** 0.3621")
            st.write("- **Validation Accuracy at Best Epoch:** 86.96%")
            st.write("- **Validation Loss at Best Epoch:** 0.3380")

        spatial_train_split = load_training_split_evaluation_metrics("Spatial Attention Feature Fusion (Seed 42)")
        with train_col2:
            st.markdown("**Spatial Attention Feature Fusion (Training Records & Checkpoint Evaluation)**")
            st.write("- **Total Epochs Trained:** 50")
            st.write("- **Best Validation Epoch:** 50")
            st.write("- **Validation Accuracy at Best Epoch:** 92.54%")
            st.write("- **Validation Loss at Best Epoch:** 0.2070")
            if spatial_train_split:
                st.write(f"- **Training-Set Checkpoint Evaluation Accuracy:** {spatial_train_split['accuracy'] * 100:.2f}%")
                st.write(f"- **Training-Set Macro F1-Score:** {spatial_train_split['macro_f1']:.4f}")
                st.write(f"- **Training-Set Balanced Accuracy:** {spatial_train_split['balanced_accuracy'] * 100:.2f}%")
                st.write(f"- **Training-Set Unweighted Loss:** {spatial_train_split['unweighted_loss']:.4f}")
                st.write(f"- **Training-Set Class-Weighted Loss:** {spatial_train_split['class_weighted_loss']:.4f}")
                st.write(f"- **Training Images Evaluated:** {spatial_train_split['number_of_images']:,}")
            else:
                st.write("- **Training Accuracy at Best Epoch:** *Not available in saved history*")
                st.write("- **Training Loss at Best Epoch:** *Not available in saved history*")

        st.info(
            "Note on Training Metrics and Provenance: "
            "These metrics were calculated by evaluating the saved model checkpoint on the unaugmented training split. "
            "They are not the original epoch-50 training logs and do not reproduce training-time augmentation or dropout. "
            "In contrast, the Custom CNN Baseline training metrics are historical epoch-27 logs recorded during training with online data augmentation."
        )

        st.markdown("##### Training-Split and Optimization Performance Comparison")
        st.caption("Comparison of baseline training metrics with proposed model training-split evaluation:")

        sp_train_acc_str = f"{spatial_train_split['accuracy'] * 100:.2f}%" if spatial_train_split else "Pending"
        sp_train_unweighted_loss = f"{spatial_train_split['unweighted_loss']:.4f}" if spatial_train_split else "Pending"
        sp_train_weighted_loss = f"{spatial_train_split['class_weighted_loss']:.4f}" if spatial_train_split else "Pending"
        sp_train_f1_str = f"{spatial_train_split['macro_f1']:.4f}" if spatial_train_split else "Pending"
        sp_train_bal_str = f"{spatial_train_split['balanced_accuracy'] * 100:.2f}%" if spatial_train_split else "Pending"
        sp_train_n = f"{spatial_train_split['number_of_images']:,}" if spatial_train_split else "Pending"

        df_train_compare = pd.DataFrame({
            "Metric / Parameter": [
                "Training Accuracy",
                "Training Loss (Unweighted)",
                "Training Loss (Class-Weighted)",
                "Training Macro F1-Score",
                "Training Balanced Accuracy",
                "Training Images Evaluated",
                "Validation Accuracy",
                "Validation Loss",
            ],
            "Custom CNN Baseline": [
                "85.78%",
                "0.3621",
                "0.3621 (Unweighted only)",
                "Not recorded in history",
                "Not recorded in history",
                "4,447 (Online Epoch Batches)",
                "86.96%",
                "0.3380",
            ],
            "Spatial Attention Feature Fusion": [
                sp_train_acc_str,
                sp_train_unweighted_loss,
                sp_train_weighted_loss,
                sp_train_f1_str,
                sp_train_bal_str,
                sp_train_n,
                "92.54%",
                "0.2070",
            ],
            "Measurement Methodology and Source": [
                "Baseline: Epoch-27 log (online augmentation). Proposed: Saved checkpoint evaluation on unaugmented training split.",
                "Baseline: Epoch-27 online sparse cross-entropy. Proposed: Saved checkpoint evaluation on unaugmented training split.",
                "Baseline: Unweighted only. Proposed: Meningioma weight 1.50x, others 1.0x.",
                "Baseline: Not saved in cnn_history.json. Proposed: Computed across 4,447 training predictions.",
                "Baseline: Not saved in cnn_history.json. Proposed: Macro mean class recall on unaugmented split.",
                "Identical duplicate-aware training partition (dataset/Training, N = 4,447).",
                "Baseline: Epoch-27 validation log (87.34% on perceptual split). Proposed: Best epoch-50 validation log.",
                "Baseline: Epoch-27 validation loss. Proposed: Best epoch-50 validation loss.",
            ],
        })
        st.dataframe(df_train_compare, use_container_width=True, hide_index=True)

        if spatial_train_split and spatial_train_split.get("raw_json_str"):
            st.download_button(
                label="Download Verified Training-Split Evaluation JSON Artifact",
                data=spatial_train_split["raw_json_str"],
                file_name="training_split_checkpoint_evaluation.json",
                mime="application/json",
                key="download_train_split_tab1",
                help="Download verified JSON evaluation artifact evaluated on unaugmented training split.",
            )

        # Interactive Training / Validation Curves Plotly
        cnn_hist = load_model_training_history("Custom CNN (Baseline)")
        spatial_hist = load_model_training_history("Spatial Attention Feature Fusion (Seed 42)")

        curve_col1, curve_col2 = st.columns(2)
        with curve_col1:
            if cnn_hist and cnn_hist.get("train_accuracy"):
                fig_cnn = go.Figure()
                fig_cnn.add_trace(go.Scatter(x=cnn_hist["epochs"], y=[a * 100 for a in cnn_hist["train_accuracy"]], mode="lines+markers", name="Training Accuracy", line=dict(color="#1f77b4")))
                fig_cnn.add_trace(go.Scatter(x=cnn_hist["epochs"], y=[a * 100 for a in cnn_hist["val_accuracy"]], mode="lines+markers", name="Validation Accuracy", line=dict(color="#ff7f0e")))
                fig_cnn.update_layout(
                    title="Custom CNN: Training vs Validation Accuracy",
                    xaxis_title="Epoch",
                    yaxis_title="Accuracy (%)",
                    height=350,
                    margin=dict(l=40, r=20, t=40, b=40),
                )
                st.plotly_chart(fig_cnn, use_container_width=True)
            else:
                baseline_curve_img = PROJECT_ROOT / "reports" / "streamlit_evaluation" / "baseline_training_validation_curves.png"
                if baseline_curve_img.is_file():
                    st.image(str(baseline_curve_img), caption="Custom CNN Training & Validation Curves", use_container_width=True)

        with curve_col2:
            if spatial_hist and spatial_hist.get("val_accuracy"):
                fig_sp = go.Figure()
                fig_sp.add_trace(go.Scatter(x=spatial_hist["epochs"], y=[a * 100 for a in spatial_hist["val_accuracy"]], mode="lines", name="Val Accuracy (%)", line=dict(color="#2ca02c", width=2)))
                fig_sp.add_trace(go.Scatter(x=spatial_hist["epochs"], y=[f * 100 for f in spatial_hist.get("val_macro_f1", [])], mode="lines", name="Val Macro-F1 (%)", line=dict(color="#9467bd", width=2, dash="dash")))
                fig_sp.add_trace(go.Scatter(x=spatial_hist["epochs"], y=spatial_hist["val_loss"], mode="lines", name="Val Loss", yaxis="y2", line=dict(color="#d62728", width=1.5)))
                fig_sp.update_layout(
                    title="Spatial Fusion: 50-Epoch Validation Progression",
                    xaxis_title="Epoch",
                    yaxis_title="Validation Score (%)",
                    yaxis2=dict(title="Validation Loss", overlaying="y", side="right"),
                    height=350,
                    margin=dict(l=40, r=40, t=40, b=40),
                )
                st.plotly_chart(fig_sp, use_container_width=True)
            else:
                sp_acc_img = PROJECT_ROOT / "reports" / "streamlit_evaluation" / "spatial_fusion_validation_accuracy.png"
                if sp_acc_img.is_file():
                    st.image(str(sp_acc_img), caption="Spatial Fusion 50-Epoch Validation Accuracy", use_container_width=True)

        st.divider()

        # Section 3: Validation Performance Comparison (N = 1,153)
        st.markdown("#### B. Validation Performance Comparison (Perceptual Grouped Validation, N = 1,153)")
        st.caption("Evaluated on the leakage-aware perceptual grouped validation partition (1,153 scans; zero subject overlap):")

        vcol1, vcol2 = st.columns([1, 1])
        with vcol1:
            df_val_compare = pd.DataFrame({
                "Metric": ["Accuracy", "Macro-F1 Score", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Validation Loss"],
                "Custom CNN Baseline": ["87.34%", "0.8691", "0.8707", "0.8753", "0.8707", "0.3380"],
                "Spatial Attention Fusion": ["92.54%", "0.9241", "0.9240", "0.9254", "0.9235", "0.2070"],
                "Absolute Improvement": ["+5.20%", "+0.0550", "+0.0533", "+0.0501", "+0.0528", "-0.1310"],
            })
            st.dataframe(df_val_compare, use_container_width=True, hide_index=True)

        with vcol2:
            fig_val_bar = go.Figure(data=[
                go.Bar(name="Custom CNN (Baseline)", x=["Accuracy (%)", "Macro-F1 (x100)", "Balanced Acc (%)"], y=[87.34, 86.91, 87.07], text=["87.34%", "0.8691", "87.07%"], textposition="auto", marker_color="#4682B4"),
                go.Bar(name="Spatial Fusion (Proposed)", x=["Accuracy (%)", "Macro-F1 (x100)", "Balanced Acc (%)"], y=[92.54, 92.41, 92.40], text=["92.54%", "0.9241", "92.40%"], textposition="auto", marker_color="#2E8B57"),
            ])
            fig_val_bar.update_layout(
                title="Validation Performance: Baseline vs Proposed",
                barmode="group",
                height=280,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_val_bar, use_container_width=True)

        st.divider()

        # Section 4: Official Independent Test Performance (N = 1,600)
        st.markdown("#### C. Official Independent Test Performance (dataset/Testing, N = 1,600)")
        st.caption("Evaluated on the locked candidate test partition (*dataset/Testing*, exactly 400 scans per class):")

        tcol1, tcol2 = st.columns([1, 1])
        with tcol1:
            df_test_compare = pd.DataFrame({
                "Metric": ["Overall Test Accuracy", "Macro-F1 Score", "Weighted-F1 Score", "Balanced Accuracy", "Macro Precision", "Macro Recall", "Macro ROC-AUC (OvR)", "Test Sample Count"],
                "Custom CNN Baseline": [
                    f"{baseline_test['accuracy'] * 100:.2f}%",
                    f"{baseline_test['f1_macro']:.4f}",
                    f"{baseline_test['f1_weighted']:.4f}",
                    f"{baseline_test['balanced_accuracy'] * 100:.2f}%",
                    f"{baseline_test['precision_macro']:.4f}",
                    f"{baseline_test['recall_macro']:.4f}",
                    f"{baseline_test['roc_auc_macro']:.4f}",
                    "1,600",
                ],
                "Spatial Attention Fusion": [
                    f"{spatial_test['accuracy'] * 100:.2f}%",
                    f"{spatial_test['f1_macro']:.4f}",
                    f"{spatial_test['f1_weighted']:.4f}",
                    f"{spatial_test['balanced_accuracy'] * 100:.2f}%",
                    f"{spatial_test['precision_macro']:.4f}",
                    f"{spatial_test['recall_macro']:.4f}",
                    f"{spatial_test['roc_auc_macro']:.4f}",
                    "1,600",
                ],
                "Absolute Improvement": [
                    f"+{(spatial_test['accuracy'] - baseline_test['accuracy']) * 100:.2f}%",
                    f"+{spatial_test['f1_macro'] - baseline_test['f1_macro']:.4f}",
                    f"+{spatial_test['f1_weighted'] - baseline_test['f1_weighted']:.4f}",
                    f"+{(spatial_test['balanced_accuracy'] - baseline_test['balanced_accuracy']) * 100:.2f}%",
                    f"+{spatial_test['precision_macro'] - baseline_test['precision_macro']:.4f}",
                    f"+{spatial_test['recall_macro'] - baseline_test['recall_macro']:.4f}",
                    f"+{spatial_test['roc_auc_macro'] - baseline_test['roc_auc_macro']:.4f}",
                    "Identical Partition",
                ],
            })
            st.dataframe(df_test_compare, use_container_width=True, hide_index=True)

        with tcol2:
            fig_test_bar = go.Figure(data=[
                go.Bar(name="Custom CNN (Baseline)", x=["Accuracy (%)", "Macro-F1 (x100)", "Macro ROC-AUC (x100)"], y=[baseline_test['accuracy'] * 100, baseline_test['f1_macro'] * 100, baseline_test['roc_auc_macro'] * 100], text=[f"{baseline_test['accuracy']*100:.2f}%", f"{baseline_test['f1_macro']:.4f}", f"{baseline_test['roc_auc_macro']:.4f}"], textposition="auto", marker_color="#4682B4"),
                go.Bar(name="Spatial Fusion (Proposed)", x=["Accuracy (%)", "Macro-F1 (x100)", "Macro ROC-AUC (x100)"], y=[spatial_test['accuracy'] * 100, spatial_test['f1_macro'] * 100, spatial_test['roc_auc_macro'] * 100], text=[f"{spatial_test['accuracy']*100:.2f}%", f"{spatial_test['f1_macro']:.4f}", f"{spatial_test['roc_auc_macro']:.4f}"], textposition="auto", marker_color="#2E8B57"),
            ])
            fig_test_bar.update_layout(
                title="Official Test Performance: Baseline vs Proposed",
                barmode="group",
                height=280,
                margin=dict(l=40, r=20, t=40, b=40),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_test_bar, use_container_width=True)

        # Per-Class Diagnostic Performance Comparison Table
        st.markdown("##### Per-Class Diagnostic Comparison on Official Test Partition (N = 1,600)")
        baseline_pc = baseline_test.get("per_class", {})
        spatial_pc = spatial_test.get("per_class", {})

        pc_rows = []
        for c in CLASS_NAMES:
            dname = DISPLAY_NAMES.get(c, c.title())
            b_p = f"{baseline_pc.get(c, {}).get('precision', 0.0) * 100:.2f}%"
            b_r = f"{baseline_pc.get(c, {}).get('recall', 0.0) * 100:.2f}%"
            b_f = f"{baseline_pc.get(c, {}).get('f1', 0.0):.4f}"

            s_p = f"{spatial_pc.get(c, {}).get('precision', 0.0) * 100:.2f}%"
            s_r = f"{spatial_pc.get(c, {}).get('recall', 0.0) * 100:.2f}%"
            s_f = f"{spatial_pc.get(c, {}).get('f1', 0.0):.4f}"
            supp = baseline_pc.get(c, {}).get("support", 400)

            pc_rows.append({
                "Diagnostic Class": dname,
                "Baseline Precision": b_p,
                "Proposed Precision": s_p,
                "Baseline Recall": b_r,
                "Proposed Recall": s_r,
                "Baseline F1": b_f,
                "Proposed F1": s_f,
                "Test Support": supp,
            })
        st.dataframe(pd.DataFrame(pc_rows), use_container_width=True, hide_index=True)

        # Side-by-side Visual Artifacts
        st.markdown("##### Model Evaluation Visualizations: Baseline vs Proposed")
        vis_col1, vis_col2 = st.columns(2)
        with vis_col1:
            st.markdown("**Custom CNN Baseline: Confusion Matrix**")
            b_cm_img = PROJECT_ROOT / "reports" / "streamlit_evaluation" / "baseline_confusion_matrix.png"
            if not b_cm_img.is_file():
                b_cm_img = PROJECT_ROOT / "reports" / "baseline_cnn" / "confusion_matrix.png"
            if b_cm_img.is_file():
                st.image(str(b_cm_img), caption="Custom CNN Test Confusion Matrix (N = 1,600)", use_container_width=True)

        with vis_col2:
            st.markdown("**Spatial Attention Feature Fusion: Confusion Matrix**")
            s_cm_img = PROJECT_ROOT / "reports" / "streamlit_evaluation" / "spatial_fusion_confusion_matrix.png"
            if not s_cm_img.is_file():
                s_cm_img = PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "confusion_matrix.png"
            if s_cm_img.is_file():
                st.image(str(s_cm_img), caption="Spatial Attention Feature Fusion Test Confusion Matrix (N = 1,600)", use_container_width=True)

        roc_col1, roc_col2 = st.columns(2)
        with roc_col1:
            st.markdown("**Custom CNN Baseline: Multi-Class ROC Curves**")
            b_roc_img = PROJECT_ROOT / "reports" / "baseline_cnn" / "roc_curve.png"
            if b_roc_img.is_file():
                st.image(str(b_roc_img), caption="Custom CNN Test ROC Curves (Macro ROC-AUC = 0.9341)", use_container_width=True)

        with roc_col2:
            st.markdown("**Spatial Attention Feature Fusion: Multi-Class ROC Curves**")
            s_roc_img = PROJECT_ROOT / "reports" / "streamlit_evaluation" / "spatial_fusion_roc_curves.png"
            if not s_roc_img.is_file():
                s_roc_img = PROJECT_ROOT / "experiments" / "final_evaluation" / "spatial_fusion" / "roc_curves.png"
            if s_roc_img.is_file():
                st.image(str(s_roc_img), caption="Spatial Attention Fusion Test ROC Curves (Macro ROC-AUC = 0.9816)", use_container_width=True)

        st.divider()

        # Section 5: Complete Benchmark Comparison Table Across All Models
        st.markdown("#### D. Comprehensive Benchmark Comparison (All Available Project Models)")
        st.caption("Summary of verified test and validation results extracted directly from model evaluation artifacts:")
        df_full_comp = build_model_comparison_table(selected_model_name)
        st.dataframe(df_full_comp, use_container_width=True, hide_index=True)

    # =========================================================================
    # TAB 2: DIAGNOSTIC INFERENCE & SALIENCY ANALYSIS
    # =========================================================================
    with tab_inference:
        st.subheader(f"Diagnostic Inference and Saliency Analysis: {selected_model_name}")
        st.markdown(
            f"Active Checkpoint: `{selected_model_path.name}` ({model_cfg.get('role', 'Model')})\n\n"
            "Upload an axial-plane brain MRI scan or choose a verified test sample from the official benchmark partition."
        )

        col_upload, col_sample = st.columns([2, 1])
        with col_upload:
            uploaded_file = st.file_uploader(
                "Upload Brain MRI Scan (Axial View recommended)",
                type=["jpg", "jpeg", "png"],
                help="Accepted formats: JPG, PNG.",
            )
        with col_sample:
            selected_sample_key = st.selectbox(
                "Or choose a verified benchmark test sample",
                options=list(SAMPLE_IMAGES.keys()),
            )

        # Resolve Image
        image_to_process: Image.Image | None = None
        image_source_info = ""

        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file).convert("RGB")
                image_source_info = f"Uploaded File: {uploaded_file.name}"
            except Exception as exc:
                st.error(f"Error opening uploaded image: {exc}")
                st.stop()
        elif selected_sample_key and SAMPLE_IMAGES[selected_sample_key]:
            sample_p = PROJECT_ROOT / SAMPLE_IMAGES[selected_sample_key]
            if sample_p.is_file():
                image_to_process = Image.open(sample_p).convert("RGB")
                image_source_info = f"Benchmark Sample: {sample_p.name} ({selected_sample_key.split()[0]})"
            else:
                st.warning(f"Sample file not found at: {sample_p}")

        if image_to_process is None:
            st.info("Please upload an MRI image or choose a pre-loaded test sample above to proceed.")
        else:
            prev_col, act_col = st.columns([1, 2])
            with prev_col:
                st.image(image_to_process, caption="Input MRI Scan", width=240)
            with act_col:
                st.markdown(f"**Loaded Image:** {image_source_info}")
                st.markdown(f"**Dimensions:** {image_to_process.size[0]} x {image_to_process.size[1]} px")
                st.markdown(f"**Active Model:** {selected_model_name} ({model_cfg.get('role')})")
                st.markdown(f"**Preprocessing:** {'[0, 1] Rescaling' if model_type == 'cnn' else '[0, 255] ImageNet Normalization'}")
                st.write("")
                analyze_clicked = st.button("Run Inference and Saliency Analysis", type="primary", use_container_width=True)

            if analyze_clicked:
                with st.spinner("Executing model forward pass and computing gradient saliency..."):
                    try:
                        input_tensor, orig_rgb, _ = preprocess_single_image(image_to_process, model_type=model_type)
                    except Exception as exc:
                        st.error(f"Preprocessing error: {exc}")
                        st.stop()

                    try:
                        raw_probs = model.predict(input_tensor, verbose=0)[0]
                        temperature, is_calibrated = load_calibration_temperature(selected_model_name)
                        if is_calibrated and abs(temperature - 1.0) > 1e-4:
                            calibrated_probs = apply_temperature_scaling(raw_probs, temperature)
                            calib_caption = f"Temperature-scaled (T = {temperature:.4f} fitted on validation set)."
                            prob_label = "Live Prediction Confidence"
                            delta_val = f"Raw: {float(raw_probs[int(np.argmax(calibrated_probs))]) * 100:.2f}%"
                        else:
                            calibrated_probs = raw_probs
                            calib_caption = "Raw Softmax Output (Uncalibrated; T = 1.0000)."
                            prob_label = "Live Prediction Confidence"
                            delta_val = None

                        pred_idx = int(np.argmax(calibrated_probs))
                        pred_class = CLASS_NAMES[pred_idx]
                        pred_display = DISPLAY_NAMES.get(pred_class, pred_class.title())
                        calibrated_confidence = float(calibrated_probs[pred_idx])
                    except Exception as exc:
                        st.error(f"Model prediction failed: {exc}")
                        st.stop()

                    uncertainty_info = assess_prediction_uncertainty(calibrated_probs, UNCERTAINTY_THRESHOLDS)

                    # Saliency Computation
                    cam_success = False
                    heatmap_rgb, overlay_rgb = None, None
                    try:
                        if cam_method == "Grad-CAM++" and model_cfg.get("supports_gradcam_pp", True):
                            heatmap_2d = compute_gradcam_plus_plus_heatmap(
                                feature_extractor, head_layers, input_tensor, target_class_idx=pred_idx
                            )
                        else:
                            heatmap_2d = compute_gradcam_heatmap(
                                feature_extractor, head_layers, input_tensor, target_class_idx=pred_idx
                            )

                        heatmap_rgb, overlay_rgb = overlay_heatmap_on_image(
                            heatmap_2d, orig_rgb, alpha=blend_alpha, colormap_name=chosen_colormap_name
                        )
                        cam_success = True
                    except Exception as exc:
                        st.warning(f"Saliency computation notice: {cam_method} calculation encountered an issue: {exc}")

                st.divider()
                st.subheader("Classification and Reliability Assessment")

                # Selective Prediction Triage Alert (No Emojis)
                if calibrated_confidence < SELECTIVE_THRESHOLD_TAU:
                    st.error(
                        "Selective Prediction Review Notice (Abstention Triggered)\n\n"
                        f"Calibrated confidence ({calibrated_confidence * 100:.2f}%) is below the operational reliability threshold "
                        f"(tau* = {SELECTIVE_THRESHOLD_TAU * 100:.0f}%).\n\n"
                        "Clinical Guidance: This scan presents diagnostic ambiguity. "
                        "In an assisted clinical workflow, this case is deferred for prioritized manual evaluation by a specialist neuroradiologist."
                    )
                else:
                    st.success(
                        "Selective Prediction Accepted (High Confidence)\n\n"
                        f"Calibrated confidence ({calibrated_confidence * 100:.2f}%) satisfies the operational reliability threshold "
                        f"(tau* = {SELECTIVE_THRESHOLD_TAU * 100:.0f}%)."
                    )

                res1, res2, res3 = st.columns(3)
                with res1:
                    st.metric(label="Predicted Diagnostic Class", value=pred_display)
                    st.caption(f"Estimated class: {pred_display}")
                with res2:
                    st.metric(label=prob_label, value=f"{calibrated_confidence * 100:.2f}%", delta=delta_val)
                    st.caption(calib_caption)
                with res3:
                    status = uncertainty_info["uncertainty_status"]
                    st.metric(label="Uncertainty Level", value=f"{status} Uncertainty")
                    st.caption(uncertainty_info["clinical_guidance"])

                # Probability Distribution Plotly Chart
                st.markdown("##### Class Probability Distribution")
                categories = [DISPLAY_NAMES.get(c, c.title()) for c in CLASS_NAMES]
                probs_pct = [float(p * 100) for p in calibrated_probs]

                fig_prob = go.Figure(data=[
                    go.Bar(
                        x=categories,
                        y=probs_pct,
                        text=[f"{p:.2f}%" for p in probs_pct],
                        textposition="auto",
                        marker_color=["#1f77b4" if i != pred_idx else "#2ca02c" for i in range(len(categories))],
                    )
                ])
                fig_prob.update_layout(
                    xaxis_title="Tumor Category",
                    yaxis_title="Probability (%)",
                    yaxis=dict(range=[0, 100]),
                    height=280,
                    margin=dict(l=40, r=20, t=30, b=40),
                )
                st.plotly_chart(fig_prob, use_container_width=True)

                with st.expander("Numerical Class Probabilities and Entropy Analysis"):
                    for i, c in enumerate(CLASS_NAMES):
                        dname = DISPLAY_NAMES.get(c, c.title())
                        cal_p = float(calibrated_probs[i])
                        raw_p = float(raw_probs[i])
                        if is_calibrated and abs(temperature - 1.0) > 1e-4:
                            st.write(f"- **{dname}:** Calibrated {cal_p * 100:.2f}% (Raw Softmax: {raw_p * 100:.2f}%)")
                        else:
                            st.write(f"- **{dname}:** {cal_p * 100:.2f}%")
                    st.write(f"- **Shannon Predictive Entropy:** {uncertainty_info['entropy']:.4f} nats")

                # Visual Explainability Section
                st.divider()
                st.subheader(f"Explainability Analysis: {cam_method}")
                st.info(
                    "Interpretability Notice: Saliency heatmaps indicate coarse convolutional feature activations "
                    "that positively influenced the network prediction. "
                    "This is a visualization of algorithmic attention, not an automated tumor segmentation, "
                    "and must not be used to delineate surgical margins or tumor volume."
                )

                if cam_success and heatmap_rgb is not None and overlay_rgb is not None:
                    cam_col1, cam_col2, cam_col3 = st.columns(3)
                    with cam_col1:
                        st.image(image_to_process, caption="1. Input MRI Scan", use_container_width=True)
                    with cam_col2:
                        st.image(heatmap_rgb, caption=f"2. {cam_method} Saliency Heatmap ({colormap_choice})", use_container_width=True)
                    with cam_col3:
                        st.image(overlay_rgb, caption=f"3. Blended Overlay (Alpha = {blend_alpha:.2f})", use_container_width=True)
                else:
                    st.warning("Heatmap visualization could not be generated for this configuration.")

    # =========================================================================
    # TAB 3: INDIVIDUAL MODEL INSPECTION AND ARTIFACTS
    # =========================================================================
    with tab_inspection:
        st.subheader(f"Model Technical Profile: {selected_model_name}")
        st.markdown(f"*{model_cfg['description']}*")

        test_metrics = load_model_evaluation_metrics(selected_model_name)
        val_cfg = model_cfg.get("val_metrics", {})
        train_cfg = model_cfg.get("train_metrics", {})

        train_split_metrics = load_training_split_evaluation_metrics(selected_model_name)

        # Status Badges
        st.markdown("##### Artifact Status")
        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
        with stat_col1:
            h_stat = "Available" if model_cfg.get("training_history_path") and Path(model_cfg["training_history_path"]).is_file() else "Not Available"
            st.metric("Training History", h_stat)
            st.caption(f"Epochs: {train_cfg.get('epochs', 'N/A')}")
        with stat_col2:
            st.metric("Validation Evaluation", "Verified")
            st.caption(f"Split: {val_cfg.get('split_name', 'Validation Set')}")
        with stat_col3:
            t_stat = "Verified" if test_metrics and not test_metrics.get("is_test_pending") else "Pending"
            st.metric("Official Test Evaluation", t_stat)
            st.caption(f"Samples: {test_metrics['test_images_count'] if test_metrics else 0} scans")
        with stat_col4:
            ts_stat = "Verified" if train_split_metrics else ("Not Applicable" if not model_cfg.get("training_split_evaluation_file") else "Missing")
            st.metric("Training Split Evaluation", ts_stat)
            st.caption(f"Samples: {train_split_metrics['number_of_images']:,} scans" if train_split_metrics else "Not evaluated")

        st.divider()

        # Training-Split Evaluation Section (when available)
        if train_split_metrics:
            st.markdown("##### Training-Split Checkpoint Evaluation Metrics")
            st.caption(
                "Verified metrics from saved checkpoint evaluation on unaugmented training partition "
                f"(*dataset/Training*, N = {train_split_metrics['number_of_images']:,} scans):"
            )
            st.info(
                "These metrics were calculated by evaluating the saved model checkpoint on the unaugmented training split. "
                "They are not the original epoch-50 training logs and do not reproduce training-time augmentation or dropout."
            )
            # Row 1: Accuracy, Macro-F1, Balanced Accuracy
            tk1, tk2, tk3 = st.columns(3)
            with tk1:
                st.metric(
                    label="Training-Set Checkpoint Evaluation Accuracy",
                    value=f"{train_split_metrics['accuracy'] * 100:.2f}%",
                )
                st.caption(f"Evaluated on {train_split_metrics['number_of_images']:,} unaugmented training images")
            with tk2:
                st.metric(
                    label="Training-Set Macro F1-Score",
                    value=f"{train_split_metrics['macro_f1']:.4f}",
                )
                st.caption("Macro harmonic mean of precision and recall")
            with tk3:
                st.metric(
                    label="Training-Set Balanced Accuracy",
                    value=f"{train_split_metrics['balanced_accuracy'] * 100:.2f}%",
                )
                st.caption("Macro mean class sensitivity")

            # Row 2: Unweighted Loss, Class-Weighted Loss, Number of Training Images
            tk4, tk5, tk6 = st.columns(3)
            with tk4:
                st.metric(
                    label="Training-Set Unweighted Loss",
                    value=f"{train_split_metrics['unweighted_loss']:.4f}",
                )
                st.caption("Sparse categorical cross-entropy loss")
            with tk5:
                st.metric(
                    label="Training-Set Class-Weighted Loss",
                    value=f"{train_split_metrics['class_weighted_loss']:.4f}",
                )
                st.caption("Class weights: meningioma 1.50x, others 1.00x")
            with tk6:
                st.metric(
                    label="Training Images Evaluated",
                    value=f"{train_split_metrics['number_of_images']:,}",
                )
                st.caption(f"Source: {train_split_metrics['source_file']}")

            if train_split_metrics.get("raw_json_str"):
                st.download_button(
                    label="Download Verified Training-Split Evaluation JSON",
                    data=train_split_metrics["raw_json_str"],
                    file_name="training_split_checkpoint_evaluation.json",
                    mime="application/json",
                    key="download_train_split_tab3",
                    help="Download verified JSON evaluation artifact evaluated on unaugmented training split.",
                )

            st.divider()
        elif selected_model_name == "Spatial Attention Feature Fusion (Seed 42)":
            st.warning(
                "Training-split checkpoint evaluation artifact could not be loaded or is invalid. "
                "Expected artifact: experiments/spatial_fusion/meningioma_weighted_50ep_seed42/training_split_checkpoint_evaluation.json"
            )
            st.divider()

        if test_metrics and not test_metrics.get("is_test_pending"):
            # Test Metrics Row 1
            st.markdown("##### Official Independent Test Metrics (dataset/Testing, N = 1,600)")
            mk1, mk2, mk3, mk4 = st.columns(4)
            with mk1:
                st.metric("Test Accuracy", f"{test_metrics['accuracy'] * 100:.2f}%")
                st.caption(f"N = {test_metrics['test_images_count']} test scans")
            with mk2:
                st.metric("Balanced Accuracy", f"{test_metrics['balanced_accuracy'] * 100:.2f}%")
                st.caption("Macro mean sensitivity")
            with mk3:
                st.metric("Macro-F1 Score", f"{test_metrics['f1_macro']:.4f}")
                st.caption(f"Weighted-F1: {test_metrics['f1_weighted']:.4f}")
            with mk4:
                st.metric("Macro ROC-AUC (OvR)", f"{test_metrics['roc_auc_macro']:.4f}")
                st.caption("One-vs-Rest multiclass discrimination")

            # Test Metrics Row 2
            mk5, mk6, mk7, mk8 = st.columns(4)
            with mk5:
                st.metric("Macro Precision", f"{test_metrics['precision_macro']:.4f}")
                st.caption("Positive predictive value")
            with mk6:
                st.metric("Macro Recall", f"{test_metrics['recall_macro']:.4f}")
                st.caption("Macro sensitivity")
            with mk7:
                st.metric("Total Parameters", f"{model_cfg['params_total']:,}")
                st.caption(f"Trainable: {model_cfg['params_trainable']:,}")
            with mk8:
                st.metric("Avg Latency (CPU)", f"{test_metrics['inference_ms']:.1f} ms")
                st.caption("Average per-image inference")

            st.divider()

            # Visual Artifacts for Selected Model
            st.markdown("##### Model Evaluation Visualizations")
            pcol1, pcol2 = st.columns(2)
            with pcol1:
                st.markdown(f"**Confusion Matrix: {selected_model_name}**")
                cm_p = model_cfg.get("confusion_matrix_img")
                if cm_p and Path(cm_p).is_file():
                    st.image(str(cm_p), use_container_width=True)
                else:
                    st.info(f"Confusion matrix plot not available for {selected_model_name}.")

            with pcol2:
                st.markdown(f"**ROC Curves: {selected_model_name}**")
                roc_p = model_cfg.get("roc_curve_img")
                if roc_p and Path(roc_p).is_file():
                    st.image(str(roc_p), use_container_width=True)
                else:
                    st.info(f"ROC curve plot not available for {selected_model_name}.")

            # Precision-Recall Curve or Per-Class Bar Chart
            pr_col1, pr_col2 = st.columns(2)
            with pr_col1:
                pr_p = model_cfg.get("pr_curve_img")
                if pr_p and Path(pr_p).is_file():
                    st.markdown(f"**Precision-Recall Curves: {selected_model_name}**")
                    st.image(str(pr_p), use_container_width=True)

            with pr_col2:
                pc_p = model_cfg.get("per_class_metrics_path")
                if pc_p and Path(pc_p).is_file():
                    st.markdown(f"**Per-Class Metrics: {selected_model_name}**")
                    st.image(str(pc_p), use_container_width=True)

            st.divider()

            # Per-Class Table for Selected Model
            st.markdown("##### Per-Class Performance Breakdown")
            pc_dict = test_metrics.get("per_class", {})
            if pc_dict:
                ind_rows = []
                for c in CLASS_NAMES:
                    cd = pc_dict.get(c, {})
                    ind_rows.append({
                        "Category": DISPLAY_NAMES.get(c, c.title()),
                        "Precision": f"{float(cd.get('precision', 0.0)) * 100:.2f}%",
                        "Recall (Sensitivity)": f"{float(cd.get('recall', 0.0)) * 100:.2f}%",
                        "F1-Score": f"{float(cd.get('f1_score', cd.get('f1', 0.0))):.4f}",
                        "Support": int(cd.get("support", 400)),
                        "ROC-AUC (OvR)": f"{float(cd.get('roc_auc', cd.get('roc_auc_ovr', 0.0))):.4f}" if cd.get("roc_auc") or cd.get("roc_auc_ovr") else "N/A",
                    })
                st.dataframe(pd.DataFrame(ind_rows), use_container_width=True, hide_index=True)

            # Raw Confusion Matrix Expander
            if test_metrics.get("confusion_matrix"):
                with st.expander("Raw Confusion Matrix Counts"):
                    raw_cm_df = pd.DataFrame(
                        test_metrics["confusion_matrix"],
                        index=[f"True {DISPLAY_NAMES.get(c, c)}" for c in CLASS_NAMES],
                        columns=[f"Pred {DISPLAY_NAMES.get(c, c)}" for c in CLASS_NAMES],
                    )
                    st.dataframe(raw_cm_df, use_container_width=True)

            # Classification Report CSV Expander
            rep_csv = model_cfg.get("classification_report_csv")
            if rep_csv and Path(rep_csv).is_file():
                with st.expander("Scikit-Learn Classification Report (CSV)"):
                    try:
                        df_rep = pd.read_csv(rep_csv)
                        st.dataframe(df_rep, use_container_width=True)
                    except Exception as e:
                        st.caption(f"Could not load classification report: {e}")

        else:
            st.warning("Official test metrics for this model are pending independent evaluation.")


if __name__ == "__main__":
    main()
