"""Perturbation robustness testing for Brain Tumor MRI classification models."""

from __future__ import annotations
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

from config import CLASS_NAMES, REPORTS_DIR


def apply_brightness_perturbation(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    """Modulate brightness by additive offset."""
    delta = int(factor * 255)
    return np.clip(image_rgb.astype(np.int16) + delta, 0, 255).astype(np.uint8)


def apply_contrast_perturbation(image_rgb: np.ndarray, factor: float) -> np.ndarray:
    """Modulate contrast by linear scaling around midpoint 128."""
    scaled = (image_rgb.astype(np.float32) - 128.0) * factor + 128.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def apply_gaussian_noise(image_rgb: np.ndarray, sigma: float = 15.0) -> np.ndarray:
    """Add zero-mean Gaussian noise."""
    noise = np.random.normal(0, sigma, image_rgb.shape)
    return np.clip(image_rgb.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_jpeg_compression(image_rgb: np.ndarray, quality: int = 50) -> np.ndarray:
    """Simulate lossy JPEG compression artifacts."""
    pil_img = Image.fromarray(image_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    reloaded = Image.open(buffer).convert("RGB")
    return np.array(reloaded, dtype=np.uint8)


def apply_resizing_perturbation(image_rgb: np.ndarray, scale: float = 0.85) -> np.ndarray:
    """Simulate resolution degradation via downscaling and re-upscaling."""
    h, w = image_rgb.shape[:2]
    inter_w = max(16, int(w * scale))
    inter_h = max(16, int(h * scale))
    pil_img = Image.fromarray(image_rgb)
    downscaled = pil_img.resize((inter_w, inter_h), Image.Resampling.BILINEAR)
    restored = downscaled.resize((w, h), Image.Resampling.BILINEAR)
    return np.array(restored, dtype=np.uint8)


PERTURBATIONS = {
    "brightness_+15%": lambda img: apply_brightness_perturbation(img, 0.15),
    "brightness_-15%": lambda img: apply_brightness_perturbation(img, -0.15),
    "contrast_+20%": lambda img: apply_contrast_perturbation(img, 1.20),
    "contrast_-20%": lambda img: apply_contrast_perturbation(img, 0.80),
    "gaussian_noise_sigma15": lambda img: apply_gaussian_noise(img, sigma=15.0),
    "jpeg_quality_50": lambda img: apply_jpeg_compression(img, quality=50),
    "jpeg_quality_75": lambda img: apply_jpeg_compression(img, quality=75),
    "resize_scale_85%": lambda img: apply_resizing_perturbation(img, scale=0.85),
}


def run_robustness_test(
    model: tf.keras.Model,
    image_paths: list[Path],
    true_labels: list[int],
    sample_limit: int = 200,
    model_type: str = "efficientnet",
    output_dir: Path = REPORTS_DIR,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate prediction invariance under perturbations."""
    from src.data.preprocessing import preprocess_single_image

    records = []
    sampled_indices = np.linspace(0, len(image_paths) - 1, min(len(image_paths), sample_limit), dtype=int)

    for idx in sampled_indices:
        img_path = image_paths[idx]
        y_true = true_labels[idx]
        true_name = CLASS_NAMES[y_true]

        orig_tensor, orig_rgb, _ = preprocess_single_image(img_path, model_type=model_type)
        orig_probs = model.predict(orig_tensor, verbose=0)[0]
        orig_pred_idx = int(np.argmax(orig_probs))
        orig_conf = float(orig_probs[orig_pred_idx])

        for p_name, p_fn in PERTURBATIONS.items():
            try:
                perturbed_rgb = p_fn(orig_rgb)
                pert_tensor, _, _ = preprocess_single_image(Image.fromarray(perturbed_rgb), model_type=model_type)
                pert_probs = model.predict(pert_tensor, verbose=0)[0]
                pert_pred_idx = int(np.argmax(pert_probs))
                pert_conf = float(pert_probs[pert_pred_idx])
                changed = (orig_pred_idx != pert_pred_idx)

                records.append({
                    "image": img_path.name,
                    "true_class": true_name,
                    "perturbation": p_name,
                    "orig_pred": CLASS_NAMES[orig_pred_idx],
                    "pert_pred": CLASS_NAMES[pert_pred_idx],
                    "orig_prob": round(orig_conf, 4),
                    "pert_prob": round(pert_conf, 4),
                    "prob_shift": round(float(abs(orig_conf - pert_conf)), 4),
                    "prediction_changed": "YES" if changed else "NO",
                })
            except Exception:
                continue

    df = pd.DataFrame(records)
    csv_path = output_dir / "robustness_results.csv"
    df.to_csv(csv_path, index=False)

    summary = {}
    for p_name in PERTURBATIONS.keys():
        subset = df[df["perturbation"] == p_name]
        if len(subset) > 0:
            changed_rate = float((subset["prediction_changed"] == "YES").mean())
            mean_shift = float(subset["prob_shift"].mean())
            summary[p_name] = {
                "n_samples": len(subset),
                "prediction_flip_rate": round(changed_rate, 4),
                "stability_rate": round(1.0 - changed_rate, 4),
                "mean_probability_shift": round(mean_shift, 4),
            }

    report = {
        "total_images_tested": len(sampled_indices),
        "total_perturbation_trials": len(df),
        "perturbation_summary": summary,
        "clinical_note": (
            "Robustness metrics measure numerical stability under synthetic noise and compression. "
            "They do not validate clinical scanner invariance or diagnostic equivalence."
        ),
    }

    json_path = output_dir / "robustness_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report, df
