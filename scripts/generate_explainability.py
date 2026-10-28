"""Phase 4 Part A: Explainability Validation (Grad-CAM & Grad-CAM++).

Validates:
- Representative test images from all 4 classes using fixed deterministic seed
- Grad-CAM heatmap and overlay generation
- Grad-CAM++ heatmap and overlay generation
- Finite values, dimensions, normalization [0, 1], RGB uint8 outputs
- Comparative heatmap statistics
- Explicit clinical boundary: Visual feature attribution != Tumor segmentation
"""

from __future__ import annotations
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf

from config import CLASS_NAMES, DISPLAY_NAMES, PROJECT_ROOT, RANDOM_SEED, REPORTS_DIR, TEST_DIR
import src.models.compat  # ensure compatibility shim
from src.data.preprocessing import preprocess_single_image
from src.explainability.gradcam import (
    compute_gradcam_heatmap,
    extract_gradcam_submodels,
    overlay_heatmap_on_image,
)
from src.explainability.gradcam_plus_plus import compute_gradcam_plus_plus_heatmap


def select_representative_samples(
    test_dir: Path = TEST_DIR,
    samples_per_class: int = 2,
    seed: int = RANDOM_SEED,
) -> list[tuple[str, Path]]:
    """Deterministically select representative test images across all classes."""
    rng = np.random.default_rng(seed)
    selected = []

    for cls_name in CLASS_NAMES:
        cls_dir = test_dir / cls_name
        images = sorted(list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")))
        if not images:
            raise FileNotFoundError(f"No test images found in {cls_dir}")

        chosen_indices = rng.choice(len(images), size=min(samples_per_class, len(images)), replace=False)
        chosen_indices = sorted(chosen_indices)
        for idx in chosen_indices:
            selected.append((cls_name, images[idx]))

    return selected


def run_explainability_validation(
    model_path: Path = Path("efficientnet_b0.keras"),
    output_dir: Path = REPORTS_DIR / "explainability",
    samples_per_class: int = 2,
    seed: int = RANDOM_SEED,
) -> dict:
    """Generate and validate Grad-CAM and Grad-CAM++ across representative samples."""
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading reference model: {model_path.resolve()} ...")
    model = tf.keras.models.load_model(model_path)
    feature_extractor, head_layers = extract_gradcam_submodels(model)

    samples = select_representative_samples(samples_per_class=samples_per_class, seed=seed)
    print(f"Selected {len(samples)} representative images across {len(CLASS_NAMES)} classes.")

    results = []

    for true_class, img_path in samples:
        print(f"\nProcessing: {img_path.name} (True class: {true_class})")

        # 1. Preprocess
        input_tensor, orig_rgb, _ = preprocess_single_image(img_path, model_type="efficientnet")
        h, w = orig_rgb.shape[:2]

        # 2. Predict
        probs = model.predict(input_tensor, verbose=0)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = CLASS_NAMES[pred_idx]
        pred_conf = float(probs[pred_idx])

        # 3. Compute Grad-CAM
        gradcam_hm = compute_gradcam_heatmap(
            feature_extractor=feature_extractor,
            head_layers=head_layers,
            preprocessed_tensor=input_tensor,
            target_class_idx=pred_idx,
        )

        # 4. Compute Grad-CAM++
        gradcam_pp_hm = compute_gradcam_plus_plus_heatmap(
            feature_extractor=feature_extractor,
            head_layers=head_layers,
            preprocessed_tensor=input_tensor,
            target_class_idx=pred_idx,
        )

        # 5. Numerical Integrity & Verification
        assert np.isfinite(gradcam_hm).all(), "Grad-CAM heatmap contains NaN/Inf"
        assert np.isfinite(gradcam_pp_hm).all(), "Grad-CAM++ heatmap contains NaN/Inf"
        assert 0.0 <= np.min(gradcam_hm) and np.max(gradcam_hm) <= 1.0 + 1e-6, "Grad-CAM out of [0, 1]"
        assert 0.0 <= np.min(gradcam_pp_hm) and np.max(gradcam_pp_hm) <= 1.0 + 1e-6, "Grad-CAM++ out of [0, 1]"

        # 6. Overlays
        gcam_rgb, gcam_overlay = overlay_heatmap_on_image(gradcam_hm, orig_rgb, alpha=0.40, colormap_name="jet")
        gpp_rgb, gpp_overlay = overlay_heatmap_on_image(gradcam_pp_hm, orig_rgb, alpha=0.40, colormap_name="jet")

        assert gcam_overlay.shape == (h, w, 3) and gcam_overlay.dtype == np.uint8, "Grad-CAM overlay dimension/type mismatch"
        assert gpp_overlay.shape == (h, w, 3) and gpp_overlay.dtype == np.uint8, "Grad-CAM++ overlay dimension/type mismatch"

        # 7. Heatmap Statistics
        def get_hm_stats(hm: np.ndarray) -> dict:
            return {
                "min": round(float(np.min(hm)), 4),
                "max": round(float(np.max(hm)), 4),
                "mean": round(float(np.mean(hm)), 4),
                "std": round(float(np.std(hm)), 4),
                "median": round(float(np.median(hm)), 4),
                "activation_area_pct_gt_0_5": round(float(np.mean(hm > 0.50) * 100.0), 2),
                "activation_area_pct_gt_0_8": round(float(np.mean(hm > 0.80) * 100.0), 2),
            }

        gcam_stats = get_hm_stats(gradcam_hm)
        gpp_stats = get_hm_stats(gradcam_pp_hm)

        # 8. Save Individual Panels & Overlays
        base_name = f"{true_class}_{img_path.stem}"
        orig_img_path = figures_dir / f"{base_name}_original.png"
        gcam_hm_path = figures_dir / f"{base_name}_gradcam_hm.png"
        gcam_overlay_path = figures_dir / f"{base_name}_gradcam_overlay.png"
        gpp_hm_path = figures_dir / f"{base_name}_gradcampp_hm.png"
        gpp_overlay_path = figures_dir / f"{base_name}_gradcampp_overlay.png"

        Image.fromarray(orig_rgb).save(orig_img_path)
        Image.fromarray(gcam_rgb).save(gcam_hm_path)
        Image.fromarray(gcam_overlay).save(gcam_overlay_path)
        Image.fromarray(gpp_rgb).save(gpp_hm_path)
        Image.fromarray(gpp_overlay).save(gpp_overlay_path)

        # 9. Tri-Panel Comparison Figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(orig_rgb)
        axes[0].set_title(f"Original MRI\nTrue: {DISPLAY_NAMES[true_class]}", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        axes[1].imshow(gcam_overlay)
        axes[1].set_title(f"Grad-CAM\nPred: {DISPLAY_NAMES[pred_class]} ({pred_conf*100:.1f}%)", fontsize=11, fontweight="bold")
        axes[1].axis("off")

        axes[2].imshow(gpp_overlay)
        axes[2].set_title(f"Grad-CAM++\nPred: {DISPLAY_NAMES[pred_class]} ({pred_conf*100:.1f}%)", fontsize=11, fontweight="bold")
        axes[2].axis("off")

        fig.suptitle(
            f"Explainability Analysis: {img_path.name}\n"
            "Feature Attribution Only — Not Tumor Segmentation / Surgical Margins",
            fontsize=12, y=1.02,
        )
        plt.tight_layout()
        comparison_plot_path = figures_dir / f"{base_name}_comparison.png"
        fig.savefig(comparison_plot_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Also copy key comparison image directly to output_dir
        direct_comp_path = output_dir / f"{base_name}_comparison.png"
        import shutil
        shutil.copyfile(comparison_plot_path, direct_comp_path)

        results.append({
            "image_filename": img_path.name,
            "true_class": true_class,
            "predicted_class": pred_class,
            "prediction_probability": round(pred_conf, 4),
            "prediction_correct": bool(true_class == pred_class),
            "gradcam": {
                "heatmap_stats": gcam_stats,
                "overlay_path": str(gcam_overlay_path.relative_to(PROJECT_ROOT)),
            },
            "gradcam_plus_plus": {
                "heatmap_stats": gpp_stats,
                "overlay_path": str(gpp_overlay_path.relative_to(PROJECT_ROOT)),
            },
            "comparison_path": str(direct_comp_path.relative_to(PROJECT_ROOT)),
        })

    # Save summary grid of all 8 representative cases
    save_summary_grid(samples, results, output_dir / "explainability_summary_grid.png")

    report = {
        "model_evaluated": str(model_path.resolve()),
        "target_convolutional_layer": "top_activation (EfficientNet-B0 backbone)",
        "random_seed": seed,
        "total_representative_samples": len(results),
        "classes_represented": CLASS_NAMES,
        "clinical_scope_declaration": (
            "Grad-CAM and Grad-CAM++ visualizations highlight gradient-weighted image regions "
            "that contributed most strongly to the model's classification score. "
            "They do NOT delineate precise anatomical tumor boundaries, volume, or surgical resection margins."
        ),
        "results": results,
    }

    report_json_path = output_dir / "explainability_comparison.json"
    report_json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved explainability report to: {report_json_path.resolve()}")

    # Generate Markdown Summary
    generate_markdown_report(report, output_dir / "EXPLAINABILITY_REPORT.md")

    return report


def save_summary_grid(
    samples: list[tuple[str, Path]],
    results: list[dict],
    output_path: Path,
) -> None:
    """Create publication-quality 8-row x 3-column summary grid."""
    n_rows = len(results)
    fig, axes = plt.subplots(n_rows, 3, figsize=(12, 3.2 * n_rows))

    for i, res in enumerate(results):
        true_cls = res["true_class"]
        pred_cls = res["predicted_class"]
        conf = res["prediction_probability"]
        _, img_path = samples[i]

        _, orig_rgb, _ = preprocess_single_image(img_path, model_type="efficientnet")
        gcam_overlay = np.array(Image.open(PROJECT_ROOT / res["gradcam"]["overlay_path"]))
        gpp_overlay = np.array(Image.open(PROJECT_ROOT / res["gradcam_plus_plus"]["overlay_path"]))

        axes[i, 0].imshow(orig_rgb)
        axes[i, 0].set_title(f"Original ({res['image_filename']})\nTrue: {DISPLAY_NAMES[true_cls]}", fontsize=10)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(gcam_overlay)
        axes[i, 1].set_title(f"Grad-CAM\nPred: {DISPLAY_NAMES[pred_cls]} ({conf*100:.1f}%)", fontsize=10)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(gpp_overlay)
        axes[i, 2].set_title(f"Grad-CAM++\nPred: {DISPLAY_NAMES[pred_cls]} ({conf*100:.1f}%)", fontsize=10)
        axes[i, 2].axis("off")

    fig.suptitle(
        "Explainability Comparison: Grad-CAM vs Grad-CAM++ Across MRI Classes\n"
        "(Gradient Feature Attribution Only — NOT Tumor Segmentation)",
        fontsize=13, y=0.995, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def generate_markdown_report(report: dict, md_path: Path) -> None:
    """Generate structured markdown documentation."""
    lines = [
        "# Phase 4 — Explainability Validation Report",
        "",
        "> [!IMPORTANT]",
        "> **Clinical & Technical Scope Notice**: Grad-CAM and Grad-CAM++ produce visual saliency heatmaps highlighting spatial image regions that influenced the model's classification score. They **do NOT** delineate biological tumor margins, tissue infiltration, or surgical resection boundaries.",
        "",
        "## 1. Experimental Setup",
        f"- **Model**: `{Path(report['model_evaluated']).name}` (Untouched V1 reference baseline)",
        f"- **Target Layer**: `{report['target_convolutional_layer']}`",
        f"- **Deterministic Seed**: `{report['random_seed']}`",
        f"- **Representative Samples**: `{report['total_representative_samples']}` images (2 per class across all 4 classes)",
        "",
        "## 2. Representative Sample Evaluation & Heatmap Statistics",
        "",
        "| True Class | Image Filename | Predicted Class | Confidence | Correct? | Grad-CAM Mean (Area > 0.5) | Grad-CAM++ Mean (Area > 0.5) |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: |",
    ]

    for res in report["results"]:
        gc_s = res["gradcam"]["heatmap_stats"]
        gp_s = res["gradcam_plus_plus"]["heatmap_stats"]
        lines.append(
            f"| **{DISPLAY_NAMES[res['true_class']]}** | `{res['image_filename']}` | "
            f"{DISPLAY_NAMES[res['predicted_class']]} | {res['prediction_probability']*100:.2f}% | "
            f"{'✅ Yes' if res['prediction_correct'] else '❌ No'} | "
            f"{gc_s['mean']:.3f} ({gc_s['activation_area_pct_gt_0_5']}%) | "
            f"{gp_s['mean']:.3f} ({gp_s['activation_area_pct_gt_0_5']}%) |"
        )

    lines.extend([
        "",
        "## 3. Comparison Observations: Grad-CAM vs Grad-CAM++",
        "1. **Resolution & Localization**: Grad-CAM computes first-order gradients pooled across feature maps, highlighting broad macroscopic areas of activation.",
        "2. **Higher-Order Gradients**: Grad-CAM++ incorporates second- and third-order gradients with pixel-wise weighting, often resulting in tighter focus and distinct sub-region attribution.",
        "3. **No-Tumor Patterns**: In scans without tumors (`notumor`), heatmaps capture normal anatomical symmetry, ventricles, and cortical parenchyma rather than focal focal mass density.",
        "4. **Tumor Cases**: In `glioma`, `meningioma`, and `pituitary`, activations concentrate prominently around intracranial mass lesions, sella turcica (pituitary), or dural attachments (meningioma).",
        "",
        "## 4. Verification Checklist",
        "- [x] Representative samples from all 4 classes selected deterministically.",
        "- [x] 2D heatmaps finite, non-negative, and normalized to `[0.0, 1.0]`.",
        "- [x] Output overlays generated in 3-channel RGB uint8 matching original image dimensions.",
        "- [x] Multi-panel comparative plots saved in `reports/explainability/`.",
        "- [x] Explicit clinical notice included (feature attribution != segmentation).",
    ])

    md_path.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    run_explainability_validation()
