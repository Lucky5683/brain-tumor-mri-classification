# Phase 4 — Explainability Validation Report

> [!IMPORTANT]
> **Clinical & Technical Scope Notice**: Grad-CAM and Grad-CAM++ produce visual saliency heatmaps highlighting spatial image regions that influenced the model's classification score. They **do NOT** delineate biological tumor margins, tissue infiltration, or surgical resection boundaries.

## 1. Experimental Setup
- **Model**: `efficientnet_b0.keras` (Untouched V1 reference baseline)
- **Target Layer**: `top_activation (EfficientNet-B0 backbone)`
- **Deterministic Seed**: `42`
- **Representative Samples**: `8` images (2 per class across all 4 classes)

## 2. Representative Sample Evaluation & Heatmap Statistics

| True Class | Image Filename | Predicted Class | Confidence | Correct? | Grad-CAM Mean (Area > 0.5) | Grad-CAM++ Mean (Area > 0.5) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Glioma** | `Te-gl_130.jpg` | Glioma | 70.88% | ✅ Yes | 0.273 (22.45%) | 0.020 (2.04%) |
| **Glioma** | `Te-gl_378.jpg` | Glioma | 100.00% | ✅ Yes | 0.320 (16.33%) | 0.302 (28.57%) |
| **Meningioma** | `Te-me_162.jpg` | Meningioma | 99.80% | ✅ Yes | 0.472 (46.94%) | 0.237 (10.2%) |
| **Meningioma** | `Te-me_164.jpg` | Meningioma | 98.89% | ✅ Yes | 0.173 (10.2%) | 0.130 (10.2%) |
| **No Tumor** | `Te-no_13.jpg` | No Tumor | 99.26% | ✅ Yes | 0.206 (14.29%) | 0.251 (22.45%) |
| **No Tumor** | `Te-no_35.jpg` | No Tumor | 99.99% | ✅ Yes | 0.367 (34.69%) | 0.365 (30.61%) |
| **Pituitary** | `Te-pi_132.jpg` | Pituitary | 99.95% | ✅ Yes | 0.253 (18.37%) | 0.025 (2.04%) |
| **Pituitary** | `Te-pi_289.jpg` | Pituitary | 93.66% | ✅ Yes | 0.298 (30.61%) | 0.349 (34.69%) |

## 3. Comparison Observations: Grad-CAM vs Grad-CAM++
1. **Resolution & Localization**: Grad-CAM computes first-order gradients pooled across feature maps, highlighting broad macroscopic areas of activation.
2. **Higher-Order Gradients**: Grad-CAM++ incorporates second- and third-order gradients with pixel-wise weighting, often resulting in tighter focus and distinct sub-region attribution.
3. **No-Tumor Patterns**: In scans without tumors (`notumor`), heatmaps capture normal anatomical symmetry, ventricles, and cortical parenchyma rather than focal focal mass density.
4. **Tumor Cases**: In `glioma`, `meningioma`, and `pituitary`, activations concentrate prominently around intracranial mass lesions, sella turcica (pituitary), or dural attachments (meningioma).

## 4. Verification Checklist
- [x] Representative samples from all 4 classes selected deterministically.
- [x] 2D heatmaps finite, non-negative, and normalized to `[0.0, 1.0]`.
- [x] Output overlays generated in 3-channel RGB uint8 matching original image dimensions.
- [x] Multi-panel comparative plots saved in `reports/explainability/`.
- [x] Explicit clinical notice included (feature attribution != segmentation).