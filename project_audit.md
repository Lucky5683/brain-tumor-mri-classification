# Project Technical Audit: Brain Tumor MRI Classification Using EfficientNet-B0 with Grad-CAM

**Institution:** The Apollo University, School of Technology, Department of Computer Science & Engineering (AI & DS), Chittoor  
**Audit Date:** October 2026  
**Auditor:** Senior ML Engineer & DL Research Mentor  
**Project Directory:** `X:\Mini Project\brain-tumor-mri-classification`  

---

## 1. Executive Summary

An exhaustive technical audit of the codebase, dataset splits, trained model checkpoints, evaluation pipelines, explainability modules, Streamlit web application, and presentation artifacts was conducted.

The project implements a deep learning classification system for brain magnetic resonance imaging (MRI) scans categorized into four distinct classes:
1. **Glioma**
2. **Meningioma**
3. **Pituitary Tumor**
4. **No Tumor (Healthy Control)**

### Verified Baseline Benchmark Results (Official Test Set $N = 1,600$ images)
- **Custom 4-Block Sequential CNN**: **68.75% Test Accuracy**, **0.6624 Macro F1-Score** (Severe bottleneck in Glioma recall: $41.25\%$).
- **ImageNet Pretrained EfficientNet-B0 (Transfer Learning)**: **87.50% Test Accuracy**, **0.8699 Macro F1-Score**, **0.9718 Macro OvR ROC-AUC** (Glioma recall: $65.75\%$, Meningioma: $85.25\%$, Pituitary: $99.00\%$, No Tumor: $100.00\%$).

Both models are genuine, saved on disk, and their test evaluations match the reported historical records. However, several critical architectural bugs, calibration leakage risks, and interface compatibility issues were discovered and cataloged below.

---

## 2. File and Directory Inventory

| Path / Directory | Role / Description | Status |
| :--- | :--- | :--- |
| `app.py` | Streamlit interactive web application entry point | Needs repair (calibration temperature leakage, Grad-CAM Sequential model crash) |
| `config.py` | Centralized paths, hyperparameters, and class definitions | Functional; requires update for checkpoint-specific calibration mapping |
| `dataset/` | Kaggle Brain Tumor MRI dataset (7,200 images total) | Intact (5,600 Training, 1,600 Testing; 4 classes perfectly balanced) |
| `baseline_cnn.keras` | Saved Keras 3 checkpoint for custom CNN baseline | Functional (Input: $224 \times 224 \times 3$, Output: 4) |
| `efficientnet_b0.keras` | Saved Keras 3 checkpoint for reference EfficientNet-B0 | Functional (Input: $224 \times 224 \times 3$, Output: 4) |
| `models/` | Checkpoints for fine-tuning (`efficientnet_b0_finetuned.keras`, etc.) | Functional |
| `preprocess.py` | Root wrapper for image loading, resizing, and normalization | Functional |
| `gradcam.py` | Root CLI interface for Grad-CAM generation | **Broken** (`TypeError: unexpected keyword 'colormap'`) |
| `evaluate.py` | Root evaluation wrapper calling `scripts/evaluate.py` | Functional |
| `train_cnn.py` | Training pipeline for custom baseline CNN | Functional |
| `train_efficientnet.py` | Root wrapper for two-stage transfer learning training | Functional |
| `src/data/` | Modular dataset loading, stratified splitting, and augmentation | Functional |
| `src/models/` | Model architectures and Keras 3 deserialization compatibility shim | Functional |
| `src/evaluation/` | Metrics calculation, bootstrapping, reliability diagrams, uncertainty | Functional |
| `src/explainability/` | Grad-CAM and Grad-CAM++ implementations | **Broken** for Sequential models in Keras 3 |
| `reports/` | Generated metrics, ROC curves, confusion matrices, and audit logs | Intact |
| `experiments/` | Experiment log and summary CSV | Intact |
| `tests/` | Automated test suite | **Missing** (Prior to audit) |

---

## 3. Confirmed Code & Architecture Issues

### Issue 1: Grad-CAM Submodel Extraction Failure for Sequential CNN (High Severity)
- **Location:** `src/explainability/gradcam.py`, function `extract_gradcam_submodels()`.
- **Symptom:** In Keras 3.14+, deserialized `Sequential` models do not maintain a symbolic `.input` property on the model container. When selecting the `Baseline Custom CNN` in the Streamlit app or running Grad-CAM, the function raised:
  ```text
  AttributeError: The layer baseline_cnn has never been called and thus has no defined input.
  ```
- **Root Cause:** The code attempted `tf.keras.Model(inputs=full_model.input, outputs=target_conv_layer.output)` which only works for Functional API models or eager Sequential models that have traced their inputs symbolically.
- **Fix:** Dynamically reconstruct a functional feature extractor using `tf.keras.Input(shape=(224, 224, 3))` and pipe through layers up to `conv_idx + 1`. This provides 100% compatibility across both Functional (EfficientNet) and Sequential (Custom CNN) models.

### Issue 2: Keyword Argument Mismatch in Root `gradcam.py` CLI (Medium Severity)
- **Location:** `gradcam.py`, line 63, function `overlay_heatmap()`.
- **Symptom:** Running `python gradcam.py --image <path>` crashed with:
  ```text
  TypeError: overlay_heatmap_on_image() got an unexpected keyword argument 'colormap'
  ```
- **Root Cause:** `overlay_heatmap_on_image` in `src/explainability/gradcam.py` accepts `colormap_name: str = "jet"`, but `gradcam.py` was passing `colormap=cm` where `cm` was an OpenCV integer constant.
- **Fix:** Update `gradcam.py` to forward `colormap_name=colormap_name.lower()` directly.

### Issue 3: Indiscriminate Calibration Temperature Leakage (Methodological Severity)
- **Location:** `app.py`, lines 56–65 and 223–224.
- **Symptom:** `app.py` loaded `T = 1.0993` from `reports/calibration_report.json` (fitted exclusively on validation probabilities from `efficientnet_b0.keras`) and applied it to every selected model, including `baseline_cnn.keras`.
- **Methodological Problem:** Temperature scaling is strictly model-dependent. A temperature parameter fitted to an overconfident or underconfident EfficientNet-B0 cannot be applied to a custom CNN without distorting its predictive uncertainty.
- **Fix:** Make calibration model-aware. If `Baseline Custom CNN` is selected, use its own calibrated temperature if available, or maintain raw softmax probabilities ($T = 1.0$) with an explicit UI notice stating that temperature scaling is active only for checkpoints where it was fitted on validation data.

### Issue 4: Missing Automated Unit and Integration Test Suite (Quality Severity)
- **Location:** Project root.
- **Symptom:** No automated regression or validation tests existed to verify file format validation, corrupt image handling, model inference bounds, probability validity, or Grad-CAM tensor constraints.
- **Fix:** Construct a comprehensive `pytest` suite under `tests/test_pipeline.py` verifying:
  1. Valid MRI classification inference.
  2. Unsupported file extension rejection.
  3. Corrupted / truncated image handling.
  4. Non-standard image resolution resizing ($224 \times 224$).
  5. Missing checkpoint error handling.
  6. Output shape and class probability simplex sum ($1.0 \pm 10^{-5}$).
  7. Grad-CAM 2D heatmap normalization ($[0.0, 1.0]$) and overlay dimensions.
  8. Model-specific calibration behavior.

---

## 5. Summary of Recommended Actions

1. Apply the Sequential Grad-CAM extraction fix in `src/explainability/gradcam.py`.
2. Fix the keyword argument in `gradcam.py`.
3. Update `app.py` to decouple calibration temperatures across models and improve UI reliability.
4. Construct and execute the automated test suite in `tests/test_pipeline.py`.
5. Generate verified visual outputs in `evaluation/`, `gradcam_examples/`, `diagrams/`, and `screenshots/`.
6. Populate the 20-slide Apollo University PowerPoint presentation and complete the technical Word report.
