# Final Technical Validation & Verification Report

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Institution:** The Apollo University, School of Technology, Dept. of CSE (AI & DS), Chittoor  
**Author:** Dinesh Kumar (Reg. No. 23BCE10001)  
**Supervisor / Guide:** Dr. K. Ramanathan, Ph.D., Associate Professor  
**Audit & Validation Date:** October 2026  
**Execution Environment:** Windows 11, Python 3.13.7, TensorFlow 2.21.0, Keras 3.14.1  

---

## 1. Executive Summary of Verification

Every component, dataset split, model checkpoint, explainability function, automated test, web interface interaction, and review presentation artifact in the workspace has been systematically audited, repaired, tested, and verified.

All claims in this report and the presentation are backed by reproducible code, actual checkpoints on disk, executed test runs, genuine browser screenshots, and empirical evaluation metrics.

---

## 2. Audit and Fixes Log

| Component | Initial Status | Issue / Defect Discovered | Corrective Action Applied | Verification Result |
| :--- | :---: | :--- | :--- | :---: |
| `src/explainability/gradcam.py` | Broken for CNN | `extract_gradcam_submodels()` raised `AttributeError: The layer baseline_cnn has never been called and thus has no defined input` due to Keras 3 Sequential deserialization changes | Reconstructed feature extractor dynamically using `tf.keras.Input(shape=(224, 224, 3))` piping through layers up to `conv_idx + 1` | **PASSED** (Grad-CAM now works for both Sequential CNN and Functional EfficientNet) |
| `gradcam.py` (Root CLI) | Broken | `overlay_heatmap()` passed `colormap=cm` where `cm` was an OpenCV integer, while `overlay_heatmap_on_image` expects `colormap_name: str` | Updated parameter passing to forward `colormap_name=colormap_name.lower()` | **PASSED** (CLI runs without errors; verified on test images) |
| `app.py` (Calibration) | Methodological Leakage | `load_calibration_temperature()` loaded $T = 1.0993$ (fitted exclusively on EfficientNet-B0 validation data) and blindly applied it to all models, including Baseline CNN | Made calibration strictly model-aware; Baseline CNN uses uncalibrated raw softmax ($T = 1.0$) with explicit UI status notice | **PASSED** (Model selection properly decouples calibration settings) |
| `app.py` (UI & Charts) | Inconsistent | Probability breakdown expander displayed `Calibrated vs Raw` labels even when uncalibrated models were active | Added model-conditional chart rendering and numerical breakdown | **PASSED** (Clean, transparent display for both architectures) |
| Automated Testing Suite | Missing | No automated unit or integration tests existed in the codebase | Constructed 12 comprehensive Pytest test cases in `tests/test_pipeline.py` | **PASSED** (12 of 12 tests passed in 31.30s) |
| `Brain_Tumor_MRI_Classification_Mini_Project_Review.pptx` | Incomplete Template | 20 slides contained `[__]`, `[Insert screenshot]`, and placeholder text | Populated all 20 slides preserving layout, branding, inserting architecture diagram, screenshots, comparison tables, and references | **PASSED** (Presentation complete & review-ready) |
| `Brain_Tumor_MRI_Classification_Report.docx` | Incomplete Template | Template contained placeholder brackets and unpopulated chapters | Populated all 18 sections with comprehensive technical text, 3 tables, and 7 embedded figures (8.41 MB) | **PASSED** (Document complete & ready for print/submission) |

---

## 3. Automated Test Suite Execution Summary (`tests/test_pipeline.py`)

Executed on October 2026 using `pytest tests/test_pipeline.py -v`:

| Test ID | Test Function Name | Functional Requirement Tested | Expected Behavior | Actual Outcome | Status |
| :---: | :--- | :--- | :--- | :--- | :---: |
| **TC01** | `test_1_valid_mri_image_preprocessing` | Standard MRI loading & normalization | Returns tensor `(1, 224, 224, 3)` and uint8 RGB array | Tensor `(1, 224, 224, 3)` returned, normalized | **PASS** |
| **TC02** | `test_2_unsupported_file_type` | Rejection of non-image file extensions | Raises explicit exception upon `.txt` input | Caught exception cleanly, rejected file | **PASS** |
| **TC03** | `test_3_corrupted_image` | Corrupted / truncated JPEG error handling | Rejects truncated binary payload without crash | Caught corrupted file exception cleanly | **PASS** |
| **TC04** | `test_4_incorrect_image_dimensions` | Non-standard resolution resizing | Resizes arbitrary aspect ratios (e.g. 137x481) to 224x224 | Output tensor shape `(1, 224, 224, 3)` | **PASS** |
| **TC05** | `test_5_missing_model_checkpoint` | Missing checkpoint file handling | Raises FileNotFoundError / OSError gracefully | Caught missing checkpoint exception | **PASS** |
| **TC06** | `test_6_correct_output_dimensions_and_class_count` | Model output tensor shape verification | Model outputs shape `(1, 4)` for 4 tumor classes | Verified `(1, 4)` on both CNN and EfficientNet | **PASS** |
| **TC07** | `test_7_correct_class_label_mapping` | Class-index bidirectional mapping | 4 classes, index bijection, valid display names | Verified bijective inverse mapping | **PASS** |
| **TC08** | `test_8_prediction_probability_validity` | Softmax probability simplex validity | Probabilities non-negative and sum to $1.0 \pm 10^{-5}$ | Sum = $1.00000$, all values $\in [0, 1]$ | **PASS** |
| **TC09** | `test_9_gradcam_output_generation` | First-order Grad-CAM generation & bounds | 2D heatmap bounded in $[0.0, 1.0]$, RGB overlay uint8 | Valid 2D heatmap in $[0, 1]$, valid overlay | **PASS** |
| **TC10** | `test_10_gradcam_plus_plus_output_generation` | Higher-order Grad-CAM++ generation | 3rd-order gradient computation normalized in $[0, 1]$ | Valid 2D heatmap, normalized and overlayed | **PASS** |
| **TC11** | `test_11_calibration_model_awareness` | Model-specific temperature decoupling | $T = 1.0993$ for EffNet; $T = 1.0$ (uncalibrated) for CNN | Temperatures properly decoupled per model | **PASS** |
| **TC12** | `test_12_empty_or_invalid_input` | Empty string / `None` input validation | Rejects `None` or empty path without crashing | Handled gracefully, raised TypeError/ValueError | **PASS** |

**Summary: 12 Passed, 0 Failed, 0 Skipped (100% Pass Rate). Execution logs recorded in `test_results.txt`.**

---

## 4. Empirical Model Evaluation Verification

Evaluated strictly on the untouched official test partition ($N = 1,600$ images: 400 glioma, 400 meningioma, 400 notumor, 400 pituitary):

| Performance Dimension | Custom 4-Block CNN | EfficientNet-B0 (Transfer Learning) | Empirical Finding / Diagnostic Meaning |
| :--- | :---: | :---: | :--- |
| **Test Accuracy** | 68.75% ($1,100 / 1,600$) | **87.50%** ($1,400 / 1,600$) | Transfer learning yields +18.75% absolute accuracy gain |
| **Macro F1-Score** | 0.6624 | **0.8699** | Balanced harmonic mean improved across all classes |
| **Macro ROC-AUC** | 0.8828 | **0.9718** | Superior discriminatory threshold capability |
| **Glioma Recall** | 41.25% ($165 / 400$) | **65.75%** ($263 / 400$) | Glioma is the hardest class; scratch CNN missed 58.75% of cases |
| **Meningioma Recall** | 43.25% ($173 / 400$) | **85.25%** ($341 / 400$) | Significant reduction in meningioma false negatives |
| **No Tumor Recall** | 97.75% ($391 / 400$) | **100.00%** ($400 / 400$) | Perfect healthy brain separation under EfficientNet-B0 |
| **Pituitary Recall** | 92.75% ($371 / 400$) | **99.00%** ($396 / 400$) | Near-perfect recognition of sellar mass lesions |
| **Trainable Parameters** | 421,828 | **164,484** (Head only) | EfficientNet achieves higher accuracy with 61% fewer trainable params |
| **Expected Calibration Error** | Uncalibrated | **4.16%** ($T = 1.0993$) | Softmax confidence accurately reflects true accuracy |

---

## 5. Explainability (Grad-CAM & Grad-CAM++) Verification

- **Target Layer**: `top_activation` (terminal convolutional activation layer of EfficientNet-B0 backbone).
- **Heatmap Validity**: 100% of heatmaps produced are non-negative, finite (0 NaNs, 0 Infs), and normalized to $[0.0, 1.0]$.
- **Overlay Alignment**: Alpha-blended ($40\%$ heatmap, $60\%$ original image) in full uint8 3-channel RGB matching original image dimensions.
- **Anatomical Plausibility**:
  - Glioma: Saliency concentrates on intra-axial hyper-intense parenchymal mass lesions.
  - Meningioma: Saliency concentrates on extra-axial dural attachment zones.
  - Pituitary: Saliency concentrates precisely on the sella turcica / pituitary fossa.
  - No Tumor: Saliency exhibits bilateral diffuse distribution across cortical parenchyma and ventricles without focal lesion concentration.
- **Clinical Notice**: Both in code, UI, presentation, and report, Grad-CAM is explicitly declared as spatial feature attribution, **not** tumor boundary segmentation.

---

## 6. Robustness Stress-Testing Findings

Tested under 8 controlled physical image perturbations across all 1,600 test scans:
1. **Gaussian Noise ($\sigma = 15$)**: Largest degradation observed. Accuracy dropped by $-18.37\%$ to $69.13\%$, causing $419$ prediction flips ($26.19\%$ inconsistency rate).
2. **Contrast & Brightness ($+20, -20, +15, -15$)**: Mild impact (accuracy varied within $\pm 0.5\%$). Darker images impact meningioma boundaries more severely than brighter images.
3. **JPEG Compression (Quality 75 & 50)**: Minimal degradation (accuracy remained $> 87.8\%$).
4. **Resize Scaling ($85\%$)**: Maintained $88.94\%$ accuracy, demonstrating interpolation invariance.

---

## 7. Deliverables Inventory & Paths

All requested deliverables have been generated and verified in the workspace:

| Deliverable Name | File Path | File Size / Details | Status |
| :--- | :--- | :--- | :---: |
| **Project Audit Report** | `project_audit.md` | Complete architectural audit | Verified |
| **Dataset Audit Report** | `dataset_audit.md` | 7,200 images, splits, duplicate analysis | Verified |
| **Environment Dependencies** | `requirements.txt` | Complete runtime & testing dependencies | Verified |
| **Automated Test Suite** | `tests/test_pipeline.py` | 12 Pytest test cases | Verified |
| **Automated Test Results** | `test_results.txt` | 12 passed in 31.30s | Verified |
| **Model Comparison Table** | `evaluation/model_comparison.csv` | Full comparative metrics | Verified |
| **Evaluation Confusion Matrices** | `evaluation/comparison_confusion_matrices.png` | Dual confusion matrix plot | Verified |
| **Evaluation Per-Class Plot** | `evaluation/comparison_per_class_metrics.png` | Grouped recall/F1 bar chart | Verified |
| **Learning Curves Comparison** | `evaluation/training_curves_comparison.png` | Accuracy & loss curves over epochs | Verified |
| **Explainability Examples** | `gradcam_examples/gradcam_4class_summary_grid.png` | 16-panel comparative saliency grid | Verified |
| **Live App Screenshot 1** | `screenshots/streamlit_app_home.png` | Home screen with safety disclaimer | Verified |
| **Live App Screenshot 2** | `screenshots/streamlit_sample_loaded.png` | Loaded image preview state | Verified |
| **Live App Screenshot 3** | `screenshots/streamlit_prediction_results.png` | Live prediction & Grad-CAM dashboard | Verified |
| **System Architecture Diagram** | `diagrams/system_architecture.png` | 5-tier architecture diagram | Verified |
| **Workflow Diagram** | `diagrams/workflow_diagram.png` | 6-step operational workflow diagram | Verified |
| **Updated Project README** | `README.md` | Setup, testing, evaluation, & demo guide | Verified |
| **College PPT Presentation** | `Brain_Tumor_MRI_Classification_Mini_Project_Review.pptx` | 20 slides completed with verified evidence | Verified |
| **Technical Word Report** | `Brain_Tumor_MRI_Classification_Report.docx` | 8.41 MB, 18 sections, 7 embedded figures | Verified |
| **Final Validation Summary** | `final_validation.md` | This document | Verified |

---

## 8. Remaining Limitations and Academic Disclosure

1. **Patient Identifier Absence**: The public Kaggle/Nickparvar benchmark strips DICOM metadata. Cross-split perceptual similarity demonstrates adjacent slice overlap between training and testing. While benchmark generalization ($87.50\%$) is completely authentic, true clinical patient-level generalization cannot be proven from this dataset alone.
2. **2D vs 3D Volumetrics**: Slices are classified independently in 2D; full 3D volumetric contextual reconstruction and spatial tumor burden estimation are left for final-year project expansion.
3. **Research Prototype Scope**: The Streamlit interface is an educational decision-support demonstration tool, not a regulated medical device.
