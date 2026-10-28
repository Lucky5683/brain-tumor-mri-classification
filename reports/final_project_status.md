# Final Project Status & Technical Verification Report

**Project Title:** BRAIN TUMOR MRI IMAGE CLASSIFICATION USING EFFICIENTNET-B0 WITH GRAD-CAM  
**Institution:** The Apollo University, School of Technology, Dept. of CSE (AI & DS), Chittoor  
**Author:** Dinesh Kumar (Reg. No. 23BCE10001)  
**Supervisor / Guide:** Dr. K. Ramanathan, Ph.D., Associate Professor  
**Final Release Date:** October 2026  
**Primary Checkpoint:** `models/multiscale_fusion_grouped_best.keras`  
**Application Entry Point:** `app.py` (Streamlit)  

---

## 1. Project Overview & Execution Summary

All eight phases of the autonomous engineering and research mandate have been completed, verified with executed code, and documented with genuine empirical results:

1. **Phase 1 (Recovery & Verification):** Repaired `scripts/evaluate_grouped_checkpoint.py` and reproduced validation metrics for `models/multiscale_fusion_grouped_best.keras` with 100% precision ($0.0000\%$ discrepancy: Accuracy $86.3833\%$, Macro-F1 $85.8211\%$).
2. **Phase 2 (Data Quality & Split Integrity):** Executed comprehensive dataset audit across all 5,600 training images and 1,600 test images. Verified $0$ exact SHA-256 duplicate crossings, $0$ similarity group crossings across the 4,005 pHash groups, and confirmed the official test set remained completely untouched during model training and selection.
3. **Phase 3 (Fair Controlled Baseline Comparison):** Evaluated Custom 4-Block CNN, EfficientNet-B0 (Two-Stage), Multi-Scale Fusion (Random Split), and Multi-Scale Fusion (Grouped Split) on the identical 1,153 validation images. Demonstrated that un-grouped training exhibits inflated validation accuracy ($88.90\%$) due to near-duplicate slice leakage, whereas the true group-aware baseline is $86.38\%$.
4. **Phase 4 & 5 (Research Contribution & Implementation):** Implemented **Reliability-Aware Multi-Scale Brain Tumor MRI Classification via Calibrated Selective Prediction and Explanation Stability**:
   - Fitted post-hoc Temperature Scaling strictly on validation logits ($T^* = 1.1322$), reducing calibration error (ECE) by $15.28\%$ on validation data and $27.2\%$ on test data.
   - Designed Calibrated Selective Prediction (Abstention / Triage): Scans with calibrated confidence below $\tau^* = 0.60$ are deferred to human neuroradiologists. Because $61.4\%$ of rejected cases are difficult Meningiomas, selective accuracy leaps to $>91.7\%$ on validation and $87.05\%$ on test data.
   - Tested Grad-CAM Explanation Stability across 3 perturbation regimes (contrast +15%, brightness +10%, Gaussian noise sigma=10.0 on N=16 sampled MRI scans, 48 total evaluations). High-confidence predictions (c >= 0.80) empirically exhibited higher mean spatial saliency consistency (IoU 0.6239 +/- 0.23 vs. 0.5440 +/- 0.27, Pearson r = 0.8454 +/- 0.20 vs. 0.7997 +/- 0.25) than borderline predictions (c < 0.80). Statistical hypothesis testing (Welch's t-test p=0.2818, Mann-Whitney U test p=0.1904) indicates that this is a directional empirical trend rather than a formally statistically significant difference at alpha=0.05 with N=24 evaluations per cohort.
5. **Phase 6 (Checkpoint & Model Integrity):** Verified model architecture ($4,427,687$ parameters; intermediate `block6d_add` 192-d + deep `top_activation` 1280-d; 31.70 MB) and serialization compatibility with Keras 3.
6. **Phase 7 (Single Locked Test Evaluation):** Evaluated the final selected model on the untouched official test set ($N = 1,600$) exactly once with validation-locked hyperparameters. Full coverage accuracy is $80.62\%$ (Macro ROC-AUC $0.9584$). With selective prediction ($\tau^* = 0.60$), diagnostic accuracy on accepted scans leaps to **$87.05\%$** with Macro-F1 **$0.8512$** at $84.94\%$ coverage.
7. **Phase 8 (Final Deliverables & Streamlit App):** Upgraded `app.py` with multi-scale model support, dynamic temperature scaling, automated selective prediction triage alert banners, interactive Grad-CAM / Grad-CAM++ visualization, and verified 12/12 automated pytest tests passing.

---

## 2. Key Empirical Results Summary

| Dataset Partition | Coverage ($\phi$) | Operating Threshold $\tau$ | Accuracy | Macro-F1 | Glioma Recall | Meningioma Recall | Pituitary Recall | No Tumor Recall | Expected Calibration Error (ECE) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Validation Set (Grouped)** | 100.00% | 0.00 | 86.38% | 0.8582 | 86.27% | 63.80% | 99.02% | 95.05% | 0.0244 (Calibrated) |
| **Validation Set (Selective)** | 86.30% | 0.60 | 91.76% | 0.9029 | 95.08% | 65.93% | 99.67% | 97.40% | — |
| **Official Test Set (100% Coverage)** | 100.00% | 0.00 | **80.62%** | **0.7965** | 62.50% | 62.00% | 99.25% | 98.75% | **0.0458** (Calibrated) |
| **Official Test Set (Selective Triage)** | **84.94%** | **0.60** | **87.05%** | **0.8512** | — | — | — | — | — |

*Bootstrap 95% Confidence Intervals on Official Test Set (1,000 resamples):*
- Test Accuracy: $[78.69\%, 82.50\%]$
- Test Macro-F1: $[0.7771, 0.8159]$
- Test Macro ROC-AUC: $0.9584$

---

## 3. Directory of Final Deliverables

| Deliverable | Path | Description |
| :--- | :--- | :--- |
| **Final Experiment Summary** | `reports/final_experiment_summary.json` | Complete machine-readable metadata, architecture, metrics, and limitations. |
| **Final Classification Report** | `reports/final_classification_report.csv` | Official test set precision, recall, F1, and support for all 4 classes. |
| **Final Confusion Matrix Image** | `reports/final_confusion_matrix.png` | Publication-ready confusion matrix visualization on official test data. |
| **Final Confusion Matrix CSV** | `reports/final_confusion_matrix.csv` | Exact integer counts for test set class predictions. |
| **Final Test Predictions** | `reports/final_test_predictions.csv` | Item-by-item predictions, calibrated confidences, and triage status for all 1,600 test images. |
| **Data Quality Audit** | `reports/data_quality_audit.json` | Verification of 5,600 training images, 4,005 groups, and 0 cross-split leakage. |
| **Controlled Baseline Comparison** | `reports/baseline_comparison.csv` | Comparative evaluation of CNN, EffNet, and Multi-Scale models on the same validation set. |
| **Calibration Analysis** | `reports/final_calibration_results.json` | Temperature scaling parameters, uncalibrated vs. calibrated ECE and Brier scores. |
| **Risk-Coverage Analysis** | `reports/selective_prediction_results.csv` | Accuracy-coverage trade-off sweeping thresholds from 0.0 to 0.95. |
| **Explanation Stability Report** | `reports/explanation_stability_results.csv` | Quantitative IoU and Pearson $r$ saliency stability under 3 perturbation modes. |
| **Research & Novelty Report** | `reports/research_novelty_and_ablation.md` | Full academic review, distinction from QGC-CNN, and ablation discussion. |
| **Streamlit Web Application** | `app.py` | Interactive clinical decision-support prototype with triage alerts and Grad-CAM. |
| **Automated Test Suite** | `tests/test_pipeline.py` | 12 automated pytest unit and integration tests (100% pass rate). |

---

## 4. Commands to Reproduce Experiments

All commands should be executed from the project root (`X:\Mini Project\brain-tumor-mri-classification`):

### 1. Launch the Streamlit Web Application
```bash
streamlit run app.py
```

### 2. Run Automated Pytest Suite
```bash
pytest tests/test_pipeline.py -v
```

### 3. Reproduce Data Quality Audit
```bash
python scripts/audit_dataset_quality_complete.py
```

### 4. Reproduce Fair Baseline Comparison (Phase 3)
```bash
python scripts/compare_baseline_models.py
```

### 5. Reproduce Calibration, Selective Prediction & Explanation Stability (Phase 5)
```bash
python scripts/run_reliability_selective_prediction.py
```

### 6. Reproduce Final Locked Test Evaluation (Phase 7)
```bash
python scripts/evaluate_final_locked_test.py
```

---

## 5. Remaining Limitations & Clinical Boundaries

1. **Educational/Research Prototype Only:** This system is NOT certified as a medical diagnostic device under FDA/CE-MDR standards.
2. **Absence of Patient Metadata:** The underlying public Kaggle dataset lacks verified patient identifiers, slice thickness, magnetic field strength (1.5T vs 3.0T), and contrast-enhancement documentation. Perceptual hashing (pHash) serves as an empirical proxy, not clinical proof of subject-level independence.
3. **Meningioma / Pituitary Morphological Overlap:** Axial slices near the sellar, parasellar, and cavernous sinus regions exhibit high visual similarity. While selective prediction routes ambiguous scans for expert human review, differentiating these entities without multi-planar (coronal/sagittal) series remains challenging.
4. **Grad-CAM Saliency Interpretation:** Saliency maps visualize gradients of the classification loss with respect to convolutional feature maps; they do not represent ground-truth tumor segmentations.
