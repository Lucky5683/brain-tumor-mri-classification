# Spatial Fusion Model: Official Test Set Evaluation Report

**Evaluation Date:** 2026-10-28T08:25:37.426053
**Evaluated Checkpoint:** `experiments\spatial_fusion\meningioma_weighted_50ep_seed42\best_model.keras`
**Test Dataset Path:** `dataset\Testing`
**Architecture:** `EfficientNetB0_SpatialFusion` (4,428,295 parameters)
**Execution Environment:** Python 3.13.7 on Windows-11-10.0.26300-SP0 (TensorFlow 2.21.0, Keras 3.14.1)

---

## 1. Executive Summary & Verification Standard

This report provides the reproducible evaluation of the selected **Spatial Attention Feature Fusion** model on the official independent test partition ($N = 1,600$ MRI scans, exactly $400$ per class).

- **Evaluation Protocol:** Strictly single-pass, feed-forward inference on unmodified image tensors.
- **Model Invariance:** No weights, biases, class indices, or hyperparameters were adjusted.
- **Data Integrity:** All 1,600 scans were read and decoded with zero corrupted files.
- **Independence & Data Leakage Note:** Zero exact SHA-256 byte duplicates exist between training and testing sets. However, per the established dataset audit, volumetric adjacent 2D MRI slices from common patient scan sessions are present across the public Kaggle benchmark. Therefore, these metrics represent **benchmark generalization** and must not be conflated with clinical patient-level generalization.

---

## 2. Key Aggregate Metrics Summary

| Primary Evaluation Metric | Test Score (N = 1,600) | Historical Validation Score (N = 1,153) | Difference (Val â†’ Test) |
| :--- | :---: | :---: | :---: |
| **Overall Test Accuracy** | **88.19%** (1411 / 1,600) | 92.54% (1,067 / 1,153) | -4.35% |
| **Balanced Accuracy** | **88.19%** | 92.35% | -4.16% |
| **Macro-Averaged F1-Score** | **0.8777** | 0.9241 | -0.0464 |
| **Weighted-Averaged F1-Score** | **0.8777** | 0.9252 | -0.0475 |
| **Macro Precision** | **0.8872** | 0.9254 | -0.0382 |
| **Macro Recall (Sensitivity)** | **0.8819** | 0.9235 | -0.0416 |
| **Macro ROC-AUC (OvR)** | **0.9816** | â€” | â€” |

---

## 3. Class-by-Class Diagnostic Performance

| Diagnostic Class | Precision | Recall (Sensitivity) | Specificity | F1-Score | Support | ROC-AUC (OvR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Glioma** | 95.52% | 69.25% | 98.92% | **0.8029** | 400 | 0.9605 |
| **Meningioma** | 82.04% | 84.50% | 93.83% | **0.8325** | 400 | 0.9695 |
| **No Tumor (Healthy)** | 87.47% | 99.50% | 95.25% | **0.9310** | 400 | 0.9979 |
| **Pituitary Tumor** | 89.84% | 99.50% | 96.25% | **0.9442** | 400 | 0.9986 |

---

## 4. Confusion Matrix Analysis

```text
Actual \ Predicted     glioma    meningioma     notumor   pituitary       Total
glioma                    277       72            38          13          400
meningioma                11        338           19          32          400
notumor                   1         1             398         0           400
pituitary                 1         1             0           398         400
Total Predicted           290       412           455         443         1600
```

### Key Error Observations:
- **Total Misclassified Samples:** 189 / 1,600 (11.81%)
- **Most Common Error Pair:** glioma -> meningioma (72 occurrences)
- **Healthy Brain Safety (False Positives for No Tumor):** 2 healthy scans misclassified as tumors.
- **Healthy Brain False Negatives (Tumors misclassified as Healthy):** 57 diseased scans misclassified as normal.

---

## 5. Artifact Manifest

All generated evaluation artifacts are stored deterministically in `X:\Mini Project\brain-tumor-mri-classification\experiments\final_evaluation\spatial_fusion`:
- `metrics_summary.json`: Complete serialized metrics and metadata.
- `confusion_matrix.csv`: Raw count confusion matrix.
- `normalized_confusion_matrix.csv`: Row-normalized confusion matrix.
- `confusion_matrix.png`: High-resolution dual-panel visualization.
- `roc_curves.png`: One-vs-Rest ROC curves with per-class and macro AUC.
- `per_class_metrics.png`: Grouped comparative bar chart of class metrics.
- `misclassified_samples.csv`: Complete index of misclassified scans with true/predicted labels, confidence, and margin.
