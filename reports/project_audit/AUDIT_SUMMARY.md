# Technical Audit Summary: Brain Tumor MRI Classification Project

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Audit Date:** October 2026  
**Audit Mode:** Read-Only Audit (No code, weights, manifests, or environments modified)  

---

## 1. Executive Summary

This technical audit provides an exhaustive, evidence-backed evaluation of the brain tumor MRI image classification codebase, models, training protocols, evaluation artifacts, and dataset integrity. The system classifies brain MRI scans into four diagnostic categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary Tumor**.

The primary objective of this audit is to identify the real bottlenecks governing the performance of the current multi-scale feature fusion model, critically examine the previously reported results across different data splits, evaluate the feasibility of achieving an aspirational target of $\ge 95\%$ accuracy, and provide an actionable, prioritized experimental improvement plan.

---

## 2. Key Verified Empirical Results

All reported project metrics were audited against original JSON and CSV evaluation artifacts. The metrics below represent verified ground-truth values from the project files:

### 2.1 Official Test Set Benchmarks ($N = 1,600$, `dataset/Testing`)

| Model Identifier | Checkpoint Path | Test Accuracy | Macro F1 | Balanced Accuracy | Macro ROC-AUC | Primary Evidence Artifact |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **EfficientNet-B0 (V1 Baseline)** | `efficientnet_b0.keras` | **87.50%** | **0.8699** | **87.50%** | **0.9718** | `reports/final_multiseed/final_test_summary.json` |
| **Multi-Scale Fusion (Seed 42)** | `experiments/multiseed_fusion/seed_42/best_model.keras` | **85.75%** | **0.8524** | **85.75%** | **0.9675** | `reports/final_multiseed/final_test_summary.json` |
| **EfficientNet-B0 (V2 Fine-Tuned)** | `models/efficientnet_b0_finetuned.keras` | **81.50%** | **0.8064** | **81.50%** | **0.9630** | `reports/efficientnet_b0_finetuned/final_metrics.json` |
| **Multi-Scale Fusion (Grouped Base)** | `models/multiscale_fusion_grouped_best.keras` | **80.63%** | **0.7965** | **80.63%** | **0.9584** | `reports/multiscale_fusion_grouped/final_metrics.json` |
| **Baseline Custom 4-Block CNN** | `baseline_cnn.keras` | **78.06%** | **0.7675** | **78.06%** | **0.9341** | `reports/baseline_cnn/final_metrics.json` |

> [!IMPORTANT]
> **Key Finding on Benchmark Accuracy:**
> The original frozen EfficientNet-B0 baseline (`efficientnet_b0.keras`) achieves **87.50%** accuracy on the official 1,600-image test set, outperforming the Multi-Scale Fusion Seed 42 checkpoint (**85.75%**) by **+1.75%** ($+0.0175$ Macro F1). Claims that multi-scale fusion improves overall classification accuracy over the baseline were based on validation-set performance, not the official test set.

### 2.2 Multi-Seed Grouped Validation Split ($N = 1,153$, `experiments/duplicate_aware_split/training_manifest.csv`)

| Seed | Checkpoint Path | Val Accuracy | Val Macro F1 | Val Balanced Acc | Meningioma Recall | Evidence Artifact |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| **Seed 42** | `experiments/multiseed_fusion/seed_42/best_model.keras` | **88.73%** | **0.8859** | **88.47%** | 76.34% | `experiments/multiseed_fusion/final_summary.json` |
| **Seed 123** | `experiments/multiseed_fusion/seed_123/best_model.keras` | **87.86%** | **0.8765** | **87.57%** | 72.04% | `experiments/multiseed_fusion/final_summary.json` |
| **Seed 2026** | `experiments/multiseed_fusion/seed_2026/best_model.keras` | **87.94%** | **0.8779** | **87.66%** | 73.48% | `experiments/multiseed_fusion/final_summary.json` |
| **Multi-Seed Mean** | Across Seeds 42, 123, 2026 | **88.18% $\pm$ 0.39%** | **0.8801 $\pm$ 0.41%** | **87.90% $\pm$ 0.41%** | 73.95% $\pm$ 1.8% | `experiments/multiseed_fusion/final_summary.json` |

---

## 3. Top Five Bottlenecks Governing Model Performance

Ranked strictly by verified empirical evidence from code, logs, and evaluation reports:

### Bottleneck 1: Glioma vs. Meningioma Diagnostic Confusion (Confidence: HIGH)
- **Measured Evidence:** On the official test set, 89 out of 400 gliomas (22.25%) are misclassified as meningioma by Multi-Scale Fusion (and 90 out of 400 by the baseline). Glioma recall is capped at **65.25%** (Fusion) and **65.75%** (Baseline), whereas No Tumor (98.50%) and Pituitary (98.75%) are virtually solved.
- **Root Cause:** Slices with small tumor cross-sections or peripheral axial cuts lack distinguishing morphological markers. Furthermore, `experiments/duplicate_aware_split/cross_class_near_matches.csv` reveals 20+ near-duplicate image pairs in the dataset that are labeled as glioma in one image and meningioma in another (e.g., `Tr-gl_1046.jpg` vs `Tr-me_1285.jpg`, perceptual hash distance $d = 2$).

### Bottleneck 2: "Pseudo-Multi-Scale" Architectural Design (Confidence: HIGH)
- **Code Observation:** In `src/models/multiscale_fusion.py`, the two extracted layers are `block6d_add` and `top_activation`.
- **Dimensional Verification:** Both layers share the exact same spatial dimensions: **$7 \times 7$** (`(None, 7, 7, 192)` and `(None, 7, 7, 1280)`).
- **Consequence:** True multi-scale architectures (such as FPN or U-Net) fuse high-resolution low-level features (e.g., $28 \times 28$ or $56 \times 56$) with semantic low-resolution features ($7 \times 7$). Here, both feature maps are at $7 \times 7$, and both are immediately globally pooled into 1D vectors before concatenation ($192 + 1280 = 1472$). No fine-grained spatial boundary information is preserved or fused.

### Bottleneck 3: Head Overparameterization & Regularization Imbalance (Confidence: HIGH)
- **Code Observation:** The fusion head introduces a 256-unit dense layer with two 0.35 dropout layers, adding **378,116 parameters** in the classifier head alone (more than double the baseline's 164,484 head parameters).
- **Measured Effect:** When fine-tuned on 4,447 images, training loss continues to drop while validation loss hits a minimum at epochs 9–11 and begins rising (e.g., Seed 42 val loss rises from 0.2973 at epoch 11 to 0.3296 at epoch 12). The larger head overfits the training manifest without improving test-set generalization.

### Bottleneck 4: Pre-Augmentation & Slice Leakage in Public Benchmark (Confidence: HIGH)
- **Data Audit Finding:** The meningioma class contains pre-augmented images embedded in the raw Kaggle distribution: **100 images** in `Training/meningioma` (`Tr-aug-me_*.jpg`) and **103 images** in `Testing/meningioma` (`Te-aug-me_*.jpg`). Over 25% of the test meningioma images are pre-augmented variations.
- **Cross-Split Similarity:** There are **2,009 cross-split near-duplicate candidate pairs** ($d \le 4$) between Training and Testing. 1,539 of these pairs (76.6%) reside in the `notumor` class. This explains why all models achieve $\ge 98.5\%$ recall on `notumor`: near-identical slices from identical scans exist in both sets.

### Bottleneck 5: Split Incomparability and Repeated Test Set Evaluation (Confidence: MEDIUM)
- **Observation:** The project uses multiple non-identical evaluation splits:
  1. Historical random stratified 80/20 split ($N = 1,120$).
  2. Duplicate-aware grouped validation manifest ($N = 1,153$).
  3. Official Kaggle Testing set ($N = 1,600$).
- Models tuned on the grouped validation set experienced distribution shift when evaluated on the official test set. Additionally, the official test set has been evaluated repeatedly across multiple development iterations.

---

## 4. Assessment of the $\ge 95\%$ Accuracy Target

The aspirational target of $\ge 95\%$ accuracy requires rigorous scientific qualification:

1. **Current Reality:** The best verified test accuracy in this repository is **87.50%** (EfficientNet-B0 Reference Baseline) and **85.75%** (Multi-Scale Fusion Seed 42). A gap of **+7.50% to +9.25%** remains.
2. **Mathematical Error Budget:** On the 1,600-image test set, achieving 95.0% accuracy permits a maximum of **80 total errors** across all four classes. Currently:
   - Glioma alone accounts for **139 errors** in Multi-Scale Fusion (137 in Baseline).
   - Meningioma accounts for **78 errors** in Multi-Scale Fusion (59 in Baseline).
   - Total errors currently stand at **228 errors** (Fusion) and **200 errors** (Baseline).
3. **Bayes Error Bound:** Due to cross-class identical pairs ($d \le 2$) and 2D axial slice ambiguity without volumetric or contrast-enhanced context, the irreducible label noise on this Kaggle benchmark likely prevents reaching 95% test accuracy without cleaning label errors in the dataset.
4. **Required Evidence to Support 95% Claim:**
   - Visual inspection and correction/quarantine of contradictory cross-class duplicate pairs.
   - Incorporation of earlier, higher-resolution spatial feature maps (e.g., $28 \times 28$ from stage 3).
   - Evaluation on an independent clinical cohort with verifiable patient identifiers.

---

## 5. Highest-Priority Next Step (Recommended Experiment)

**Experiment 1: True Spatial Multi-Scale Feature Fusion with Spatial Attention**
- Replace `block6d_add` ($7 \times 7$) with an intermediate feature map from stage 3 (`block3b_add`, shape $28 \times 28 \times 40$).
- Apply a $1 \times 1$ conv projection and spatial attention before pooling, providing genuine fine-grained tumor margin representation to complement the $7 \times 7 \times 1280$ semantic features.
- Evaluate on the fixed duplicate-aware validation manifest ($N = 1,153$) to measure whether glioma recall and macro-F1 improve.
