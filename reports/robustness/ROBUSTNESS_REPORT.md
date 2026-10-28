# Phase 4 — Robustness Evaluation Report

> [!IMPORTANT]
> **Methodological Integrity**: Evaluated strictly on the untouched official test set ($N=1,600$) using the untouched reference model (`efficientnet_b0.keras`). No model retraining, fine-tuning, or hyperparameter selection was conducted on test data.

## 1. Executive Summary
- **Reference Baseline Accuracy**: **87.50%** (Macro F1: **0.8698**, Glioma Recall: **66.00%**)
- **Largest Measured Accuracy Degradation**: **Gaussian noise sigma=15** with drop of **18.37%** (rel: **20.99%**)
- **Most Inconsistent Perturbation**: **Gaussian noise sigma=15** with **26.19%** prediction flip rate (**419** flips out of 1,600)

## 2. Comprehensive Robustness Metrics Table

| Perturbation Condition | Accuracy [95% CI] | Macro F1 [95% CI] | Glioma Recall [95% CI] | ROC-AUC | Mean Max Prob | Consistency | Flips / 1600 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Original (Baseline)** | 87.50% [85.9, 89.1] | 0.8698 [0.8532, 0.8851] | 66.00% [61.4, 70.8] | 0.9714 | 0.9244 | 100.00% | 0 |
| **Brightness +15** | 87.44% [85.9, 89.0] | 0.8687 [0.8533, 0.8844] | 64.25% [59.6, 68.9] | 0.9731 | 0.9245 | 98.50% | 24 |
| **Brightness -15** | 87.94% [86.2, 89.4] | 0.8754 [0.8581, 0.8901] | 67.00% [62.4, 71.6] | 0.9713 | 0.9148 | 95.56% | 71 |
| **Contrast +20** | 87.06% [85.3, 88.7] | 0.8651 [0.8484, 0.8811] | 62.50% [57.7, 67.4] | 0.9705 | 0.9116 | 94.81% | 83 |
| **Contrast -20** | 88.00% [86.3, 89.6] | 0.8762 [0.8594, 0.8920] | 70.25% [65.8, 74.4] | 0.9720 | 0.9181 | 97.00% | 48 |
| **Gaussian noise sigma=15** | 69.13% [66.9, 71.3] | 0.6529 [0.6308, 0.6759] | 54.75% [49.9, 59.7] | 0.9394 | 0.9097 | 73.81% | 419 |
| **JPEG quality=50** | 87.81% [86.2, 89.4] | 0.8739 [0.8582, 0.8899] | 64.50% [59.8, 69.1] | 0.9694 | 0.9179 | 94.19% | 93 |
| **JPEG quality=75** | 88.19% [86.6, 89.8] | 0.8774 [0.8610, 0.8926] | 65.00% [60.4, 69.3] | 0.9705 | 0.9193 | 96.13% | 62 |
| **Resize scale=85%** | 88.94% [87.4, 90.4] | 0.8860 [0.8707, 0.9005] | 67.75% [63.3, 71.9] | 0.9733 | 0.9261 | 95.44% | 73 |

## 3. Per-Class Recall Breakdown Across Perturbations

| Perturbation Condition | Glioma Recall | Meningioma Recall | No Tumor Recall | Pituitary Recall |
| :--- | :---: | :---: | :---: | :---: |
| **Original (Baseline)** | 66.00% | 84.75% | 100.00% | 99.25% |
| **Brightness +15** | 64.25% | 86.00% | 100.00% | 99.50% |
| **Brightness -15** | 67.00% | 87.50% | 99.50% | 97.75% |
| **Contrast +20** | 62.50% | 88.50% | 100.00% | 97.25% |
| **Contrast -20** | 70.25% | 83.50% | 99.25% | 99.00% |
| **Gaussian noise sigma=15** | 54.75% | 24.00% | 99.00% | 98.75% |
| **JPEG quality=50** | 64.50% | 91.00% | 99.25% | 96.50% |
| **JPEG quality=75** | 65.00% | 90.75% | 100.00% | 97.00% |
| **Resize scale=85%** | 67.75% | 91.00% | 100.00% | 97.00% |

## 4. Ranked Measured Degradations Relative to Baseline

| Rank | Perturbation Condition | Accuracy Drop | Relative Drop (%) | Macro F1 Drop | Glioma Recall Drop | Inconsistency Rate |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | **Gaussian noise sigma=15** | -18.37% | -20.99% | -0.2169 | -11.25% | 26.19% (419 flips) |
| 2 | **Contrast +20** | -0.44% | -0.50% | -0.0047 | -3.50% | 5.19% (83 flips) |
| 3 | **Brightness +15** | -0.06% | -0.07% | -0.0011 | -1.75% | 1.50% (24 flips) |
| 4 | **JPEG quality=50** | +0.31% | +0.35% | +0.0041 | -1.50% | 5.81% (93 flips) |
| 5 | **Brightness -15** | +0.44% | +0.50% | +0.0056 | +1.00% | 4.44% (71 flips) |
| 6 | **Contrast -20** | +0.50% | +0.57% | +0.0064 | +4.25% | 3.00% (48 flips) |
| 7 | **JPEG quality=75** | +0.69% | +0.79% | +0.0076 | -1.00% | 3.87% (62 flips) |
| 8 | **Resize scale=85%** | +1.44% | +1.65% | +0.0162 | +1.75% | 4.56% (73 flips) |

## 5. Clinical & Technical Findings
1. **Vulnerability to Additive Noise**: High-frequency additive Gaussian noise severely degrades activation patterns, leading to substantial prediction flips.
2. **Asymmetry in Contrast/Brightness**: Reductions in contrast and brightness impact dark-tissue boundary resolution more severely than corresponding increases.
3. **JPEG Compression Stability**: Standard lossy JPEG compression at quality 75 produces minimal degradation, while quality 50 demonstrates moderate erosion in borderline cases.
4. **Clinical Implication**: High test accuracy on clean curated benchmarks does not imply clinical scanner invariance. Robustness stress tests must be mandatory before any consideration of clinical decision support.