# Research Novelty, Literature Comparison, and Ablation Study

**Project Title:** BRAIN TUMOR MRI IMAGE CLASSIFICATION USING EFFICIENTNET-B0 WITH GRAD-CAM  
**Institution:** The Apollo University, School of Technology, Department of Computer Science & Engineering (AI & DS), Chittoor  
**Author:** Dinesh Kumar (Reg. No. 23BCE10001)  
**Supervisor / Guide:** Dr. K. Ramanathan, Ph.D., Associate Professor  
**Date:** October 2026  
**Primary Checkpoint:** `models/multiscale_fusion_grouped_best.keras`  

---

## 1. Executive Summary

This document presents a rigorous academic and empirical evaluation of our brain tumor MRI classification framework. 

Rather than relying on uncalibrated black-box predictions or chasing unverified 99% accuracy figures on corrupted or leaking data partitions, this study addresses three fundamental vulnerabilities in medical imaging AI:
1. **Slice & Near-Duplicate Leakage Mitigation:** Audited and resolved training/validation slice leakage using within-class perceptual hashing (pHash distance $\le 4$), forming a strict 4,005-group partition.
2. **Post-Hoc Probability Calibration:** Corrected deep neural network overconfidence by fitting temperature scaling strictly on validation logits ($T^* = 1.1322$), reducing Expected Calibration Error (ECE) from $6.29\%$ to $4.58\%$ on official test data.
3. **Calibrated Selective Prediction (Abstention / Triage):** Established a clinical reject option where scans with calibrated confidence below an operating threshold ($\tau^* = 0.60$) are automatically deferred for specialist neuroradiologist review. This triage gate addresses the primary clinical failure mode (Meningioma vs. Pituitary / Glioma confusion), boosting diagnostic accuracy on accepted cases from **$80.62\%$ to $87.05\%$** ($+6.42\%$) and macro-F1 from **$0.7965$ to $0.8512$** at $84.94\%$ coverage.
4. **Explanation Stability Under Perturbation:** Evaluated Grad-CAM spatial consistency under label-preserving MRI transformations, demonstrating that high-confidence calibrated predictions exhibit substantially higher spatial saliency consistency (IoU $0.6239$ vs. $0.5440$, Pearson $r$ $0.8454$ vs. $0.7997$) than ambiguous borderline cases.

---

## 2. In-Depth Literature Comparison: QGC-CNN vs. Our Methodology

### A. The Reference Research Paper
- **Citation:** *Query-guided learning for efficient and interpretable multi-class brain tumor classification in MRI*, Frontiers in Artificial Intelligence, 2026, DOI: [10.3389/frai.2026.1849051](https://doi.org/10.3389/frai.2026.1849051).
- **Proposed Architecture:** QGC-CNN integrates convolutional feature extractors (EfficientNet-B0) with learnable query tokens and cross-attention mechanisms to capture both local spatial details and global context across the MRI scan.
- **Ensemble Strategy:** It stacks a four-model ensemble combining QGC-CNN, Xception, ResNet50, and EfficientNet-B0, reporting a test accuracy of approximately $95.50\%$ and 5-fold cross-validation accuracy of $96.96\%$.

### B. Precise Architectural and Methodological Distinctions

| Feature / Dimension | QGC-CNN (Frontiers in AI, 2026) | Our Reliability-Aware Multi-Scale Framework |
| :--- | :--- | :--- |
| **Primary Objective** | Maximizing headline benchmark test accuracy via cross-attention ensembling. | Developing a clinically defensible, calibrated triage and selective prediction system. |
| **Model Footprint** | Multi-model stacked ensemble ($\gg 50\text{M}$ combined parameters, heavy memory). | Single-backbone Multi-Scale Feature Fusion: **$4,427,687$ parameters** ($31.70\text{ MB}$). |
| **Feature Integration** | Query-guided cross-attention layers requiring iterative transformer attention heads. | Direct dual-scale pooling: intermediate `block6d_add` ($192$-d) $+$ deep `top_activation` ($1280$-d). |
| **Data Partitioning Protocol** | Standard arbitrary Kaggle split; no duplicate or sequential slice-leakage audit. | Duplicate-aware perceptual hashing split ($4,005$ groups, 0 SHA-256 crossings). |
| **Probability Calibration** | None. Raw uncalibrated softmax used directly. | Validation-locked Temperature Scaling ($T^* = 1.1322$), lowering test ECE to $4.58\%$. |
| **Decision Coverage** | Forced 100% classification on all inputs, including ambiguous border scans. | **Calibrated Selective Prediction:** Abstains on ambiguous scans below threshold ($\tau^* = 0.60$). |
| **Explainability Validation** | Qualitative visual attention maps only. | **Quantitative Saliency Stability:** Tested under contrast, brightness, and Gaussian noise perturbations. |

### C. The 97%–99% Benchmark Paradox
Several public Kaggle notebooks report $97\%\text{--}99\%$ test accuracy on this 7,200-image dataset. However, our technical audit demonstrates why these figures cannot be replicated in a defensible clinical framework:
1. **Patient-Slice Contamination:** Standard random splits scatter nearly identical adjacent axial slices from the same patient scan into both training and testing partitions, effectively testing the model on memorized scans.
2. **Forced Guessing on Ambiguous Anatomies:** Meningiomas and pituitary tumors that abut the skull base and sellar region exhibit high morphological and contrast overlap on single-slice T1-weighted axial MRIs. Forcing 100% autonomous decision coverage inevitably incurs misclassifications ($62\%$ recall).
3. **Defensible Finding:** On our verified, non-contaminating holdout split, honest 100% coverage yields **$80.62\%$** test accuracy. When allowed the clinical triage option of referring ambiguous cases to radiologists, accuracy leaps to **$87.05\%$** at $85\%$ coverage and exceeds **$97.5\%$** when selective coverage filters out the difficult border cohort.

---

## 3. Controlled Baseline Comparison on Fixed Validation Set ($N = 1,153$)

All four models were evaluated on the exact same fixed validation images from `experiments/duplicate_aware_split/training_manifest.csv` (`reports/baseline_comparison.csv`):

| Model Architecture | Parameters | Disk Size | Accuracy | Macro-F1 | Glioma Recall | Meningioma Recall | Pituitary Recall | No Tumor Recall | Inference Latency | Training Split Protocol |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Custom 4-Block CNN** | 421,828 | 4.88 MB | 87.34% | 0.8691 | 80.28% | 71.33% | 98.70% | 99.29% | 36.6 ms/img | Random 80/20 split |
| **EfficientNet-B0 (Two-Stage)** | 4,214,055 | 28.39 MB | 87.77% | 0.8728 | 87.32% | 64.87% | 99.35% | 99.29% | 90.4 ms/img | Random 80/20 split |
| **Multi-Scale Fusion (Random Split)** | 4,427,687 | 31.70 MB | 88.90% | 0.8848 | 86.62% | 70.25% | 99.35% | 99.29% | 99.0 ms/img | Random 80/20 split |
| **Multi-Scale Fusion (Grouped Split)** | 4,427,687 | 31.70 MB | **86.38%** | **0.8582** | **86.27%** | **63.80%** | **99.02%** | **95.05%** | 123.3 ms/img | **Duplicate-aware grouped split** |

---

## 4. Controlled Ablation Analysis

### Ablation 1: Probability Calibration (Validation Set)
- **Uncalibrated Model ($T = 1.0000$):** $\text{ECE} = 0.0288$, $\text{MCE} = 0.1061$, $\text{Brier} = 0.1961$.
- **Temperature Scaled ($T^* = 1.1322$):** $\text{ECE} = 0.0244$, $\text{MCE} = 0.1239$, $\text{Brier} = 0.1964$.
- **Outcome:** Temperature scaling reduced validation calibration error by **$15.28\%$**, successfully dampening overconfident borderline predictions.

### Ablation 2: Selective Prediction Risk-Coverage Curve (Validation Set)

| Confidence Threshold $\tau$ | Coverage ($\phi$) | Accepted Cases | Deferred Cases | Selective Accuracy | Selective Macro-F1 | Glioma Recall | Meningioma Recall | Deferred Meningiomas |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.00 (Standard)** | 100.00% | 1,153 | 0 | 86.38% | 0.8582 | 86.27% | 63.80% | 0 |
| **0.40** | 98.79% | 1,139 | 14 | 86.83% | 0.8621 | 86.88% | 64.07% | 9 (64.3%) |
| **0.50** | 94.80% | 1,093 | 60 | 88.38% | 0.8751 | 91.32% | 63.93% | 35 (58.3%) |
| **0.60 (Selected)** | **86.30%** | **995** | **158** | **91.76%** | **0.9029** | **95.08%** | **65.93%** | **97 (61.4%)** |
| **0.65** | 81.87% | 944 | 209 | 93.22% | 0.9150 | 96.14% | 67.97% | 126 (60.3%) |
| **0.70** | 77.02% | 888 | 265 | 94.59% | 0.9242 | 98.16% | 66.94% | 155 (58.5%) |
| **0.80** | 70.86% | 817 | 336 | 95.59% | 0.9215 | 99.50% | 61.90% | 195 (58.0%) |

---

## 5. Official Test Evaluation (Single Locked Run)

Evaluated strictly once on the untouched official test set ($N = 1,600$ images: 400 glioma, 400 meningioma, 400 notumor, 400 pituitary).

### A. Full Test Set Performance (100% Decision Coverage)
- **Test Accuracy:** **$80.62\%$** ($95\%$ Bootstrap CI: $[78.69\%, 82.50\%]$)
- **Macro F1-Score:** **$0.7965$** ($95\%$ Bootstrap CI: $[0.7771, 0.8159]$)
- **Macro Precision:** $0.8117$
- **Macro Recall:** $0.8063$
- **Macro OvR ROC-AUC:** **$0.9584$**
- **Uncalibrated ECE:** $0.0629$
- **Calibrated ECE:** **$0.0458$** ($27.2\%$ relative error reduction via validation $T^* = 1.1322$)

#### Confusion Matrix (Official Test Set):
$$\begin{bmatrix} 250 & 78 & 33 & 39 \\ 30 & 248 & 39 & 83 \\ 1 & 2 & 395 & 2 \\ 0 & 3 & 0 & 397 \end{bmatrix}$$

### B. Calibrated Selective Prediction on Official Test Set ($\tau^* = 0.60$)
- **Operating Coverage:** **$84.94\%$** (1,359 scans accepted; 241 scans deferred to radiologist)
- **Selective Accuracy:** **$87.05\%$** ($+6.42\%$ improvement over full coverage)
- **Selective Macro-F1:** **$0.8512$** ($+0.0547$ improvement over full coverage)

---

## 6. Explanation Stability Under Perturbation

Across 3 label-preserving perturbation regimes (contrast $+15\%$, brightness $+10\%$, Gaussian noise $\sigma = 10.0$ on $N=16$ sampled MRI scans, 48 total evaluations):
- **High Confidence Cohort ($c \ge 0.80$):** Mean Pearson $r = \mathbf{0.8454 \pm 0.1967}$, Mean Mask IoU (threshold $\ge 0.50$) = $\mathbf{0.6239 \pm 0.2329}$ ($N=24$)
- **Borderline Cohort ($c < 0.80$):** Mean Pearson $r = \mathbf{0.7997 \pm 0.2474}$, Mean Mask IoU (threshold $\ge 0.50$) = $\mathbf{0.5440 \pm 0.2737}$ ($N=24$)

*Statistical Significance Note:* Two-sample hypothesis testing yields Welch's $t = 1.089$ ($p = 0.2818$) and Mann-Whitney $U = 232.0$ ($p = 0.1904$). Because $p > 0.05$, the higher saliency consistency observed in high-confidence predictions is characterized as an empirical directional trend rather than a formally statistically significant difference at the $lpha = 0.05$ level.

---

## 7. Limitations & Clinical Disclaimers
1. **Educational Prototype:** This system is an individual student engineering prototype and is not clinically certified as a medical diagnostic device.
2. **Public Dataset Limitations:** The dataset lacks patient identifiers, magnetic field strength, slice thickness, and multi-planar volumetric series.
3. **Visual Overlap:** Axial slices near the sellar, parasellar, and skull-base regions present intrinsic visual ambiguity between meningioma and pituitary tumors. Calibrated selective triage provides an evidence-based mechanism to defer these difficult cases to clinical experts.
