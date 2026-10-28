# Comprehensive Pre-Training Audit Report: Brain Tumor MRI Multi-Scale Feature Fusion

**Project:** Brain Tumor MRI Classification Using EfficientNet-B0 with Grad-CAM  
**Audit Scope:** Phase 1 — Complete Pre-Training Verification and Baseline Reproduction  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Date of Audit:** October 23, 2026  
**Auditor:** Antigravity Autonomous Coding & Research Agent  
**Audit Protocol Status:** PASSED (Audit Only — Zero Weights Altered, Official Test Set Preserved)

---

## 1. Executive Summary

This audit establishes the pre-training integrity, dataset validity, checkpoint health, architectural structure, and baseline performance of the **EfficientNet-B0 Multi-Scale Feature Fusion** architecture before executing further fine-tuning experiments.

### Key Audit Findings:
1. **Split Manifest Integrity (100% Passed):** The duplicate-aware training manifest (`experiments/duplicate_aware_split/training_manifest.csv`) contains exactly 5,600 records (4,447 training, 1,153 validation). All SHA-256 hashes match physical files with zero bitwise errors. Exactly zero SHA-256 duplicates and zero perceptual hash (`group_id`) clusters cross the train-validation split. Zero images from `dataset/Testing` appear in the training manifest.
2. **Checkpoint Availability:** 13 out of 14 scanned checkpoints exist and load cleanly using the custom Keras 3 deserialization compatibility shim (`src.models.compat`). The only non-existent path is `models/efficientnet_b0.keras`, which is expected because the reference baseline checkpoint is located at the project root (`efficientnet_b0.keras`).
3. **Reproduced Validation Baseline:** Evaluating `models/multiscale_fusion_grouped_best.keras` on the exact 1,153 validation manifest images yielded:
   - **Validation Accuracy:** $86.3833\%$
   - **Validation Macro-F1:** $0.858211$
   - **Validation Balanced Accuracy:** $86.0357\%$
   - Discrepancy with historical logged training metrics: **$0.0000\%$** (identical to 16 decimal places).
4. **Comparison with Baseline Backbone:** Evaluating `models/efficientnet_b0_finetuned.keras` on the exact same 1,153 validation images yielded:
   - **Validation Accuracy:** $87.7710\%$
   - **Validation Macro-F1:** $0.872802$
   - **Validation Balanced Accuracy:** $87.4382\%$
5. **Primary Diagnostic Bottleneck:** Both models suffer from two specific clinical confusion modes:
   - **Meningioma $\rightarrow$ Pituitary:** Multi-Scale misclassifies 64 meningiomas as pituitary (EfficientNet-B0 misclassifies 74).
   - **Glioma $\leftrightarrow$ Meningioma:** Multi-Scale misclassifies 29 gliomas as meningioma, and 24 meningiomas as glioma.
6. **Feasibility of Safe Fine-Tuning:** The multi-scale fusion architecture can be safely fine-tuned from `models/multiscale_fusion_grouped_best.keras` without retraining from scratch. The backbone layers can be partially unfrozen while keeping all `BatchNormalization` layers strictly frozen (`trainable=False`), preventing covariate shift on small batches.
7. **Official Test Set Isolation:** The 1,600 images in `dataset/Testing` were not loaded, preprocessed, or evaluated during this audit.

---

## 2. Task 1: Environment & Codebase Inspection

### 2.1 Hardware and Runtime Environment
- **Project Root Directory:** `X:\Mini Project\brain-tumor-mri-classification`
- **Operating System:** Windows 11 Home Single Language (10.0.26100, AMD64)
- **Python Executable:** `C:\Users\dines\AppData\Local\Programs\Python\Python313\python.exe`
- **Python Version:** `3.13.7` (CPython 64-bit)
- **TensorFlow Version:** `2.21.0`
- **Keras Version:** `3.13.0`
- **Available Hardware:** 12 logical CPU cores (AMD/Intel x86_64 architecture).
- **GPU Availability:** No physical GPU visible to TensorFlow (`tf.config.list_physical_devices('GPU') == []`). TensorFlow is running on **CPU fallback** utilizing native oneDNN (MKL-DNN) multithreaded vector primitives.
- **Package Modifications:** Zero packages were installed or upgraded during this audit.

### 2.2 Core Project Components
- `config.py`: Defines global configuration constants, class labels (`glioma`, `meningioma`, `notumor`, `pituitary`), default image dimensions ($224 \times 224$), directory paths, random seed ($42$), and uncertainty rejection thresholds ($\tau^* = 0.60$).
- `src/data/dataset.py`: Implements manifest-aware dataset pipeline (`get_train_val_datasets`). Uses `tf.data.Dataset` with deterministic mapping for validation (`training=False`, no augmentation) and random flips/rotations/zooms strictly for training (`training=True`).
- `src/data/preprocessing.py`: Implements robust image reading, RGB conversion, bilinear resizing to $224 \times 224$, and EfficientNet-compatible scale normalization ($[0, 255]$ for EfficientNet pass-through, $[-1, 1]$ or $[0, 1]$ for custom CNNs).
- `src/models/multiscale_fusion.py`: Implements two-branch multi-scale feature extractor combining intermediate layer `block6d_add` ($7 \times 7 \times 192$) with final activation layer `top_activation` ($7 \times 7 \times 1280$) via Global Average Pooling into a $1472$-dimensional fused representation.
- `scripts/train_multiscale_fusion_grouped.py`: Implements two-stage training (Stage 1 head warmup, Stage 2 partial backbone fine-tuning) on `training_manifest.csv` with validation Macro-F1 checkpointing.
- `scripts/train_efficientnet.py`: Standard baseline training script supporting unweighted and class-weighted two-stage fine-tuning.
- `scripts/compare_baseline_models.py`: Comparative evaluation script aggregating validation metrics across models.
- `app.py`: Streamlit interactive web interface supporting model switching, dual Grad-CAM / Grad-CAM++ overlays, selective classification triage ($\tau^* = 0.60$), and temperature calibration.
- `src/explainability/gradcam.py`: Contains custom `compute_multiscale_fusion_gradcam` implementation targeting the nested `top_activation` tensor of `multiscale_feature_extractor` with gradient tracking through the fusion head.

---

## 3. Task 2: Checkpoint Inventory

A complete inspection was performed across all model files in `models/` and the project root. Fourteen candidate model paths were evaluated for physical existence, file size, deserialization capability, parameter counts, training history, and test set exposure.

### 3.1 Checkpoint Inventory Table

| Checkpoint Name | File Path | Exists | File Size | Loadable | Total Params | Trainable Params | Training Manifest / Split | Recorded Val Acc | Recorded Val Macro-F1 | Test Metrics Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :---: | :---: | :--- |
| `efficientnet_b0.keras` | `efficientnet_b0.keras` | Yes | 18.15 MB | Yes | 4,214,055 | 164,484 | `dataset/Training` (Historical 80/20) | 91.79% | 0.9130 | Historical (Test Acc: 87.50%, Macro-F1: 0.8699) |
| `baseline_cnn.keras` | `baseline_cnn.keras` | Yes | 4.88 MB | Yes | 421,828 | 421,828 | Random 80/20 Stratified Split | 72.50% | 0.6840 | Historical (Test Acc: 68.75%, Macro-F1: 0.6624) |
| `models/efficientnet_b0.keras` | `models/efficientnet_b0.keras` | **No** | 0 MB | No | N/A | N/A | N/A (Root file used instead) | N/A | N/A | None |
| `efficientnet_b0_finetuned.keras` | `models/efficientnet_b0_finetuned.keras` | Yes | 28.39 MB | Yes | 4,214,055 | 1,507,252 | Random 80/20 Stratified Split | 85.18% (rand) / **87.77%** (grouped) | 0.8469 (rand) / **0.8728** (grouped) | Historical (Test Acc: 83.38%, Macro-F1: 0.8271) |
| `efficientnet_b0_classweighted.keras` | `models/efficientnet_b0_classweighted.keras` | Yes | 28.39 MB | Yes | 4,214,055 | 1,507,252 | Random 80/20 Stratified Split | 85.27% | 0.8483 | None |
| `exp1_efficientnet_b0_best.keras` | `models/exp1_efficientnet_b0_best.keras` | Yes | 18.15 MB | Yes | 4,214,055 | 164,484 | `training_manifest.csv` (Grouped) | Stage 1 Only | Stage 1 Only | None (Stage 2 Never Run) |
| `multiscale_fusion_best.keras` | `models/multiscale_fusion_best.keras` | Yes | 31.70 MB | Yes | 4,427,687 | 1,832,676 | Random 80/20 Stratified Split | 85.36% (rand) / 88.90% (grouped) | 0.8472 (rand) / 0.8848 (grouped) | None |
| `multiscale_fusion_grouped_best.keras` | `models/multiscale_fusion_grouped_best.keras` | Yes | 31.70 MB | Yes | 4,427,687 | 1,832,676 | `training_manifest.csv` (Grouped) | **86.38%** | **0.8582** | Historical Locked Eval (Test Acc: 80.63%, Macro-F1: 0.7965) |
| `multiscale_fusion_stage1_best.keras` | `models/multiscale_fusion_stage1_best.keras` | Yes | 20.59 MB | Yes | 4,427,687 | 378,116 | Random 80/20 Stratified Split | 83.57% | 0.8290 | None |
| `multiscale_fusion_grouped_stage1_best.keras` | `models/multiscale_fusion_grouped_stage1_best.keras` | Yes | 20.59 MB | Yes | 4,427,687 | 378,116 | `training_manifest.csv` (Grouped) | 84.13% | 0.8345 | None |
| `efficientnet_b0_stage1.keras` | `models/efficientnet_b0_stage1.keras` | Yes | 18.15 MB | Yes | 4,214,055 | 164,484 | Random 80/20 Stratified Split | 84.20% | 0.8360 | None |
| `efficientnet_b0_stage1_weighted.keras` | `models/efficientnet_b0_stage1_weighted.keras` | Yes | 18.15 MB | Yes | 4,214,055 | 164,484 | Random 80/20 Stratified Split | 84.11% | 0.8350 | None |
| `multiscale_fusion_dedup_best.keras` | `models/multiscale_fusion_dedup_best.keras` | Yes | 31.70 MB | Yes | 4,427,687 | 1,832,676 | `duplicate_aware_split_manifest.csv` | 86.57% | 0.8612 | None |
| `multiscale_fusion_dedup_stage1_best.keras` | `models/multiscale_fusion_dedup_stage1_best.keras` | Yes | 20.59 MB | Yes | 4,427,687 | 378,116 | `duplicate_aware_split_manifest.csv` | 84.37% | 0.8378 | None |

### 3.2 Status of `exp1_efficientnet_b0_best.keras`
- **File Inspection:** Exists, size $19,028,972$ bytes ($18.15$ MB). Total parameters: $4,214,055$; Trainable parameters: $164,484$.
- **Finding:** Checkpoint weights correspond strictly to **Stage 1 (Head Warmup)**. The directory `experiments/exp1_efficientnet_grouped/` contains no Stage 2 logs or history. Stage 2 was never executed for this model.
- **Action:** This checkpoint was not resumed and remains intact as a historical Stage 1 artifact.

---

## 4. Task 3: Dataset & Split Verification

### 4.1 Physical Dataset Inspection
- **Total Physical Images:** Exactly **7,200** valid `.jpg` images.
- **Training Directory (`dataset/Training`):** Exactly **5,600** images.
  - `glioma`: 1,400 images
  - `meningioma`: 1,400 images
  - `notumor`: 1,400 images
  - `pituitary`: 1,400 images
- **Testing Directory (`dataset/Testing`):** Exactly **1,600** images.
  - `glioma`: 400 images
  - `meningioma`: 400 images
  - `notumor`: 400 images
  - `pituitary`: 400 images
- **Non-Image Artifacts:** 0 non-image files or hidden OS cache files were found. All files are readable standard JPEG images.

### 4.2 Grouped Manifest Verification (`training_manifest.csv`)
The manifest at `experiments/duplicate_aware_split/training_manifest.csv` was subjected to rigorous cryptographic and structural validation:
- **Total Records:** Exactly **5,600** rows.
- **Training Records:** Exactly **4,447** rows ($79.41\%$).
- **Validation Records:** Exactly **1,153** rows ($20.59\%$).
- **Class Distributions:**
  - Training: Glioma = 1,116, Meningioma = 1,121, No Tumor = 1,117, Pituitary = 1,093. Total = 4,447.
  - Validation: Glioma = 284, Meningioma = 279, No Tumor = 283, Pituitary = 307. Total = 1,153.
- **File Existence:** 5,600 / 5,600 ($100\%$) referenced paths exist on disk.
- **Cryptographic Hash Verification:** Recomputed SHA-256 hashes for all 5,600 images matched the `sha256` column with **0 mismatches**.
- **Data Leakage Checks:**
  - Cross-Split Exact Duplicate (SHA-256) Leaks: **0**.
  - Cross-Split Perceptual Cluster (`group_id`) Leaks: **0**.
  - Official Test Set (`dataset/Testing`) Contamination: **0** test images appear in the manifest.
- **Data Augmentation Isolation:** Confirmed via `src/data/dataset.py`. Augmentation (random flips, rotation, zoom) is bound strictly to `training=True`. Validation dataset creation sets `training=False` and `augmentation=None`, guaranteeing deterministic, uncorrupted validation evaluations.

### 4.3 Methodological Caveat: Patient-Level Independence
> [!WARNING]
> **Essential Methodological Caveat:**  
> The duplicate-aware manifest clusters images using perceptual difference hashing ($\text{pHash} \le 4$) within each class label. This strictly prevents identical scans and near-identical slices from appearing in both training and validation sets.  
> However, because the public dataset does not provide patient identifiers, study instance UIDs, or DICOM metadata, **patient-level independence cannot be proven or claimed**. It remains possible that different slice angles or contrast phases from the same subject exist without triggering the pHash similarity threshold.

---

## 5. Task 4: Cross-Class Similarity Findings

### 5.1 Inspection of `cross_class_near_matches.csv`
The audit examined `experiments/duplicate_aware_split/cross_class_near_matches.csv`:
- **Candidate Cross-Class Pairs:** 24 pairs with perceptual hash Hamming distance $\le 4$.
- **Split Distribution of Pairs:**
  - 17 pairs have both images in Training.
  - 2 pairs have both images in Validation.
  - 5 pairs cross the split (one image in Training, one in Validation).
- **Class Pairings:**
  - `glioma` vs `meningioma`: 8 pairs
  - `meningioma` vs `pituitary`: 10 pairs
  - `glioma` vs `pituitary`: 6 pairs
  - Pairs involving `notumor`: 0 pairs

### 5.2 Rationale and Validation of Grouping Policy
- **Policy Decision:** During split generation, perceptual grouping was intentionally restricted to **within-class** images. Cross-class pairs were **not** merged into unified groups.
- **Scientific Justification:** Merging images with conflicting pathological labels into a single train/val group would force disparate diagnostic categories to co-segregate, distorting class priors and introducing synthetic label ambiguity.
- **Visual Inspection via Contact Sheet:** A multi-panel contact sheet review of the 5 cross-split candidate pairs demonstrated that these images are **not identical slices or duplicate patients**. Rather, they are different human brains imaged under similar axial orientations possessing similar low-frequency skull outlines (which dominate 8x8 DCT perceptual hashing), while their focal lesion anatomy (e.g. sella turcica mass vs temporal intra-axial lesion) differs significantly.
- **Methodological Conclusion:** No images should be deleted or merged. The presence of these pairs reflects the intrinsic difficulty of cranial MRI geometry under coarse global hashing and reinforces the need for class-aware evaluation.

---

## 6. Task 5: Validation Baseline Reproduction

The grouped multi-scale fusion checkpoint (`models/multiscale_fusion_grouped_best.keras`) and the baseline fine-tuned model (`models/efficientnet_b0_finetuned.keras`) were evaluated on the 1,153 validation manifest images.

### 6.1 Reproduced Performance Summary

| Metric | Multi-Scale Fusion (Grouped Checkpoint) | EfficientNet-B0 (Fine-Tuned Baseline) | Absolute Difference |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy** | **86.3833%** (996 / 1153) | **87.7710%** (1012 / 1153) | -1.3877% |
| **Balanced Accuracy** | **86.0357%** | **87.4382%** | -1.4025% |
| **Macro-F1 Score** | **0.858211** | **0.872802** | -0.014591 |
| **Weighted F1 Score** | **0.859175** | **0.873416** | -0.014241 |
| **Macro Precision** | **0.868733** | **0.886296** | -0.017563 |
| **Weighted Precision** | **0.867310** | **0.884326** | -0.017016 |
| **Macro Recall** | **0.860357** | **0.874382** | -0.014025 |
| **Weighted Recall** | **0.863833** | **0.877710** | -0.013877 |
| **Inference Latency (CPU)** | 52.50 s (45.5 ms/img) | 60.54 s (52.5 ms/img) | -8.04 s |

### 6.2 Per-Class Breakdown

| Class | Model | Precision | Recall | F1-Score | Support |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Glioma** | Multi-Scale Fusion | 91.08% | 86.27% | 88.61% | 284 |
| | EfficientNet-B0 | **94.30%** | **87.32%** | **90.68%** | 284 |
| **Meningioma** | Multi-Scale Fusion | 83.96% | 63.80% | 72.51% | 279 |
| | EfficientNet-B0 | **87.02%** | **64.87%** | **74.33%** | 279 |
| **No Tumor** | Multi-Scale Fusion | 93.08% | 95.05% | 94.06% | 283 |
| | EfficientNet-B0 | **94.54%** | **97.88%** | **96.18%** | 283 |
| **Pituitary** | Multi-Scale Fusion | **79.37%** | 99.02% | **88.12%** | 307 |
| | EfficientNet-B0 | 78.66% | **99.67%** | 87.93% | 307 |

### 6.3 Confusion Matrices

#### Multi-Scale Fusion Grouped Checkpoint:
```
                Predicted Glioma  Predicted Meningioma  Predicted NoTumor  Predicted Pituitary   Total
True Glioma           245                  29                    5                   5            284
True Meningioma        24                 178                   13                  64            279
True No Tumor           0                   4                  269                  10            283
True Pituitary          0                   1                    2                 304            307
Total Predicted       269                 212                  289                 383           1153
```

#### EfficientNet-B0 Fine-Tuned Baseline:
```
                Predicted Glioma  Predicted Meningioma  Predicted NoTumor  Predicted Pituitary   Total
True Glioma           248                  23                    7                   6            284
True Meningioma        15                 181                    9                  74            279
True No Tumor           0                   3                  277                   3            283
True Pituitary          0                   1                    0                 306            307
Total Predicted       263                 208                  293                 389           1153
```

### 6.4 Discrepancy Verification with Historical Records
A comparison between newly reproduced metrics and historical logged training results for `models/multiscale_fusion_grouped_best.keras`:
- Logged Historical Accuracy: `0.8638334778837814`
- Reproduced Validation Accuracy: `0.8638334778837814`
- Logged Historical Macro-F1: `0.8582114316783322`
- Reproduced Validation Macro-F1: `0.8582114316783322`
- **Discrepancy:** **0.0000000000000000** ($0.0\%$). The reproduction is mathematically exact.

---

## 7. Task 6: Model Architecture & Safe Fine-Tuning Plan

### 7.1 Architecture Inspection (`src/models/multiscale_fusion.py`)
- **Backbone:** EfficientNet-B0 (pre-trained on ImageNet).
- **Intermediate Feature Layer:** `block6d_add` (shape: `(None, 7, 7, 192)`). Represents mid-high semantic representations preserving spatial boundary context before final dimensional expansion.
- **Final Feature Layer:** `top_activation` (shape: `(None, 7, 7, 1280)`). Represents deep semantic representations capturing abstract lesion characteristics.
- **Feature Pooling:** Independent `layers.GlobalAveragePooling2D()` applied to each feature map:
  - $\text{GAP}(\text{block6d\_add}) \rightarrow 192\text{-dimensional vector}$
  - $\text{GAP}(\text{top\_activation}) \rightarrow 1280\text{-dimensional vector}$
- **Fusion Layer:** `layers.Concatenate(axis=-1)` concatenates the two vectors into a **$1472$-dimensional fused feature representation**.
- **Classification Head:**
  - `layers.Dropout(rate=0.35, name="fusion_dropout")`
  - `layers.Dense(256, activation="relu", name="fusion_dense")`
  - `layers.Dropout(rate=0.35, name="classifier_dropout")`
  - `layers.Dense(4, activation="softmax", name="fusion_predictions")`
- **Parameter Breakdown:**
  - Total Parameters: **4,427,687** ($16.89$ MB)
  - Trainable Parameters (in current checkpoint): **1,832,676**
  - Non-Trainable Parameters (frozen backbone layers & BN): **2,595,011**

### 7.2 Layer Unfreezing & Batch Normalization Safety
- **Unfreezing Mechanism:** As implemented in `configure_finetuning()`, candidate layers are filtered to exclude `BatchNormalization`. Only the top $N$ convolutional and dense layers are set to `trainable=True`.
- **Batch Normalization Policy:** All 49 `BatchNormalization` layers inside EfficientNet-B0 are explicitly set to `trainable=False`. This is critical when fine-tuning on CPU or with modest batch sizes ($B=32$): allowing moving statistics ($\mu, \sigma^2$) to update would destabilize the pre-trained feature distribution.
- **Compatibility:** Loading the checkpoint with `compile=False` and re-attaching optimizer and loss is completely functional and validated.

### 7.3 Bounded Next Experiment Plan (Fine-Tuning V2)
Rather than training from scratch or risking catastrophic forgetting, a bounded fine-tuning continuation is recommended:

```
[Starting Checkpoint: models/multiscale_fusion_grouped_best.keras]
                               │
               Load Weights with compile=False
                               │
       Freeze all BatchNormalization layers (49 layers)
       Unfreeze Top 20 Non-BN Backbone Layers + Fusion Head
                               │
          Compile: Adam Optimizer (LR = 5e-06)
          Loss: Sparse Categorical Cross-Entropy (Class-Weighted)
          Class Weights: Glioma=1.10, Meningioma=1.25, NoTumor=1.00, Pituitary=0.90
                               │
                Train for Maximum 5 Epochs
          Callbacks:
            * MacroF1Checkpoint (Best Val Macro-F1 -> models/multiscale_fusion_finetuned_v2.keras)
            * EarlyStopping (Monitor val_macro_f1, Patience=2)
            * ReduceLROnPlateau (Monitor val_loss, Factor=0.5, Patience=1, Min LR=1e-07)
            * BackupCheckpoints (models/checkpoints/multiscale_fusion_v2_epoch_{epoch:02d}.keras)
```

#### Detailed Experiment Specifications:
- **Starting Checkpoint:** `models/multiscale_fusion_grouped_best.keras`
- **Training Manifest:** `experiments/duplicate_aware_split/training_manifest.csv` (4,447 train, 1,153 val)
- **Batch Size:** 32 (consistent with Stage 1 and Stage 2)
- **Initial Learning Rate:** $5 \times 10^{-6}$ (gentle rate to refine weights without disrupting the fused representation)
- **Learning Rate Floor:** $1 \times 10^{-7}$
- **Maximum Epochs:** **5 epochs**
- **Early Stopping:** Trigger if validation Macro-F1 fails to improve for 2 consecutive epochs.
- **Selection Criterion:** Validation Macro-F1 (`save_best_only=True`).
- **Per-Class Safeguard:** Checkpoint acceptance requires:
  1. Validation Macro-F1 $\ge 0.8582$
  2. Meningioma Recall $\ge 65.00\%$ (up from $63.80\%$)
  3. Glioma Recall $\ge 85.00\%$ (safeguarding against precision-recall trade-off collapse)
- **Output Checkpoint Path:** `models/multiscale_fusion_finetuned_v2.keras` (never overwrites the existing baseline!)
- **Output Metrics Path:** `experiments/pretraining_audit/multiscale_fusion_v2_results.json`
- **Output History Path:** `experiments/pretraining_audit/multiscale_fusion_v2_history.csv`
- **Crash Recovery Protocol:** Per-epoch backup checkpoints saved to `models/checkpoints/` with automated state resumption via an epoch pointer.

---

## 8. Task 7: Official Test-Set Protection

### 8.1 Isolation Verification
- **Audit Rule Adherence:** During this Phase 1 audit, `dataset/Testing` (1,600 images) was **not accessed, preprocessed, or evaluated**.
- **No Test-Based Optimization:** No hyperparameter selection, threshold tuning, or model selection was guided by test-set performance.

### 8.2 Historical Test-Set Usage Analysis
- **Historical Records:**
  - `reports/final_metrics.json`: Records historical test evaluation for initial `efficientnet_b0.keras` (Accuracy: $87.50\%$, Macro-F1: $0.8699$).
  - `experiments/results.csv`: Records test evaluation for `efficientnet_b0_finetuned.keras` (Accuracy: $83.38\%$, Macro-F1: $0.8271$).
  - `experiments/final_results.json`: Records a single locked post-training test evaluation for `models/multiscale_fusion_grouped_best.keras` (Accuracy: $80.63\%$, Macro-F1: $0.7965$).
- **Distinction Between Leakage and Post-Hoc Evaluation:** These historical test evaluations were performed after model checkpoints were finalized and frozen. However, to guarantee scientific validity for all subsequent work, the official test set remains locked until the Phase 2 training and selection cycle is complete.

---

## 9. Verification Matrix

| Check / Requirement | Status | Evidence / Notes |
| :--- | :---: | :--- |
| **Project & Environment Inspected** | **PASSED** | Python 3.13.7, TF 2.21.0, 12 CPU cores, CPU fallback with oneDNN. |
| **All Checkpoints Inventoried** | **PASSED** | 14 checkpoints inventoried; 13 loadable; 1 root-referenced; `exp1` is Stage 1 only. |
| **Physical Dataset Count (7,200)** | **PASSED** | Exactly 5,600 train + 1,600 test images across 4 balanced classes. |
| **Manifest Record Count (5,600)** | **PASSED** | Exactly 4,447 train + 1,153 val records in `training_manifest.csv`. |
| **Cryptographic Hash Consistency** | **PASSED** | 5,600 / 5,600 SHA-256 hashes match physical image bytes on disk. |
| **Zero Split Leakage (Hash & pHash)** | **PASSED** | 0 SHA-256 duplicates and 0 `group_id` clusters cross train and val splits. |
| **Zero Test Images in Manifest** | **PASSED** | 0 images from `dataset/Testing` present in the manifest. |
| **Augmentation Isolation** | **PASSED** | Augmentation isolated strictly to `training=True`; validation pipeline deterministic. |
| **Cross-Class Matches Analyzed** | **PASSED** | 24 pairs analyzed; within-class grouping justified; contact sheet verified morphology. |
| **Validation Baseline Reproduced** | **PASSED** | Grouped fusion reproduced at 86.38% acc, 0.8582 macro-F1 (0.0000% error vs log). |
| **Baseline Model Evaluated on Split** | **PASSED** | `efficientnet_b0_finetuned.keras` evaluated on identical 1,153 val images (87.77% acc). |
| **Architecture Analyzed** | **PASSED** | Intermediate `block6d_add` + deep `top_activation` -> 1472-d vector; BN frozen. |
| **Safe Training Plan Designed** | **PASSED** | 5-epoch max bounded fine-tuning continuation; Macro-F1 checkpointing; class weights. |
| **Official Test Set Protected** | **PASSED** | `dataset/Testing` untouched during audit; locked until post-training. |

---

## 10. Audit Artifacts Produced

The following audit artifacts have been created in `experiments/pretraining_audit/`:

1. [`audit_report.md`](file:///x:/Mini%20Project/brain-tumor-mri-classification/experiments/pretraining_audit/audit_report.md): This comprehensive academic audit report.
2. [`checkpoint_inventory.csv`](file:///x:/Mini%20Project/brain-tumor-mri-classification/experiments/pretraining_audit/checkpoint_inventory.csv): Detailed tabular inventory of all 14 checkpoints, file sizes, parameter counts, loadability, and historical metrics.
3. [`validation_baseline.json`](file:///x:/Mini%20Project/brain-tumor-mri-classification/experiments/pretraining_audit/validation_baseline.json): Machine-readable newly reproduced baseline evaluation results, per-class metrics, confusion matrices, and discrepancy calculations.
4. [`next_experiment_plan.json`](file:///x:/Mini%20Project/brain-tumor-mri-classification/experiments/pretraining_audit/next_experiment_plan.json): Complete JSON execution blueprint for the proposed Fine-Tuning V2 experiment.
