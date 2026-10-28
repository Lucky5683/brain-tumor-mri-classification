# Experiment and Artifact Inventory

## 1. Overview & Purpose

This document catalogs all machine learning checkpoints, training manifests, training histories, evaluation reports, and cross-validation artifacts in the repository.

> [!IMPORTANT]
> **Strict Verification Standard**:
> - All performance metrics are explicitly categorized by dataset partition: **Training Set**, **Grouped Validation Set** ($N = 1,153$), or **Official Independent Test Set** ($N = 1,600$).
> - Historical validation scores are never promoted to or conflated with final test performance.
> - Artifacts marked as *Smoke Tests* are clearly distinguished from completed research experiments.

---

## 2. Model Checkpoint Inventory

| Checkpoint Path | Architecture Type | File Size | Parameter Count | Evaluation Partition | Verified Metric(s) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `baseline_cnn.keras` | 4-Block Custom CNN | 5.16 MB | 421,828 | Official Test Set ($N=1,600$) | Accuracy: **68.75%**, Macro-F1: **0.6624** | Evaluated (Baseline) |
| `efficientnet_b0.keras` | EfficientNet-B0 (Transfer) | 19.01 MB | 4,213,972 | Official Test Set ($N=1,600$) | Accuracy: **87.50%**, Macro-F1: **0.8699** | Evaluated (Reference) |
| `models/efficientnet_b0_finetuned.keras` | EfficientNet-B0 (Fine-Tuned) | 19.01 MB | 4,213,972 | Official Test Set ($N=1,600$) | Accuracy: **87.50%**, Macro-F1: **0.8699** | Evaluated (Fine-Tuned) |
| `models/multiscale_fusion_grouped_best.keras` | Multi-Scale Feature Fusion | 33.80 MB | 4,427,687 | Official Test Set ($N=1,600$) | Accuracy: **80.62%**, Macro-F1: **0.7965** | Evaluated (Review III) |
| `experiments/multiseed_fusion/seed_42/best_model.keras` | Multi-Scale Fusion (Seed 42) | 33.80 MB | 4,427,687 | Grouped Validation ($N=1,153$) | Accuracy: **88.73%**, Macro-F1: **0.8859** | Multi-Seed Run |
| `experiments/multiseed_fusion/seed_123/best_model.keras` | Multi-Scale Fusion (Seed 123) | 33.80 MB | 4,427,687 | Grouped Validation ($N=1,153$) | Completed Run Checkpoint | Multi-Seed Run |
| `experiments/multiseed_fusion/seed_456/best_model.keras` | Multi-Scale Fusion (Seed 456) | 33.80 MB | 4,427,687 | Grouped Validation ($N=1,153$) | Completed Run Checkpoint | Multi-Seed Run |
| `experiments/spatial_fusion/meningioma_weighted_50ep_seed42/best_model.keras` | Spatial Attention Fusion | 33.81 MB | 4,428,295 | Grouped Validation ($N=1,153$) | Val Accuracy: **92.54%**, Val Macro-F1: **92.41%** | **Protected Candidate** (Pending Final Test Eval) |

---

## 3. Detailed Inspection of Selected Spatial Fusion Artifacts

**Directory:** `experiments/spatial_fusion/meningioma_weighted_50ep_seed42/`

### 3.1 Primary Model Checkpoint (`best_model.keras`)
- **File Size:** 33,806,024 bytes (~33.8 MB)
- **Last Modified:** 2026-10-23
- **Preservation Status:** **PROTECTED**. Under Non-Negotiable Rule 6, this checkpoint must never be overwritten, retrained, or altered.

### 3.2 Metrics and Training History Records
The experiment directory contains two separate JSON status records:

1. **`best_metrics.json` (Epoch 50 Final Evaluation)**:
   ```json
   {
     "epoch": 50,
     "val_loss": 0.207039645934449,
     "val_accuracy": 0.9254119992256165,
     "val_macro_f1": 0.9241270271576402,
     "val_weighted_f1": 0.925232709292881,
     "val_macro_precision": 0.9254395347313045,
     "val_macro_recall": 0.9234907086884877,
     "per_class_f1": {
       "glioma": 0.8879668049792531,
       "meningioma": 0.887719298245614,
       "notumor": 0.9632352941176471,
       "pituitary": 0.9575867012880466
     }
   }
   ```
2. **`results.json` (Epoch 31 Interim Checkpoint)**:
   ```json
   {
     "epoch": 31,
     "val_loss": 0.233630650639534,
     "val_accuracy": 0.9167389273643494,
     "val_macro_f1": 0.91508215467453
   }
   ```
3. **`history.json` (Full 50-Epoch Trajectory)**:
   - Contains arrays of length 50 for `loss`, `accuracy`, `val_loss`, and `val_accuracy`.
   - Records smooth convergence from initial epoch (`val_loss: 0.542`, `val_accuracy: 0.812`) to epoch 50 (`val_loss: 0.207`, `val_accuracy: 0.9254`).

### 3.3 Explanation of Discrepancy & Verification Status
- **Why do `results.json` and `best_metrics.json` differ?**
  `results.json` was serialized at epoch 31 when an early checkpoint criterion was reached during training. As training continued across the full 50-epoch schedule with class weighting, the model attained superior performance at epoch 50, recorded in `best_metrics.json`.
- **Are these results test-set results?**
  **NO.** Both numbers (91.67% and 92.54%) are strictly **validation-set** metrics evaluated on the 1,153-image grouped validation partition of `experiments/duplicate_aware_split/training_manifest.csv`. Independent evaluation on the locked 1,600-image `dataset/Testing` split has not yet been executed for this checkpoint.

---

## 4. Dataset Partition & Split Manifests

### 4.1 Duplicate-Aware Split Manifest
- **File:** `experiments/duplicate_aware_split/training_manifest.csv`
- **Total Images:** 5,600
- **Training Subset:** 4,447 images (79.41%)
- **Validation Subset:** 1,153 images (20.59%)
- **Leakage Status:** **Verified 0% Leakage**. No exact SHA-256 duplicate or perceptual hash ($d \le 6$) group spans across training and validation splits.
- **Class Breakdown:**
  - `glioma`: 1,114 train / 286 val
  - `meningioma`: 1,111 train / 289 val
  - `notumor`: 1,109 train / 291 val
  - `pituitary`: 1,113 train / 287 val

### 4.2 Official Test Split
- **Directory:** `dataset/Testing/`
- **Total Images:** 1,600 (400 per class: glioma, meningioma, notumor, pituitary).
- **Integrity:** Kept locked during all training routines, checkpoint selections, and hyperparameter tuning.

---

## 5. Five-Fold Cross-Validation Artifacts

**Directory:** `experiments/five_fold_cv/`

### 5.1 Existing Manifests
- **`fold_manifest.csv`**: Contains complete 5,600-image fold assignments (`fold_0` through `fold_4`), constructed using perceptual hash clustering to ensure zero inter-fold slice leakage.
- **`fold_class_counts.csv`**: Verifies balanced class distribution across all 5 folds (~1,120 images per fold).

### 5.2 Existing Execution Artifacts & Verification
- **`smoke_tests/`**: Contains dry-run logs and single-batch checkpoints created to verify the training loop, learning rate scheduling, and tensor shapes.
- **`custom_cnn/fold_1/`**: Contains partial single-fold test output for the baseline architecture.
- **Current Completion Status:** **INCOMPLETE (Manifests & Smoke Tests Only)**.
  - The repository does **NOT** contain completed 5-fold cross-validation runs or aggregate fold metrics for either EfficientNet-B0 or the Spatial Fusion architecture.
  - Full 5-fold execution is designated as a separate, subsequent phase.

---

## 6. Saliency and Saliency Verification Artifacts

| Saliency Asset | Path | Description | Layer Target |
| :--- | :--- | :--- | :--- |
| Saliency Script | `src/explainability/gradcam.py` | Grad-CAM (first-order gradients) | `top_activation` |
| Saliency++ Script | `src/explainability/gradcam_plus_plus.py` | Grad-CAM++ (weighted second/third-order) | `top_activation` |
| CLI Generator | `gradcam.py` | Command-line image saliency generation | Auto-detected terminal conv |
| Test Overlays | `gradcam_examples/` | Diagnostic overlays across all 4 classes | Confirmed focused on anatomical tumors |
| Saliency Investigation | `scripts/investigate_spatial_gradcam.py` | Resolution & layer gradient propagation audit | Verified compatibility with spatial attention block |

---

## 7. Automated Test Suite Artifacts

- **Path:** `tests/test_pipeline.py` & `tests/test_spatial_fusion.py`
- **Total Tests:** 13 unit and integration tests.
- **Recent Execution Result:** **13 PASSED** in 181.74s (Windows CPU execution).
- **Scope:**
  - Class index ordering consistency across modules.
  - Image preprocessing pipelines and shape invariants ($224 \times 224 \times 3$).
  - Model deserialization and prediction shapes ($B \times 4$).
  - Grad-CAM and Grad-CAM++ output generation and range constraints ($[0, 1]$).
  - Temperature scaling calibration math and probability conservation ($\sum p_i = 1$).
