# Experiment Plan: Evidence-Based Next Experiments for Model Improvement

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Target:** Maximize generalization performance on the verified duplicate-aware split; establish empirical feasibility of the aspirational $\ge 95\%$ accuracy goal.

---

## 1. Experimental Protocol and Model Selection Rules

To maintain scientific integrity and prevent test-set contamination:

1. **Iterative Model Selection:** All candidate architectures, loss functions, and hyperparameters must be compared **strictly on the fixed duplicate-aware validation manifest** (`experiments/duplicate_aware_split/training_manifest.csv`, $N = 1,153$).
2. **Official Test Set Reservation:** The official 1,600-image test set (`dataset/Testing`) must **NOT** be evaluated during exploration. It is evaluated only once on the single, validation-selected model as a final confirmatory check.
3. **Primary Selection Metric:** **Validation Macro-F1** (with Validation Balanced Accuracy and Glioma/Meningioma Recall as secondary constraints). Due to class confusion, accuracy alone is insufficient.
4. **Multi-Seed Rule:** Any candidate showing positive improvement on Seed 42 must be evaluated across three random seeds (42, 123, 2026) to verify that gains exceed stochastic training variance ($\sigma \approx 0.40\%$).

---

## 2. Ranked Experimental Suite

```
Rank 1: True Spatial Multi-Scale Feature Fusion (Stage 3 + Stage 7)
   │
Rank 2: Classification Head Simplification & Regularization
   │
Rank 3: Focal Loss / Margin-Based Loss for Glioma-Meningioma Confusion
   │
Rank 4: Clean Manifest Ablation (Quarantine 24 Conflicting Cross-Class Pairs)
   │
Rank 5: Baseline EfficientNet-B0 Benchmark on Grouped Split
```

---

### Experiment 1: True Spatial Multi-Scale Feature Fusion (Stage 3 + Stage 7)

- **Hypothesis:** Fusing a high-resolution feature map from stage 3 (`block3b_add`, shape $28 \times 28 \times 40$) with the semantic feature map (`top_activation`, shape $7 \times 7 \times 1280$) will supply genuine spatial boundary cues that current $7 \times 7$ concatenation lacks, improving glioma and meningioma recall.
- **Source Files to Modify:**
  - `src/models/multiscale_fusion.py` (update `build_multiscale_fusion_model()`)
  - `src/explainability/gradcam.py` (ensure multi-scale gradient tracing maps to top features)
- **Initialization Checkpoint:** `models/efficientnet_b0_finetuned.keras` (or ImageNet backbone)
- **Data Split & Seed:** `experiments/duplicate_aware_split/training_manifest.csv` ($N_{train}=4447, N_{val}=1153$), Seed 42.
- **Parameters to Change:**
  - Intermediate extraction layer: `block3b_add` ($28 \times 28 \times 40$) instead of `block6d_add` ($7 \times 7 \times 192$).
  - Spatial projection: Apply $1 \times 1$ Conv2D (reducing 40 to 64 channels), followed by Spatial Attention or Channel Attention before GlobalAveragePooling2D.
- **Parameters to Hold Constant:**
  - Input resolution: $(224, 224, 3)$
  - Batch size: 32
  - Optimizer: Adam, Stage 1 $\text{lr}=1\times 10^{-3}$, Stage 2 $\text{lr}=1\times 10^{-5}$
  - Epoch bounds: 10–12 epochs with `MacroF1EarlyStopping`
  - Augmentation: Conservative flip, zoom 8%, rotation 5%, shift 4%
- **Validation Metrics:** Val Macro-F1, Val Accuracy, Glioma Recall, Meningioma Recall.
- **Success Criterion:** Validation Macro-F1 $\ge 89.20\%$ (an improvement of $+0.60\%$ over current best 88.59%) with Glioma Recall $\ge 86.0\%$.
- **Rejection Criterion:** Validation Macro-F1 $< 88.40\%$ or severe training instability.
- **Estimated Computational Cost:** $\approx 45\text{--}55$ minutes per seed on CPU (~8–10 minutes on CUDA GPU).

---

### Experiment 2: Head Simplification & Regularization

- **Hypothesis:** The current 256-unit dense head with 378k parameters overfits the 4,447-image training set. Reducing the dense bottleneck to 128 units and adding LayerNormalization (or L2 weight decay $1\times 10^{-4}$) will reduce validation loss divergence and stabilize post-peak epochs.
- **Source Files to Modify:**
  - `src/models/multiscale_fusion.py` (lines 48–57)
- **Initialization Checkpoint:** `models/multiscale_fusion_grouped_best.keras`
- **Data Split & Seed:** Grouped validation manifest, Seed 42.
- **Parameters to Change:**
  - Head architecture: `Concat(1472) -> LayerNormalization() -> Dropout(0.25) -> Dense(128, relu, kernel_regularizer=L2(1e-4)) -> Dropout(0.25) -> Dense(4, softmax)`.
  - Head parameter count drops from 378,116 to **190,084** ($-50\%$ reduction).
- **Parameters to Hold Constant:** Backbone weights, learning rate schedule, data pipeline.
- **Success Criterion:** Validation loss at epoch 12 remains lower than epoch 10 without divergence; Validation Macro-F1 $\ge 88.80\%$.
- **Rejection Criterion:** Underfitting (Val Macro-F1 $< 87.50\%$).
- **Estimated Cost:** $\approx 40$ minutes on CPU.

---

### Experiment 3: Focal Loss for Diagnostic Ambiguity

- **Hypothesis:** Standard cross-entropy treats all errors equally, causing the optimizer to over-index on easy No-Tumor and Pituitary samples while struggling with hard Glioma/Meningioma boundaries. A multi-class Focal Loss ($\gamma = 2.0, \alpha = [1.2, 1.2, 0.8, 0.8]$) will down-weight easy confident examples and concentrate gradients on hard tumor margins.
- **Source Files to Modify:**
  - `scripts/train_multiscale_fusion_multiseed.py` (loss function definition)
- **Initialization Checkpoint:** `models/multiscale_fusion_grouped_best.keras`
- **Data Split & Seed:** Grouped validation manifest, Seed 42.
- **Parameters to Change:**
  - Loss: Multi-Class Categorical Focal Loss with focusing parameter $\gamma = 2.0$.
- **Parameters to Hold Constant:** Learning rates, model architecture, data pipeline.
- **Success Criterion:** Glioma recall increases from $84.15\%$ to $\ge 88.0\%$ on validation without degrading No Tumor recall below $92.0\%$.
- **Rejection Criterion:** Overall validation Macro-F1 drops below $87.80\%$.
- **Estimated Cost:** $\approx 40$ minutes on CPU.

---

### Experiment 4: Conflicting Cross-Class Duplicate Quarantine

- **Hypothesis:** The 24 near-duplicate image pairs with contradictory class labels (e.g., `Tr-gl_1046.jpg` vs `Tr-me_1285.jpg` with $d = 2$) inject contradictory gradients. Quarantining these 24 ambiguous images from the training manifest will eliminate ground-truth label noise.
- **Source Files to Modify:**
  - `experiments/duplicate_aware_split/training_manifest_clean.csv` (exclude 24 flagged paths)
- **Initialization Checkpoint:** Same base checkpoint.
- **Data Split & Seed:** Cleaned manifest ($N_{train} = 4,423, N_{val} = 1,153$).
- **Success Criterion:** Reduced mutual false-positive rate between Glioma and Meningioma.
- **Rejection Criterion:** No measurable change in validation Macro-F1 ($\Delta < 0.1\%$).
- **Estimated Cost:** $\approx 40$ minutes on CPU.

---

### Experiment 5: Baseline EfficientNet-B0 Re-evaluation on Grouped Split

- **Hypothesis:** To establish a scientifically fair comparison, the frozen-backbone baseline (`efficientnet_b0.keras`) must be trained and evaluated on the exact same 4,447 / 1,153 grouped split used for the fusion model, rather than relying on historical random-split numbers.
- **Source Files to Modify:**
  - `scripts/train_efficientnet.py` (pass `--split-manifest`)
- **Success Criterion:** Produces a verified, apples-to-apples baseline delta $\Delta = \text{Fusion} - \text{Baseline}$ on the grouped validation set.
- **Estimated Cost:** $\approx 35$ minutes on CPU.

---

## 3. Realism Check: The Path to $\ge 95\%$ Accuracy

Can this model reach $\ge 95\%$ accuracy on this benchmark?

| Factor | Current State | Required for $\ge 95\%$ Accuracy | Feasibility Assessment |
| :--- | :--- | :--- | :--- |
| **Error Budget ($N=1,600$)** | 228 errors (Fusion) / 200 errors (Baseline) | $\le 80$ total errors across all classes | Requires eliminating $>60\%$ of all current errors |
| **Glioma Error Count** | 139 errors (Recall = 65.25%) | $\le 30$ errors (Recall $\ge 92.5\%$) | **Unlikely with single 2D slices alone** without 3D volumetric context |
| **Meningioma Error Count** | 78 errors (Recall = 80.50%) | $\le 30$ errors (Recall $\ge 92.5\%$) | Feasible with focal loss and true multi-scale spatial features |
| **No-Tumor / Pituitary** | 6 errors (No Tumor) / 5 errors (Pituitary) | Maintain $\le 15$ errors | Already achieved ($\ge 98.5\%$) |
| **Label Noise Floor** | 24 verified contradictory pairs in training | Manual radiologist curation required | Irreducible noise in public Kaggle test set |

### Scientific Verdict:
Aiming for 95% accuracy is an excellent aspirational benchmark. However:
1. Reaching $\approx 89\text{--}91\%$ test accuracy is realistic through true multi-scale spatial fusion, focal loss, and head regularization.
2. Reaching genuine $\ge 95\%$ clinical accuracy requires **volumetric 3D MRI inputs** (or multi-parametric T1+T2+FLAIR sequences) and curated clinical labels with confirmed histopathology, rather than single-slice 2D JPEG benchmarks.
