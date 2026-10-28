# Training Audit: Multi-Scale Fusion and Baseline Training Strategies

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Inspected Training Scripts & Logs:**
- `scripts/train_multiscale_fusion_multiseed.py`
- `scripts/train_multiscale_fusion_grouped.py`
- `scripts/train_efficientnet.py`
- `train_cnn.py`
- `experiments/multiseed_fusion/final_summary.json`
- `experiments/multiseed_fusion/seed_42/results.json`
- `experiments/multiseed_fusion/seed_123/results.json`
- `experiments/multiseed_fusion/seed_2026/results.json`

---

## 1. Comparative Training Configuration Matrix

| Parameter / Protocol | Baseline Custom CNN (`train_cnn.py`) | EfficientNet-B0 V2-E1 (`scripts/train_efficientnet.py`) | Multi-Scale Fusion Base (`scripts/train_multiscale_fusion_grouped.py`) | Multi-Seed Fusion (`scripts/train_multiscale_fusion_multiseed.py`) |
| :--- | :--- | :--- | :--- | :--- |
| **Split Manifest / Mechanism** | Random 80/20 Stratified (`random_seed=42`) | Random 80/20 Stratified (`random_seed=42`) | Duplicate-Aware Grouped Manifest | Duplicate-Aware Grouped Manifest |
| **Training Images ($N_{train}$)** | 4,480 images | 4,480 images | 4,447 images | 4,447 images |
| **Validation Images ($N_{val}$)** | 1,120 images | 1,120 images | 1,153 images | 1,153 images |
| **Official Test Set ($N_{test}$)** | Untouched during training | Untouched during training | Untouched during training | Untouched during training |
| **Batch Size** | 32 | 32 | 32 | 32 |
| **Stage 1 (Head Warmup)** | N/A (Trained from scratch) | 5 epochs, Adam $\text{lr}=1\times 10^{-3}$ | 5 epochs, Adam $\text{lr}=1\times 10^{-3}$ | Inherited from base checkpoint |
| **Stage 2 (Fine-Tuning)** | 15 epochs, Adam $\text{lr}=1\times 10^{-3}$ | 4 epochs, Adam $\text{lr}=1\times 10^{-5}$ | 4 epochs, Adam $\text{lr}=1\times 10^{-5}$ | 8 to 12 epochs, Adam $\text{lr}=1\times 10^{-5}$ |
| **Unfrozen Layers** | All layers trainable | Top 20 non-BatchNorm layers | Top 20 non-BatchNorm layers | Top 20 non-BatchNorm layers |
| **BatchNorm Behavior** | N/A (No BatchNorm in CNN) | Strictly frozen (`layer.trainable=False`) | Strictly frozen (`layer.trainable=False`) | Strictly frozen (`layer.trainable=False`) |
| **LR Scheduler** | None | ReduceLROnPlateau (`factor=0.5, pat=2`) | ReduceLROnPlateau (`factor=0.5, pat=2`) | Custom `MacroF1ReduceLR` (`factor=0.5, pat=2`) |
| **Early Stopping Monitor** | Validation loss (`patience=5`) | Validation Macro-F1 (`patience=3`) | Validation Macro-F1 (`patience=3`) | Custom `MacroF1EarlyStopping` (`pat=3` after ep 8) |
| **Checkpoint Selection Metric**| Lowest Validation Loss | **Highest Validation Macro-F1** | **Highest Validation Macro-F1** | **Highest Validation Macro-F1** (tie: val loss) |
| **Loss Function** | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy | Sparse Categorical Crossentropy |
| **Class Weighting** | Unweighted | Tested unweighted vs. error-based | Unweighted | Unweighted |
| **Training Augmentation** | Flip, Rotation 5%, Zoom 8%, Shift 5% | Flip, Rotation 5%, Zoom 8%, Shift 5% | Flip, Rotation 5%, Zoom 8%, Shift 4% | Flip, Rotation 5%, Zoom 8%, Shift 4% |
| **Validation Augmentation** | **Disabled** (Strictly deterministic) | **Disabled** (Strictly deterministic) | **Disabled** (Strictly deterministic) | **Disabled** (Strictly deterministic) |

---

## 2. Split Protocol and Data Integrity

### 2.1 Augmentation Application Timing
- **Audit Verification:** Augmentation layers are instantiated as a `tf.keras.Sequential` block and mapped strictly onto `train_ds` via `dataset.map(lambda x, y: (augmentation(x, training=True), y))` **after** splitting.
- **Result:** Validation and test datasets are decoded with pure bilinear resizing and no augmentation. There is zero augmentation leakage into the validation or testing pipelines.

### 2.2 Manifest Determinism and Hash Verification
In `scripts/train_multiscale_fusion_multiseed.py` (lines 151–193), runtime assertions verify every image's SHA-256 hash against `training_manifest.csv`:
- Exact duplicate hash overlap between train and validation: **0**.
- Similarity group overlap between train and validation: **0**.
- Official test images present in training manifest: **0**.

---

## 3. Training Dynamics and Learning Curve Audit

### 3.1 Multi-Seed Fine-Tuning Analysis (Seeds 42, 123, 2026)

Detailed inspection of epoch-by-epoch validation logs (`results.json` for each seed) reveals distinct convergence patterns:

#### Seed 42:
- **Trajectory:** Val loss steadily decreased from 0.4171 (epoch 1) down to a global minimum of **0.2973** at **epoch 11** (Val Acc: **88.73%**, Macro-F1: **88.59%**).
- **Overfitting Onset:** At epoch 12, validation loss jumped to **0.3296**, validation accuracy dropped to 87.68%, and meningioma recall dropped from 76.34% to 69.18%.
- **Checkpoint Selected:** Epoch 11 was correctly selected as the best model checkpoint.

#### Seed 123:
- **Trajectory:** Val loss fluctuated between 0.3859 and 0.3213, reaching its peak validation Macro-F1 of **87.65%** at **epoch 10** (Val Acc: **87.86%**, Val Loss: 0.3263).
- **Post-Peak Decay:** At epoch 12, accuracy dropped to 86.90% and Macro-F1 dropped to 86.61%.
- **Checkpoint Selected:** Epoch 10 was correctly selected.

#### Seed 2026:
- **Trajectory:** Reached lowest validation loss of **0.3219** and peak Macro-F1 of **87.79%** at **epoch 9** (Val Acc: **87.94%**).
- **Post-Peak Decay:** From epoch 10 to 12, loss rose to 0.3331–0.3475 and meningioma recall dropped from 73.48% to 65.59%.
- **Checkpoint Selected:** Epoch 9 was correctly selected.

### 3.2 Key Training Diagnoses

1. **Epoch Ceiling:** In all three seeds, peak generalization occurs strictly between **epochs 9 and 11**. Training beyond 11 epochs leads to parameter over-specialization and loss divergence.
2. **Meningioma Instability:** Across all runs, meningioma recall is the most volatile metric. In Seed 42, it jumped from 65.23% (epoch 6) to 76.34% (epoch 11), then plunged back to 69.18% (epoch 12). This volatility reflects the subtle boundary features of meningioma and conflicting training signals from near-duplicate slices.
3. **No-Tumor and Pituitary Plateau:** Both classes consistently hit $\ge 94\%$ and $\ge 98\%$ recall by epoch 3 and remain flat throughout fine-tuning, confirming that optimization capacity is primarily contested between glioma and meningioma.

---

## 4. Class Weighting vs. Unweighted Training

In `scripts/train_efficientnet.py`, Experiment V2-E2 tested error-based class weighting:
```python
W_k = (1 - Recall_k) / sum(1 - Recall_j) * C
```
- **Hypothesis:** Amplifying glioma weight would counteract low glioma recall.
- **Empirical Outcome:** Reweighting boosted glioma recall slightly on the training set, but caused a sharp increase in false positives on meningioma and no-tumor, degrading overall validation Macro-F1 compared to the unweighted run.
- **Finding:** Simple static class weighting does not address the underlying spatial ambiguity or label noise between glioma and meningioma; focal loss or margin-based ranking losses are more appropriate.

---

## 5. Checkpoint Selection & Evaluation Integrity

- **Evaluation Protocol:** Checkpoint selection was performed **strictly on the validation set** using validation Macro-F1.
- **Test Set Protection:** The official test set (`dataset/Testing`, $N=1,600$) was **never** evaluated inside any training loop, callback, or hyperparameter selection routine. It was loaded only once after training completion for the final locked evaluation report (`scripts/evaluate_final_multiseed.py`).
- **Conclusion:** Training execution adhered to standard leakage-prevention protocols.
