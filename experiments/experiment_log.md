# Experiment Log: Brain Tumor MRI Classification V2

Structured registry of controlled training, fine-tuning, and evaluation experiments.

---

## Controlled Model Selection Protocol (Phase 2 & Phase 7)
1. **Model Selection**: Conducted strictly using the 20% stratified validation set ($1,120$ images in `dataset/Training`).
2. **Test Set Isolation**: The official test set ($1,600$ images in `dataset/Testing`) is strictly reserved for single-pass evaluation of the selected candidate model.
3. **Reproducibility**: All experiments use `seed=42` with stratified class representation.

---

## Logged Experiments

### Experiment EXP-01-CNN: Custom CNN Baseline (V1)
- **Date**: 2026-09-07
- **Architecture**: 4-Block Sequential ConvNet + GAP + Dropout (0.3/0.4) + Dense(128) + Softmax(4)
- **Hyperparameters**: Image Size $224 \times 224$, Batch Size 32, Adam lr=$10^{-3}$, 30 Epochs
- **Augmentation**: Horizontal flip, Rotation ($5\%$), Zoom ($10\%$)
- **Class Weights**: None (Unweighted)
- **Validation Accuracy**: $72.50\%$
- **Test Performance**:
  - Test Accuracy: **$68.75\%$** ($1,100 / 1,600$)
  - Test Macro-F1: **$0.6624$**
  - Glioma Recall: **$41.25\%$** ($165 / 400$)
  - Glioma Precision: **$88.24\%$**
- **Notes**: High false negative rate for glioma and meningioma; healthy brain (`notumor`) easily separated ($97.75\%$).

---

### Experiment EXP-02-EFFNET-V1: EfficientNet-B0 Frozen Feature Extractor (V1)
- **Date**: 2026-09-07
- **Architecture**: ImageNet Pretrained EfficientNet-B0 (Frozen Backbone) + GAP Head + Dropout (0.3) + Dense(128) + Dropout (0.3) + Softmax(4)
- **Parameters**: 4,214,055 total; 164,484 trainable; 4,049,571 frozen
- **Hyperparameters**: Adam lr=$10^{-3}$, 20 Epochs, Batch Size 32
- **Validation Performance**:
  - Validation Accuracy: **$91.79\%$**
  - Validation Loss: $0.2096$
- **Test Performance**:
  - Test Accuracy: **$87.50\%$** ($1,400 / 1,600$)
  - Test Macro-F1: **$0.8699$**
  - ROC-AUC (Macro OvR): **$0.9718$**
  - Glioma Recall: **$65.75\%$** ($263 / 400$)
  - Meningioma Recall: **$85.25\%$** ($341 / 400$)
  - No Tumor Recall: **$100.00\%$** ($400 / 400$)
  - Pituitary Recall: **$99.00\%$** ($396 / 400$)
- **Key Finding**: Glioma recall remains the primary bottleneck ($65.75\%$), with 90 cases misdiagnosed as meningioma and 34 as notumor.

---

### Experiment EXP-03-EFFNET-V2-TWOPASS: Two-Stage Fine-Tuning (Unweighted)
- **Status**: Queued / In Progress
- **Architecture**: Two-stage transfer learning. Stage 1 head warmup (lr=$10^{-3}$), Stage 2 backbone unfreeze (lr=$10^{-5}$).
- **Target Metrics**: Validation accuracy, validation macro-F1, validation glioma recall.

---

### Experiment EXP-04-EFFNET-V2-WEIGHTED: Two-Stage Fine-Tuning + Balanced Class Weighting
- **Status**: Queued / In Progress
- **Architecture**: Two-stage transfer learning with scikit-learn computed balanced training class weights.
- **Objective**: Improve glioma recall above the baseline $65.75\%$ without sacrificing overall precision.
