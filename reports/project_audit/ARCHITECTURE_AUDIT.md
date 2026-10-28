# Architecture Audit: Multi-Scale Feature Fusion vs. Baselines

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Inspected Source Files:**
- `src/models/multiscale_fusion.py`
- `src/models/efficientnet.py`
- `src/models/custom_cnn.py`
- `src/models/compat.py`
- `app.py`
- `scripts/train_multiscale_fusion_multiseed.py`

---

## 1. Multi-Scale Fusion Architecture Analysis

The multi-scale fusion architecture is defined in `src/models/multiscale_fusion.py` via `build_multiscale_fusion_model()`.

```python
# Extracted from src/models/multiscale_fusion.py (lines 19-58)
feature_extractor = Model(
    inputs=backbone.input,
    outputs=[
        backbone.get_layer("block6d_add").output,
        backbone.get_layer("top_activation").output,
    ],
    name="multiscale_feature_extractor",
)

image_input = layers.Input(shape=(224, 224, 3), name="fusion_image_input")
intermediate_features, final_features = feature_extractor(image_input)

intermediate_vector = layers.GlobalAveragePooling2D(name="intermediate_pool")(intermediate_features)
final_vector = layers.GlobalAveragePooling2D(name="final_pool")(final_features)

fused = layers.Concatenate(name="multiscale_fusion")([intermediate_vector, final_vector])
x = layers.Dropout(0.35, name="fusion_dropout")(fused)
x = layers.Dense(256, activation="relu", name="fusion_dense")(x)
x = layers.Dropout(0.35, name="classifier_dropout")(x)
outputs = layers.Dense(4, activation="softmax", name="fusion_predictions")(x)
```

### 1.1 Extracted Feature Layers & Dimensional Verification

| Feature Stream | Extraction Layer Name | Block / Stage in EfficientNet-B0 | Spatial Dimension | Channel Depth | Total Feature Elements |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Intermediate Stream** | `block6d_add` | Stage 6 (Final inverted residual add) | **$7 \times 7$** | 192 | 9,408 |
| **Final Stream** | `top_activation` | Stage 7 / Head Conv (Post 1x1 conv) | **$7 \times 7$** | 1280 | 62,720 |

### 1.2 The "Pseudo-Multi-Scale" Finding (Architectural Critique)

In foundational computer vision literature (e.g., Feature Pyramid Networks, U-Net, BiFPN), "multi-scale fusion" implies fusing **high-resolution / low-semantic** features (e.g., $56 \times 56$ or $28 \times 28$ from early/mid stages capturing precise tumor boundaries, microcalcifications, and margins) with **low-resolution / high-semantic** features (e.g., $7 \times 7$ capturing global anatomical orientation).

In this implementation:
1. **Identical Spatial Resolutions:** Both `block6d_add` and `top_activation` operate at the exact same spatial grid: **$7 \times 7$ pixels**. The operation separating them is essentially a single $1 \times 1$ pointwise convolution (`top_conv`) expanding 192 channels to 1,280 channels, followed by batch normalization and Swish activation.
2. **Loss of Spatial Localization via Immediate GAP:** Both feature maps are independently processed with `GlobalAveragePooling2D()`. This collapses the $7 \times 7 \times 192$ and $7 \times 7 \times 1280$ tensors into 1D vectors of length 192 and 1280, completely eliminating spatial localization before concatenation.
3. **Information Overlap:** `top_activation` is directly computed from `block7a`, which in turn directly consumes `block6d_add`. The 192 features from `block6d_add` are therefore already largely embedded in the 1280-dimensional `top_activation` representation.

---

## 2. Parameter Budget and Head Complexity

| Architecture | Total Parameters | Trainable Parameters | Head Parameters | Head Description |
| :--- | :---: | :---: | :---: | :--- |
| **Baseline Custom CNN** | 421,828 | 421,828 | 33,284 | GAP -> Dropout(0.30) -> Dense(128) -> Dropout(0.40) -> Dense(4) |
| **EfficientNet-B0 Baseline** | 4,214,055 | 164,484 (Stage 1) / 1,507,252 (Stage 2) | **164,484** | GAP -> Dropout(0.30) -> Dense(128) -> Dropout(0.30) -> Dense(4) |
| **Multi-Scale Feature Fusion** | 4,427,687 | 378,116 (Stage 1) / 1,832,676 (Stage 2) | **378,116** | GAP x2 -> Concat(1472) -> Dropout(0.35) -> Dense(256) -> Dropout(0.35) -> Dense(4) |

### Head Analysis:
- The fusion head adds **213,632 extra parameters** compared to the baseline head (+130% increase in classification head capacity).
- For a small medical training manifest (4,447 images), a 256-unit dense layer with 378k parameters increases the risk of co-adaptation and validation loss divergence without supplying genuinely new spatial cues.

---

## 3. Pretraining, Layer Freezing, and Batch Normalization Policy

### 3.1 Backbone Initialization
- The EfficientNet-B0 backbone is initialized with standard ImageNet weights (`weights="imagenet"`).
- In `scripts/train_multiscale_fusion_grouped.py`, the initial model loaded `models/efficientnet_b0_finetuned.keras` as its backbone weights rather than raw ImageNet weights.
- In `scripts/train_multiscale_fusion_multiseed.py`, all three seeds (42, 123, 2026) were consistently re-initialized from the exact same checkpoint: `models/multiscale_fusion_grouped_best.keras`.

### 3.2 Batch Normalization Freezing Policy
- In Keras/TensorFlow, fine-tuning pre-trained networks with small batch sizes ($B = 32$) on domain-shifted data can destabilize pre-trained running mean and variance if BatchNormalization layers are made trainable.
- **Verification:** In `scripts/train_multiscale_fusion_multiseed.py` (lines 538–555), explicit code enforces that BatchNormalization layers remain frozen:
  ```python
  for layer in backbone.layers:
      if isinstance(layer, tf.keras.layers.BatchNormalization):
          layer.trainable = False
  ```
  This is a correct, best-practice implementation that preserves running statistics during fine-tuning.

### 3.3 Layer Unfreezing
- Exactly 20 non-BatchNorm layers are unfrozen in the backbone during fine-tuning:
  `selected = eligible[-UNFREEZE_LAST_LAYERS:]`
- These 20 layers encompass stages 6 and 7 (including `block6c`, `block6d`, `block7a`, and `top_conv`).

---

## 4. Pipeline Consistency Audit: Training vs. Evaluation vs. Streamlit App

| Step / Property | Training Pipeline (`scripts/train_multiscale_fusion_multiseed.py`) | Test Evaluation (`scripts/evaluate_final_multiseed.py`) | Streamlit Web App (`app.py`) | Consistency Status |
| :--- | :--- | :--- | :--- | :---: |
| **Input Shape** | $(224, 224, 3)$ | $(224, 224, 3)$ | $(224, 224, 3)$ | **MATCH** |
| **Interpolation** | Bilinear resize via `tf.image.resize` | Bilinear resize via `tf.image.resize` | Bilinear resize via PIL (`Resampling.BILINEAR`) | **MATCH** |
| **Color Channels** | 3-channel RGB (`decode_image`) | 3-channel RGB (`tf.io.decode_image`) | 3-channel RGB (`Image.open().convert("RGB")`) | **MATCH** |
| **Pixel Intensity Range** | $[0, 255]$ float32 | $[0, 255]$ float32 | $[0, 255]$ float32 (`model_type='efficientnet'`) | **MATCH** |
| **Normalization Layer** | Internal `Rescaling` layer in backbone graph | Internal `Rescaling` layer in backbone graph | Internal `Rescaling` layer in backbone graph | **MATCH** |
| **Class Label Ordering** | Alphabetical: `glioma, meningioma, notumor, pituitary` | Alphabetical: `glioma, meningioma, notumor, pituitary` | Alphabetical: `glioma, meningioma, notumor, pituitary` | **MATCH** |
| **Model Deserialization** | Uses `src.models.compat` shim | Uses `src.models.compat` shim | Uses `src.models.compat` shim | **MATCH** |
| **Augmentation** | Enabled (training set only) | Disabled | Disabled | **MATCH** |

### Critical Finding on Checkpoint Integrity:
The model loaded by `app.py` for Option 5 (`experiments/multiseed_fusion/seed_42/best_model.keras`) is **byte-for-byte identical** to the checkpoint evaluated in `reports/final_multiseed/final_test_summary.json`. There is zero substitution or discrepancy.

---

## 5. Explainability Architecture Compatibility

### 5.1 Multi-Scale Fusion Grad-CAM
In `src/explainability/gradcam.py`:
- Standard Grad-CAM builds a sub-model mapping `backbone.input -> conv_layer.output`.
- For the multi-scale fusion model, the backbone is nested inside `multiscale_feature_extractor`.
- The specialized function `compute_multiscale_fusion_gradcam()` targets `final_features` (output of `top_activation`), taking gradients of the target class score with respect to the $7 \times 7 \times 1280$ activation map.
- This produces valid, well-localized heatmaps without gradient disconnection.

### 5.2 Baseline Custom CNN Constraint
- In Keras 3, deserialized Sequential models lack symbolic `.input` attributes.
- Higher-order Grad-CAM++ requires higher-order graph tracing that fails on this sequential structure.
- `app.py` correctly disables Grad-CAM++ for the Custom CNN while retaining standard first-order Grad-CAM.

---

## 6. Summary of Architectural Strengths and Concrete Weaknesses

### Strengths:
1. Rigorous BatchNormalization freezing prevents divergence.
2. Complete end-to-end consistency across input resolution, RGB decoding, pixel scaling ($[0, 255]$), and alphabetical label mapping.
3. Keras 3 deserialization compatibility shim (`src.models.compat`) cleanly handles legacy checkpoints.
4. Explainability extraction cleanly routes gradients through the nested fusion graph.

### Concrete Weaknesses:
1. **Lack of true multi-scale spatial resolution:** Fusing two $7 \times 7$ representations provides channel diversity but zero spatial multiscale benefit.
2. **Information loss prior to fusion:** Immediate Global Average Pooling on both feature maps discards spatial tumor topography before concatenation.
3. **Overparameterized classification head:** 378,116 parameters in a 256-unit dense layer with double 0.35 dropout invites overfitting on 4,447 training images.
