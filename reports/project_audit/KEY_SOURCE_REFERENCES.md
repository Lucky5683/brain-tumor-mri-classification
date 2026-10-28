# Key Source References & Code Navigation Index

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Purpose:** Quick technical index mapping architectural components, data pipelines, training routines, and evaluation functions to their exact file paths and primary line ranges.

---

## 1. Core Application & Configuration

### `config.py`
- [config.py:L7-L19](file:///X:/Mini%20Project/brain-tumor-mri-classification/config.py#L7-L19): Path constants (`PROJECT_ROOT`, `DATA_DIR`, `MODELS_DIR`, `REPORTS_DIR`, `EXPERIMENTS_DIR`).
- [config.py:L21-L39](file:///X:/Mini%20Project/brain-tumor-mri-classification/config.py#L21-L39): Image dimensions (`(224, 224, 3)`), batch size (`32`), class names (`glioma, meningioma, notumor, pituitary`), and display names.
- [config.py:L41-L51](file:///X:/Mini%20Project/brain-tumor-mri-classification/config.py#L41-L51): Augmentation constraints (`AUGMENTATION_CONFIG`), explicitly forbidding vertical flips to maintain anatomical superior/inferior alignment.
- [config.py:L63-L74](file:///X:/Mini%20Project/brain-tumor-mri-classification/config.py#L63-L74): Uncertainty thresholds and mandatory research medical disclaimer.

### `app.py`
- [app.py:L55-L177](file:///X:/Mini%20Project/brain-tumor-mri-classification/app.py#L55-L177): `MODEL_REGISTRY` mapping all five dropdown options to checkpoints, preprocessing, parameters, latency, calibration temperatures, metrics files, and visual artifacts.
- [app.py:L196-L230](file:///X:/Mini%20Project/brain-tumor-mri-classification/app.py#L196-L230): `load_calibration_temperature()`: Decouples model-aware temperatures fitted strictly on validation ($T^* = 1.0993$ for EffNet, $1.1322$ for Fusion, $1.0000$ for CNN).
- [app.py:L233-L243](file:///X:/Mini%20Project/brain-tumor-mri-classification/app.py#L233-L243): `apply_temperature_scaling()`: Applies temperature-scaled softmax.
- [app.py:L248-L263](file:///X:/Mini%20Project/brain-tumor-mri-classification/app.py#L248-L263): `load_selected_model()`: Cached model and sub-model loader decorated with `@st.cache_resource`.
- [app.py:L267-L315](file:///X:/Mini%20Project/brain-tumor-mri-classification/app.py#L267-L315): `load_model_evaluation_metrics()`: Robust parser extracting test metrics, per-class tables, and confusion matrices from saved JSON artifacts.
- [app.py:L520-L533](file:///X:/Mini%20Project/brain-tumor-mri-classification/app.py#L520-L533): Selective prediction referral trigger ($\tau^* = 0.60$), flagging ambiguous scans for neuroradiologist priority triage.
- [app.py:L572-L730](file:///X:/Mini%20Project/brain-tumor-mri-classification/app.py#L572-L730): Tab 2 rendering dynamic evaluation cards, validation vs test distinction, confusion matrix, ROC curve, PR curve, and classification reports.

---

## 2. Model Architectures & Compatibility

### `src/models/compat.py`
- [compat.py:L1-L45](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/models/compat.py#L1-L45): Keras 3 compatibility shim intercepting `GlorotUniform` and `VarianceScaling` initializers to discard unsupported `input_axes` kwargs on legacy models.

### `src/models/multiscale_fusion.py`
- [multiscale_fusion.py:L6-L63](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/models/multiscale_fusion.py#L6-L63): `build_multiscale_fusion_model()`:
  - Lines 21–24: Extracts `block6d_add` ($7 \times 7 \times 192$) and `top_activation` ($7 \times 7 \times 1280$).
  - Lines 37–43: Independent `GlobalAveragePooling2D` on both streams.
  - Lines 45–47: Concatenation layer yielding 1472-d feature vector.
  - Lines 49–57: Classification head (`Dropout(0.35) -> Dense(256) -> Dropout(0.35) -> Dense(4)`).

### `src/models/efficientnet.py`
- [efficientnet.py:L9-L37](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/models/efficientnet.py#L9-L37): `build_efficientnet_b0_stage1()`: Stage 1 frozen backbone model with 128-unit dense head.
- [efficientnet.py:L40-L67](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/models/efficientnet.py#L40-L67): `prepare_stage2_finetuning()`: Unfreezes top 20 non-BatchNorm layers; strictly freezes BatchNorm layers.
- [efficientnet.py:L69-L76](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/models/efficientnet.py#L69-L76): `count_parameters()`: Utility auditing total, trainable, and frozen parameters.

### `src/models/custom_cnn.py`
- [custom_cnn.py:L8-L41](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/models/custom_cnn.py#L8-L41): `build_custom_cnn()`: 4-block CNN architecture (32, 64, 128, 256 filters) with global pooling and 128-unit dense head.

---

## 3. Data Pipelines, Preprocessing & Augmentation

### `src/data/preprocessing.py`
- [preprocessing.py:L12-L19](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/preprocessing.py#L12-L19): `load_image_and_label()`: Decodes image, enforces 3 channels, resizes to $224 \times 224$ via bilinear interpolation.
- [preprocessing.py:L22-L24](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/preprocessing.py#L22-L24): `normalize_for_cnn()`: Scales pixel values to $[0, 1]$.
- [preprocessing.py:L27-L29](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/preprocessing.py#L27-L29): `normalize_for_efficientnet()`: Preserves $[0, 255]$ for internal EfficientNet normalization.
- [preprocessing.py:L32-L73](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/preprocessing.py#L32-L73): `preprocess_single_image()`: Unified single-image inference preprocessor for `app.py` and evaluation.

### `src/data/augmentation.py`
- [augmentation.py:L15-L55](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/augmentation.py#L15-L55): `build_training_augmentation()`: Instantiates horizontal flip, rotation $\pm 5\%$, zoom $\pm 8\%$, and shift $\pm 4\%$.

### `src/data/dataset.py`
- [dataset.py:L63-L91](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/dataset.py#L63-L91): `make_stratified_split()`: Generates deterministic stratified 80/20 train/validation splits.
- [dataset.py:L108-L274](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/dataset.py#L108-L274): `load_manifest_split()`: Parses fixed manifest, verifies SHA-256 hashes, and enforces zero cross-split duplicate groups.
- [dataset.py:L317-L380](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/dataset.py#L317-L380): `build_tf_dataset()`: Builds optimized `tf.data.Dataset` pipelines with prefetching, batching, and conditional training augmentation.
- [dataset.py:L382-L448](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/dataset.py#L382-L448): `get_train_val_datasets()`: High-level dataset entry point.
- [dataset.py:L450-L470](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/data/dataset.py#L450-L470): `get_test_dataset()`: Pure test set loader with deterministic preprocessing and augmentation disabled.

---

## 4. Explainability & Saliency

### `src/explainability/gradcam.py`
- [gradcam.py:L14-L40](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/explainability/gradcam.py#L14-L40): `get_target_conv_layer()`: Locates target conv layer in sequential, functional, or nested backbone containers.
- [gradcam.py:L42-L84](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/explainability/gradcam.py#L42-L84): `extract_gradcam_submodels()`: Deconstructs model into feature extractor and head layer chain.
- [gradcam.py:L86-L118](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/explainability/gradcam.py#L86-L118): `compute_multiscale_fusion_gradcam()`: Computes gradients specifically targeting the $7 \times 7 \times 1280$ `final_features` activation map inside the nested multi-scale model.
- [gradcam.py:L120-L160](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/explainability/gradcam.py#L120-L160): `compute_gradcam_heatmap()`: General first-order Grad-CAM gradient tape pooling.
- [gradcam.py:L162-L190](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/explainability/gradcam.py#L162-L190): `overlay_heatmap_on_image()`: Resizes heatmap using PIL bilinear interpolation and blends with Matplotlib colormaps.

### `src/explainability/gradcam_plus_plus.py`
- [gradcam_plus_plus.py:L14-L80](file:///X:/Mini%20Project/brain-tumor-mri-classification/src/explainability/gradcam_plus_plus.py#L14-L80): `compute_gradcam_plus_plus_heatmap()`: Higher-order Grad-CAM++ taking 2nd and 3rd order gradients for finer pixel-attribution weighting.

---

## 5. Training & Evaluation Scripts

### `scripts/train_multiscale_fusion_multiseed.py`
- [train_multiscale_fusion_multiseed.py:L100-L195](file:///X:/Mini%20Project/brain-tumor-mri-classification/scripts/train_multiscale_fusion_multiseed.py#L100-L195): Integrity assertions verifying all 5,600 manifest rows, image file existence, SHA-256 hashes, and zero split leakage.
- [train_multiscale_fusion_multiseed.py:L293-L451](file:///X:/Mini%20Project/brain-tumor-mri-classification/scripts/train_multiscale_fusion_multiseed.py#L293-L451): `ValidationMetrics` callback tracking epoch-by-epoch per-class metrics and checkpoint saving based strictly on validation Macro-F1.
- [train_multiscale_fusion_multiseed.py:L453-L470](file:///X:/Mini%20Project/brain-tumor-mri-classification/scripts/train_multiscale_fusion_multiseed.py#L453-L470): `MacroF1EarlyStopping`: Enforces a minimum of 8 epochs and stops training after 3 stagnant epochs.
- [train_multiscale_fusion_multiseed.py:L516-L575](file:///X:/Mini%20Project/brain-tumor-mri-classification/scripts/train_multiscale_fusion_multiseed.py#L516-L575): `prepare_fine_tuning()`: Freezes backbone BatchNorm layers and unfreezes the top 20 non-BatchNorm layers.
- [train_multiscale_fusion_multiseed.py:L580-L680](file:///X:/Mini%20Project/brain-tumor-mri-classification/scripts/train_multiscale_fusion_multiseed.py#L580-L680): Multi-seed execution loop training Seeds 42, 123, and 2026 sequentially from `BASE_CHECKPOINT`.

### `scripts/evaluate_final_multiseed.py`
- [evaluate_final_multiseed.py:L1-L350](file:///X:/Mini%20Project/brain-tumor-mri-classification/scripts/evaluate_final_multiseed.py#L1-L350): Independent locked evaluation script computing official test metrics, confusion matrices, ROC curves, and PR curves for Seed 42 and Baseline V1.

### `tests/test_pipeline.py`
- [test_pipeline.py:L1-L260](file:///X:/Mini%20Project/brain-tumor-mri-classification/tests/test_pipeline.py#L1-L260): Complete 13-test regression suite covering model loading, forward inference, probability bounds, Grad-CAM, Grad-CAM++, temperature scaling, and Streamlit components.
