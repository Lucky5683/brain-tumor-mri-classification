"""Comprehensive Automated Test Suite for Brain Tumor MRI Classification Project.

Validates:
1. Valid MRI image preprocessing & normalization
2. Unsupported file type rejection
3. Corrupted / truncated image error handling
4. Non-standard image dimensions resizing to (1, 224, 224, 3)
5. Missing model checkpoint handling
6. Correct output dimensions & class count (4 classes)
7. Correct class-label mapping & inverse consistency
8. Prediction probability simplex validity (sum to 1.0, non-negative)
9. Grad-CAM generation & bounds for both EfficientNet-B0 and Custom CNN
10. Grad-CAM++ generation & higher-order gradient stability
11. Model-specific calibration temperature decoupling
12. Empty / invalid input error handling
"""

from __future__ import annotations
import os
from pathlib import Path
import tempfile
import numpy as np
from PIL import Image
import pytest
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import tensorflow as tf

from config import (
    CLASS_NAMES,
    CLASS_TO_INDEX,
    DISPLAY_NAMES,
    IMAGE_SIZE,
    INDEX_TO_CLASS,
    NUM_CLASSES,
)
import src.models.compat
from src.data.preprocessing import preprocess_single_image
from src.explainability.gradcam import (
    compute_gradcam_heatmap,
    extract_gradcam_submodels,
    overlay_heatmap_on_image,
)
from src.explainability.gradcam_plus_plus import compute_gradcam_plus_plus_heatmap
from app import load_calibration_temperature, apply_temperature_scaling


VALID_TEST_IMAGE = Path("dataset/Testing/glioma/Te-gl_10.jpg")
EFFNET_CHECKPOINT = Path("efficientnet_b0.keras")
CNN_CHECKPOINT = Path("baseline_cnn.keras")


@pytest.fixture(scope="session")
def loaded_effnet():
    if not EFFNET_CHECKPOINT.is_file():
        pytest.skip(f"EfficientNet checkpoint missing at {EFFNET_CHECKPOINT}")
    return tf.keras.models.load_model(EFFNET_CHECKPOINT)


@pytest.fixture(scope="session")
def loaded_cnn():
    if not CNN_CHECKPOINT.is_file():
        pytest.skip(f"Baseline CNN checkpoint missing at {CNN_CHECKPOINT}")
    return tf.keras.models.load_model(CNN_CHECKPOINT)


def test_1_valid_mri_image_preprocessing():
    """Test 1: Valid MRI image loading and preprocessing."""
    assert VALID_TEST_IMAGE.is_file(), f"Sample test image not found: {VALID_TEST_IMAGE}"
    tensor_eff, orig_rgb, orig_size = preprocess_single_image(VALID_TEST_IMAGE, model_type="efficientnet")

    assert tensor_eff.shape == (1, 224, 224, 3), f"Expected (1, 224, 224, 3), got {tensor_eff.shape}"
    assert tensor_eff.dtype == np.float32 or tensor_eff.dtype == np.float64
    assert orig_rgb.ndim == 3 and orig_rgb.shape[2] == 3
    assert orig_rgb.dtype == np.uint8
    assert orig_size == (orig_rgb.shape[1], orig_rgb.shape[0])

    # CNN preprocessing scaling to [0, 1]
    tensor_cnn, _, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="cnn")
    assert tensor_cnn.shape == (1, 224, 224, 3)
    assert 0.0 <= tensor_cnn.min() <= tensor_cnn.max() <= 1.0


def test_2_unsupported_file_type():
    """Test 2: Rejection of unsupported file types."""
    with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as tmp:
        tmp.write(b"This is a text file, not an MRI image.")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(Exception):
            preprocess_single_image(tmp_path)
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


def test_3_corrupted_image():
    """Test 3: Corrupted / truncated image error handling."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        # Write corrupted JPEG header with incomplete payload
        tmp.write(b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xFF\xDBcorrupted_bytes_here")
        tmp_path = Path(tmp.name)

    try:
        with pytest.raises(Exception):
            preprocess_single_image(tmp_path)
    finally:
        if tmp_path.exists():
            os.remove(tmp_path)


def test_4_incorrect_image_dimensions():
    """Test 4: Non-standard resolutions resize to (1, 224, 224, 3)."""
    # Create synthetic PIL image with odd aspect ratio (e.g., 137 x 481)
    odd_image = Image.new("RGB", (137, 481), color=(128, 64, 32))
    tensor, orig_rgb, orig_size = preprocess_single_image(odd_image, model_type="efficientnet")

    assert tensor.shape == (1, 224, 224, 3), f"Expected (1, 224, 224, 3), got {tensor.shape}"
    assert orig_size == (137, 481)
    assert orig_rgb.shape == (481, 137, 3)


def test_5_missing_model_checkpoint():
    """Test 5: Missing checkpoint handling."""
    fake_path = Path("non_existent_model_checkpoint.keras")
    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        tf.keras.models.load_model(fake_path)


def test_6_correct_output_dimensions_and_class_count(loaded_effnet, loaded_cnn):
    """Test 6: Correct output dimensions & class count (4 classes)."""
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)

    pred_eff = loaded_effnet.predict(dummy_input, verbose=0)
    assert pred_eff.shape == (1, 4), f"EfficientNet output shape was {pred_eff.shape}, expected (1, 4)"

    pred_cnn = loaded_cnn.predict(dummy_input, verbose=0)
    assert pred_cnn.shape == (1, 4), f"Custom CNN output shape was {pred_cnn.shape}, expected (1, 4)"


def test_7_correct_class_label_mapping():
    """Test 7: Correct class-label mapping & inverse consistency."""
    assert NUM_CLASSES == 4
    assert CLASS_NAMES == ["glioma", "meningioma", "notumor", "pituitary"]
    for idx, cname in enumerate(CLASS_NAMES):
        assert CLASS_TO_INDEX[cname] == idx
        assert INDEX_TO_CLASS[idx] == cname
        assert cname in DISPLAY_NAMES


def test_8_prediction_probability_validity(loaded_effnet, loaded_cnn):
    """Test 8: Prediction probability simplex constraints."""
    tensor_eff, _, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="efficientnet")
    probs_eff = loaded_effnet.predict(tensor_eff, verbose=0)[0]

    assert len(probs_eff) == 4
    assert np.all(probs_eff >= 0.0), f"Negative probabilities found: {probs_eff}"
    assert np.all(probs_eff <= 1.0), f"Probabilities exceed 1.0: {probs_eff}"
    assert np.isclose(np.sum(probs_eff), 1.0, atol=1e-5), f"Probabilities do not sum to 1.0: {np.sum(probs_eff)}"

    tensor_cnn, _, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="cnn")
    probs_cnn = loaded_cnn.predict(tensor_cnn, verbose=0)[0]
    assert np.isclose(np.sum(probs_cnn), 1.0, atol=1e-5)


def test_9_gradcam_output_generation(loaded_effnet, loaded_cnn):
    """Test 9: Grad-CAM output generation & bounds for both EfficientNet and CNN."""
    tensor_eff, orig_rgb, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="efficientnet")
    feat_eff, head_eff = extract_gradcam_submodels(loaded_effnet)
    hm_eff = compute_gradcam_heatmap(feat_eff, head_eff, tensor_eff, target_class_idx=0)

    assert hm_eff.ndim == 2, f"Grad-CAM heatmap should be 2D, got shape {hm_eff.shape}"
    assert 0.0 <= hm_eff.min() <= hm_eff.max() <= 1.0, f"Heatmap out of bounds [0, 1]: min={hm_eff.min()}, max={hm_eff.max()}"

    colored_rgb, overlay = overlay_heatmap_on_image(hm_eff, orig_rgb, alpha=0.40)
    assert colored_rgb.shape == orig_rgb.shape
    assert overlay.shape == orig_rgb.shape
    assert colored_rgb.dtype == np.uint8
    assert overlay.dtype == np.uint8

    # Sequential CNN Grad-CAM test
    tensor_cnn, _, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="cnn")
    feat_cnn, head_cnn = extract_gradcam_submodels(loaded_cnn)
    hm_cnn = compute_gradcam_heatmap(feat_cnn, head_cnn, tensor_cnn, target_class_idx=0)
    assert hm_cnn.ndim == 2
    assert 0.0 <= hm_cnn.min() <= hm_cnn.max() <= 1.0


def test_10_gradcam_plus_plus_output_generation(loaded_effnet):
    """Test 10: Grad-CAM++ higher order gradient generation."""
    tensor_eff, orig_rgb, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="efficientnet")
    feat_eff, head_eff = extract_gradcam_submodels(loaded_effnet)
    hm_pp = compute_gradcam_plus_plus_heatmap(feat_eff, head_eff, tensor_eff, target_class_idx=0)

    assert hm_pp.ndim == 2
    assert 0.0 <= hm_pp.min() <= hm_pp.max() <= 1.0
    _, overlay_pp = overlay_heatmap_on_image(hm_pp, orig_rgb, alpha=0.40)
    assert overlay_pp.shape == orig_rgb.shape


def test_11_calibration_model_awareness():
    """Test 11: Calibration temperature is decoupled per model architecture."""
    # EfficientNet should return calibrated temperature (T ~ 1.0993)
    temp_eff, is_calib_eff = load_calibration_temperature("EfficientNet-B0 (Reference Baseline - V1)")
    assert is_calib_eff is True
    assert 1.0 < temp_eff < 1.5, f"Unexpected temperature for EfficientNet: {temp_eff}"

    # CNN should return uncalibrated baseline (T = 1.0, is_calibrated = False)
    temp_cnn, is_calib_cnn = load_calibration_temperature("Baseline Custom CNN")
    assert is_calib_cnn is False
    assert temp_cnn == 1.0

    # Temperature scaling on dummy distribution
    dummy_probs = np.array([0.70, 0.10, 0.10, 0.10])
    scaled = apply_temperature_scaling(dummy_probs, temp_eff)
    assert np.isclose(np.sum(scaled), 1.0, atol=1e-5)
    # Temperature > 1.0 softens probabilities
    assert scaled[0] < dummy_probs[0]


def test_12_empty_or_invalid_input():
    """Test 12: Empty or invalid input handling."""
    with pytest.raises((TypeError, ValueError)):
        preprocess_single_image(None)

    with pytest.raises(Exception):
        preprocess_single_image("")


def test_13_selected_multiseed_fusion_checkpoint():
    """Test 13: Selected Multi-Scale Fusion Checkpoint (Seed 42) validation."""
    fusion_ckpt = PROJECT_ROOT / "experiments" / "multiseed_fusion" / "seed_42" / "best_model.keras"
    assert fusion_ckpt.is_file(), f"Selected checkpoint not found: {fusion_ckpt}"

    model = tf.keras.models.load_model(fusion_ckpt, compile=False)
    assert model.name == "EfficientNetB0_MultiScaleFusion"
    assert model.count_params() == 4427687

    # Verify inference and simplex constraints
    tensor, orig_rgb, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="efficientnet")
    probs = model.predict(tensor, verbose=0)[0]
    assert len(probs) == 4
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)

    # Verify Grad-CAM support on the multi-scale architecture
    feat_ext, head_layers = extract_gradcam_submodels(model)
    hm = compute_gradcam_heatmap(feat_ext, head_layers, tensor, target_class_idx=0)
    assert hm.ndim == 2
    assert 0.0 <= hm.min() <= hm.max() <= 1.0

    # Verify calibration temperature retrieval
    temp_fusion, is_calib_fusion = load_calibration_temperature("Multi-Scale Feature Fusion (Seed 42 - Selected Model)")
    assert is_calib_fusion is True
    assert 1.0 < temp_fusion < 1.5


def test_14_spatial_attention_feature_fusion_model():
    """Test 14: Spatial Attention Feature Fusion model registration, loading, and integrity."""
    from app import MODEL_REGISTRY, load_model_evaluation_metrics, build_model_comparison_table

    # 1. Exactly 4 models in registry
    assert len(MODEL_REGISTRY) == 4
    assert "Spatial Attention Feature Fusion (Seed 42)" in MODEL_REGISTRY

    # 2. Checkpoint exists and loads
    cfg = MODEL_REGISTRY["Spatial Attention Feature Fusion (Seed 42)"]
    ckpt_path = cfg["checkpoint"]
    assert ckpt_path.is_file(), f"Spatial checkpoint not found: {ckpt_path}"

    model = tf.keras.models.load_model(ckpt_path, compile=False)
    assert model.count_params() == 4472487

    # 3. Output shape (None, 4) in canonical order
    dummy_x = np.zeros((1, 224, 224, 3), dtype=np.float32)
    out = model(dummy_x)
    assert out.shape == (1, 4)

    # Inference on valid sample MRI
    tensor, orig_rgb, _ = preprocess_single_image(VALID_TEST_IMAGE, model_type="efficientnet")
    probs = model.predict(tensor, verbose=0)[0]
    assert len(probs) == 4
    assert np.isclose(np.sum(probs), 1.0, atol=1e-5)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)

    # 4. Validation metrics read from correct artifact
    val_metrics = cfg["val_metrics"]
    assert abs(val_metrics["accuracy"] - 0.9254) < 1e-3
    assert abs(val_metrics["macro_f1"] - 0.9241) < 1e-3
    assert abs(val_metrics["balanced_accuracy"] - 0.9240) < 1e-3
    assert val_metrics["best_epoch"] == 50
    assert cfg["val_metrics_file"].is_file()
    assert cfg["val_classification_report_csv"].is_file()
    assert cfg["val_confusion_matrix_csv"].is_file()

    # 5. Verified official test metrics
    metrics = load_model_evaluation_metrics("Spatial Attention Feature Fusion (Seed 42)")
    assert metrics["is_test_pending"] is False
    assert abs(metrics["accuracy"] - 0.8819) < 1e-3
    assert abs(metrics["f1_macro"] - 0.8777) < 1e-3
    assert abs(metrics["roc_auc_macro"] - 0.9816) < 1e-3
    assert metrics["test_images_count"] == 1600

    # 6. Comparison table has 4 rows with active status and verified test accuracy
    df_comp = build_model_comparison_table("Spatial Attention Feature Fusion (Seed 42)")
    assert len(df_comp) == 4
    spatial_row = df_comp.loc[df_comp["Model Architecture"] == "Spatial Attention Feature Fusion (Seed 42)"]
    assert len(spatial_row) == 1
    assert spatial_row["Status"].values[0] == "Active"
    assert "88.19%" in spatial_row["Test Accuracy"].values[0]
    assert spatial_row["Test Samples (N)"].values[0] == "1,600"

    # 7. Supported explainability features (Grad-CAM & Grad-CAM++) work
    feat_ext, head_layers = extract_gradcam_submodels(model)
    hm = compute_gradcam_heatmap(feat_ext, head_layers, tensor, target_class_idx=0)
    assert hm.ndim == 2
    assert 0.0 <= hm.min() <= hm.max() <= 1.0

    hm_pp = compute_gradcam_plus_plus_heatmap(feat_ext, head_layers, tensor, target_class_idx=0)
    assert hm_pp.ndim == 2
    assert 0.0 <= hm_pp.min() <= hm_pp.max() <= 1.0
