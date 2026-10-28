"""End-to-end verification script for app.py and MODEL_REGISTRY (4 Core Project Models).

Tests:
1. Exactly the 4 requested models exist in MODEL_REGISTRY:
   - Custom CNN (Baseline)
   - EfficientNet-B0
   - Multi-Scale Feature Fusion (Seed 42)
   - Spatial Attention Feature Fusion (Seed 42)
2. All 4 checkpoints exist and load successfully via load_selected_model.
3. verify_model_health passes for all 4 models.
4. All metric JSONs and artifacts exist and load via load_model_evaluation_metrics.
5. Correct test accuracies match verified values (78.06% for CNN, 87.50% for EffNet, 85.75% for Seed 42),
   and Spatial model honestly reports test evaluation pending while reading verified validation metrics (92.54% val acc, 0.9241 val macro-F1).
6. build_model_comparison_table produces a complete 4-row comparison table.
7. Preprocessing and inference on a real test image works for all 4 models.
8. Grad-CAM and Grad-CAM++ generation work for compatible models.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import app
from src.data.preprocessing import preprocess_single_image
from src.explainability.gradcam import (
    compute_gradcam_heatmap,
    overlay_heatmap_on_image,
)
from src.explainability.gradcam_plus_plus import compute_gradcam_plus_plus_heatmap


def test_registry():
    print("=== 1. Verifying MODEL_REGISTRY ===")
    expected_models = [
        "Custom CNN (Baseline)",
        "EfficientNet-B0",
        "Multi-Scale Feature Fusion (Seed 42)",
        "Spatial Attention Feature Fusion (Seed 42)",
    ]

    assert list(app.MODEL_REGISTRY.keys()) == expected_models, f"Dropdown keys mismatch! Got {list(app.MODEL_REGISTRY.keys())}"
    print("PASS: Exactly the 4 expected dropdown keys match.")

    # Find a sample image
    sample_img_path = ROOT_DIR / "dataset" / "Testing" / "glioma" / "Te-gl_10.jpg"
    if not sample_img_path.exists():
        sample_img_path = next((ROOT_DIR / "dataset" / "Testing").glob("*/*.jpg"))
    print(f"Using test sample: {sample_img_path}")
    test_pil = Image.open(sample_img_path).convert("RGB")

    for model_name in expected_models:
        print("\n" + "=" * 55)
        print(f"Testing: {model_name}")
        print("=" * 55)
        cfg = app.MODEL_REGISTRY[model_name]

        # Check checkpoint
        ckpt_path = cfg["checkpoint"]
        assert ckpt_path.exists(), f"Checkpoint missing: {ckpt_path}"
        print(f"Checkpoint exists: {ckpt_path} ({ckpt_path.stat().st_size / (1024*1024):.1f} MB)")

        # Verify model health
        healthy, msg = app.verify_model_health(str(ckpt_path))
        assert healthy, f"Health check failed for {model_name}: {msg}"
        print(f"Health check PASS: {msg}")

        # Load model via app loader
        model, feat_extractor, head_layers, loaded_name = app.load_selected_model(str(ckpt_path))
        assert model is not None, f"Failed to load model: {model_name}"
        print(f"Model loaded successfully. Total params: {model.count_params():,}")

        # Load metrics via app loader
        metrics = app.load_model_evaluation_metrics(model_name)
        assert metrics is not None, f"Metrics missing for: {model_name}"

        if cfg.get("is_test_pending"):
            assert metrics["is_test_pending"] is True
            assert metrics["accuracy"] is None
            assert metrics["f1_macro"] is None
            assert metrics["roc_auc_macro"] is None
            print("Official test evaluation pending correctly reported.")

            # Verify validation metrics from artifacts
            val_cfg = cfg["val_metrics"]
            assert abs(val_cfg["accuracy"] - 0.9254) < 1e-3, f"Validation accuracy mismatch: {val_cfg['accuracy']}"
            assert abs(val_cfg["macro_f1"] - 0.9241) < 1e-3, f"Validation macro-F1 mismatch: {val_cfg['macro_f1']}"
            assert abs(val_cfg["balanced_accuracy"] - 0.9240) < 1e-3, f"Validation balanced accuracy mismatch: {val_cfg['balanced_accuracy']}"
            print(f"Validation metrics verified: Val Acc = {val_cfg['accuracy']*100:.2f}%, Val Macro F1 = {val_cfg['macro_f1']:.4f}, Epoch = {val_cfg.get('best_epoch')}")

            # Check validation artifact files
            assert cfg["val_metrics_file"].exists(), f"Validation metrics missing: {cfg['val_metrics_file']}"
            assert cfg["val_classification_report_csv"].exists(), f"Validation report missing: {cfg['val_classification_report_csv']}"
            assert cfg["val_confusion_matrix_csv"].exists(), f"Validation CM missing: {cfg['val_confusion_matrix_csv']}"
            print("Validation artifact files exist and verified.")
        else:
            test_acc = metrics["accuracy"]
            f1_macro = metrics["f1_macro"]
            roc_auc = metrics["roc_auc_macro"]
            test_count = metrics["test_images_count"]
            inference_ms = metrics["inference_ms"]
            print(f"Metrics parsed: Test Acc = {test_acc*100:.2f}%, Macro F1 = {f1_macro:.4f}, ROC-AUC = {roc_auc:.4f}, N = {test_count}, Latency = {inference_ms} ms")

            if model_name == "Custom CNN (Baseline)":
                assert abs(test_acc - 0.7806) < 1e-3, f"CNN test accuracy mismatch: {test_acc}"
            elif model_name == "EfficientNet-B0":
                assert abs(test_acc - 0.875) < 1e-4, f"EffNet test accuracy mismatch: {test_acc}"
            elif model_name == "Multi-Scale Feature Fusion (Seed 42)":
                assert abs(test_acc - 0.8575) < 1e-4, f"Seed 42 test accuracy mismatch: {test_acc}"
            elif model_name == "Spatial Attention Feature Fusion (Seed 42)":
                assert abs(test_acc - 0.8819) < 1e-3, f"Spatial test accuracy mismatch: {test_acc}"
                train_eval = app.load_training_split_evaluation_metrics(model_name)
                assert train_eval is not None, "Spatial training-split evaluation metrics failed to load!"
                assert abs(train_eval["accuracy"] - 0.9528) < 1e-3
                assert abs(train_eval["macro_f1"] - 0.9526) < 1e-3
                assert abs(train_eval["balanced_accuracy"] - 0.9530) < 1e-3
                assert abs(train_eval["unweighted_loss"] - 0.1265) < 1e-3
                assert abs(train_eval["class_weighted_loss"] - 0.1619) < 1e-3
                assert train_eval["number_of_images"] == 4447
                print(f"Verified training-split evaluation: Acc = {train_eval['accuracy']*100:.2f}%, F1 = {train_eval['macro_f1']:.4f}, N = {train_eval['number_of_images']}")

            # Check visual artifacts
            assert cfg["confusion_matrix_img"].exists(), f"CM missing: {cfg['confusion_matrix_img']}"
            assert cfg["roc_curve_img"].exists(), f"ROC missing: {cfg['roc_curve_img']}"
            assert cfg["classification_report_csv"].exists(), f"Report CSV missing: {cfg['classification_report_csv']}"
            print("Visual/tabular artifact files exist.")

        # Preprocessing & Inference
        tensor, orig_rgb, _ = preprocess_single_image(test_pil, model_type=cfg["model_type"])
        raw_probs = model.predict(tensor, verbose=0)[0]
        assert len(raw_probs) == 4, f"Expected 4 class outputs, got {len(raw_probs)}"
        assert np.isclose(np.sum(raw_probs), 1.0, atol=1e-5), "Probabilities do not sum to 1.0"

        temp, is_cal = app.load_calibration_temperature(model_name)
        calibrated_probs = app.apply_temperature_scaling(raw_probs, temp) if is_cal else raw_probs
        pred_idx = int(np.argmax(calibrated_probs))
        conf = float(calibrated_probs[pred_idx])
        pred_class = app.CLASS_NAMES[pred_idx]
        print(f"Inference output: class={pred_class}, confidence={conf*100:.2f}% (T={temp:.4f}, is_cal={is_cal})")

        # Grad-CAM test
        if cfg["supports_gradcam"]:
            heatmap_2d = compute_gradcam_heatmap(feat_extractor, head_layers, tensor, target_class_idx=pred_idx)
            assert heatmap_2d is not None and heatmap_2d.ndim == 2, "Grad-CAM heatmap invalid"
            h_rgb, o_rgb = overlay_heatmap_on_image(heatmap_2d, orig_rgb, alpha=0.4)
            assert o_rgb.shape == orig_rgb.shape, "Overlay generation failed"
            print("Grad-CAM generation PASS")

        # Grad-CAM++ test
        if cfg.get("supports_gradcam_pp", False):
            heatmap_pp = compute_gradcam_plus_plus_heatmap(feat_extractor, head_layers, tensor, target_class_idx=pred_idx)
            assert heatmap_pp is not None and heatmap_pp.ndim == 2, "Grad-CAM++ heatmap invalid"
            h_pp, o_pp = overlay_heatmap_on_image(heatmap_pp, orig_rgb, alpha=0.4)
            assert o_pp.shape == orig_rgb.shape, "Grad-CAM++ overlay generation failed"
            print("Grad-CAM++ generation PASS")

    print("\n=== 2. Verifying Model Comparison Table ===")
    df_comp = app.build_model_comparison_table("EfficientNet-B0")
    print(df_comp.to_string())
    assert len(df_comp) == 4, f"Expected 4 rows in comparison table, got {len(df_comp)}"
    assert df_comp.loc[df_comp["Model Architecture"] == "EfficientNet-B0", "Status"].values[0] == "Active"

    spatial_row = df_comp.loc[df_comp["Model Architecture"] == "Spatial Attention Feature Fusion (Seed 42)"]
    assert len(spatial_row) == 1, "Spatial Attention Feature Fusion row missing"
    assert "88.19%" in spatial_row["Test Accuracy"].values[0]
    assert spatial_row["Test Samples (N)"].values[0] == "1,600"
    assert "95.28%" in spatial_row["Training Accuracy"].values[0]
    assert "Checkpoint Eval" in spatial_row["Training Methodology"].values[0]
    print("PASS: Comparison table contains 4 rows and reports verified test status and training split status for all models.")

    print("\n[PASS] ALL END-TO-END VERIFICATION CHECKS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_registry()
