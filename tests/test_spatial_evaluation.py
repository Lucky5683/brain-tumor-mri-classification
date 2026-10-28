"""Unit tests for Spatial Fusion evaluation metrics and label-order correctness."""

import numpy as np
import pytest
from pathlib import Path

from config import CLASS_NAMES, CLASS_TO_INDEX, TEST_DIR
from scripts.evaluate_spatial_fusion_test import (
    compute_per_class_specificity,
    verify_test_dataset,
)


def test_canonical_class_ordering():
    """Verify that canonical 4-class ordering is strictly maintained."""
    expected_classes = ["glioma", "meningioma", "notumor", "pituitary"]
    assert CLASS_NAMES == expected_classes
    for idx, name in enumerate(expected_classes):
        assert CLASS_TO_INDEX[name] == idx


def test_compute_per_class_specificity():
    """Verify specificity calculation on a known synthetic confusion matrix."""
    # Synthetic 4x4 confusion matrix
    # Rows = True, Cols = Pred
    # Total = 100 per class = 400 total
    cm = np.array([
        [90, 5, 2, 3],    # Class 0: TP=90, FN=10
        [4, 85, 6, 5],    # Class 1: TP=85, FN=15
        [1, 2, 95, 2],    # Class 2: TP=95, FN=5
        [2, 3, 1, 94],    # Class 3: TP=94, FN=6
    ])

    specs = compute_per_class_specificity(cm)

    # For Class 0 ('glioma'):
    # Total samples = 400
    # TP = 90
    # FN = 5 + 2 + 3 = 10
    # FP = 4 + 1 + 2 = 7
    # TN = 400 - (90 + 10 + 7) = 293
    # Specificity = TN / (TN + FP) = 293 / (293 + 7) = 293 / 300 = 0.97666... -> 0.9767
    assert specs["glioma"] == round(293 / 300, 4)

    # For Class 2 ('notumor'):
    # TP = 95
    # FN = 1 + 2 + 2 = 5
    # FP = 2 + 6 + 1 = 9
    # TN = 400 - (95 + 5 + 9) = 291
    # Specificity = TN / (TN + FP) = 291 / (291 + 9) = 291 / 300 = 0.9700
    assert specs["notumor"] == round(291 / 300, 4)


def test_verify_test_dataset_structure():
    """Verify that dataset/Testing exists and has exactly 400 valid images per class."""
    if not TEST_DIR.is_dir():
        pytest.skip("Test directory not found on disk.")

    paths, labels, counts = verify_test_dataset(TEST_DIR)

    assert len(paths) == 1600
    assert len(labels) == 1600
    assert counts == {
        "glioma": 400,
        "meningioma": 400,
        "notumor": 400,
        "pituitary": 400,
    }

    # Verify deterministic label ordering: 400 of 0, 400 of 1, 400 of 2, 400 of 3
    assert labels[:400] == [0] * 400
    assert labels[400:800] == [1] * 400
    assert labels[800:1200] == [2] * 400
    assert labels[1200:1600] == [3] * 400


def test_spatial_fusion_training_split_evaluation_loading():
    """Verify that verified training-split checkpoint evaluation metrics load accurately and safely."""
    from app import load_training_split_evaluation_metrics, MODEL_REGISTRY

    # Checkpoint evaluation file exists
    cfg = MODEL_REGISTRY["Spatial Attention Feature Fusion (Seed 42)"]
    assert "training_split_evaluation_file" in cfg
    eval_file = cfg["training_split_evaluation_file"]
    assert eval_file.is_file(), f"Evaluation JSON missing: {eval_file}"

    # Load metrics via app loader
    m = load_training_split_evaluation_metrics("Spatial Attention Feature Fusion (Seed 42)")
    assert m is not None

    # Verify exact verified numerical values
    assert abs(m["accuracy"] - 0.9527771531369462) < 1e-6
    assert abs(m["macro_f1"] - 0.952570304976383) < 1e-6
    assert abs(m["balanced_accuracy"] - 0.9530450992143806) < 1e-6
    assert abs(m["unweighted_loss"] - 0.12648823857307434) < 1e-6
    assert abs(m["class_weighted_loss"] - 0.16188783588364514) < 1e-6
    assert m["number_of_images"] == 4447

    # Verify formatting precision
    assert f"{m['accuracy'] * 100:.2f}%" == "95.28%"
    assert f"{m['macro_f1']:.4f}" == "0.9526"
    assert f"{m['balanced_accuracy'] * 100:.2f}%" == "95.30%"
    assert f"{m['unweighted_loss']:.4f}" == "0.1265"
    assert f"{m['class_weighted_loss']:.4f}" == "0.1619"

    # Verify explanatory note presence
    assert "evaluating the saved checkpoint on the unaugmented training split" in m["note"]
    assert "not historical epoch-50 training logs" in m["note"]

    # Verify graceful degradation on missing / invalid model
    assert load_training_split_evaluation_metrics("Custom CNN (Baseline)") is None
    assert load_training_split_evaluation_metrics("Nonexistent Architecture") is None
