"""Uncertainty estimation and rejection gating for Brain Tumor MRI predictions."""

from __future__ import annotations
import numpy as np
from config import UNCERTAINTY_THRESHOLDS, MEDICAL_DISCLAIMER


def compute_prediction_entropy(probabilities: np.ndarray) -> float:
    """Compute Shannon entropy of probability distribution (in nats)."""
    probs = np.clip(probabilities, 1e-12, 1.0)
    return float(-np.sum(probs * np.log(probs)))


def assess_prediction_uncertainty(
    probabilities: np.ndarray,
    thresholds: dict = UNCERTAINTY_THRESHOLDS,
) -> dict:
    """Evaluate model certainty for an inference output.

    CRITICAL CLINICAL NOTICE:
    Heuristic uncertainty thresholds are mathematical scoring boundaries based on softmax distribution
    entropy and margin. They DO NOT represent medically validated risk thresholds.
    """
    probs = np.asarray(probabilities, dtype=float).flatten()
    max_prob = float(np.max(probs))
    pred_idx = int(np.argmax(probs))
    entropy = compute_prediction_entropy(probs)

    high_min = thresholds.get("high_confidence_min", 0.85)
    mod_min = thresholds.get("moderate_confidence_min", 0.60)
    max_entropy = thresholds.get("max_entropy_threshold", 1.10)

    if max_prob >= high_min and entropy < max_entropy:
        status = "High"
        guidance = "Prediction exhibits strong model feature concordance."
        requires_review = False
    elif max_prob >= mod_min:
        status = "Moderate"
        guidance = "Prediction shows moderate agreement; ambiguity exists with secondary classes."
        requires_review = True
    else:
        status = "Uncertain"
        guidance = "The model prediction is uncertain and should not be relied upon without expert review."
        requires_review = True

    return {
        "predicted_index": pred_idx,
        "max_probability": round(max_prob, 4),
        "entropy": round(entropy, 4),
        "confidence_level": status,
        "uncertainty_status": status,
        "requires_expert_review": requires_review,
        "clinical_guidance": guidance,
        "disclaimer": "Research/Educational Use Only. Not a Medical Diagnosis.",
    }
