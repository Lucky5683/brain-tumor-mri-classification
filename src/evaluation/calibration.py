"""Model calibration evaluation, reliability diagrams, ECE, and temperature scaling."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

from config import FIGURES_DIR, REPORTS_DIR


def compute_ece_and_reliability(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute Expected Calibration Error (ECE), Maximum Calibration Error (MCE), and bin stats.

    Softmax outputs quantify model confidence/probability, NOT clinical certainty.
    """
    confidences = np.max(y_probs, axis=1)
    predictions = np.argmax(y_probs, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    mce = 0.0
    bin_details = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = float(np.mean(in_bin))

        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(accuracies[in_bin]))
            avg_confidence_in_bin = float(np.mean(confidences[in_bin]))
            abs_diff = abs(avg_confidence_in_bin - accuracy_in_bin)
            ece += abs_diff * prop_in_bin
            mce = max(mce, abs_diff)

            bin_details.append({
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": int(np.sum(in_bin)),
                "prop_samples": round(prop_in_bin, 4),
                "accuracy": round(accuracy_in_bin, 4),
                "confidence": round(avg_confidence_in_bin, 4),
                "calibration_gap": round(abs_diff, 4),
            })
        else:
            bin_details.append({
                "bin_range": [round(bin_lower, 2), round(bin_upper, 2)],
                "count": 0,
                "prop_samples": 0.0,
                "accuracy": None,
                "confidence": None,
                "calibration_gap": None,
            })

    # Multi-class Brier score (average squared difference over all class probabilities)
    num_classes = y_probs.shape[1]
    y_one_hot = np.zeros_like(y_probs)
    for i, t in enumerate(y_true):
        y_one_hot[i, t] = 1.0
    brier = float(np.mean(np.sum((y_probs - y_one_hot) ** 2, axis=1)))

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "brier_score": round(brier, 4),
        "n_bins": n_bins,
        "bins": bin_details,
        "clinical_note": (
            "Softmax output reflects heuristic model prediction confidence, "
            "not verified clinical disease probability."
        ),
    }


class TemperatureScaler:
    """Post-hoc temperature scaling fitted STRICTLY on validation logits."""

    def __init__(self):
        self.temperature = 1.0

    def fit(self, val_logits: np.ndarray, val_labels: np.ndarray) -> float:
        """Find optimal temperature T on validation data minimizing NLL."""
        labels = np.array(val_labels, dtype=int)

        def nll_obj(t_arr):
            t = t_arr[0]
            if t <= 0:
                return 1e9
            scaled = val_logits / t
            # Stable log-sum-exp
            max_logits = np.max(scaled, axis=1, keepdims=True)
            log_sum_exp = max_logits + np.log(np.sum(np.exp(scaled - max_logits), axis=1, keepdims=True))
            log_probs = scaled - log_sum_exp
            nll = -np.mean(log_probs[np.arange(len(labels)), labels])
            return nll

        res = minimize(nll_obj, x0=[1.0], bounds=[(0.05, 10.0)], method="L-BFGS-B")
        self.temperature = float(res.x[0])
        return self.temperature

    def scale_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Apply learned temperature scaling to test/inference logits."""
        scaled = logits / self.temperature
        exp_logits = np.exp(scaled - np.max(scaled, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def plot_reliability_diagram(
    calibration_data: dict,
    output_path: Path = FIGURES_DIR / "reliability_diagram.png",
    title: str = "Model Reliability Diagram",
) -> None:
    """Render and save reliability diagram with gap visualization."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8), gridspec_kw={"height_ratios": [3, 1]})

    bins = [b for b in calibration_data["bins"] if b["count"] > 0]
    confs = [b["confidence"] for b in bins]
    accs = [b["accuracy"] for b in bins]
    counts = [b["count"] for b in bins]
    bin_centers = [np.mean(b["bin_range"]) for b in bins]

    # Perfect calibration diagonal
    ax1.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    # Empirical reliability
    ax1.bar(bin_centers, accs, width=0.08, alpha=0.7, color="#2980b9", edgecolor="black", label="Outputs")
    # Gap visualization
    for c, a, bc in zip(confs, accs, bin_centers):
        if c > a:
            ax1.bar(bc, c - a, bottom=a, width=0.08, alpha=0.4, color="#e74c3c", edgecolor="red", label="Gap" if bc == bin_centers[0] else "")

    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.set_ylabel("Empirical Accuracy", fontsize=11)
    ece_val = calibration_data.get("ece", 0.0)
    brier_val = calibration_data.get("brier_score", 0.0)
    ax1.set_title(f"{title} (ECE = {ece_val:.4f}, Brier = {brier_val:.4f})", fontsize=12, fontweight="bold")
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Histogram of sample counts
    ax2.bar(bin_centers, counts, width=0.08, color="#7f8c8d", edgecolor="black")
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Confidence", fontsize=11)
    ax2.set_ylabel("Sample Count", fontsize=11)
    ax2.grid(True, axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
