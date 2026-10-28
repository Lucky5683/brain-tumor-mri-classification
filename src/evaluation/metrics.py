"""Evaluation metrics, multi-class ROC-AUC, classification reports, and bootstrap confidence intervals."""

from __future__ import annotations
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from config import CLASS_NAMES, DISPLAY_NAMES, FIGURES_DIR, REPORTS_DIR


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
) -> dict:
    """Compute overall and per-class performance metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # Multi-class One-vs-Rest ROC-AUC
    try:
        roc_auc_macro = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro"))
        roc_auc_weighted = float(roc_auc_score(y_true, y_probs, multi_class="ovr", average="weighted"))
    except Exception:
        roc_auc_macro, roc_auc_weighted = None, None

    # Per-class metrics
    per_class = {}
    for idx, cname in enumerate(class_names):
        bin_true = (y_true == idx).astype(int)
        bin_pred = (y_pred == idx).astype(int)
        c_prec = float(precision_score(bin_true, bin_pred, zero_division=0))
        c_rec = float(recall_score(bin_true, bin_pred, zero_division=0))
        c_f1 = float(f1_score(bin_true, bin_pred, zero_division=0))
        support = int(np.sum(bin_true))
        per_class[cname] = {
            "precision": round(c_prec, 4),
            "recall": round(c_rec, 4),
            "f1": round(c_f1, 4),
            "support": support,
        }

    return {
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "roc_auc_macro_ovr": round(roc_auc_macro, 4) if roc_auc_macro is not None else None,
        "roc_auc_weighted_ovr": round(roc_auc_weighted, 4) if roc_auc_weighted is not None else None,
        "per_class": per_class,
    }


def compute_bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> tuple[dict, pd.DataFrame]:
    """Calculate non-parametric bootstrap confidence intervals (95%) for all key metrics.

    Metrics covered:
    - Accuracy
    - Macro F1
    - Macro Recall
    - Macro Precision
    - Glioma Recall

    Returns summary dict and full bootstrap DataFrame for auditability.
    """
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    boot_accuracies = []
    boot_macro_f1s = []
    boot_macro_recalls = []
    boot_macro_precisions = []
    boot_glioma_recalls = []

    glioma_idx = CLASS_NAMES.index("glioma") if "glioma" in CLASS_NAMES else 0

    for i in range(n_bootstraps):
        indices = rng.randint(0, n_samples, n_samples)
        sample_true = y_true[indices]
        sample_pred = y_pred[indices]

        acc = float(accuracy_score(sample_true, sample_pred))
        f1 = float(f1_score(sample_true, sample_pred, average="macro", zero_division=0))
        rec = float(recall_score(sample_true, sample_pred, average="macro", zero_division=0))
        prec = float(precision_score(sample_true, sample_pred, average="macro", zero_division=0))

        boot_accuracies.append(acc)
        boot_macro_f1s.append(f1)
        boot_macro_recalls.append(rec)
        boot_macro_precisions.append(prec)

        gl_mask = sample_true == glioma_idx
        if np.any(gl_mask):
            gl_rec = float(np.sum(sample_pred[gl_mask] == glioma_idx) / np.sum(gl_mask))
            boot_glioma_recalls.append(gl_rec)
        else:
            boot_glioma_recalls.append(0.0)

    alpha = (1.0 - confidence_level) / 2.0

    def get_ci(values):
        low = float(np.percentile(values, alpha * 100))
        high = float(np.percentile(values, (1.0 - alpha) * 100))
        mean = float(np.mean(values))
        std = float(np.std(values))
        return {
            "mean": round(mean, 4),
            "std": round(std, 4),
            "ci_lower": round(low, 4),
            "ci_upper": round(high, 4),
        }

    ci_summary = {
        "n_bootstraps": n_bootstraps,
        "confidence_level": confidence_level,
        "accuracy": get_ci(boot_accuracies),
        "macro_f1": get_ci(boot_macro_f1s),
        "macro_recall": get_ci(boot_macro_recalls),
        "macro_precision": get_ci(boot_macro_precisions),
        "glioma_recall": get_ci(boot_glioma_recalls),
        "note": "Bootstrap confidence intervals reflect test sampling distribution uncertainty and do not replace external clinical validation.",
    }

    boot_df = pd.DataFrame({
        "bootstrap_iteration": np.arange(1, n_bootstraps + 1),
        "accuracy": boot_accuracies,
        "macro_f1": boot_macro_f1s,
        "macro_recall": boot_macro_recalls,
        "macro_precision": boot_macro_precisions,
        "glioma_recall": boot_glioma_recalls,
    })

    return ci_summary, boot_df


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    output_path: Path = FIGURES_DIR / "confusion_matrix.png",
    title: str = "Test Set Confusion Matrix",
) -> None:
    """Save publication-quality confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    display_labels = [DISPLAY_NAMES.get(c, c.title()) for c in class_names]
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=display_labels,
        yticklabels=display_labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=11, fontweight="bold",
            )

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    output_path: Path = FIGURES_DIR / "roc_curve.png",
) -> None:
    """Generate One-vs-Rest ROC curves for all classes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, cname in enumerate(class_names):
        bin_true = (y_true == idx).astype(int)
        score = y_probs[:, idx]
        fpr, tpr, _ = roc_curve(bin_true, score)
        auc = roc_auc_score(bin_true, score)
        disp = DISPLAY_NAMES.get(cname, cname.title())
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], lw=2, label=f"{disp} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.7, label="Chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("One-vs-Rest (OvR) ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_metrics_bar(
    per_class_metrics: dict,
    output_path: Path = FIGURES_DIR / "class_metrics.png",
) -> None:
    """Plot precision, recall, and F1 comparison bar chart per class."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    classes = list(per_class_metrics.keys())
    display_classes = [DISPLAY_NAMES.get(c, c.title()) for c in classes]

    precisions = [per_class_metrics[c]["precision"] * 100 for c in classes]
    recalls = [per_class_metrics[c]["recall"] * 100 for c in classes]
    f1s = [per_class_metrics[c]["f1"] * 100 for c in classes]

    x = np.arange(len(classes))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, precisions, width, label="Precision", color="#3498db")
    ax.bar(x, recalls, width, label="Recall", color="#e67e22")
    ax.bar(x + width, f1s, width, label="F1-Score", color="#2ecc71")

    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Per-Class Precision, Recall, and F1-Score", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_classes)
    ax.set_ylim(0, 110)
    ax.legend(loc="lower right")
    ax.grid(True, axis="y", linestyle="--", alpha=0.6)

    for i in range(len(classes)):
        ax.text(x[i] - width, precisions[i] + 1.5, f"{precisions[i]:.1f}%", ha="center", fontsize=8)
        ax.text(x[i], recalls[i] + 1.5, f"{recalls[i]:.1f}%", ha="center", fontsize=8, fontweight="bold")
        ax.text(x[i] + width, f1s[i] + 1.5, f"{f1s[i]:.1f}%", ha="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_multiclass_precision_recall_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str] = CLASS_NAMES,
    output_path: Path = FIGURES_DIR / "precision_recall_curve.png",
) -> None:
    """Generate One-vs-Rest Precision-Recall curves for all classes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for idx, cname in enumerate(class_names):
        bin_true = (y_true == idx).astype(int)
        score = y_probs[:, idx]
        precision, recall, _ = precision_recall_curve(bin_true, score)
        ap = average_precision_score(bin_true, score)
        disp = DISPLAY_NAMES.get(cname, cname.title())
        ax.plot(recall, precision, color=colors[idx % len(colors)], lw=2, label=f"{disp} (AP = {ap:.4f})")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_title("Precision-Recall (PR) Curves per Class", fontsize=12, fontweight="bold")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
