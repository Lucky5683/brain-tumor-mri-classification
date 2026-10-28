import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"X:\Mini Project\brain-tumor-mri-classification")
OUT = ROOT / "reports" / "streamlit_evaluation"
OUT.mkdir(parents=True, exist_ok=True)

def load_history(path):
    """Read a saved Keras-style history JSON without retraining."""
    path = ROOT / path
    if not path.exists():
        print(f"[MISSING] History: {path}")
        return None

    try:
        obj = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ERROR] Cannot read {path}: {exc}")
        return None

    # Support common history formats.
    if isinstance(obj, dict) and isinstance(obj.get("history"), dict):
        obj = obj["history"]
    elif isinstance(obj, dict) and isinstance(obj.get("history"), list):
        rows = obj["history"]
        if rows and isinstance(rows[0], dict):
            keys = set().union(*(r.keys() for r in rows))
            obj = {k: [r.get(k) for r in rows] for k in keys}

    if not isinstance(obj, dict):
        print(f"[WARNING] Unsupported history format: {path}")
        return None

    cleaned = {}
    for key, value in obj.items():
        if isinstance(value, (list, tuple)):
            try:
                cleaned[str(key)] = [
                    float(v) if v is not None else np.nan for v in value
                ]
            except (TypeError, ValueError):
                continue

    print(f"\n[HISTORY] {path.name}")
    print("Available fields:", ", ".join(cleaned.keys()))
    return cleaned

def find_key(history, candidates):
    lower_map = {k.lower(): k for k in history}
    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]
    return None

def plot_history(history_path, model_label, prefix):
    history = load_history(history_path)
    if not history:
        return

    train_acc = find_key(history, ["accuracy", "acc", "train_accuracy"])
    val_acc = find_key(history, ["val_accuracy", "val_acc", "validation_accuracy"])
    train_loss = find_key(history, ["loss", "train_loss"])
    val_loss = find_key(history, ["val_loss", "validation_loss"])

    # Plot only metrics actually recorded in the history.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, train_key, val_key, title, ylabel in [
        (axes[0], train_acc, val_acc, "Accuracy", "Accuracy"),
        (axes[1], train_loss, val_loss, "Loss", "Loss"),
    ]:
        plotted = False
        if train_key and history[train_key]:
            y = history[train_key]
            ax.plot(range(1, len(y) + 1), y, label="Training")
            plotted = True
        if val_key and history[val_key]:
            y = history[val_key]
            ax.plot(range(1, len(y) + 1), y, label="Validation")
            plotted = True

        ax.set_title(f"{model_label}: {title}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if plotted:
            ax.legend()
        else:
            ax.text(
                0.5, 0.5, "Metric not recorded in saved history",
                ha="center", va="center", transform=ax.transAxes
            )

    fig.tight_layout()
    target = OUT / f"{prefix}_training_validation_curves.png"
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {target}")

def plot_comparison(filename, title, ylabel, values, labels):
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)
    ax.bar_label(bars, fmt="%.2f%%", padding=3)
    plt.xticks(rotation=10, ha="right")
    fig.tight_layout()
    target = OUT / filename
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {target}")

# 1. Training/validation history plots.
plot_history(
    Path("cnn_history.json"),
    "Custom CNN Baseline",
    "baseline"
)

plot_history(
    Path("experiments/spatial_fusion/meningioma_weighted_50ep_seed42/history.json"),
    "Spatial Attention Feature Fusion",
    "spatial_fusion"
)

# 2. Official test-set comparison: both models evaluated on 1,600 images.
labels = ["Custom CNN Baseline", "Spatial Attention Feature Fusion"]

plot_comparison(
    "test_accuracy_comparison.png",
    "Official Test Accuracy Comparison",
    "Accuracy (%)",
    [78.06, 88.19],
    labels
)

plot_comparison(
    "test_macro_f1_comparison.png",
    "Official Test Macro-F1 Comparison",
    "Macro-F1 (%)",
    [76.75, 87.77],
    labels
)

# 3. Copy existing genuine evaluation figures without altering originals.
existing_figures = {
    ROOT / "baseline_confusion_matrix.png":
        "baseline_confusion_matrix.png",

    ROOT / "experiments/final_evaluation/spatial_fusion/confusion_matrix.png":
        "spatial_fusion_confusion_matrix.png",

    ROOT / "experiments/final_evaluation/spatial_fusion/roc_curves.png":
        "spatial_fusion_roc_curves.png",

    ROOT / "experiments/final_evaluation/spatial_fusion/per_class_metrics.png":
        "spatial_fusion_per_class_metrics.png",
}

for source, filename in existing_figures.items():
    target = OUT / filename
    if source.exists():
        shutil.copy2(source, target)
        print(f"[COPIED] {target.name}")
    else:
        print(f"[MISSING FIGURE] {source}")

print("\nFinished. Output folder:")
print(OUT)
print("\nReview the printed history fields and any missing-file warnings.")
