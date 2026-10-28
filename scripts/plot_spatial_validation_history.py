import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path(r"X:\Mini Project\brain-tumor-mri-classification")
history_path = root / "experiments/spatial_fusion/meningioma_weighted_50ep_seed42/history.json"
out = root / "reports/streamlit_evaluation"
out.mkdir(parents=True, exist_ok=True)

with history_path.open("r", encoding="utf-8") as f:
    records = json.load(f)

if not isinstance(records, list) or not records:
    raise ValueError("Expected a non-empty list of epoch records.")

epochs = [r.get("epoch", i + 1) for i, r in enumerate(records)]
val_accuracy = [r.get("val_accuracy") for r in records]
val_loss = [r.get("val_loss") for r in records]
val_macro_f1 = [r.get("val_macro_f1") for r in records]

def plot_metric(values, title, ylabel, filename, percent=False):
    valid = [(e, v) for e, v in zip(epochs, values) if v is not None]
    if not valid:
        print(f"[MISSING] No saved values for {title}")
        return

    x, y = zip(*valid)
    if percent:
        y = [v * 100 for v in y]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, y, marker="o", markersize=3, label="Validation")
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    target = out / filename
    fig.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {target}")

plot_metric(
    val_accuracy,
    "Spatial Fusion - Validation Accuracy",
    "Accuracy (%)",
    "spatial_fusion_validation_accuracy.png",
    percent=True
)

plot_metric(
    val_loss,
    "Spatial Fusion - Validation Loss",
    "Loss",
    "spatial_fusion_validation_loss.png"
)

plot_metric(
    val_macro_f1,
    "Spatial Fusion - Validation Macro-F1",
    "Macro-F1",
    "spatial_fusion_validation_macro_f1.png"
)

print(f"\nEpoch records found: {len(records)}")
print("Note: this history records validation metrics only.")
print("It does not contain training accuracy or training loss.")
