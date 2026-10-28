import json
import shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)

root = Path.cwd()
json_path = root / "reports/final_multiseed/final_test_summary.json"
image_path = root / "reports/final_multiseed/selected_fusion_confusion_matrix.png"

if not json_path.is_file():
    raise FileNotFoundError(f"Metrics file not found: {json_path}")

data = json.loads(json_path.read_text(encoding="utf-8"))
result = data["selected_fusion_model"]
metrics = result["metrics"]

labels = ["glioma", "meningioma", "notumor", "pituitary"]
cm = np.asarray(result["confusion_matrix"], dtype=int)

if cm.shape != (4, 4) or np.any(cm < 0):
    raise ValueError(f"Unexpected confusion matrix: {cm.shape}")

total = int(cm.sum())
correct = int(np.trace(cm))
accuracy = correct / total

# Derive class-level metrics from the matrix.
y_true, y_pred = [], []
for i in range(4):
    for j in range(4):
        y_true.extend([i] * int(cm[i, j]))
        y_pred.extend([j] * int(cm[i, j]))

_, _, macro_f1, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)

print("\n=== VERIFIED SAVED FUSION RESULTS ===")
print(f"Checkpoint: {result.get('checkpoint', 'Not recorded')}")
print(f"Test images: {total}")
print(f"Correct predictions: {correct}")
print(f"Accuracy from matrix: {accuracy:.4%}")
print(f"Saved accuracy:       {metrics['accuracy']:.4%}")
print(f"Macro-F1 from matrix: {macro_f1:.4f}")
print(f"Saved Macro-F1:       {metrics['f1_macro']:.4f}")

if abs(accuracy - float(metrics["accuracy"])) > 1e-8:
    raise ValueError("Accuracy mismatch. Existing plot was not replaced.")

if abs(macro_f1 - float(metrics["f1_macro"])) > 1e-4:
    raise ValueError("Macro-F1 mismatch. Existing plot was not replaced.")

# Back up the existing image before replacing it.
if image_path.exists():
    backup = image_path.with_name(
        f"{image_path.stem}_backup_{datetime.now():%Y%m%d_%H%M%S}{image_path.suffix}"
    )
    shutil.copy2(image_path, backup)
    print(f"Previous image backed up to: {backup}")

fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(cm, cmap="Blues")

ax.set(
    xticks=np.arange(4),
    yticks=np.arange(4),
    xticklabels=labels,
    yticklabels=labels,
    xlabel="Predicted label",
    ylabel="True label",
    title=f"Final Multi-Scale Fusion Test Confusion Matrix\n"
          f"Accuracy: {accuracy * 100:.2f}% | Macro-F1: {macro_f1:.4f}",
)

threshold = cm.max() / 2
for i in range(4):
    for j in range(4):
        ax.text(
            j, i, f"{cm[i, j]}",
            ha="center", va="center",
            color="white" if cm[i, j] > threshold else "#123456",
            fontsize=12,
        )

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(image_path, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"\nUpdated confusion matrix: {image_path}")
print("Verification passed. Saved evaluation JSON was not modified.")
