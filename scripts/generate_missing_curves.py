"""Generate standalone ROC and PR curves for Multi-Scale Fusion checkpoints."""
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_NUM_INTRAOP_THREADS"] = "4"
os.environ["TF_NUM_INTEROP_THREADS"] = "2"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score, auc
import tensorflow as tf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import src.models.compat  # noqa: F401
from config import CLASS_NAMES, DISPLAY_NAMES, TEST_DIR
from src.data.dataset import get_test_dataset


def main():
    test_ds, test_paths, test_labels = get_test_dataset(test_dir=TEST_DIR, model_type="efficientnet", batch_size=32)
    y_true = np.array(test_labels, dtype=np.int32)
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=4)
    colors = ["#2b5c8f", "#d95f02", "#7570b3", "#1b9e77"]

    models_to_plot = {
        "reports/final_multiseed": {
            "ckpt": ROOT / "experiments" / "multiseed_fusion" / "seed_42" / "best_model.keras",
            "title": "Multi-Scale Feature Fusion (Seed 42 Best)"
        },
        "reports/multiscale_fusion_grouped": {
            "ckpt": ROOT / "models" / "multiscale_fusion_grouped_best.keras",
            "title": "Multi-Scale Feature Fusion (Grouped Base Checkpoint)"
        }
    }

    for dir_rel, info in models_to_plot.items():
        out_dir = ROOT / dir_rel
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Loading {info['ckpt']}...")
        model = tf.keras.models.load_model(info["ckpt"], compile=False)
        probs = model.predict(test_ds, verbose=1)

        # 1. ROC Curve
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        macro_aucs = []
        for idx, cname in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_true_oh[:, idx], probs[:, idx])
            val_auc = auc(fpr, tpr)
            macro_aucs.append(val_auc)
            ax.plot(fpr, tpr, color=colors[idx], lw=2, label=f"{DISPLAY_NAMES[cname]} (AUC = {val_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.6)
        ax.set_title(f"ROC Curves: {info['title']}\nMacro ROC-AUC = {np.mean(macro_aucs):.4f}", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
        ax.set_ylabel("True Positive Rate (Sensitivity / Recall)", fontsize=11)
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig.savefig(out_dir / "roc_curve.png", dpi=300)
        plt.close(fig)

        # 2. PR Curve
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
        for idx, cname in enumerate(CLASS_NAMES):
            prec, rec, _ = precision_recall_curve(y_true_oh[:, idx], probs[:, idx])
            ap = average_precision_score(y_true_oh[:, idx], probs[:, idx])
            ax.plot(rec, prec, color=colors[idx], lw=2, label=f"{DISPLAY_NAMES[cname]} (AP = {ap:.4f})")
        ax.set_title(f"Precision-Recall Curves: {info['title']}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall (Sensitivity)", fontsize=11)
        ax.set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
        ax.legend(loc="lower left")
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        fig.savefig(out_dir / "precision_recall_curve.png", dpi=300)
        plt.close(fig)
        print(f"Generated ROC & PR curves for {info['title']} in {dir_rel}")

    print("ALL CURVES COMPLETED!")


if __name__ == "__main__":
    main()
