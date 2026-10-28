"""Root train_efficientnet.py compatibility script.

Delegates execution to the full V2 two-stage training pipeline in scripts/train_efficientnet.py.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from config import DATA_DIR, STAGE1_EPOCHS, STAGE2_EPOCHS
from scripts.train_efficientnet import run_training_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Two-stage EfficientNet-B0 training pipeline.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--stage1-epochs", type=int, default=STAGE1_EPOCHS)
    parser.add_argument("--stage2-epochs", type=int, default=STAGE2_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--use-class-weights", action="store_true", help="Apply balanced class weighting calculated on training set.")
    parser.add_argument("--unfreeze-layers", type=int, default=20)
    args = parser.parse_args()

    run_training_pipeline(
        data_dir=args.data_dir,
        stage1_epochs=args.stage1_epochs,
        stage2_epochs=args.stage2_epochs,
        batch_size=args.batch_size,
        use_class_weights=args.use_class_weights,
        unfreeze_layers=args.unfreeze_layers,
    )


if __name__ == "__main__":
    main()
