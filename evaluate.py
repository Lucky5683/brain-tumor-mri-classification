"""Root evaluate.py compatibility script.

Delegates execution to the full V2 evaluation pipeline in scripts/evaluate.py.
"""

from __future__ import annotations
import sys
from scripts.evaluate import evaluate_model_checkpoint
from config import TEST_DIR, REPORTS_DIR
from pathlib import Path
import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on official test set.")
    parser.add_argument("--model-path", type=Path, default=Path("efficientnet_b0.keras"))
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--model-type", choices=["cnn", "efficientnet"], default="efficientnet")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--bootstraps", type=int, default=1000)
    args = parser.parse_args()

    evaluate_model_checkpoint(
        model_path=args.model_path,
        test_dir=args.test_dir,
        model_type=args.model_type,
        output_dir=args.output_dir,
        n_bootstraps=args.bootstraps,
    )


if __name__ == "__main__":
    main()
