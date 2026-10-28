"""Robustness evaluation script: tests prediction stability under controlled image perturbations."""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import tensorflow as tf

from config import DATA_DIR, REPORTS_DIR, TEST_DIR
import src.models.compat
from src.data.dataset import get_test_dataset
from src.evaluation.robustness import run_robustness_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model prediction robustness under perturbations.")
    parser.add_argument("--model-path", type=Path, default=Path("efficientnet_b0.keras"))
    parser.add_argument("--test-dir", type=Path, default=TEST_DIR)
    parser.add_argument("--model-type", choices=["cnn", "efficientnet"], default="efficientnet")
    parser.add_argument("--sample-limit", type=int, default=200)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()

    print(f"Loading model checkpoint: {args.model_path.resolve()} ...")
    model = tf.keras.models.load_model(args.model_path)

    print(f"Loading test images from: {args.test_dir.resolve()} ...")
    _, test_paths, test_labels = get_test_dataset(test_dir=args.test_dir, model_type=args.model_type)

    print(f"Running robustness audit on {min(len(test_paths), args.sample_limit)} test images across 8 perturbation modes...")
    report, df = run_robustness_test(
        model=model,
        image_paths=test_paths,
        true_labels=test_labels,
        sample_limit=args.sample_limit,
        model_type=args.model_type,
        output_dir=args.output_dir,
    )

    print("\n" + "=" * 60)
    print("PERTURBATION ROBUSTNESS SUMMARY")
    print("=" * 60)
    for p_name, stats in report["perturbation_summary"].items():
        print(f"  {p_name:25s}: Stability = {stats['stability_rate']*100:5.1f}%, Flip Rate = {stats['prediction_flip_rate']*100:4.1f}%, Mean Delta-P = {stats['mean_probability_shift']:.4f}")
    print("=" * 60)
    print(f"Saved artifacts to {args.output_dir.resolve()}:")
    print("  - robustness_report.json")
    print("  - robustness_results.csv")


if __name__ == "__main__":
    main()
