"""Root data_quality.py wrapper.

Delegates execution to scripts/data_quality.py.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from config import DATA_DIR, REPORTS_DIR
from scripts.data_quality import run_dataset_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run complete dataset quality audit.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run_dataset_audit(data_dir=args.data_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
