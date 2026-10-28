
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT_DIR = ROOT / "experiments" / "duplicate_audit"


def read_csv(filename):
    with (AUDIT_DIR / filename).open(
        "r", newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def main():
    exact = read_csv("exact_duplicate_pairs.csv")
    near = read_csv("near_duplicate_candidates.csv")

    cross_near = [
        row for row in near
        if row["split_a"] != row["split_b"]
    ]

    print("\n=== EXACT DUPLICATE PAIRS BY CLASS ===")
    for class_name, count in sorted(
        Counter(row["class"] for row in exact).items()
    ):
        print(f"{class_name}: {count}")

    print("\n=== CROSS-TRAINING/TESTING CANDIDATES BY CLASS ===")
    for class_name, count in sorted(
        Counter(row["class"] for row in cross_near).items()
    ):
        print(f"{class_name}: {count}")

    print("\n=== CROSS-SPLIT PERCEPTUAL DISTANCES ===")
    for distance, count in sorted(
        Counter(int(row["phash_distance"]) for row in cross_near).items()
    ):
        print(f"Distance {distance}: {count} pairs")

    file_counts = Counter()
    for row in cross_near:
        file_counts[row["file_a"]] += 1
        file_counts[row["file_b"]] += 1

    print("\n=== MOST FREQUENTLY MATCHED FILES ===")
    for filename, count in file_counts.most_common(20):
        print(f"{count:3} matches | {filename}")

    print("\nSummary complete. No dataset files were modified.")


if __name__ == "__main__":
    main()
