
import csv
from pathlib import Path

csv_path = Path(
    "experiments/duplicate_aware_split/cross_class_near_matches.csv"
)

with csv_path.open("r", newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))

pairs = [row for row in rows if row["split_a"] != row["split_b"]]

print(f"\nCross-split candidate pairs: {len(pairs)}\n")

for number, row in enumerate(pairs, start=1):
    print(f"PAIR {number}")
    print(f"  Image A: {row['path_a']}")
    print(f"  Class A: {row['label_a']} | Split: {row['split_a']}")
    print(f"  Image B: {row['path_b']}")
    print(f"  Class B: {row['label_b']} | Split: {row['split_b']}")
    print(f"  pHash distance: {row['distance']}")
    print()

print("Use these filenames to inspect the five pairs individually.")
