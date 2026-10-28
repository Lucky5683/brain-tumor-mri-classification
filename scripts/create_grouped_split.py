
from pathlib import Path
from collections import Counter
import csv
import random

import imagehash
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

# Run from the project root.
DATASET_DIR = Path("dataset/Training")
OUTPUT_DIR = Path("experiments/duplicate_aware_split")
SEED = 42
PHASH_THRESHOLD = 4

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load image paths and class labels.
records = []
for class_dir in sorted(DATASET_DIR.iterdir()):
    if not class_dir.is_dir():
        continue

    for path in sorted(class_dir.iterdir()):
        if path.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".bmp", ".webp"
        }:
            records.append({
                "path": str(path),
                "label": class_dir.name,
                "hash": None,
            })

if not records:
    raise RuntimeError(f"No images found in {DATASET_DIR.resolve()}")

print(f"Images found: {len(records)}")
print("Computing perceptual hashes...")

for i, record in enumerate(records):
    with Image.open(record["path"]) as image:
        record["hash"] = imagehash.phash(image.convert("RGB"))

    if (i + 1) % 500 == 0:
        print(f"Processed {i + 1}/{len(records)}")

# Union-find: connect similar images into the same group.
parent = list(range(len(records)))
rank = [0] * len(records)


def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(a, b):
    root_a, root_b = find(a), find(b)
    if root_a == root_b:
        return

    if rank[root_a] < rank[root_b]:
        root_a, root_b = root_b, root_a

    parent[root_b] = root_a
    if rank[root_a] == rank[root_b]:
        rank[root_a] += 1


# Compare within each class first, as in the earlier audit.
# Cross-class near matches are reported separately for review.
by_class = {}
for i, record in enumerate(records):
    by_class.setdefault(record["label"], []).append(i)

candidate_pairs = 0

for label, indices in by_class.items():
    print(f"Comparing class: {label} ({len(indices)} images)")

    for position, i in enumerate(indices):
        for j in indices[position + 1:]:
            distance = records[i]["hash"] - records[j]["hash"]

            if distance <= PHASH_THRESHOLD:
                union(i, j)
                candidate_pairs += 1

# Assign a stable group ID to each connected component.
root_to_group = {}
for i in range(len(records)):
    root = find(i)
    if root not in root_to_group:
        root_to_group[root] = f"G{len(root_to_group):06d}"
    records[i]["group"] = root_to_group[root]

# Group split with approximate class stratification.
labels = [r["label"] for r in records]
groups = [r["group"] for r in records]

splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED,
)

train_idx, val_idx = next(
    splitter.split(records, labels, groups)
)

train_set, val_set = set(train_idx), set(val_idx)

assert not (
    {groups[i] for i in train_set}
    & {groups[i] for i in val_set}
), "Leakage: a group appears in both splits!"

# Write the manifest.
manifest_path = OUTPUT_DIR / "manifest.csv"
with manifest_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f, fieldnames=["path", "label", "group", "split"]
    )
    writer.writeheader()

    for i, record in enumerate(records):
        writer.writerow({
            "path": record["path"],
            "label": record["label"],
            "group": record["group"],
            "split": "train" if i in train_set else "validation",
        })

print("\nSplit complete")
print(f"Total images: {len(records)}")
print(f"Candidate similar pairs: {candidate_pairs}")
print(f"Groups created: {len(root_to_group)}")
print(f"Training images: {len(train_idx)}")
print(f"Validation images: {len(val_idx)}")
print(f"Manifest: {manifest_path}")
print("\nClass counts:")
for split_name, indices in [
    ("train", train_idx),
    ("validation", val_idx),
]:
    print(split_name, dict(Counter(labels[i] for i in indices)))
