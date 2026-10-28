
from pathlib import Path
from collections import Counter, defaultdict
import csv

import imagehash
from PIL import Image

DATASET_DIR = Path("dataset/Training")
MANIFEST = Path("experiments/duplicate_aware_split/manifest.csv")
OUTPUT_DIR = Path("experiments/duplicate_aware_split")
THRESHOLD = 4

with MANIFEST.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Inspect group sizes and label consistency.
group_members = defaultdict(list)

for row in rows:
    group_members[row["group"]].append(row)

sizes = sorted(
    ((group, members) for group, members in group_members.items()),
    key=lambda item: len(item[1]),
    reverse=True,
)

print("\nLargest similarity groups:")
for group, members in sizes[:20]:
    labels = Counter(r["label"] for r in members)
    print(f"{group}: {len(members)} images, labels={dict(labels)}")

# Compute hashes and search only across different classes.
records = []
for row in rows:
    path = Path(row["path"])
    with Image.open(path) as image:
        records.append({
            "path": str(path),
            "label": row["label"],
            "group": row["group"],
            "split": row["split"],
            "hash": imagehash.phash(image.convert("RGB")),
        })

by_class = defaultdict(list)
for record in records:
    by_class[record["label"]].append(record)

output_path = OUTPUT_DIR / "cross_class_near_matches.csv"
matches = 0

with output_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "path_a", "label_a", "split_a",
            "path_b", "label_b", "split_b",
            "distance", "same_group",
        ],
    )
    writer.writeheader()

    labels = sorted(by_class)
    for i, label_a in enumerate(labels):
        for label_b in labels[i + 1:]:
            print(f"Checking {label_a} vs {label_b}...")
            for a in by_class[label_a]:
                for b in by_class[label_b]:
                    distance = a["hash"] - b["hash"]
                    if distance <= THRESHOLD:
                        writer.writerow({
                            "path_a": a["path"],
                            "label_a": label_a,
                            "split_a": a["split"],
                            "path_b": b["path"],
                            "label_b": label_b,
                            "split_b": b["split"],
                            "distance": distance,
                            "same_group": a["group"] == b["group"],
                        })
                        matches += 1

print(f"\nCross-class candidate pairs: {matches}")
print(f"Saved: {output_path}")
print("Review these pairs visually before changing any labels.")
