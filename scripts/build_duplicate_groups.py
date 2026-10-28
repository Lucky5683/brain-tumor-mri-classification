
import sys
import csv
import hashlib
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import TRAIN_DIR, CLASS_NAMES, IMAGE_EXTENSIONS
from src.data.dataset import find_images, make_stratified_split
from config import RANDOM_SEED, VALIDATION_FRACTION

OUT_DIR = ROOT / "experiments" / "duplicate_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    images_by_class = find_images(TRAIN_DIR)
    train_paths, _, val_paths, _ = make_stratified_split(
        images_by_class,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    train_set = {p.resolve() for p in train_paths}
    val_set = {p.resolve() for p in val_paths}

    print("Hashing Training images...")
    groups = defaultdict(list)
    total = sum(len(paths) for paths in images_by_class.values())
    done = 0

    for class_name in CLASS_NAMES:
        for path in images_by_class[class_name]:
            groups[sha256(path)].append(path)
            done += 1
            if done % 500 == 0:
                print(f"Processed {done}/{total}")

    rows = []
    cross_split_groups = 0

    for digest, paths in groups.items():
        if len(paths) < 2:
            continue

        splits = {
            "train" if path.resolve() in train_set else "validation"
            for path in paths
        }
        if len(splits) > 1:
            cross_split_groups += 1

        for path in paths:
            rows.append({
                "sha256": digest,
                "file": str(path.relative_to(ROOT)),
                "class": path.parent.name,
                "split": (
                    "train" if path.resolve() in train_set else "validation"
                ),
                "group_size": len(paths),
                "crosses_split": len(splits) > 1,
            })

    output = OUT_DIR / "training_exact_duplicate_groups.csv"
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sha256", "file", "class", "split",
                "group_size", "crosses_split",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n=== EXACT DUPLICATE GROUP AUDIT ===")
    print(f"Training images: {total}")
    print(f"Images belonging to duplicate groups: {len(rows)}")
    print(f"Duplicate groups crossing train/validation: {cross_split_groups}")
    print(f"Report saved: {output}")
    print("No images were modified.")


if __name__ == "__main__":
    main()
