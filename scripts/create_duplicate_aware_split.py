
import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

# Run this script from the project root.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = PROJECT_ROOT / "dataset" / "Training"
OUTPUT_DIR = PROJECT_ROOT / "experiments" / "duplicate_audit"

VALIDATION_FRACTION = 0.20
RANDOM_SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if not TRAIN_DIR.is_dir():
        raise FileNotFoundError(
            f"Training directory not found: {TRAIN_DIR}\n"
            "Check the dataset path before running this script."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect images by class and group identical files by SHA-256.
    records = []
    groups = defaultdict(list)
    label_names = sorted(
        p.name for p in TRAIN_DIR.iterdir() if p.is_dir()
    )

    if not label_names:
        raise RuntimeError(f"No class folders found in {TRAIN_DIR}")

    for label in label_names:
        class_dir = TRAIN_DIR / label
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                image_hash = sha256_file(path)
                record = {
                    "path": str(path.resolve()),
                    "label": label,
                    "sha256": image_hash,
                }
                records.append(record)
                groups[image_hash].append(record)

    if not records:
        raise RuntimeError(f"No images found in {TRAIN_DIR}")

    # Report identical image content assigned to different class labels.
    label_conflicts = []
    for image_hash, members in groups.items():
        labels = sorted({m["label"] for m in members})
        if len(labels) > 1:
            label_conflicts.append({
                "sha256": image_hash,
                "labels": labels,
                "paths": [m["path"] for m in members],
            })

    # One group per unique SHA-256 hash: a group can never be split
    # between train and validation.
    group_items = []
    for image_hash, members in groups.items():
        class_counts = Counter(m["label"] for m in members)
        group_items.append({
            "sha256": image_hash,
            "members": members,
            "class_counts": class_counts,
            "size": len(members),
        })

    total_by_class = Counter(r["label"] for r in records)
    target_by_class = {
        label: total_by_class[label] * VALIDATION_FRACTION
        for label in label_names
    }
    target_total = len(records) * VALIDATION_FRACTION

    # Try multiple deterministic randomized greedy assignments and keep
    # the split closest to the desired per-class validation counts.
    best_assignment = None
    best_score = float("inf")

    for attempt in range(300):
        rng = random.Random(RANDOM_SEED + attempt)
        shuffled = group_items.copy()
        rng.shuffle(shuffled)

        # Larger groups first; randomization breaks ties across attempts.
        shuffled.sort(key=lambda g: g["size"], reverse=True)

        validation_groups = set()
        validation_counts = Counter()
        validation_total = 0

        def score(counts, total):
            class_error = sum(
                ((counts[label] - target_by_class[label])
                 / max(target_by_class[label], 1)) ** 2
                for label in label_names
            )
            total_error = (
                (total - target_total) / max(target_total, 1)
            ) ** 2
            return class_error + total_error

        for group in shuffled:
            # Compare assigning this whole group to validation versus train.
            candidate_counts = validation_counts.copy()
            candidate_counts.update(group["class_counts"])
            candidate_total = validation_total + group["size"]

            current_score = score(validation_counts, validation_total)
            candidate_score = score(candidate_counts, candidate_total)

            if candidate_score < current_score:
                validation_groups.add(group["sha256"])
                validation_counts = candidate_counts
                validation_total = candidate_total

        final_score = score(validation_counts, validation_total)
        if final_score < best_score:
            best_score = final_score
            best_assignment = validation_groups

    if best_assignment is None:
        raise RuntimeError("Could not construct a split.")

    # Write a manifest only. Original images are not moved or modified.
    manifest_path = OUTPUT_DIR / "duplicate_aware_split_manifest.csv"
    class_split_counts = defaultdict(Counter)

    with manifest_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["path", "label", "split", "sha256", "group_id"],
        )
        writer.writeheader()

        for record in records:
            split = (
                "validation"
                if record["sha256"] in best_assignment
                else "train"
            )
            class_split_counts[record["label"]][split] += 1
            writer.writerow({
                **record,
                "split": split,
                "group_id": record["sha256"],
            })

    # Verify that no exact-hash group crosses the new split.
    hash_splits = defaultdict(set)
    for record in records:
        split = (
            "validation"
            if record["sha256"] in best_assignment
            else "train"
        )
        hash_splits[record["sha256"]].add(split)

    crossing_groups = [
        image_hash for image_hash, splits in hash_splits.items()
        if len(splits) > 1
    ]

    summary = {
        "training_directory": str(TRAIN_DIR),
        "total_images": len(records),
        "unique_sha256_groups": len(groups),
        "train_images": sum(
            counts["train"] for counts in class_split_counts.values()
        ),
        "validation_images": sum(
            counts["validation"] for counts in class_split_counts.values()
        ),
        "requested_validation_fraction": VALIDATION_FRACTION,
        "random_seed": RANDOM_SEED,
        "class_counts": {
            label: {
                "total": total_by_class[label],
                "train": class_split_counts[label]["train"],
                "validation": class_split_counts[label]["validation"],
                "validation_target": round(target_by_class[label], 2),
            }
            for label in label_names
        },
        "cross_split_exact_duplicate_groups": len(crossing_groups),
        "cross_label_exact_duplicate_groups": len(label_conflicts),
        "cross_label_conflicts": label_conflicts,
        "optimization_score": best_score,
        "manifest": str(manifest_path),
        "official_testing_directory_modified": False,
    }

    summary_path = OUTPUT_DIR / "duplicate_aware_split_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nDuplicate-aware split created.")
    print(f"Images: {len(records)}")
    print(f"Train: {summary['train_images']}")
    print(f"Validation: {summary['validation_images']}")
    print(f"Exact duplicate groups crossing splits: {len(crossing_groups)}")
    print(f"Exact duplicate groups with conflicting labels: {len(label_conflicts)}")
    print("\nClass distribution:")
    for label, counts in summary["class_counts"].items():
        print(
            f"  {label}: train={counts['train']}, "
            f"validation={counts['validation']}, "
            f"target validation≈{counts['validation_target']}"
        )
    print(f"\nManifest: {manifest_path}")
    print(f"Summary:  {summary_path}")
    print("\nNo images were moved or deleted.")


if __name__ == "__main__":
    main()
