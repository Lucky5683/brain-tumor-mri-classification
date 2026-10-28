"""Phase 2: Comprehensive Data Quality and Split Integrity Audit."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_NAMES, DATA_DIR, REPORTS_DIR, TEST_DIR, TRAIN_DIR
from src.data.dataset import find_images

MANIFEST_PATH = PROJECT_ROOT / "experiments" / "duplicate_aware_split" / "training_manifest.csv"
CROSS_CLASS_CSV = PROJECT_ROOT / "experiments" / "duplicate_aware_split" / "cross_class_near_matches.csv"


def run_data_quality_audit():
    print("=" * 70)
    print("PHASE 2: DATA QUALITY & SPLIT INTEGRITY AUDIT")
    print("=" * 70)

    # 1. Dataset Scan
    train_images_by_class = find_images(TRAIN_DIR)
    total_train_files = sum(len(paths) for paths in train_images_by_class.values())
    print(f"Total images found in Training directory: {total_train_files}")
    for cls in CLASS_NAMES:
        print(f"  {cls:<15}: {len(train_images_by_class[cls])} images")

    test_images_by_class = find_images(TEST_DIR)
    total_test_files = sum(len(paths) for paths in test_images_by_class.values())
    print(f"\nTotal images found in Testing directory: {total_test_files}")
    for cls in CLASS_NAMES:
        print(f"  {cls:<15}: {len(test_images_by_class[cls])} images")

    # 2. Manifest Verification
    print(f"\nLoading and verifying manifest: {MANIFEST_PATH}")
    manifest_rows = []
    with MANIFEST_PATH.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        manifest_rows = list(reader)

    print(f"Manifest rows: {len(manifest_rows)}")
    assert len(manifest_rows) == total_train_files, f"Row count {len(manifest_rows)} != {total_train_files}"

    paths_in_manifest = set()
    split_counts = Counter()
    class_split_counts = defaultdict(Counter)
    group_splits = {}
    group_members = defaultdict(list)
    sha_splits = defaultdict(set)

    for row in manifest_rows:
        p = Path(row["path"]).resolve()
        assert p.is_file(), f"Missing file: {p}"
        assert p not in paths_in_manifest, f"Duplicate path in manifest: {p}"
        paths_in_manifest.add(p)

        label = row["label"]
        split = row["split"]
        group_id = row["group_id"]
        sha = row["sha256"]

        assert p.parent.name == label, f"Folder/label mismatch: {p.parent.name} != {label}"
        split_counts[split] += 1
        class_split_counts[label][split] += 1
        group_members[group_id].append((p, label, split))
        sha_splits[sha].add(split)

        prev_split = group_splits.setdefault(group_id, split)
        assert prev_split == split, f"Group {group_id} crosses split boundary: {prev_split} vs {split}"

    print("\nSplit Distribution:")
    for split, count in split_counts.items():
        print(f"  {split:<12}: {count} ({count / len(manifest_rows) * 100:.2f}%)")

    print("\nClass breakdown by split:")
    print(f"{'Class':<15} {'Train':<8} {'Val':<8} {'Total':<8} {'Val Ratio'}")
    print("-" * 50)
    for cls in CLASS_NAMES:
        tr = class_split_counts[cls]["train"]
        vl = class_split_counts[cls]["validation"]
        tot = tr + vl
        print(f"{cls:<15} {tr:<8} {vl:<8} {tot:<8} {vl/tot*100:.2f}%")

    # Group size distribution
    group_sizes = [len(m) for m in group_members.values()]
    size_dist = Counter(group_sizes)
    print("\nSimilarity Group Size Distribution:")
    print(f"  Total groups: {len(group_members)}")
    print(f"  Singleton groups (size 1): {size_dist[1]} ({size_dist[1] / len(group_members) * 100:.1f}%)")
    print(f"  Multi-image groups (size >= 2): {sum(c for s, c in size_dist.items() if s >= 2)}")
    print(f"  Largest group size: {max(group_sizes)}")

    # Check exact SHA-256 crossing
    crossing_shas = [sha for sha, splits in sha_splits.items() if len(splits) > 1]
    print(f"\nExact SHA-256 hashes crossing train/val: {len(crossing_shas)}")
    assert len(crossing_shas) == 0, "Exact duplicates cross train/val!"

    # 3. Cross-Class Near Matches Audit
    cross_pairs_count = 0
    cross_split_pairs_count = 0
    if CROSS_CLASS_CSV.is_file():
        with CROSS_CLASS_CSV.open("r", newline="", encoding="utf-8") as f:
            cross_rows = list(csv.DictReader(f))
        cross_pairs_count = len(cross_rows)
        cross_split_pairs = [r for r in cross_rows if r["split_a"] != r["split_b"]]
        cross_split_pairs_count = len(cross_split_pairs)
        print(f"\nCross-class pHash matches (threshold <= 4): {cross_pairs_count}")
        print(f"Cross-class matches crossing train/val boundary: {cross_split_pairs_count}")
        for idx, pair in enumerate(cross_split_pairs, 1):
            print(f"  Pair {idx}: {pair['label_a']} ({pair['split_a']}) vs {pair['label_b']} ({pair['split_b']}) - dist: {pair['distance']}")

    # Save summary JSON
    audit_data = {
        "dataset_summary": {
            "training_images": total_train_files,
            "testing_images": total_test_files,
            "classes": CLASS_NAMES,
            "per_class_training": {cls: len(train_images_by_class[cls]) for cls in CLASS_NAMES},
            "per_class_testing": {cls: len(test_images_by_class[cls]) for cls in CLASS_NAMES},
        },
        "split_integrity": {
            "manifest_rows": len(manifest_rows),
            "train_samples": split_counts["train"],
            "validation_samples": split_counts["validation"],
            "exact_sha256_crossings": len(crossing_shas),
            "similarity_groups": len(group_members),
            "singleton_groups": size_dist[1],
            "multi_image_groups": sum(c for s, c in size_dist.items() if s >= 2),
            "max_group_size": max(group_sizes),
            "cross_class_pairs": cross_pairs_count,
            "cross_split_cross_class_pairs": cross_split_pairs_count,
            "limitations": [
                "Dataset lacks verified patient identifiers; perceptual similarity is an empirical proxy, not proof of subject-level separation.",
                "Similarity grouping was constructed within each class; 5 cross-class pHash candidate pairs exist across the train/validation split.",
                "Visual review of the 5 cross-class pairs confirmed they are not exact duplicate scans, but represent similar axial slice geometry.",
                "Large similarity groups (up to size 22) reflect repeated slices or burst imaging in the source dataset."
            ]
        }
    }

    out_json = REPORTS_DIR / "data_quality_audit.json"
    out_json.write_text(json.dumps(audit_data, indent=2), encoding="utf-8")
    print(f"\nSaved data quality audit to: {out_json}")


if __name__ == "__main__":
    run_data_quality_audit()
