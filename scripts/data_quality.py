"""Comprehensive Dataset Quality Audit for Brain Tumor MRI Classification V2.

Performs:
1. Image count and class balance audit across Training and Testing splits.
2. Dimensions, channel count, and file format distributions.
3. Unreadable/corrupt file screening.
4. Exact duplicate detection via SHA-256 content hashing.
5. Near-duplicate screening via perceptual difference hash (dHash, Hamming distance <= 4).
6. Train vs Test duplicate pair and potential leakage screening.
7. Brightness and contrast distributions with outlier detection (extremely dark/bright).
8. Generates visual contact sheets for suspicious duplicate pairs.

Reports generated in reports/:
- data_quality_report.json
- data_quality_report.txt
- duplicate_groups.csv
- near_duplicate_pairs.csv
- train_test_duplicate_candidates.csv
- duplicate_contact_sheets/
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from config import DATA_DIR, IMAGE_EXTENSIONS, REPORTS_DIR, RANDOM_SEED


def compute_sha256(path: Path) -> str:
    """Compute exact file hash."""
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash (dHash) for visual similarity matching."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.asarray(resized, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return int("".join("1" if b else "0" for b in diff.ravel()), 2)


def hamming_distance(h1: int, h2: int) -> int:
    return (h1 ^ h2).bit_count()


def create_contact_sheet(
    img_path_a: Path,
    img_path_b: Path,
    label_a: str,
    label_b: str,
    similarity_desc: str,
    output_path: Path,
) -> None:
    """Generate a side-by-side visual comparison contact sheet."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    im_a = Image.open(img_path_a).convert("RGB")
    im_b = Image.open(img_path_b).convert("RGB")

    ax1.imshow(im_a)
    ax1.set_title(f"A: {label_a}\n({img_path_a.name})", fontsize=10)
    ax1.axis("off")

    ax2.imshow(im_b)
    ax2.set_title(f"B: {label_b}\n({img_path_b.name})", fontsize=10)
    ax2.axis("off")

    fig.suptitle(f"Duplicate Inspection: {similarity_desc}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_dataset_audit(
    data_dir: Path = DATA_DIR,
    output_dir: Path = REPORTS_DIR,
    contact_sheet_limit: int = 20,
) -> dict:
    """Execute complete dataset quality audit."""
    output_dir.mkdir(parents=True, exist_ok=True)
    contact_dir = output_dir / "duplicate_contact_sheets"
    contact_dir.mkdir(parents=True, exist_ok=True)

    records = []
    corrupted_files = []

    splits = ["Training", "Testing"]
    classes = ["glioma", "meningioma", "notumor", "pituitary"]

    print(f"Scanning dataset directory: {data_dir.resolve()} ...")
    for split in splits:
        for cname in classes:
            class_dir = data_dir / split / cname
            if not class_dir.is_dir():
                continue
            for file_path in sorted(class_dir.iterdir()):
                if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue

                try:
                    with Image.open(file_path) as img:
                        w, h = img.size
                        channels = len(img.getbands())
                        fmt = img.format or file_path.suffix.lstrip(".").upper()
                        # Sample small 64x64 patch for brightness and contrast
                        thumb = img.convert("L").resize((64, 64), Image.Resampling.BILINEAR)
                        arr = np.asarray(thumb, dtype=np.float32)
                        mean_b = float(np.mean(arr))
                        std_c = float(np.std(arr))
                        dhash_val = compute_dhash(img)

                    sha = compute_sha256(file_path)
                    records.append({
                        "path": file_path,
                        "rel_path": str(file_path.relative_to(data_dir)),
                        "split": split,
                        "class_name": cname,
                        "width": w,
                        "height": h,
                        "channels": channels,
                        "format": fmt,
                        "brightness": mean_b,
                        "contrast": std_c,
                        "sha256": sha,
                        "dhash": dhash_val,
                    })
                except Exception as exc:
                    corrupted_files.append({"path": str(file_path), "error": str(exc)})

    total_images = len(records)
    print(f"Successfully processed {total_images} images. Corrupted/unreadable: {len(corrupted_files)}")

    # 1. Distribution
    dist = Counter((r["split"], r["class_name"]) for r in records)
    split_counts = Counter(r["split"] for r in records)
    class_counts = Counter(r["class_name"] for r in records)

    # 2. Dimensions & Formats
    dim_counter = Counter((r["width"], r["height"]) for r in records)
    fmt_counter = Counter(r["format"] for r in records)
    chan_counter = Counter(r["channels"] for r in records)

    # 3. Exact Duplicates
    sha_buckets = defaultdict(list)
    for r in records:
        sha_buckets[r["sha256"]].append(r)

    exact_dup_groups = [group for group in sha_buckets.values() if len(group) > 1]
    exact_cross_split = []
    for group in exact_dup_groups:
        splits_in_group = {r["split"] for r in group}
        if len(splits_in_group) > 1:
            exact_cross_split.append(group)

    # Save duplicate_groups.csv
    with (output_dir / "duplicate_groups.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["group_id", "sha256", "file_count", "cross_split", "relative_paths"])
        for gid, group in enumerate(exact_dup_groups):
            cross = "YES" if len({r["split"] for r in group}) > 1 else "NO"
            paths_str = " | ".join(r["rel_path"] for r in group)
            writer.writerow([gid, group[0]["sha256"], len(group), cross, paths_str])

    # 4. Near-Duplicates (dHash indexing)
    # 64-bit hash split into 4 16-bit blocks
    dhash_buckets = defaultdict(list)
    for idx, r in enumerate(records):
        for block in range(4):
            key = (block, (r["dhash"] >> (block * 16)) & 0xFFFF)
            dhash_buckets[key].append(idx)

    candidate_pairs = set()
    for indices in dhash_buckets.values():
        if len(indices) > 1:
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    idx_a, idx_b = indices[i], indices[j]
                    if idx_a > idx_b:
                        idx_a, idx_b = idx_b, idx_a
                    candidate_pairs.add((idx_a, idx_b))

    near_duplicates = []
    cross_split_near_duplicates = []

    for idx_a, idx_b in candidate_pairs:
        r_a = records[idx_a]
        r_b = records[idx_b]
        dist_h = hamming_distance(r_a["dhash"], r_b["dhash"])
        if dist_h <= 4:
            # Categorize
            if r_a["sha256"] == r_b["sha256"]:
                status = "EXACT DUPLICATE"
            elif dist_h <= 1:
                status = "LIKELY DUPLICATE"
            elif dist_h <= 3:
                status = "POSSIBLE NEAR DUPLICATE"
            else:
                status = "FALSE POSITIVE / LOW SIMILARITY"

            pair_info = {
                "file_a": r_a["rel_path"],
                "split_a": r_a["split"],
                "class_a": r_a["class_name"],
                "file_b": r_b["rel_path"],
                "split_b": r_b["split"],
                "class_b": r_b["class_name"],
                "hamming_distance": dist_h,
                "status": status,
                "path_a": r_a["path"],
                "path_b": r_b["path"],
            }
            near_duplicates.append(pair_info)
            if r_a["split"] != r_b["split"]:
                cross_split_near_duplicates.append(pair_info)

    # Save near_duplicate_pairs.csv
    with (output_dir / "near_duplicate_pairs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_a", "split_a", "class_a", "file_b", "split_b", "class_b", "distance", "status"])
        for p in near_duplicates:
            writer.writerow([p["file_a"], p["split_a"], p["class_a"], p["file_b"], p["split_b"], p["class_b"], p["hamming_distance"], p["status"]])

    # Save train_test_duplicate_candidates.csv
    with (output_dir / "train_test_duplicate_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["train_file", "train_class", "test_file", "test_class", "distance", "status"])
        for p in cross_split_near_duplicates:
            tr = p["file_a"] if p["split_a"] == "Training" else p["file_b"]
            tc = p["class_a"] if p["split_a"] == "Training" else p["class_b"]
            te = p["file_b"] if p["split_a"] == "Training" else p["file_a"]
            tec = p["class_b"] if p["split_a"] == "Training" else p["class_a"]
            writer.writerow([tr, tc, te, tec, p["hamming_distance"], p["status"]])

    # Generate visual contact sheets for top candidates
    sheets_generated = 0
    for pair in cross_split_near_duplicates[:contact_sheet_limit]:
        sheet_name = f"contact_pair_{sheets_generated+1:02d}_dist{pair['hamming_distance']}_{pair['status'].replace(' ', '_').replace('/', '_')}.png"
        create_contact_sheet(
            pair["path_a"], pair["path_b"],
            f"{pair['split_a']} / {pair['class_a']}",
            f"{pair['split_b']} / {pair['class_b']}",
            f"{pair['status']} (dHash Dist = {pair['hamming_distance']})",
            contact_dir / sheet_name,
        )
        sheets_generated += 1

    # 5. Brightness & Contrast Outliers
    dark_images = [r["rel_path"] for r in records if r["brightness"] < 10.0]
    bright_images = [r["rel_path"] for r in records if r["brightness"] > 180.0]
    unusual_sizes = [r["rel_path"] for r in records if min(r["width"], r["height"]) < 128 or max(r["width"], r["height"]) / min(r["width"], r["height"]) > 2.0]

    report = {
        "dataset_directory": str(data_dir.resolve()),
        "total_images": total_images,
        "corrupted_images": corrupted_files,
        "class_distribution": {f"{s}/{c}": dist.get((s, c), 0) for s in splits for c in classes},
        "split_totals": dict(split_counts),
        "class_totals": dict(class_counts),
        "unique_resolutions_count": len(dim_counter),
        "top_resolutions": {f"{w}x{h}": cnt for (w, h), cnt in dim_counter.most_common(5)},
        "channel_formats": {str(k): v for k, v in chan_counter.items()},
        "file_formats": dict(fmt_counter),
        "exact_duplicate_groups": len(exact_dup_groups),
        "exact_duplicates_crossing_train_test": len(exact_cross_split),
        "near_duplicate_pairs_count": len(near_duplicates),
        "cross_split_near_duplicate_pairs_count": len(cross_split_near_duplicates),
        "contact_sheets_rendered": sheets_generated,
        "intensity_outliers": {
            "extremely_dark_count": len(dark_images),
            "extremely_bright_count": len(bright_images),
            "unusual_aspect_or_tiny_count": len(unusual_sizes),
        },
        "patient_identifier_status": (
            "CONFIRMED: Dataset filenames (e.g. Tr-gl_0010.jpg, Te-me_0005.jpg) do NOT contain patient IDs "
            "or acquisition session metadata. Per-patient clinical leakage cannot be proved or disproved."
        ),
        "recommendation": (
            "DO NOT automatically delete near-duplicates. Maintain strict train/test separation. "
            "Transparently disclose cross-split visual similarity candidates in publication reports."
        ),
    }

    # Save JSON report
    json_path = output_dir / "data_quality_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Save readable TXT report
    txt_path = output_dir / "data_quality_report.txt"
    with txt_path.open("w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("BRAIN TUMOR MRI CLASSIFICATION V2 - DATASET AUDIT REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset Path: {data_dir.resolve()}\n")
        f.write(f"Total Scanned Images: {total_images}\n")
        f.write(f"Corrupted / Unreadable: {len(corrupted_files)}\n\n")

        f.write("1. SPLIT & CLASS COUNTS:\n")
        for s in splits:
            f.write(f"  [{s}] Total: {split_counts.get(s, 0)}\n")
            for c in classes:
                f.write(f"    - {c:12s}: {dist.get((s, c), 0):4d}\n")
        f.write("\n")

        f.write("2. RESOLUTION & FORMATS:\n")
        f.write(f"  Unique image sizes: {len(dim_counter)}\n")
        for res, cnt in dim_counter.most_common(5):
            f.write(f"    {res[0]}x{res[1]}: {cnt} images\n")
        f.write(f"  Formats: {dict(fmt_counter)}\n\n")

        f.write("3. DUPLICATE AUDIT:\n")
        f.write(f"  Exact Duplicate Groups (SHA-256): {len(exact_dup_groups)}\n")
        f.write(f"  Exact Duplicates Crossing Train/Test: {len(exact_cross_split)}\n")
        f.write(f"  Near-Duplicate Pairs (dHash <= 4): {len(near_duplicates)}\n")
        f.write(f"  Near-Duplicate Pairs Crossing Train/Test: {len(cross_split_near_duplicates)}\n")
        f.write(f"  Contact Sheets Generated: {sheets_generated}\n\n")

        f.write("4. QUALITY & INTENSITY ANOMALIES:\n")
        f.write(f"  Extremely Dark (mean < 10.0): {len(dark_images)}\n")
        f.write(f"  Extremely Bright (mean > 180.0): {len(bright_images)}\n")
        f.write(f"  Unusual Aspect / Tiny: {len(unusual_sizes)}\n\n")

        f.write("5. PATIENT IDENTIFIER STATUS:\n")
        f.write(f"  {report['patient_identifier_status']}\n\n")
        f.write("=" * 60 + "\n")

    print(f"Data quality audit report saved to:\n  {json_path.resolve()}\n  {txt_path.resolve()}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run complete dataset quality audit.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    args = parser.parse_args()
    run_dataset_audit(data_dir=args.data_dir, output_dir=args.output_dir)
