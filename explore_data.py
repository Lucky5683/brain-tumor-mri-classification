"""Explore the Brain Tumor MRI dataset before model training.

Expected dataset layout:
dataset/
    Training/<class name>/<image files>
    Testing/<class name>/<image files>

Run from the project folder:
    python explore_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, UnidentifiedImageError


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def image_dhash(image: Image.Image) -> int:
    """Create a small perceptual hash for finding visually similar images."""
    gray = image.convert("L").resize((9, 8))
    pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    return int("".join("1" if bit else "0" for bit in bits.ravel()), 2)


def hamming_distance(first: int, second: int) -> int:
    return (first ^ second).bit_count()


def find_near_duplicate_pairs(records: list[dict], max_distance: int = 4) -> list[tuple[dict, dict, int]]:
    """Find dHash-similar pairs without comparing every possible image pair.

    Any two 64-bit hashes differing in at most four bits share at least one of
    their four 16-bit blocks, so only records in the same block are compared.
    """
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        for block in range(4):
            value = (record["dhash"] >> (block * 16)) & 0xFFFF
            buckets[(block, value)].append(index)

    candidate_pairs: set[tuple[int, int]] = set()
    for indices in buckets.values():
        for first, second in itertools.combinations(indices, 2):
            candidate_pairs.add((first, second))

    matches = []
    for first, second in candidate_pairs:
        distance = hamming_distance(records[first]["dhash"], records[second]["dhash"])
        if distance <= max_distance:
            matches.append((records[first], records[second], distance))
    return matches


def save_sample_grid(records: list[dict], output_path: Path) -> None:
    """Save one example from each available split/class combination."""
    grouped: dict[tuple[str, str], dict] = {}
    for record in records:
        grouped.setdefault((record["split"], record["class_name"]), record)

    samples = list(grouped.values())
    columns = 4
    cell_width, cell_height, label_height = 260, 220, 40
    rows = max(1, int(np.ceil(len(samples) / columns)))
    canvas = Image.new("RGB", (columns * cell_width, rows * (cell_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)

    for index, record in enumerate(samples):
        with Image.open(record["path"]) as image:
            preview = image.convert("RGB")
            preview.thumbnail((cell_width - 10, cell_height - 10))
        row, column = divmod(index, columns)
        x = column * cell_width + (cell_width - preview.width) // 2
        y = row * (cell_height + label_height) + (cell_height - preview.height) // 2
        canvas.paste(preview, (x, y))
        draw.text((column * cell_width + 8, row * (cell_height + label_height) + cell_height + 8),
                  f"{record['split']}: {record['class_name']}", fill="black")

    canvas.save(output_path)


def explore_dataset(data_dir: Path, output_dir: Path) -> None:
    split_dirs = {"Training": data_dir / "Training", "Testing": data_dir / "Testing"}
    missing = [name for name, path in split_dirs.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} folder(s) in {data_dir}. "
            "Download and extract the dataset so it contains Training/ and Testing/."
        )

    records: list[dict] = []
    corrupt_files: list[str] = []
    dimensions: Counter = Counter()
    formats: Counter = Counter()
    class_counts: dict[str, Counter] = {name: Counter() for name in split_dirs}
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_square_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for split_name, split_dir in split_dirs.items():
        for class_dir in sorted(path for path in split_dir.iterdir() if path.is_dir()):
            for path in sorted(class_dir.rglob("*")):
                if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        rgb = image.convert("RGB")
                        # A 64x64 thumbnail gives fast, comparable dataset-level
                        # intensity statistics without loading every full image into
                        # a large floating-point array.
                        array = np.asarray(rgb.resize((64, 64)), dtype=np.float64)
                        width, height = rgb.size
                        file_hash = hashlib.md5(path.read_bytes()).hexdigest()
                        record = {
                            "path": path,
                            "split": split_name,
                            "class_name": class_dir.name,
                            "file_hash": file_hash,
                            "dhash": image_dhash(rgb),
                        }
                        records.append(record)
                        class_counts[split_name][class_dir.name] += 1
                        dimensions[(width, height)] += 1
                        formats[image.format or path.suffix.upper()] += 1
                        pixel_sum += array.sum(axis=(0, 1))
                        pixel_square_sum += np.square(array).sum(axis=(0, 1))
                        pixel_count += array.shape[0] * array.shape[1]
                        if len(records) % 500 == 0:
                            print(f"Scanned {len(records)} images...", flush=True)
                except (UnidentifiedImageError, OSError, ValueError) as error:
                    corrupt_files.append(f"{path}: {error}")

    if not records:
        raise RuntimeError("No supported image files were found.")

    output_dir.mkdir(parents=True, exist_ok=True)
    save_sample_grid(records, output_dir / "sample_images.png")

    by_file_hash: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_file_hash[record["file_hash"]].append(record)
    exact_groups = [group for group in by_file_hash.values() if len(group) > 1]
    cross_split_exact = [
        group for group in exact_groups if {item["split"] for item in group} == {"Training", "Testing"}
    ]

    print("Checking potential near-duplicates...", flush=True)
    near_pairs = find_near_duplicate_pairs(records)
    cross_split_near = [
        pair for pair in near_pairs if pair[0]["split"] != pair[1]["split"]
    ]

    mean = pixel_sum / pixel_count
    variance = pixel_square_sum / pixel_count - np.square(mean)
    standard_deviation = np.sqrt(np.maximum(variance, 0))

    lines = ["BRAIN TUMOR MRI DATASET EXPLORATION", "=" * 38, ""]
    lines.append(f"Dataset directory: {data_dir.resolve()}")
    lines.append(f"Readable images: {len(records)}")
    lines.append(f"Unreadable/corrupt images: {len(corrupt_files)}")
    lines.append("")
    lines.append("Images per class")
    all_classes = sorted(set(class_counts["Training"]) | set(class_counts["Testing"]))
    for class_name in all_classes:
        train_count = class_counts["Training"][class_name]
        test_count = class_counts["Testing"][class_name]
        lines.append(f"  {class_name}: training={train_count}, testing={test_count}, total={train_count + test_count}")
    lines.append("")
    lines.append(f"Image formats: {dict(formats)}")
    lines.append(f"Unique dimensions: {len(dimensions)}")
    lines.append("Most common dimensions: " + ", ".join(f"{size}: {count}" for size, count in dimensions.most_common(10)))
    lines.append("Pixel mean (RGB, 0-255; 64x64 sample per image): " + ", ".join(f"{value:.2f}" for value in mean))
    lines.append("Pixel std (RGB, 0-255; 64x64 sample per image): " + ", ".join(f"{value:.2f}" for value in standard_deviation))
    lines.append("")
    lines.append(f"Exact duplicate groups (same file bytes): {len(exact_groups)}")
    lines.append(f"Exact duplicate groups crossing train/test: {len(cross_split_exact)}")
    lines.append(f"Potential near-duplicate pairs (dHash distance <= 4): {len(near_pairs)}")
    lines.append(f"Potential near-duplicate pairs crossing train/test: {len(cross_split_near)}")
    lines.append("")
    lines.append("Notes")
    lines.append("- Near-duplicate results are a screening signal; visually inspect matches before deciding they are leakage.")
    lines.append("- Patient-level leakage cannot be ruled out because this dataset does not provide patient IDs.")

    if corrupt_files:
        lines.extend(["", "Unreadable files:"] + [f"  {item}" for item in corrupt_files])
    if cross_split_exact:
        lines.extend(["", "Train/test exact-overlap examples:"])
        for group in cross_split_exact[:10]:
            lines.append("  " + " | ".join(str(item["path"].relative_to(data_dir)) for item in group))
    if cross_split_near:
        lines.extend(["", "Train/test near-duplicate examples:"])
        for first, second, distance in cross_split_near[:10]:
            lines.append(f"  dHash={distance}: {first['path'].relative_to(data_dir)} | {second['path'].relative_to(data_dir)}")

    report_path = output_dir / "exploration_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved report: {report_path.resolve()}")
    print(f"Saved samples: {(output_dir / 'sample_images.png').resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore the Brain Tumor MRI dataset.")
    parser.add_argument("--data-dir", type=Path, default=Path("dataset"), help="Folder containing Training/ and Testing/.")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Where to save the report and sample image grid.")
    arguments = parser.parse_args()
    explore_dataset(arguments.data_dir, arguments.output_dir)


if __name__ == "__main__":
    main()
