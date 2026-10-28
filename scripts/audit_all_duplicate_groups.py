
import sys
import csv
import hashlib
from pathlib import Path
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import imagehash
from PIL import Image

from config import TRAIN_DIR, TEST_DIR, CLASS_NAMES, IMAGE_EXTENSIONS


OUTPUT_DIR = ROOT / "experiments" / "duplicate_audit"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path):
    with Image.open(path) as image:
        return imagehash.phash(image.convert("RGB"))


def collect_images():
    records = []
    for split_name, base in [("Training", TRAIN_DIR), ("Testing", TEST_DIR)]:
        for class_name in CLASS_NAMES:
            folder = Path(base) / class_name
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing class folder: {folder}")

            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    records.append({
                        "path": path,
                        "split": split_name,
                        "class": class_name,
                    })
    return records


def main():
    records = collect_images()
    print(f"Images found: {len(records)}")
    print("Hashing images; this may take a few minutes...")

    for i, item in enumerate(records, start=1):
        item["sha256"] = sha256(item["path"])
        item["phash"] = phash(item["path"])
        if i % 500 == 0:
            print(f"Processed {i}/{len(records)}")

    exact_pairs = []
    near_pairs = []

    for i, j in combinations(range(len(records)), 2):
        a, b = records[i], records[j]

        # Avoid comparing every pair of unrelated classes.
        if a["class"] != b["class"]:
            continue

        same_split = a["split"] == b["split"]
        if a["sha256"] == b["sha256"]:
            exact_pairs.append({
                "file_a": str(a["path"].relative_to(ROOT)),
                "file_b": str(b["path"].relative_to(ROOT)),
                "class": a["class"],
                "split_a": a["split"],
                "split_b": b["split"],
                "relationship": "same_split" if same_split else "cross_split",
                "phash_distance": a["phash"] - b["phash"],
            })
        else:
            distance = a["phash"] - b["phash"]
            if distance <= 4:
                near_pairs.append({
                    "file_a": str(a["path"].relative_to(ROOT)),
                    "file_b": str(b["path"].relative_to(ROOT)),
                    "class": a["class"],
                    "split_a": a["split"],
                    "split_b": b["split"],
                    "relationship": "same_split" if same_split else "cross_split",
                    "phash_distance": distance,
                })

    def save_csv(filename, rows):
        path = OUTPUT_DIR / filename
        fields = [
            "file_a", "file_b", "class", "split_a",
            "split_b", "relationship", "phash_distance"
        ]
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved: {path}")

    save_csv("exact_duplicate_pairs.csv", exact_pairs)
    save_csv("near_duplicate_candidates.csv", near_pairs)

    print("\n=== SUMMARY ===")
    print(f"Exact duplicate pairs: {len(exact_pairs)}")
    print(f"  Cross Training/Testing: {sum(r['split_a'] != r['split_b'] for r in exact_pairs)}")
    print(f"Near-duplicate candidates (excluding exact matches): {len(near_pairs)}")
    print(f"  Cross Training/Testing: {sum(r['split_a'] != r['split_b'] for r in near_pairs)}")
    print("\nPerceptual matches are candidates, not proof of duplicates.")
    print("Review the CSV files before making any dataset changes.")


if __name__ == "__main__":
    main()
