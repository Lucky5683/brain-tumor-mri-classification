import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hashlib
import imagehash
from PIL import Image

from config import RANDOM_SEED, TRAIN_DIR, VALIDATION_FRACTION
from src.data.dataset import find_images, make_stratified_split


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def phash(path):
    with Image.open(path) as image:
        return imagehash.phash(image.convert("RGB"))


def main():
    print("Reproducing the project's exact train/validation split...")

    images_by_class = find_images(TRAIN_DIR)
    train_paths, _, val_paths, _ = make_stratified_split(
        images_by_class,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )

    train_set = {p.resolve() for p in train_paths}
    val_set = {p.resolve() for p in val_paths}

    class_dir = Path(TRAIN_DIR) / "meningioma"
    augmented = sorted(class_dir.glob("Tr-aug-me_*.jpg"))
    others = sorted(
        p for p in class_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and p not in augmented
    )

    print(f"Augmented-name files: {len(augmented)}")
    print(f"Other images: {len(others)}")
    print(f"Training images overall: {len(train_paths)}")
    print(f"Validation images overall: {len(val_paths)}")

    print("\nHashing images...")
    other_hashes = {}
    other_phashes = {}

    for i, path in enumerate(others, start=1):
        other_hashes[path] = sha256(path)
        other_phashes[path] = phash(path)
        if i % 200 == 0:
            print(f"Hashed {i}/{len(others)} other images")

    exact_cross = []
    exact_same = []
    near_cross = []
    near_same = []

    for i, aug_path in enumerate(augmented, start=1):
        aug_sha = sha256(aug_path)
        aug_phash = phash(aug_path)
        aug_resolved = aug_path.resolve()
        aug_split = "train" if aug_resolved in train_set else "validation"

        for other_path in others:
            other_resolved = other_path.resolve()
            other_split = (
                "train" if other_resolved in train_set else "validation"
            )
            crosses_split = aug_split != other_split

            if aug_sha == other_hashes[other_path]:
                pair = (aug_path.name, other_path.name,
                        aug_split, other_split)
                (exact_cross if crosses_split else exact_same).append(pair)

            distance = aug_phash - other_phashes[other_path]
            if distance <= 4:
                pair = (aug_path.name, other_path.name, distance,
                        aug_split, other_split)
                (near_cross if crosses_split else near_same).append(pair)

        if i % 10 == 0:
            print(f"Compared {i}/{len(augmented)} augmented images")

    print("\n=== EXACT DUPLICATE PAIRS ===")
    print(f"Across train/validation: {len(exact_cross)}")
    for a, b, split_a, split_b in exact_cross:
        print(f"CROSS: {a} [{split_a}] <-> {b} [{split_b}]")

    print(f"\nWithin the same split: {len(exact_same)}")
    for a, b, split_a, split_b in exact_same:
        print(f"SAME: {a} [{split_a}] <-> {b} [{split_b}]")

    print("\n=== PERCEPTUAL HASH PAIRS (distance <= 4) ===")
    print(f"Across train/validation: {len(near_cross)}")
    for a, b, distance, split_a, split_b in near_cross:
        print(
            f"CROSS (distance={distance}): "
            f"{a} [{split_a}] <-> {b} [{split_b}]"
        )

    print(f"\nWithin the same split: {len(near_same)}")
    for a, b, distance, split_a, split_b in near_same:
        print(
            f"SAME (distance={distance}): "
            f"{a} [{split_a}] <-> {b} [{split_b}]"
        )

    print(
        "\nNote: perceptual-hash matches are candidates, "
        "not proof of related images or leakage."
    )


if __name__ == "__main__":
    main()
