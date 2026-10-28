
from pathlib import Path
import hashlib

import imagehash
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CLASS_DIR = ROOT / "dataset" / "Training" / "meningioma"
AUGMENTED = sorted(CLASS_DIR.glob("Tr-aug-me_*.jpg"))
OTHERS = sorted(
    p for p in CLASS_DIR.rglob("*")
    if p.is_file()
    and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    and p not in AUGMENTED
)


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def phash(path):
    with Image.open(path) as im:
        return imagehash.phash(im.convert("RGB"))


def main():
    print("Audit started", flush=True)
    print("Named augmented files:", len(AUGMENTED), flush=True)
    print("Other meningioma files:", len(OTHERS), flush=True)

    other_hashes = {}
    other_phashes = {}

    print("Hashing other images...", flush=True)
    for i, path in enumerate(OTHERS, start=1):
        other_hashes.setdefault(sha256(path), []).append(path)
        other_phashes[path] = phash(path)
        if i % 200 == 0:
            print(f"Processed {i}/{len(OTHERS)} other images", flush=True)

    exact_matches = []
    near_matches = []

    print("Comparing augmented images...", flush=True)
    for i, aug_path in enumerate(AUGMENTED, start=1):
        digest = sha256(aug_path)
        if digest in other_hashes:
            for match in other_hashes[digest]:
                exact_matches.append((aug_path.name, match.name))

        aug_phash = phash(aug_path)
        for path, other_phash in other_phashes.items():
            distance = aug_phash - other_phash
            if distance <= 4:
                near_matches.append(
                    (aug_path.name, path.name, distance)
                )

        if i % 10 == 0:
            print(f"Compared {i}/{len(AUGMENTED)} augmented images", flush=True)

    print("\nExact duplicate pairs:", len(exact_matches))
    for a, b in exact_matches[:30]:
        print(f"EXACT: {a} <-> {b}")

    print(
        "\nPerceptual-hash pairs with distance <= 4:",
        len(near_matches),
    )
    for a, b, distance in near_matches[:100]:
        print(f"NEAR (distance={distance}): {a} <-> {b}")

    print(
        "\nNote: perceptual-hash matches are candidates, "
        "not proof of the same original image."
    )


if __name__ == "__main__":
    main()
