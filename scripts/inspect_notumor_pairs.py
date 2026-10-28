
from pathlib import Path
import hashlib
from PIL import Image
import imagehash

ROOT = Path(__file__).resolve().parent.parent

PAIRS = [
    ("Training", "Tr-no_1039.jpg", "Testing", "Te-no_163.jpg"),
    ("Training", "Tr-no_1051.jpg", "Testing", "Te-no_163.jpg"),
    ("Training", "Tr-no_11.jpg", "Testing", "Te-no_24.jpg"),
    ("Training", "Tr-no_19.jpg", "Testing", "Te-no_24.jpg"),
    ("Training", "Tr-no_1170.jpg", "Testing", "Te-no_61.jpg"),
    ("Training", "Tr-no_1013.jpg", "Testing", "Te-no_316.jpg"),
]


def inspect(path):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with Image.open(path) as image:
        dimensions = image.size
        mode = image.mode
        phash = str(imagehash.phash(image.convert("RGB")))
    return digest, dimensions, mode, phash


def main():
    for train_split, train_name, test_split, test_name in PAIRS:
        train_path = ROOT / "dataset" / train_split / "notumor" / train_name
        test_path = ROOT / "dataset" / test_split / "notumor" / test_name

        if not train_path.exists() or not test_path.exists():
            print(f"\nMISSING: {train_path} or {test_path}")
            continue

        a = inspect(train_path)
        b = inspect(test_path)

        print(f"\n{train_name} <-> {test_name}")
        print(f"  Training dimensions/mode: {a[1]} / {a[2]}")
        print(f"  Testing dimensions/mode:  {b[1]} / {b[2]}")
        print(f"  SHA-256 identical: {a[0] == b[0]}")
        print(f"  pHash distance: {imagehash.hex_to_hash(a[3]) - imagehash.hex_to_hash(b[3])}")


if __name__ == "__main__":
    main()
