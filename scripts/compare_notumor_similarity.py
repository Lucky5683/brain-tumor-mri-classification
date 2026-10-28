
from pathlib import Path
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent.parent

PAIRS = [
    ("Tr-no_1039.jpg", "Te-no_163.jpg"),
    ("Tr-no_1051.jpg", "Te-no_163.jpg"),
    ("Tr-no_11.jpg", "Te-no_24.jpg"),
    ("Tr-no_19.jpg", "Te-no_24.jpg"),
    ("Tr-no_1170.jpg", "Te-no_61.jpg"),
    ("Tr-no_1013.jpg", "Te-no_316.jpg"),
]


def load_resized(path):
    with Image.open(path) as image:
        image = image.convert("L").resize(
            (224, 224), Image.Resampling.LANCZOS
        )
        return np.asarray(image, dtype=np.float32)


def main():
    print("Starting MRI similarity comparison...", flush=True)

    for train_name, test_name in PAIRS:
        train_path = ROOT / "dataset" / "Training" / "notumor" / train_name
        test_path = ROOT / "dataset" / "Testing" / "notumor" / test_name

        if not train_path.exists() or not test_path.exists():
            print(f"Missing: {train_name} or {test_name}")
            continue

        a = load_resized(train_path)
        b = load_resized(test_path)

        ssim = structural_similarity(a, b, data_range=255)
        mae = float(np.mean(np.abs(a - b)) / 255.0)

        print(f"\n{train_name} <-> {test_name}")
        print(f"SSIM: {ssim:.4f}")
        print(f"Normalized pixel difference: {mae:.4f}")


if __name__ == "__main__":
    main()
