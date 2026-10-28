
from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    CLASS_NAMES,
    TRAIN_DIR,
    RANDOM_SEED,
    VALIDATION_FRACTION,
)
from src.data.dataset import get_train_val_datasets

MODEL_PATH = ROOT / "models" / "multiscale_fusion_best.keras"
OUTPUT_DIR = (
    ROOT / "experiments" / "multiscale_validation" / "meningioma_errors"
)

IMAGE_SIZE = (224, 224)
IMAGES_PER_PAGE = 20


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Re-create the same deterministic validation split.
    _, val_ds, _, _, val_paths, val_labels = get_train_val_datasets(
        train_dir=TRAIN_DIR,
        model_type="efficientnet",
        batch_size=32,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
        use_augmentation=False,
    )

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    probabilities = model.predict(val_ds, verbose=1)

    y_true = np.asarray(val_labels)
    y_pred = np.argmax(probabilities, axis=1)

    meningioma_idx = CLASS_NAMES.index("meningioma")
    pituitary_idx = CLASS_NAMES.index("pituitary")

    # Actual meningioma, predicted as pituitary.
    mask = (y_true == meningioma_idx) & (y_pred == pituitary_idx)
    error_indices = np.flatnonzero(mask)

    print(f"\nMeningioma -> Pituitary errors: {len(error_indices)}")

    if len(error_indices) == 0:
        print("No matching errors found.")
        return

    records = []

    for idx in error_indices:
        probs = probabilities[idx]
        records.append({
            "image_path": str(val_paths[idx]),
            "actual_class": CLASS_NAMES[y_true[idx]],
            "predicted_class": CLASS_NAMES[y_pred[idx]],
            "predicted_confidence": float(probs[y_pred[idx]]),
            "meningioma_probability": float(probs[meningioma_idx]),
            "pituitary_probability": float(probs[pituitary_idx]),
            "glioma_probability": float(probs[CLASS_NAMES.index("glioma")]),
            "notumor_probability": float(probs[CLASS_NAMES.index("notumor")]),
        })

    df = pd.DataFrame(records)
    df = df.sort_values(
        "pituitary_probability", ascending=False
    ).reset_index(drop=True)

    csv_path = OUTPUT_DIR / "meningioma_predicted_as_pituitary.csv"
    df.to_csv(csv_path, index=False)
    print("Saved error details:", csv_path)

    # Create contact sheets of the most confusing images.
    page_count = math.ceil(len(df) / IMAGES_PER_PAGE)

    for page in range(page_count):
        start = page * IMAGES_PER_PAGE
        page_df = df.iloc[start:start + IMAGES_PER_PAGE]

        columns = 4
        rows = math.ceil(len(page_df) / columns)
        cell_w, cell_h = 250, 285

        sheet = Image.new(
            "RGB",
            (columns * cell_w, rows * cell_h),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for local_idx, (_, record) in enumerate(page_df.iterrows()):
            image_path = Path(record["image_path"])
            try:
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((cell_w - 16, 205))

                col = local_idx % columns
                row = local_idx // columns
                x = col * cell_w
                y = row * cell_h

                sheet.paste(image, (x + 8, y + 8))

                filename = image_path.name
                confidence = record["pituitary_probability"]

                draw.text(
                    (x + 8, y + 218),
                    filename[:34],
                    fill="black",
                )
                draw.text(
                    (x + 8, y + 238),
                    f"Actual: meningioma",
                    fill="black",
                )
                draw.text(
                    (x + 8, y + 258),
                    f"Pituitary probability: {confidence:.1%}",
                    fill="black",
                )

            except Exception as exc:
                print(f"Could not read {image_path}: {exc}")

        sheet_path = OUTPUT_DIR / f"contact_sheet_{page + 1}.jpg"
        sheet.save(sheet_path, quality=90)
        print("Saved contact sheet:", sheet_path)

    print("\nReview the contact sheets and CSV.")
    print("Do not relabel or delete images based on model predictions alone.")


if __name__ == "__main__":
    main()
