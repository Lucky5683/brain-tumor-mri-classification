
from pathlib import Path
import csv
import math

from PIL import Image, ImageDraw

CSV_PATH = Path(
    "experiments/duplicate_aware_split/cross_class_near_matches.csv"
)
OUTPUT_PATH = Path(
    "experiments/duplicate_aware_split/cross_class_contact_sheet.jpg"
)

with CSV_PATH.open(newline="", encoding="utf-8") as f:
    pairs = list(csv.DictReader(f))

if not pairs:
    print("No cross-class candidate pairs to review.")
    raise SystemExit(0)

thumb_w, thumb_h = 220, 190
label_h, row_h = 65, 2 * 190 + 65
columns = 2
rows = math.ceil(len(pairs) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * row_h),
    "white",
)
draw = ImageDraw.Draw(sheet)

for i, pair in enumerate(pairs):
    col, row = i % columns, i // columns
    x, y = col * thumb_w, row * row_h

    for side, path_key, label_key, split_key in [
        ("A", "path_a", "label_a", "split_a"),
        ("B", "path_b", "label_b", "split_b"),
    ]:
        path = Path(pair[path_key])

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w - 12, thumb_h - 8))
            px = x + (thumb_w - im.width) // 2
            py = y + (0 if side == "A" else thumb_h)
            sheet.paste(im, (px, py))

        text_y = y + (thumb_h - 18 if side == "A" else 2 * thumb_h - 18)
        draw.text(
            (x + 5, text_y),
            f"{side}: {path.name} | {pair[label_key]} | "
            f"{pair[split_key]}",
            fill="black",
        )

    draw.text(
        (x + 5, y + 2 * thumb_h + 5),
        f"Pair {i + 1} | pHash distance: {pair['distance']}",
        fill="red",
    )

sheet.save(OUTPUT_PATH, quality=92)

split_crossings = sum(
    pair["split_a"] != pair["split_b"] for pair in pairs
)

print(f"Pairs reviewed: {len(pairs)}")
print(f"Pairs crossing train/validation: {split_crossings}")
print(f"Contact sheet saved: {OUTPUT_PATH}")
