
import csv
from pathlib import Path
from PIL import Image, ImageDraw

csv_path = Path(
    "experiments/duplicate_aware_split/cross_class_near_matches.csv"
)
output_path = Path(
    "experiments/duplicate_aware_split/five_cross_split_pairs.jpg"
)

with csv_path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

pairs = [r for r in rows if r["split_a"] != r["split_b"]]

if not pairs:
    print("No cross-split pairs found.")
    raise SystemExit(0)

cell_w, cell_h = 280, 440
sheet = Image.new("RGB", (cell_w * 2, cell_h * len(pairs)), "white")
draw = ImageDraw.Draw(sheet)

for i, pair in enumerate(pairs):
    for j, side in enumerate(("a", "b")):
        path = Path(pair[f"path_{side}"])

        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((260, 350))
            x = j * cell_w + 10
            y = i * cell_h + 10
            sheet.paste(im, (x, y))

        label = (
            f"{path.name}\n"
            f"Class: {pair[f'label_{side}']}\n"
            f"Split: {pair[f'split_{side}']}"
        )
        draw.multiline_text(
            (j * cell_w + 10, i * cell_h + 365),
            label,
            fill="black",
            spacing=4,
        )

    draw.text(
        (10, i * cell_h + 420),
        f"Pair {i + 1} | pHash distance: {pair['distance']}",
        fill="red",
    )

sheet.save(output_path, quality=92)
print(f"Created: {output_path.resolve()}")
print(f"Pairs included: {len(pairs)}")
