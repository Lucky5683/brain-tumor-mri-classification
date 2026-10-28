
import csv
from collections import defaultdict, Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "experiments" / "duplicate_audit" / "near_duplicate_candidates.csv"
OUT_DIR = ROOT / "experiments" / "duplicate_audit" / "notumor_review"

THUMB_W = 200
THUMB_H = 180


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Focus on cross-split No Tumor pairs with identical perceptual hashes.
    candidates = [
        row for row in rows
        if row["class"] == "notumor"
        and row["split_a"] != row["split_b"]
        and row["phash_distance"] == "0"
    ]

    # Normalize each pair to (Training image, Testing image).
    pairs = []
    test_counts = Counter()

    for row in candidates:
        a, b = row["file_a"], row["file_b"]
        if row["split_a"] == "Testing":
            test_path, train_path = a, b
        else:
            test_path, train_path = b, a

        pairs.append((train_path, test_path))
        test_counts[test_path] += 1

    # Review the most frequently matched test images first.
    ranked_test = [p for p, _ in test_counts.most_common(12)]
    selected = []

    for test_path in ranked_test:
        matching = [
            (train_path, test_path)
            for train_path, candidate_test in pairs
            if candidate_test == test_path
        ]
        selected.extend(matching[:2])

    # Avoid duplicate pairs and cap the first review at 24 pairs.
    selected = list(dict.fromkeys(selected))[:24]

    if not selected:
        print("No matching pairs found. Check the CSV file and paths.")
        return

    pairs_per_sheet = 6
    for start in range(0, len(selected), pairs_per_sheet):
        batch = selected[start:start + pairs_per_sheet]
        sheet = Image.new(
            "RGB",
            (THUMB_W * 2 + 30, len(batch) * (THUMB_H + 65) + 10),
            "white",
        )
        draw = ImageDraw.Draw(sheet)

        for idx, (train_rel, test_rel) in enumerate(batch):
            y = idx * (THUMB_H + 65) + 5
            train_path = ROOT / train_rel
            test_path = ROOT / test_rel

            for x, path, label in [
                (5, train_path, "TRAIN"),
                (THUMB_W + 20, test_path, "TEST"),
            ]:
                draw.text((x, y), label, fill="black")
                try:
                    with Image.open(path) as im:
                        thumb = ImageOps.contain(
                            im.convert("RGB"), (THUMB_W, THUMB_H)
                        )
                    sheet.paste(thumb, (x, y + 18))
                except Exception as exc:
                    draw.text((x, y + 25), f"Could not open: {exc}", fill="red")

            draw.text(
                (5, y + THUMB_H + 20),
                f"Train: {Path(train_rel).name}",
                fill="black",
            )
            draw.text(
                (THUMB_W + 20, y + THUMB_H + 20),
                f"Test: {Path(test_rel).name}",
                fill="black",
            )

        out_path = OUT_DIR / f"notumor_phash0_sheet_{start // pairs_per_sheet + 1}.jpg"
        sheet.save(out_path, quality=92)
        print(f"Saved: {out_path}")

    print(f"\nCandidate pairs available: {len(candidates)}")
    print(f"Pairs selected for visual review: {len(selected)}")
    print("These are perceptual-hash matches, not confirmed duplicates.")


if __name__ == "__main__":
    main()
