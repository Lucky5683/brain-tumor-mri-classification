"""Prepare the perceptual-similarity grouped manifest for the project loader."""
import csv
import hashlib
from pathlib import Path

SOURCE = Path("experiments/duplicate_aware_split/manifest.csv")
DESTINATION = Path(
    "experiments/duplicate_aware_split/training_manifest.csv"
)

def main():
    if not SOURCE.is_file():
        raise FileNotFoundError(f"Source manifest not found: {SOURCE.resolve()}")

    with SOURCE.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        source_columns = set(reader.fieldnames or [])

    required = {"path", "label", "group", "split"}
    if not rows or not required.issubset(source_columns):
        raise ValueError(
            f"Expected source columns {sorted(required)}; found {sorted(source_columns)}"
        )

    seen_paths = set()
    group_splits = {}
    output_rows = []

    for row in rows:
        path = Path(row["path"]).resolve()
        label = row["label"].strip()
        split = row["split"].strip().lower()
        group_id = row["group"].strip()

        if not path.is_file():
            raise FileNotFoundError(f"Manifest image does not exist: {path}")
        if path.parent.name != label:
            raise ValueError(
                f"Label/path mismatch: label={label!r}, path={path}"
            )
        if split not in {"train", "validation"}:
            raise ValueError(f"Invalid split {split!r} for {path}")
        if not group_id:
            raise ValueError(f"Missing group ID for {path}")
        if str(path) in seen_paths:
            raise ValueError(f"Duplicate path in source manifest: {path}")
        seen_paths.add(str(path))

        previous_split = group_splits.setdefault(group_id, split)
        if previous_split != split:
            raise ValueError(
                f"Similarity group {group_id} crosses train/validation"
            )

        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        output_rows.append({
            "path": str(path),
            "label": label,
            "split": split,
            "sha256": digest,
            "group_id": group_id,
        })

    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "label", "split", "sha256", "group_id"]
        )
        writer.writeheader()
        writer.writerows(output_rows)

    counts = {
        split: sum(row["split"] == split for row in output_rows)
        for split in ("train", "validation")
    }
    print(f"Prepared manifest: {DESTINATION.resolve()}")
    print(f"Images verified: {len(output_rows)}")
    print(f"Similarity groups: {len(group_splits)}")
    print(f"Training images: {counts['train']}")
    print(f"Validation images: {counts['validation']}")
    print("Group consistency check: passed")

if __name__ == "__main__":
    main()
