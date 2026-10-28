from pathlib import Path

root = Path.cwd()
source = root / "scripts" / "train_spatial_fusion.py"
target = root / "scripts" / "train_spatial_fusion_weighted.py"

text = source.read_text(encoding="utf-8")

replacements = [
    (
        'OUT = ROOT / "experiments/spatial_fusion/seed_42"',
        'OUT = ROOT / "experiments/spatial_fusion/meningioma_weighted_seed42"'
    ),
    (
        'LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}',
        'LABEL_TO_ID = {name: i for i, name in enumerate(CLASSES)}\n'
        '\n'
        '# Targeted experiment: prioritize meningioma errors.\n'
        'CLASS_WEIGHTS = {0: 1.0, 1: 1.25, 2: 1.0, 3: 1.0}'
    ),
    (
        '    epochs=MAX_EPOCHS,\n    callbacks=[',
        '    epochs=MAX_EPOCHS,\n'
        '    class_weight=CLASS_WEIGHTS,\n'
        '    callbacks=['
    )
]

for old, new in replacements:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Expected exactly one match, found {count}: {old!r}"
        )
    text = text.replace(old, new, 1)

if source.resolve() == target.resolve():
    raise RuntimeError("Refusing to overwrite the original script.")

target.write_text(text, encoding="utf-8")
print(f"Created: {target}")
print("Original training script was not modified.")
print("Output folder: experiments/spatial_fusion/meningioma_weighted_seed42")
print("Class weights:", {0: 1.0, 1: 1.25, 2: 1.0, 3: 1.0})
