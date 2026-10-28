"""Phase 4 Part D: Quality & Integrity Verification Audit Script."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CLASS_NAMES, DATA_DIR, REPORTS_DIR, TEST_DIR, TRAIN_DIR


def verify_phase4() -> bool:
    all_passed = True
    print("=" * 70)
    print("PHASE 4 QUALITY & INTEGRITY VERIFICATION AUDIT")
    print("=" * 70)

    # 1. Model SHA256 Integrity
    model_path = PROJECT_ROOT / "efficientnet_b0.keras"
    expected_hash = "4ab9935ee0764d72737144a8989815bfccbbc22ad8616b54c1cca7c616607590"
    actual_hash = hashlib.sha256(model_path.read_bytes()).hexdigest().lower()

    if actual_hash == expected_hash:
        print(f"[PASS] V1 Reference Model Checksum unchanged: {actual_hash}")
    else:
        print(f"[FAIL] V1 Model Checksum MISMATCH! Expected: {expected_hash}, Got: {actual_hash}")
        all_passed = False

    # 2. Original Test Set Untouched
    test_counts = {}
    for c in CLASS_NAMES:
        c_dir = TEST_DIR / c
        imgs = list(c_dir.glob("*.jpg")) + list(c_dir.glob("*.png"))
        test_counts[c] = len(imgs)

    expected_counts = {"glioma": 400, "meningioma": 400, "notumor": 400, "pituitary": 400}
    if test_counts == expected_counts:
        print(f"[PASS] Official Test Set Untouched: Exactly 1,600 images ({test_counts})")
    else:
        print(f"[FAIL] Test Set Modified! Current counts: {test_counts}")
        all_passed = False

    # 3. Image Opening Check
    explainability_dir = REPORTS_DIR / "explainability"
    robustness_dir = REPORTS_DIR / "robustness"

    images_to_test = list(explainability_dir.glob("*.png")) + list((explainability_dir / "figures").glob("*.png")) + list(robustness_dir.glob("*.png"))
    corrupted = []
    for img_p in images_to_test:
        try:
            with Image.open(img_p) as im:
                im.verify()
        except Exception as exc:
            corrupted.append((img_p.name, str(exc)))

    if not corrupted and len(images_to_test) > 0:
        print(f"[PASS] All {len(images_to_test)} generated PNG images opened and verified valid.")
    else:
        print(f"[FAIL] Corrupted images found: {corrupted}")
        all_passed = False

    # 4. JSON Validation & NaN/Inf Check
    json_files = [
        explainability_dir / "explainability_comparison.json",
        robustness_dir / "robustness_report.json",
    ]
    for jf in json_files:
        if not jf.is_file():
            print(f"[FAIL] Missing JSON file: {jf}")
            all_passed = False
            continue

        try:
            data = json.loads(jf.read_text(encoding="utf-8"))
            # Stringify and check for NaN or Inf
            data_str = json.dumps(data)
            if "NaN" in data_str or "Infinity" in data_str:
                print(f"[FAIL] JSON file contains NaN or Inf: {jf.name}")
                all_passed = False
            else:
                print(f"[PASS] JSON file valid and free of NaN/Inf: {jf.name}")
        except Exception as exc:
            print(f"[FAIL] Invalid JSON in {jf.name}: {exc}")
            all_passed = False

    # 5. CSV Validation
    csv_file = robustness_dir / "robustness_results.csv"
    if csv_file.is_file():
        try:
            df = pd.read_csv(csv_file)
            if df.isna().any().any():
                print(f"[FAIL] CSV contains unexpected NaN values: {csv_file.name}")
                all_passed = False
            else:
                print(f"[PASS] CSV valid and complete ({len(df)} rows, 0 NaNs): {csv_file.name}")
        except Exception as exc:
            print(f"[FAIL] Error reading CSV: {exc}")
            all_passed = False
    else:
        print(f"[FAIL] Missing CSV file: {csv_file}")
        all_passed = False

    # 6. Representation of all classes
    exp_json = json.loads((explainability_dir / "explainability_comparison.json").read_text(encoding="utf-8"))
    represented_classes = set(r["true_class"] for r in exp_json["results"])
    if represented_classes == set(CLASS_NAMES):
        print(f"[PASS] All {len(CLASS_NAMES)} classes represented in explainability validation: {sorted(list(represented_classes))}")
    else:
        print(f"[FAIL] Missing classes in explainability! Expected: {CLASS_NAMES}, Got: {represented_classes}")
        all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL PHASE 4 QUALITY CHECKS PASSED.")
    else:
        print("SOME QUALITY CHECKS FAILED.")
    print("=" * 70)
    return all_passed


if __name__ == "__main__":
    success = verify_phase4()
    sys.exit(0 if success else 1)
