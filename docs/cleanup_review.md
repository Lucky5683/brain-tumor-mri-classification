# Safe Cleanup Review and Archive Recommendations

## 1. Executive Summary

In accordance with **Non-Negotiable Safety Rules 3, 4, and 5**, no files have been aggressively deleted or discarded. Cleaning in this repository is strictly defined as improving organization, hygiene, and reproducibility while completely preserving user work, trained models, experimental histories, and academic deliverables.

This document reviews candidates identified during the repository audit as potentially redundant, obsolete, or misplaced, detailing:
- Relative path and file size.
- Identified purpose and dependencies.
- Preservation status.
- Safe recommended action (KEEP, ARCHIVE, or MANUAL REVIEW).
- Evidence required before taking any irreversible action.

---

## 2. Review Inventory of Redundant & Archive Candidates

| Relative Path | Size | Category | Identified Purpose & Dependencies | Current Action | Recommended Future Action | Evidence Required for Permanent Cleanup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `README.zip` | 476.3 MB | Archive / Binary | Compressed snapshot of older codebase or dataset assets. | **PRESERVED** | ARCHIVE / MANUAL REVIEW | Confirm that no unique scan files or checkpoints exist inside before moving off-disk. |
| `app.py.before_final_model_update.bak` | ~25 KB | Backup File | Backup of `app.py` before Review III model dropdown expansion. | **PRESERVED** (Ignored by Git) | ARCHIVE to `cleanup_archive/backups/` | Verify current `app.py` satisfies all clinical demo requirements. |
| `app.py.before_spatial_fusion_update.bak` | ~27 KB | Backup File | Backup of `app.py` before adding the Spatial Fusion option. | **PRESERVED** (Ignored by Git) | ARCHIVE to `cleanup_archive/backups/` | Verify spatial fusion dropdown functions without regression. |
| `inspect_backbone_layers.py` | 3.4 KB | Root Script | Inspects EfficientNet-B0 layer hierarchy and conv names. | **PRESERVED** (Open in editor) | MOVE to `scripts/diagnostics/` | Confirm no external CLI references point to the root path. |
| `inspect_final_checkpoint.py` | 2.7 KB | Root Script | Deserializes and validates the seed 42 fusion model. | **PRESERVED** (Open in editor) | MOVE to `scripts/diagnostics/` | Confirm tests/scripts do not import from it directly. |
| `inspect_fusion_checkpoint.py` | 3.0 KB | Root Script | Validates grouped base checkpoint tensors and outputs. | **PRESERVED** | MOVE to `scripts/diagnostics/` | Confirm no automated pipelines invoke this script. |
| `test_spatial_fusion_gradcam.py` | 4.4 KB | Root Test | Smoke test for spatial attention Grad-CAM generation. | **PRESERVED** | INTEGRATE into `tests/` | Verified: `tests/test_spatial_fusion.py` now covers this test. |
| `scripts/train_multiscale_fusion_multiseed_fixed.py` | 8.9 KB | Duplicate Script | Copy of `train_multiscale_fusion_multiseed.py` with seed fix. | **PRESERVED** | ARCHIVE to `cleanup_archive/` | Diff comparison shows identical routine to the main script. |
| `cleanup_archive/duplicate_scripts/train_spatial_fusion_weighted_v2.py` | 9.2 KB | Historical Script | Early draft of class-weighted spatial fusion training. | **PRESERVED** | KEEP in `cleanup_archive/` | Kept for academic audit trail. |
| `baseline_evaluation.txt` | 1.7 KB | Legacy Log | Initial test evaluation log for custom CNN baseline. | **PRESERVED** | KEEP / MOVE to `reports/baseline_eval/` | Already reflected in `evaluation/model_comparison.csv`. |
| `efficientnet_evaluation.txt` | 1.7 KB | Legacy Log | Initial test evaluation log for EfficientNet-B0. | **PRESERVED** | KEEP / MOVE to `reports/efficientnet_eval/` | Preserved for benchmark provenance. |
| `exploration_report.txt` | 2.2 KB | Legacy Log | Dataset summary generated during initial data exploration. | **PRESERVED** | KEEP | Contains early slice count and resolution stats. |
| `test_results.txt` | 2.4 KB | Legacy Log | Output log from earlier automated test run. | **PRESERVED** | KEEP / REGENERATE | Replaced by live Pytest verification (13 passed). |
| `gradcam_test_cli.png` | ~150 KB | Image Output | Saliency map generated during command-line testing. | **PRESERVED** | ARCHIVE | Disposable test output; reproducible via `gradcam.py`. |
| `gradcam_test_cnn.png` | ~140 KB | Image Output | Saliency map generated during CNN baseline testing. | **PRESERVED** | ARCHIVE | Disposable test output; reproducible via `gradcam.py`. |
| `architecture_description.txt` | 5.1 KB | Documentation | Detailed text description of 5-tier architecture. | **PRESERVED** | KEEP | Referenced for thesis Chapter 3 writing. |
| `Brain_Tumor_MRI_Classification_Mini_Project_Review.pptx` | ~2.5 MB | Academic Deliverable | 20-slide presentation for college project review panel. | **PRESERVED** | KEEP (Essential deliverable) | High-priority academic presentation asset. |
| `Brain_Tumor_MRI_Classification_Report.docx` | ~3.8 MB | Academic Deliverable | Formal academic mini-project report. | **PRESERVED** | KEEP (Essential deliverable) | High-priority academic manuscript asset. |

---

## 3. Safe Actions Taken During Current Audit

1. **`.gitignore` Hardening**:
   - Safely appended rules for `*.bak`, `*.tmp`, `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/`, `.coverage`, `htmlcov/`, and `.streamlit/secrets.toml`.
   - Prevented temporary files from dirtying Git status without deleting any user data from disk.
   - Explicitly preserved `!.env.example` and all research artifacts (`.keras`, `.csv`, `.json`, `experiments/`).

2. **No Destructive Operations**:
   - Zero files were permanently deleted (`rm` / `del` was not invoked on any model, dataset, or user file).
   - No Git reset (`git reset --hard`) or cleaning (`git clean -fd`) was executed.
   - All 5 model checkpoints remain bit-identical and loadable.
   - All dataset directories and manifest rows remain intact.

---

## 4. Remediation Workflow for Future Passes

When the user is ready to declutter the root directory:
```bash
# Example safe archiving (to be executed only upon explicit user request):
mkdir -p cleanup_archive/root_diagnostics
# Move scratch inspection scripts without deleting:
# git mv inspect_*.py cleanup_archive/root_diagnostics/
```
No such moves have been executed during this audit pass to prevent breaking any open IDE sessions or relative script paths.
