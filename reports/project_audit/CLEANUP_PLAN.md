# Safe Project Cleanup Plan & Dependency Assessment

**Project Title:** Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM  
**Workspace:** `X:\Mini Project\brain-tumor-mri-classification`  
**Policy:** Strictly advisory. **No files have been deleted, moved, renamed, or modified during this audit.**

---

## 1. Safety Rules & Preservation Policy

The following categories are **strictly protected from deletion**:
1. **Model Checkpoints:** All `.keras` files in root, `models/`, and `experiments/` (all represent historical checkpoints or active app options).
2. **Training & Evaluation Manifests:** `training_manifest.csv`, `manifest.csv`, `split_metadata.json`.
3. **Primary Evaluation Reports:** All JSON and CSV reports in `reports/` and `experiments/`.
4. **App Assets & Dependencies:** Everything required by `streamlit run app.py` and `tests/test_pipeline.py`.
5. **Raw Dataset:** All 7,200 images in `dataset/Training` and `dataset/Testing`.

---

## 2. Categorized File Classification

### Category A: Required for Execution & Active Application (DO NOT TOUCH)
These files are directly imported, executed, or loaded by the running Streamlit web application or the automated testing suite:

| Relative Path | Size | Reason / Dependency | Risk if Removed |
| :--- | :---: | :--- | :--- |
| `app.py` | 36.4 KB | Active multi-model Streamlit application | **FATAL**: App fails to run |
| `config.py` | 2.5 KB | Core project configuration, class mapping, thresholds | **FATAL**: All modules crash on import |
| `requirements.txt` | 643 B | Python virtual environment package definitions | High: Environment cannot be rebuilt |
| `src/models/compat.py` | 1.6 KB | Keras 3 deserialization shim for legacy weights | **FATAL**: Checkpoints crash on load |
| `src/models/multiscale_fusion.py` | 2.0 KB | Multi-scale model architecture definition | **FATAL**: Fusion models cannot deserialize |
| `src/models/efficientnet.py` | 2.9 KB | EfficientNet-B0 architecture & two-stage logic | High: Model building/fine-tuning fails |
| `src/models/custom_cnn.py` | 1.9 KB | Custom 4-block CNN architecture | Medium: CNN baseline breaks |
| `src/data/preprocessing.py` | 2.9 KB | Standardized single-image and batch preprocessing | **FATAL**: Inferences cannot preprocess |
| `src/data/dataset.py` | 13.5 KB | tf.data pipeline and manifest loading | High: Training and evaluation fail |
| `src/data/augmentation.py` | 2.2 KB | Medically constrained augmentation pipeline | Medium: Training scripts fail |
| `src/evaluation/metrics.py` | 11.4 KB | Confusion matrix, ROC, PR curves, and reports | High: Metrics reporting breaks |
| `src/evaluation/calibration.py` | 6.1 KB | Temperature scaling and ECE calculation | Medium: Confidence calibration fails |
| `src/evaluation/uncertainty.py` | 2.2 KB | Shannon entropy and triage gating logic | Medium: App triage warnings fail |
| `src/explainability/gradcam.py` | 7.6 KB | Saliency computation and overlay rendering | **FATAL**: Saliency visualization fails |
| `src/explainability/gradcam_plus_plus.py`| 2.8 KB | Higher-order Grad-CAM++ computation | High: Grad-CAM++ option fails in app |
| `tests/test_pipeline.py` | 10.9 KB | 13-test automated regression verification suite | High: No automated regression testing |
| `efficientnet_b0.keras` | 18.2 MB | Active Checkpoint: Option 4 in app | **FATAL**: App Option 4 crashes |
| `baseline_cnn.keras` | 4.9 MB | Active Checkpoint: Option 2 in app | **FATAL**: App Option 2 crashes |
| `models/efficientnet_b0_finetuned.keras` | 28.4 MB | Active Checkpoint: Option 1 in app | **FATAL**: App Option 1 crashes |
| `models/multiscale_fusion_grouped_best.keras` | 31.7 MB | Active Checkpoint: Option 3 in app | **FATAL**: App Option 3 crashes |
| `experiments/multiseed_fusion/seed_42/best_model.keras` | 31.7 MB | Active Checkpoint: Option 5 in app | **FATAL**: App Option 5 crashes |

---

### Category B: Required for Academic Reproducibility (RETAIN)
These files are not directly loaded by `app.py`, but contain the exact empirical evidence, training histories, manifests, and diagrams supporting the project report and mini-project review:

| Relative Path | Size | Reason / Purpose | Recommendation |
| :--- | :---: | :--- | :--- |
| `experiments/duplicate_aware_split/training_manifest.csv` | 992.8 KB | Ground-truth manifest for 4,447 / 1,153 grouped split | **RETAIN** (Essential for reproduction) |
| `experiments/duplicate_aware_split/cross_class_near_matches.csv` | 3.0 KB | Evidence of 24 contradictory cross-class duplicate pairs | **RETAIN** (Core evidence for report) |
| `experiments/multiseed_fusion/final_summary.json` | 4.5 KB | Official 3-seed variance summary (Seeds 42, 123, 2026) | **RETAIN** (Key experimental proof) |
| `reports/final_multiseed/final_test_summary.json` | 6.7 KB | Official locked test set evaluation metrics ($N=1,600$) | **RETAIN** (Ground truth for paper) |
| `reports/final_multiseed/model_comparison.csv` | 605 B | Comparative table for Seed 42 vs. Baseline | **RETAIN** |
| `reports/final_multiseed/selected_fusion_confusion_matrix.png` | 136.2 KB | Final confusion matrix figure | **RETAIN** |
| `reports/final_multiseed/roc_curve.png` | 219.7 KB | Final multi-class ROC curve figure | **RETAIN** |
| `reports/final_multiseed/precision_recall_curve.png` | 222.4 KB | Final Precision-Recall curve figure | **RETAIN** |
| `reports/baseline_cnn/` (all 6 files) | ~500 KB | Evaluation artifacts for baseline CNN | **RETAIN** (Consumed by app.py) |
| `reports/efficientnet_b0/` (all 6 files) | ~500 KB | Evaluation artifacts for baseline EffNet | **RETAIN** (Consumed by app.py) |
| `reports/efficientnet_b0_finetuned/` (all 6 files) | ~500 KB | Evaluation artifacts for fine-tuned EffNet | **RETAIN** (Consumed by app.py) |
| `reports/multiscale_fusion_grouped/` (all 6 files) | ~500 KB | Evaluation artifacts for grouped fusion | **RETAIN** (Consumed by app.py) |
| `diagrams/system_architecture.png` | 1.25 MB | System architecture diagram for Chapter 3.2 | **RETAIN** |
| `diagrams/workflow_diagram.png` | 265.6 KB | End-to-end project workflow diagram | **RETAIN** |
| `scripts/train_multiscale_fusion_multiseed.py` | 22.3 KB | Script that produced Seeds 42, 123, 2026 | **RETAIN** |
| `scripts/train_multiscale_fusion_grouped.py` | 10.3 KB | Script that produced base grouped model | **RETAIN** |
| `scripts/train_efficientnet.py` | 17.6 KB | Two-stage EfficientNet training pipeline | **RETAIN** |
| `scripts/train_cnn.py` | 2.1 KB | Baseline CNN training script | **RETAIN** |
| `scripts/evaluate_final_multiseed.py` | 19.4 KB | Final locked test evaluation script | **RETAIN** |
| `scripts/verify_app_e2e.py` | 5.8 KB | Headless 5-model verification script | **RETAIN** |

---

### Category C: Candidates for Archiving (Move to `archive/` — Do Not Delete)
These files represent earlier intermediate stages, duplicate script versions, or large binary archives that clutter the active workspace without being consumed by `app.py`:

| Relative Path | Size | Reason / Redundancy | Recommended Action |
| :--- | :---: | :--- | :--- |
| `README.zip` | **476.3 MB** | Historical compressed snapshot of earlier repository state | Move to external backup or `archive/` |
| `backup_pre_audit/` (14 files) | ~50 MB | Snapshot taken prior to phase 1 audit | Move to `archive/backup_pre_audit/` |
| `scripts/train_multiscale_fusion_multiseed_fixed.py` | 18.2 KB | Near-duplicate of `train_multiscale_fusion_multiseed.py` | Move to `archive/scripts/` |
| `scripts/train_multiscale_fusion.py` | 11.1 KB | Earlier fusion script on un-grouped split | Move to `archive/scripts/` |
| `scripts/test_fusion_finetuning.py` | 1.8 KB | Short scratch script testing fine-tuning feasibility | Move to `archive/scripts/` |
| `models/history_v2_e1.json` and `e2.json` | 6.3 KB | Intermediate history logs from early runs | Move to `archive/models/` |
| `models/exp1_efficientnet_b0_best.keras` | 18.2 MB | Superseded intermediate checkpoint | Move to `archive/models/` |
| `models/multiscale_fusion_stage1_best.keras` | 21.6 MB | Intermediate Stage 1 head warmup weights | Move to `archive/models/` |
| `models/multiscale_fusion_grouped_stage1_best.keras` | 21.6 MB | Intermediate Stage 1 head warmup weights | Move to `archive/models/` |
| `models/multiscale_fusion_dedup_best.keras` | 33.2 MB | Duplicate naming convention checkpoint | Move to `archive/models/` |
| `models/multiscale_fusion_dedup_stage1_best.keras` | 21.6 MB | Duplicate naming convention checkpoint | Move to `archive/models/` |
| `baseline_confusion_matrix.png` (root) | 125 KB | Root duplicate of `reports/baseline_cnn/confusion_matrix.png` | Move to `archive/root_legacy/` |
| `efficientnet_confusion_matrix.png` (root) | 128 KB | Root duplicate of `reports/efficientnet_b0/confusion_matrix.png` | Move to `archive/root_legacy/` |
| `baseline_evaluation.txt` (root) | 2.0 KB | Root duplicate of structured JSON report | Move to `archive/root_legacy/` |
| `efficientnet_evaluation.txt` (root) | 2.2 KB | Root duplicate of structured JSON report | Move to `archive/root_legacy/` |
| `exploration_report.txt` (root) | 2.1 KB | Historical output of explore_data.py | Move to `archive/root_legacy/` |
| `cnn_history.json` (root) | 3.7 KB | Historical training log of baseline CNN | Move to `archive/root_legacy/` |
| `efficientnet_history.json` (root) | 2.8 KB | Historical training log of baseline EffNet | Move to `archive/root_legacy/` |

---

## 3. Recommended Future Repository Structure

When ready to clean up the workspace (after academic review completion), the following clean layout is recommended:

```
brain-tumor-mri-classification/
├── app.py                             # Active Streamlit Web Application
├── config.py                          # Global Project Configuration
├── requirements.txt                   # Pinned Dependencies
├── README.md                          # Primary Project Documentation
│
├── dataset/                           # Standard Kaggle Benchmark
│   ├── Training/                      # 5,600 images (1,400 / class)
│   └── Testing/                       # 1,600 images (400 / class)
│
├── src/                               # Modular Source Code
│   ├── data/                          # Preprocessing, Augmentation, tf.data loaders
│   ├── models/                        # EfficientNet, Multi-Scale Fusion, CNN, Compat
│   ├── evaluation/                    # Metrics, Calibration, Uncertainty, Robustness
│   └── explainability/                # Grad-CAM and Grad-CAM++
│
├── models/                            # Active Production Checkpoints
│   ├── efficientnet_b0.keras          # Reference Baseline V1 (87.50% test acc)
│   ├── baseline_cnn.keras             # Baseline Custom CNN (78.06% test acc)
│   ├── efficientnet_b0_finetuned.keras# Two-Stage Fine-Tuned (81.50% test acc)
│   ├── multiscale_fusion_grouped_best.keras # Grouped Base Checkpoint (80.63% test acc)
│   └── multiscale_fusion_seed42.keras # Renamed from experiments for clean deployment
│
├── experiments/                       # Experiment Records & Manifests
│   ├── duplicate_aware_split/         # training_manifest.csv (4,447 train / 1,153 val)
│   └── multiseed_fusion/              # Seeds 42, 123, 2026 logs and final_summary.json
│
├── reports/                           # Verified Reports & Visualizations
│   ├── baseline_cnn/                  # Visual & Tabular artifacts for Option 2
│   ├── efficientnet_b0/               # Visual & Tabular artifacts for Option 4
│   ├── efficientnet_b0_finetuned/     # Visual & Tabular artifacts for Option 1
│   ├── multiscale_fusion_grouped/     # Visual & Tabular artifacts for Option 3
│   ├── final_multiseed/               # Visual & Tabular artifacts for Option 5
│   └── project_audit/                 # This 9-deliverable technical audit
│
├── diagrams/                          # Publication & Review Diagrams
├── tests/                             # Automated Test Suite (test_pipeline.py)
├── scripts/                           # Active Training & Evaluation Utilities
└── archive/                           # Isolated legacy snapshots & scratch scripts (OPTIONAL)
```
