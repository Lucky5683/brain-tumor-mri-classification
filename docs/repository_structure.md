# Repository Structure: Brain Tumor MRI Classification

## 1. Executive Summary

This document describes the organizational layout of the **Brain Tumor MRI Image Classification Using EfficientNet-B0 with Grad-CAM** repository. The codebase implements deep learning classification across four MRI diagnostic categories (**glioma**, **meningioma**, **notumor**, and **pituitary**), integrated with explainable saliency analysis (Grad-CAM, Grad-CAM++), post-hoc uncertainty calibration, and an interactive Streamlit clinical decision-support application.

---

## 2. Directory Hierarchy Overview

```text
brain-tumor-mri-classification/
├── app.py                                  # Streamlit interactive web application entry point
├── config.py                               # Central project configuration, hyperparams, & class indices
├── evaluate.py                             # Root evaluation command-line interface wrapper
├── gradcam.py                              # Root Grad-CAM saliency generation CLI wrapper
├── preprocess.py                           # Root dataset loader and preprocessing CLI wrapper
├── train_cnn.py                            # Custom 4-block CNN training script
├── train_efficientnet.py                   # EfficientNet-B0 transfer learning training script
├── data_quality.py                         # Dataset integrity, duplicate detection, and hash verification
├── requirements.txt                        # Python package dependencies
├── .gitignore                              # Git exclusion rules for environments, caches, and large archives
│
├── dataset/                                # MRI Scan Benchmark (7,200 images total, excluded from Git)
│   ├── Training/                           # 5,600 axial slices (1,400 per class)
│   │   ├── glioma/                         # 1,400 glioma slices
│   │   ├── meningioma/                     # 1,400 meningioma slices
│   │   ├── notumor/                        # 1,400 healthy control slices
│   │   └── pituitary/                      # 1,400 pituitary tumor slices
│   └── Testing/                            # 1,600 independent test slices (400 per class)
│       ├── glioma/                         # 400 test slices
│       ├── meningioma/                     # 400 test slices
│       ├── notumor/                        # 400 test slices
│       └── pituitary/                      # 400 test slices
│
├── src/                                    # Modular source package for models and pipelines
│   ├── __init__.py                         # Package initialization
│   ├── data/                               # Dataset ingestion, augmentation, and split management
│   │   ├── __init__.py
│   │   ├── dataset.py                      # TF dataset generators & preprocessing loaders
│   │   └── split_manager.py                # Manifest-aware splitting & grouping logic
│   ├── models/                             # Model architectures and factory loaders
│   │   ├── __init__.py
│   │   ├── custom_cnn.py                   # 4-block custom convolutional architecture
│   │   ├── efficientnet.py                 # EfficientNet-B0 transfer learning architecture
│   │   ├── multiscale_fusion.py            # Multi-scale feature fusion architecture
│   │   ├── spatial_fusion.py               # Spatial Attention Feature Fusion architecture
│   │   └── loader.py                       # Safe multi-model checkpoint loader & registry
│   ├── evaluation/                         # Evaluation, metrics, and uncertainty calibration
│   │   ├── __init__.py
│   │   ├── metrics.py                      # Accuracy, macro F1, precision, recall, ROC-AUC
│   │   ├── calibration.py                  # Temperature Scaling and Expected Calibration Error (ECE)
│   │   └── uncertainty.py                  # Shannon entropy, predictive variance, triage rejection
│   └── explainability/                     # Saliency analysis and explainable AI (XAI)
│       ├── __init__.py
│       ├── gradcam.py                      # First-order gradient Grad-CAM implementation
│       ├── gradcam_plus_plus.py            # Higher-order gradient Grad-CAM++ implementation
│       └── visualizer.py                   # Heatmap colorization, overlay, and alpha blending
│
├── scripts/                                # Utility, evaluation, and experimental scripts
│   ├── evaluate_all_models.py              # Comparative benchmark evaluation across all checkpoints
│   ├── evaluate_final_model.py             # Evaluation of selected seed 42 fusion model
│   ├── display_metrics.py                  # Console summary of verified model metrics
│   ├── investigate_spatial_gradcam.py      # Saliency layer resolution & Grad-CAM inspection
│   ├── run_five_fold_cv.py                 # 5-fold cross-validation execution script
│   ├── generate_fold_manifests.py          # Perceptual grouping fold generation script
│   ├── train_multiscale_fusion.py          # Multi-scale fusion training routine
│   ├── train_multiscale_fusion_multiseed.py # 3-seed multi-scale training script
│   ├── train_spatial_fusion_weighted.py    # Spatial fusion training with class weighting
│   └── train_spatial_fusion_multiseed.py   # Multi-seed spatial fusion training
│
├── models/                                 # Standardized model checkpoint directory
│   ├── baseline_cnn.keras                  # Custom 4-block CNN baseline weights
│   ├── efficientnet_b0.keras               # Reference ImageNet-pretrained EfficientNet-B0
│   ├── efficientnet_b0_finetuned.keras     # Fine-tuned EfficientNet-B0 checkpoint
│   └── multiscale_fusion_grouped_best.keras # Grouped-split multi-scale fusion checkpoint
│
├── experiments/                            # Structured experiment logs, manifests, and checkpoints
│   ├── duplicate_aware_split/              # Leakage-aware dataset partition artifacts
│   │   ├── training_manifest.csv           # 5,600 rows (4,447 train, 1,153 val) with 0 group overlap
│   │   └── split_summary.json              # Class distributions and hash grouping metadata
│   ├── spatial_fusion/                     # Spatial Attention Fusion experiment runs
│   │   └── meningioma_weighted_50ep_seed42/# Candidate checkpoint (50 epochs, class-weighted)
│   │       ├── best_model.keras            # Checkpoint (33.8 MB, 4.4M params) - PROTECTED
│   │       ├── best_metrics.json           # Validation metrics at Epoch 50 (92.54% val acc)
│   │       ├── history.json                # Complete 50-epoch loss/accuracy history
│   │       └── results.json                # Interim validation checkpoint record (Epoch 31)
│   ├── multiseed_fusion/                   # 3-seed stability experiment runs
│   │   ├── seed_42/                        # Seed 42 run (best_model.keras, 88.73% val acc)
│   │   ├── seed_123/                       # Seed 123 run (best_model.keras, history.json)
│   │   └── seed_456/                       # Seed 456 run (best_model.keras, history.json)
│   └── five_fold_cv/                       # 5-Fold cross-validation artifacts
│       ├── fold_manifest.csv               # Grouped 5-fold assignment manifest (5,600 scans)
│       ├── fold_class_counts.csv           # Fold-wise class distribution matrix
│       ├── smoke_tests/                    # Verification smoke-test outputs (1 epoch, truncated)
│       └── custom_cnn/                     # Preliminary Fold 1 smoke-test output
│
├── tests/                                  # Automated regression test suite
│   ├── __init__.py
│   ├── test_pipeline.py                    # 12 functional, regression, and explainability tests
│   └── test_spatial_fusion.py              # Spatial fusion architecture and Grad-CAM tests
│
├── docs/                                   # Project documentation, audit logs, and reports
│   ├── repository_structure.md             # This document (layout and directory purposes)
│   ├── experiment_inventory.md             # Inventory of checkpoints, manifests, and metrics
│   ├── cleanup_review.md                   # Redundant and archive candidate analysis
│   ├── repository_audit.md                 # Complete audit log and validation outcomes
│   ├── final_project_status.md             # Executive project summary and review briefing
│   ├── final_verification_report.md        # Comprehensive independent verification log
│   └── novelty_and_literature_comparison.md# Architectural novelty analysis vs. published literature
│
├── reports/                                # Generated evaluation summaries and text reports
│   ├── baseline_eval/                      # Custom CNN test metrics and classification report
│   ├── efficientnet_eval/                  # EfficientNet-B0 test metrics and classification report
│   └── multiscale_fusion_eval/             # Multi-scale fusion test metrics and triage analysis
│
├── evaluation/                             # Quantitative evaluation plots and comparisons
│   ├── model_comparison.csv                # Tabular comparison across initial benchmark models
│   ├── comparison_confusion_matrices.png   # Side-by-side confusion matrix plots
│   ├── comparison_per_class_metrics.png    # Per-class recall and F1 comparison bar charts
│   └── training_curves_comparison.png      # Loss and accuracy learning curves comparison
│
├── diagrams/                               # High-resolution architectural and workflow figures
│   ├── system_architecture.png             # 5-tier academic system architecture diagram
│   └── workflow_diagram.png                # End-to-end operational inference workflow
│
├── screenshots/                            # Application demonstration screenshots
│   ├── streamlit_app_home.png              # Application landing page with safety notices
│   ├── streamlit_sample_loaded.png         # Image preview card and model selector state
│   ├── streamlit_prediction_results.png    # Live prediction and Grad-CAM output
│   ├── streamlit_metrics_card.png          # Uncertainty metrics and triage gating status
│   └── streamlit_gradcam_focus.png         # Focused Grad-CAM saliency panel view
│
├── gradcam_examples/                       # Four-class clinical saliency demonstration images
│   ├── glioma_gradcam_overlay.png
│   ├── meningioma_gradcam_overlay.png
│   ├── notumor_gradcam_overlay.png
│   ├── pituitary_gradcam_overlay.png
│   └── gradcam_4class_summary_grid.png
│
├── cleanup_archive/                        # Preserved historical backup scripts and artifacts
└── backup_pre_audit/                       # Pre-audit workspace snapshot
```

---

## 3. Component Details & Design Principles

### 3.1 Core Application (`app.py`)
- Built using **Streamlit** (v1.55.0).
- Dynamically binds to five model options defined in `src/models/loader.py`:
  1. `EfficientNet-B0 V2 (Fine-Tuned)`
  2. `Baseline Custom CNN`
  3. `Multi-Scale Feature Fusion (Grouped Base Checkpoint)`
  4. `EfficientNet-B0 (Reference Baseline - V1)`
  5. `Spatial Attention Fusion (Candidate Checkpoint - Seed 42)`
- Real-time explainability computes both Grad-CAM and Grad-CAM++ with alpha blending over original MRI scans.
- Includes temperature-scaled calibrated uncertainty and clinical triage gating.

### 3.2 Configuration (`config.py`)
- Defines canonical four-class ordering: `['glioma', 'meningioma', 'notumor', 'pituitary']`.
- Maps label names to numerical indices `[0, 1, 2, 3]`.
- Centralizes image dimensions ($224 \times 224 \times 3$), normalization defaults, and file paths.

### 3.3 Source Package (`src/`)
- Designed modularly with strict separation of concerns:
  - `src/data/`: Manages dataset loading, batching, and manifest-guided splitting without data leakage.
  - `src/models/`: Encapsulates CNN, EfficientNet-B0, Multi-Scale Fusion, and Spatial Fusion architectures, along with a Keras 3 compatibility loader.
  - `src/evaluation/`: Houses metric calculations, temperature calibration, and uncertainty scoring.
  - `src/explainability/`: Implements gradient extraction and saliency generation.

### 3.4 Experiments (`experiments/`)
- Organized hierarchically by experiment type.
- Houses training manifests, hyperparameter logs, training histories (`history.json`), and validated model checkpoints (`best_model.keras`).
- Strictly isolates intermediate research runs from production root scripts.

### 3.5 Automated Test Suite (`tests/`)
- Contains 13 automated Pytest test cases across `tests/test_pipeline.py` and `tests/test_spatial_fusion.py`.
- Validates data loading, preprocessing, model deserialization, shape compatibility, Grad-CAM generation, and temperature calibration.
