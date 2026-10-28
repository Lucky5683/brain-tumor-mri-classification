"""Global Configuration for Brain Tumor MRI Classification V2."""

from __future__ import annotations
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "dataset"
TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Ensure runtime directories exist
for directory in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR, EXPERIMENTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

# Dataset & Image Specifications
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
BATCH_SIZE = 32
RANDOM_SEED = 42
VALIDATION_FRACTION = 0.20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

# Class Definitions (Consistent fixed alphabetical ordering)
CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_INDEX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}

DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}

# Augmentation Hyperparameters (Conservative, medically plausible)
AUGMENTATION_CONFIG = {
    "rotation_factor": 0.05,        # +/- 5% (~18 degrees)
    "zoom_height_factor": 0.08,     # +/- 8% zoom
    "zoom_width_factor": 0.08,
    "translation_factor": 0.05,     # +/- 5% shift
    "brightness_factor": 0.10,      # +/- 10% brightness
    "contrast_factor": 0.10,        # +/- 10% contrast
    "horizontal_flip": True,        # Bilateral symmetry valid for axial slices
    "vertical_flip": False,         # Prohibited: inverts anatomical superior/inferior axes
}

# Training Hyperparameters
STAGE1_LEARNING_RATE = 1e-3
STAGE1_EPOCHS = 15
STAGE2_LEARNING_RATE = 1e-5
STAGE2_EPOCHS = 10
EARLY_STOPPING_PATIENCE = 5
REDUCE_LR_PATIENCE = 3
REDUCE_LR_FACTOR = 0.5
DROPOUT_RATE = 0.30

# Uncertainty / Gating Thresholds
UNCERTAINTY_THRESHOLDS = {
    "high_confidence_min": 0.85,
    "moderate_confidence_min": 0.60,
    "max_entropy_threshold": 1.10,
}

# Medical Disclaimer (Mandatory display in reports and apps)
MEDICAL_DISCLAIMER = (
    "Research and educational use only. This software is not a certified medical device, "
    "does not provide a medical diagnosis, and must not replace evaluation by a qualified healthcare professional."
)
