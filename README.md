# Brain Tumor MRI Classification with Explainable AI (Grad-CAM)

An end-to-end deep learning system for automated classification of brain magnetic resonance imaging (MRI) scans into four distinct categories: **Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**. The project features a custom CNN baseline, an advanced **EfficientNet-B0** transfer learning model achieving **87.50% test accuracy (0.9718 ROC-AUC)**, **Grad-CAM visual explainability**, and an interactive **Streamlit web application** for real-time clinical decision support.

---

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architectures & Performance](#model-architectures--performance)
  - [Model Comparison](#model-comparison)
  - [Confusion Matrices & Visuals](#confusion-matrices--visuals)
- [Explainable AI (Grad-CAM)](#explainable-ai-grad-cam)
- [Streamlit Web Application](#streamlit-web-application)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Web App](#running-the-web-app)
  - [Training & Evaluation](#training--evaluation)
- [License](#license)

---

## Overview

Brain tumors represent one of the most critical neurological conditions requiring rapid, highly accurate diagnosis. While MRI is the gold standard imaging modality, manual interpretation is time-consuming and subject to inter-observer variability.

This project delivers:
1. **Multi-Class Classification**: Accurately differentiates between 3 common tumor types (Glioma, Meningioma, Pituitary) and healthy brain scans (No Tumor).
2. **Transfer Learning**: Leverages pre-trained ImageNet feature extractors via EfficientNet-B0, fine-tuned on normalized brain MRI scans.
3. **Clinical Interpretability (Grad-CAM)**: Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps highlighting the exact anatomical regions driving the network's diagnostic predictions.
4. **Interactive Dashboard**: A full-featured Streamlit web application supporting drag-and-drop image uploads, confidence score distributions, and Grad-CAM overlays.

---

## Dataset

The dataset utilized is the widely benchmarked **Brain Tumor MRI Dataset** (combining Figshare, SARTAJ, and Br35H datasets):
- **Total Images**: 7,200 high-resolution MRI scans
- **Classes (4)**:
  - `glioma`: 1,800 images (1,400 training, 400 testing)
  - `meningioma`: 1,800 images (1,400 training, 400 testing)
  - `notumor`: 1,800 images (1,400 training, 400 testing)
  - `pituitary`: 1,800 images (1,400 training, 400 testing)
- **Split**: 5,600 Training (77.8%) / 1,600 Testing (22.2%) - perfectly balanced across all four classes.
- **Image Resolution**: Resized and normalized to $224 \times 224 \times 3$.

### Dataset Setup Instructions
Due to GitHub file size limits, the raw MRI image files are excluded from this repository. To train from scratch or evaluate on the full test set:
1. Download the dataset from Kaggle: [Brain Tumor MRI Dataset (Masoud Nickparvar)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset).
2. Extract the archive into a `dataset/` directory inside this project:
   ```text
   brain-tumor-mri-classification/
   |-- dataset/
       |-- Training/
       |   |-- glioma/
       |   |-- meningioma/
       |   |-- notumor/
       |   |-- pituitary/
       |-- Testing/
           |-- glioma/
           |-- meningioma/
           |-- notumor/
           |-- pituitary/
   ```

---

## Model Architectures & Performance

### Model Comparison

Both models were trained on normalized $224 \times 224$ images and evaluated on the independent test set of 1,600 images:

| Metric | Custom Baseline CNN | EfficientNet-B0 (Transfer Learning) |
| :--- | :---: | :---: |
| **Model Size** | 5.1 MB | 19.0 MB |
| **Test Accuracy** | **68.75%** | **87.50%** (+18.75%) |
| **Test Loss** | 1.1840 | **0.5056** |
| **Precision (Weighted)** | 0.7130 | **0.8792** |
| **Recall (Weighted)** | 0.6875 | **0.8750** |
| **F1-Score (Weighted)** | 0.6624 | **0.8699** |
| **ROC-AUC (Macro OvR)** | 0.8828 | **0.9718** |

### Per-Class Detailed Performance (EfficientNet-B0)

```text
              Precision    Recall    F1-Score   Support
      glioma     0.9261    0.6575      0.7690       400
  meningioma     0.7857    0.8525      0.8177       400
     notumor     0.8969    1.0000      0.9456       400
   pituitary     0.9083    0.9900      0.9474       400

    accuracy                           0.8750      1600
   macro avg     0.8792    0.8750      0.8699      1600
weighted avg     0.8792    0.8750      0.8699      1600
```

- **Healthy Brains (`notumor`)**: **100.0% Recall** (400/400 correct, zero false negatives).
- **Pituitary Tumors**: **99.0% Recall** (396/400 correct).
- **Overall ROC-AUC**: **0.9718**, demonstrating exceptional class discriminability.

---

## Explainable AI (Grad-CAM)

Deep neural networks in medical diagnostics must not operate as "black boxes." This system integrates **Gradient-weighted Class Activation Mapping (Grad-CAM)**:
- Visualizes the gradients flowing into the final convolutional layer (`top_conv` / `block7a_project_conv`).
- Overlays thermal heatmaps (Jet colormap) onto the original MRI grayscale slice.
- Allows clinicians to verify whether the AI focused on the true lesion or background artifacts.

Sample Grad-CAM outputs generated during evaluation are available in `gradcam_samples/`.

---

## Streamlit Web Application

The interactive web app (`app.py`) provides an intuitive clinical interface:
- **Image Upload**: Accepts JPG, JPEG, and PNG brain MRI scans.
- **Model Switcher**: Compare predictions between EfficientNet-B0 and the Baseline CNN.
- **Interactive Visualizations**:
  - Predicted diagnosis with confidence percentage.
  - Full probability distribution bar chart across all 4 classes.
  - Side-by-side display of the original MRI and the Grad-CAM saliency map.

---

## Repository Structure

```text
brain-tumor-mri-classification/
|-- app.py                             # Streamlit interactive web application
|-- baseline_cnn.keras                 # Pre-trained Baseline CNN checkpoint (5.1 MB)
|-- baseline_confusion_matrix.png      # Confusion matrix for Baseline CNN
|-- baseline_evaluation.txt            # Detailed test report for Baseline CNN
|-- cnn_history.json                   # Training loss/accuracy curves history
|-- efficientnet_b0.keras              # Pre-trained EfficientNet-B0 checkpoint (19.0 MB)
|-- efficientnet_confusion_matrix.png  # Confusion matrix for EfficientNet-B0
|-- efficientnet_evaluation.txt        # Detailed test report for EfficientNet-B0
|-- efficientnet_history.json          # Training history for EfficientNet-B0
|-- efficientnet_training_curves.png   # Accuracy & loss curves plot
|-- evaluate.py                        # Model evaluation & metrics script
|-- exploration_report.txt             # Dataset exploration statistics
|-- explore_data.py                    # Dataset distribution & near-duplicate analysis
|-- gradcam.py                         # Grad-CAM heatmap generation module
|-- gradcam_samples/                   # Example Grad-CAM visualizations
|-- preprocess.py                      # Image preprocessing & data pipeline
|-- requirements.txt                   # Project Python dependencies
|-- sample_images.png                  # Sample MRI slices grid visualization
|-- train_cnn.py                       # Baseline CNN model architecture & training
|-- train_efficientnet.py              # EfficientNet-B0 transfer learning training
|-- .gitignore                         # Git exclusion rules (dataset, caches, etc.)
|-- README.md                          # Project documentation
```

---

## Getting Started

### Prerequisites
- Python 3.10 or 3.11
- Virtual environment manager (`venv` or `conda`)
- Optional: CUDA-compatible GPU for accelerated training

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Lucky5683/brain-tumor-mri-classification.git
   cd brain-tumor-mri-classification
   ```

2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Running the Web App

Run the Streamlit application using the pre-trained model weights included in the repository:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, upload an MRI image, and inspect the classification result and Grad-CAM heatmap.

### Training & Evaluation

If the dataset has been downloaded into `dataset/`:
- **Explore dataset**:
  ```bash
  python explore_data.py
  ```
- **Train custom Baseline CNN**:
  ```bash
  python train_cnn.py
  ```
- **Train EfficientNet-B0 Transfer Learning**:
  ```bash
  python train_efficientnet.py
  ```
- **Evaluate models on the test set**:
  ```bash
  python evaluate.py
  ```
- **Generate Grad-CAM heatmaps**:
  ```bash
  python gradcam.py
  ```

---

## License

This project is open-source under the MIT License.
