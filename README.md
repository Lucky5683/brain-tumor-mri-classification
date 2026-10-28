# Brain Tumor MRI Classification Using EfficientNet-B0 with Grad-CAM

An explainable deep learning application for classifying brain MRI images into four categories: glioma, meningioma, pituitary tumor, and no tumor. The project combines image classification with Grad-CAM visualizations to help users inspect image regions that influence model predictions.

## Overview

This project explores transfer learning and convolutional neural networks for brain MRI image classification. It includes baseline models, an EfficientNet-B0 model, an experimental spatial attention feature-fusion model, evaluation utilities, and a Streamlit interface.

Grad-CAM visualizations provide an additional way to inspect model behavior. They are explanatory visualizations, not proof that a prediction is medically correct.

## Features

- Brain MRI image classification across four categories.
- Custom CNN and EfficientNet-B0 baseline models.
- Experimental spatial attention feature fusion.
- Grad-CAM-based model explainability.
- Evaluation utilities for classification metrics and confusion matrices.
- Streamlit interface for interactive predictions.
- Dataset-quality and evaluation scripts.

## Technology Stack

- Python
- TensorFlow and Keras
- Streamlit
- NumPy and Pandas
- OpenCV and Pillow
- scikit-learn
- Matplotlib

## Repository Structure

```text
brain-tumor-mri-classification/
├── app.py
├── config.py
├── preprocess.py
├── train_efficientnet.py
├── evaluate.py
├── gradcam.py
├── data_quality.py
├── requirements.txt
├── diagrams/
├── evaluation/
├── experiments/
├── scripts/
├── src/
└── tests/
```

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/Lucky5683/brain-tumor-mri-classification.git
cd brain-tumor-mri-classification
```

Create and activate a virtual environment.

**Windows PowerShell**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install dependencies in an environment compatible with the versions specified in `requirements.txt`.

## Dataset Preparation

The MRI dataset is not included in this repository. Obtain the dataset from its authorized source and organize it according to the paths and class names expected by the project's configuration and data-loading code.

Before training or evaluating a model, verify the dataset directory structure and confirm that the configured paths match your local setup.

## Run the Application

From the repository root, run:

```bash
streamlit run app.py
```

Upload an MRI image using the interface and inspect the predicted class, available evaluation information, and explainability visualization.

Some application features may require model checkpoints, evaluation artifacts, or dataset files that are not included in a fresh clone.

## Models and Evaluation

The project contains baseline classification models and an experimental spatial attention feature-fusion model. Evaluation utilities support model comparison and classification analysis.

Reported metrics should be interpreted according to the corresponding validation or test split. Results from different splits are not directly interchangeable, and test-set results should not be used to select model configurations.

Consult the evaluation artifacts and scripts for the metrics associated with each model. Results depend on dataset quality, split methodology, preprocessing, and model configuration.

## Explainability

Grad-CAM generates a heatmap highlighting image regions associated with a model's output. These visualizations can help inspect predictions, but they do not establish causal reasoning or clinical reliability.

## Limitations

- Model performance depends on the quality and representativeness of the dataset.
- Dataset and checkpoint availability may vary between environments.
- Predictions may be incorrect, including predictions made with high confidence.
- Explainability maps should not be interpreted as medical evidence.
- The application has not been established as a clinically validated diagnostic system.

## Future Improvements

- Improve reproducibility through documented environment and model configurations.
- Expand external validation across independent datasets.
- Improve checkpoint and evaluation-artifact management.
- Strengthen automated end-to-end testing.
- Improve calibration and reliability analysis.

## Disclaimer

This project is intended for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or make clinical decisions about any person.

## License

No license has been specified yet. Unless a license is added, reuse and redistribution are subject to applicable copyright restrictions.
