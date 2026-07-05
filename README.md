# Brain Tumor Classification via Localization

## Project Overview

This repository contains a CPU-optimized, clinical-grade machine learning pipeline for classifying 2D Brain MRI scans into four distinct categories: **Glioma, Meningioma, Pituitary, or No Tumor**.

Given strict computational constraints and the uncurated nature of the dataset (mixed sagittal, coronal, and axial planes), deep learning models (CNNs) proved highly susceptible to spatial overfitting and geographical bias. Consequently, this project champions a **Classical Machine Learning Pipeline**. By utilizing an engineered 163-dimensional feature space (GLCM macro-textures, LBP micro-patterns, and an 8×8 spatial grid), the Support Vector Machine (SVM) achieves highly stable, interpretable predictions with a 99% recall rate for healthy brains.

---

## Repository Structure

The `~/voxelgrids` directory currently contains:

- `data/` : Contains the Train/Test image splits organized by class folders.
- `models/` : Stores the serialized `.pkl` weights for the trained SVM and Random Forest pipelines.
- `results/` : Contains generated artifacts (ROC curves, Confusion Matrices, Saliency Heatmaps, and feature matrices).
- `assistant/` : Contains the `context.json` brain for the LLM engineering co-pilot.
- `processor.py` : Handles raw data ingestion, extreme-point geometric cropping, and CLAHE contrast enhancement.
- `train_classical.py` : Executes feature extraction and trains the classical models with 5-Fold Cross-Validation.
- `evaluate.py` : Evaluates model metrics and generates clinical audit visuals.
- `predict.py` : **CLI Entrypoint** for running fast terminal-based inference on single scans.
- `app.py` : Experimental Streamlit UI with an integrated AI Co-Pilot (Requires Gemini API Key).
- `requirements.txt` : List of all required Python libraries.

---

## Setup & Installation

It is highly recommended to run this project inside a virtual environment (e.g., `venv` or `conda`).

1. **Clone the repository and navigate to the root directory:**

   ```bash
   cd path/to/voxelgrids
   ```

2. **Install the required dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Guide

### 1. Training the Models

To rebuild the feature space from the raw images and train the classical models from scratch:

```bash
python train_classical.py
```

> **Note:** Feature extraction for ~5,600 images takes approximately 2–3 minutes on a standard CPU.

---

### 2. Evaluating the Pipeline

To run the evaluation suite against the held-out test set and generate the ROC-AUC curves, Confusion Matrices, and Feature Importance plots:

```bash
python evaluate.py
```

Visual assets will be saved automatically to the `results/` directory.

---

### 3. Running Inference (Command Line)

To test the pre-trained model weights on a single MRI scan, use the `predict.py` entrypoint. Pass the path of the image as an argument:

```bash
python predict.py data/Testing/glioma/Te-gl_0010.jpg
```

**Expected Output:**

```text
--- Analyzing Scan: Te-gl_0010.jpg ---
Extracting 163-dimensional feature space...
Loading pre-trained Classical Models...

========================================
          DIAGNOSTIC RESULTS
========================================
 SVM Prediction (Primary) : GLIOMA
 RF Ensemble (Secondary)  : GLIOMA
========================================
```

---
### 4. Interactive Engineering Co-Pilot (Web App)
The repository includes a dedicated Streamlit web application that acts as an interactive architectural white-paper. Powered by a context-aware LLM, this Co-Pilot can be interrogated to explain feature engineering decisions, defend the classical ML pipeline against deep learning alternatives, and break down clinical metrics.

*Note: This app is strictly a conversational assistant and does not run live image inference (use the `predict.py` CLI for inference).*

**To launch the Co-Pilot:**
1. Open `app.py` and insert a valid Google Gemini API Key.
2. Run the application from your terminal:
```bash
streamlit run app.py
