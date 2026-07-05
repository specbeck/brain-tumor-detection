import sys
import os
import cv2
import numpy as np
import joblib

# Import your custom feature extractor
from data.features import transform_dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_mri_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        sys.exit(1)

    print(f"--- Analyzing Scan: {os.path.basename(image_path)} ---")
    
    # 1. Load and Standardize Image
    try:
        # Read as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Resize to pipeline standard
        img_resized = cv2.resize(img, (128, 128))
    except Exception as e:
        print(f"Error processing image: {e}")
        sys.exit(1)

    # 2. Extract Features
    print("Extracting 163-dimensional feature space...")
    single_img_array = np.expand_dims(img_resized, axis=0)
    raw_features = transform_dataset(single_img_array)

    # 3. Load Models
    print("Loading pre-trained Classical Models...")
    try:
        svm_pipeline = joblib.load("./models/svm_pipeline.pkl")
        rf_pipeline = joblib.load("./models/rf_pipeline.pkl")
    except FileNotFoundError:
        print("Error: Model weights not found in './models/'. Please run train_classical.py first.")
        sys.exit(1)

    class_map = {0: 'Glioma', 1: 'Meningioma', 2: 'No Tumor', 3: 'Pituitary'}

    # 4. Predict
    svm_pred = svm_pipeline.predict(raw_features)[0]
    rf_pred = rf_pipeline.predict(raw_features)[0]

    print("\n========================================")
    print("          DIAGNOSTIC RESULTS            ")
    print("========================================")
    print(f" SVM Prediction (Primary) : {class_map[svm_pred].upper()}")
    print(f" RF Ensemble (Secondary)  : {class_map[rf_pred].upper()}")
    print("========================================")

if __name__ == "__main__":
    main()
