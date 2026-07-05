import os
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from data.features import transform_dataset
import time

def main():
    print("==================================================")
    print("Initiating Classical ML Pipeline with Cross-Validation")
    print("==================================================")

    # 1. Load the preprocessed image arrays
    print("Loading standardized image arrays...")
    X_train_img = np.load("./results/X_train.npy")
    y_train = np.load("./results/y_train.npy")

    # 2. Extract Mathematical Features (GLCM, LBP, Stats)
    print("\nExtracting texture and statistical features...")
    start_time = time.time()
    X_train_features = transform_dataset(X_train_img)
    print(f"Feature extraction completed in {time.time() - start_time:.2f} seconds.")
    print(f"Feature matrix shape: {X_train_features.shape}")

    np.save("./results/X_train_features.npy", X_train_features)

    # 3. Define the Models
    print("\nConfiguring model pipelines...")
    svm_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='rbf', probability=True, random_state=42, C=1.0))
    ])

    rf_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=15, random_state=42, n_jobs=-1))
    ])

    # 4. Perform 5-Fold Cross-Validation
    print("\nRunning 5-Fold Stratified Cross-Validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    svm_cv_scores = cross_val_score(svm_pipeline, X_train_features, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f" -> SVM CV Accuracy: {svm_cv_scores.mean()*100:.2f}% (+/- {svm_cv_scores.std()*100:.2f}%)")
    
    rf_cv_scores = cross_val_score(rf_pipeline, X_train_features, y_train, cv=cv, scoring='accuracy', n_jobs=-1)
    print(f" -> Random Forest CV Accuracy: {rf_cv_scores.mean()*100:.2f}% (+/- {rf_cv_scores.std()*100:.2f}%)")

    # 5. Train Final Models on Full Training Set
    print("\nTraining final models on full dataset for export...")
    svm_pipeline.fit(X_train_features, y_train)
    rf_pipeline.fit(X_train_features, y_train)

    # 6. Serialize Models
    os.makedirs("./models", exist_ok=True)
    joblib.dump(svm_pipeline, "./models/svm_pipeline.pkl")
    joblib.dump(rf_pipeline, "./models/rf_pipeline.pkl")
    print("\nModels successfully saved to './models/' directory.")

if __name__ == "__main__":
    main()
