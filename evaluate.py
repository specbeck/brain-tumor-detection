import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from data.features import transform_dataset

def plot_confusion_matrices(y_test, y_pred_svm, y_pred_rf, class_names):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # SVM Confusion Matrix
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('SVM (RBF) Confusion Matrix', fontsize=14)
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    
    # RF Confusion Matrix
    cm_rf = confusion_matrix(y_test, y_pred_rf)
    sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title('Random Forest Confusion Matrix', fontsize=14)
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('./results/classical_confusion_matrices.png', dpi=300)
    print(" -> Saved Confusion Matrices to './results/classical_confusion_matrices.png'")

def plot_multiclass_roc(y_test, y_prob_svm, y_prob_rf, n_classes, class_names):
    # Binarize labels for One-vs-Rest ROC calculation
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3])
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['blue', 'red', 'green', 'purple']
    
    for i, color in zip(range(n_classes), colors):
        # SVM ROC
        fpr_svm, tpr_svm, _ = roc_curve(y_test_bin[:, i], y_prob_svm[:, i])
        roc_auc_svm = auc(fpr_svm, tpr_svm)
        axes[0].plot(fpr_svm, tpr_svm, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc_svm:.2f})')
        
        # RF ROC
        fpr_rf, tpr_rf, _ = roc_curve(y_test_bin[:, i], y_prob_rf[:, i])
        roc_auc_rf = auc(fpr_rf, tpr_rf)
        axes[1].plot(fpr_rf, tpr_rf, color=color, lw=2, label=f'{class_names[i]} (AUC = {roc_auc_rf:.2f})')
        
    for ax, title in zip(axes, ['SVM (RBF) ROC Curves', 'Random Forest ROC Curves']):
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title, fontsize=14)
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.savefig('./results/classical_roc_curves.png', dpi=300)
    print(" -> Saved ROC Curves to './results/classical_roc_curves.png'")


def plot_feature_importance(rf_pipeline):
    # Extract the trained Random Forest model from the pipeline
    rf_model = rf_pipeline.named_steps['classifier']
    importances = rf_model.feature_importances_
    
    # Reconstruct the feature names based on the features.py extraction logic
    feature_names = []
    
    # 1. GLCM Features (5)
    feature_names.extend(['GLCM_Contrast', 'GLCM_Dissim', 'GLCM_Homogen', 'GLCM_Energy', 'GLCM_Correl'])
    
    # 2. LBP Features (26 bins for radius=3, n_points=24 uniform method)
    feature_names.extend([f'LBP_Texture_Bin_{i}' for i in range(26)])
    
    # 3. Global Statistics (4)
    feature_names.extend(['Global_Mean', 'Global_Std', 'Global_Skew', 'Global_Kurtosis'])
    
    # 4. Dynamic Spatial Grid Features (matches our new 8x8 setup)
    GRID_SIZE = 8 
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            feature_names.extend([f'Grid_{i},{j}_Mean', f'Grid_{i},{j}_Std'])
            
    # Convert to numpy array for easy indexing
    feature_names = np.array(feature_names)
    
    # Sort the top 20 features
    indices = np.argsort(importances)[::-1][:20]
    
    plt.figure(figsize=(12, 7))
    plt.title(f"Top 20 Feature Importances (RF - 8x8 Grid)", fontsize=16, fontweight='bold')
    
    # Plot using the mapped feature names
    plt.bar(range(20), importances[indices], color="teal", align="center", edgecolor='black')
    plt.xticks(range(20), feature_names[indices], rotation=45, ha='right', fontsize=10)
    plt.xlim([-1, 20])
    plt.ylabel("Relative Importance (%)", fontsize=12)
    
    # Add subtle grid lines for readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig('./results/classical_feature_importance.png', dpi=300)
    print(" -> Saved Feature Importances to './results/classical_feature_importance.png'")


def evaluate_models():
    print("==================================================")
    print("Executing Comprehensive Classical Audit")
    print("==================================================")
    
    print("Loading test data and models...")
    X_test_img = np.load("./results/X_test.npy")
    y_test = np.load("./results/y_test.npy")
    X_test_features = transform_dataset(X_test_img)
    
    svm_pipeline = joblib.load("./models/svm_pipeline.pkl")
    rf_pipeline = joblib.load("./models/rf_pipeline.pkl")
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    n_classes = len(class_names)
    
    # Generate Predictions and Probabilities
    y_pred_svm = svm_pipeline.predict(X_test_features)
    y_prob_svm = svm_pipeline.predict_proba(X_test_features)
    
    y_pred_rf = rf_pipeline.predict(X_test_features)
    y_prob_rf = rf_pipeline.predict_proba(X_test_features)
    
    print("\n--- Support Vector Machine (SVM) Results ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_svm) * 100:.2f}%")
    print(classification_report(y_test, y_pred_svm, target_names=class_names))
    
    print("\n--- Random Forest Results ---")
    print(f"Overall Accuracy: {accuracy_score(y_test, y_pred_rf) * 100:.2f}%")
    print(classification_report(y_test, y_pred_rf, target_names=class_names))
    
    print("\nGenerating visual audits...")
    os.makedirs("./results", exist_ok=True)
    plot_confusion_matrices(y_test, y_pred_svm, y_pred_rf, class_names)
    plot_multiclass_roc(y_test, y_prob_svm, y_prob_rf, n_classes, class_names)
    plot_feature_importance(rf_pipeline)
    print("\nAudit Complete! Visuals are ready in the './results' folder.")

if __name__ == "__main__":
    evaluate_models()
