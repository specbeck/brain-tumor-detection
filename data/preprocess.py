# train.py
import os
import numpy as np
from data.processor import load_medical_dataset

def main():
    # Define absolute path structure according to your system environment
    DATA_DIR = "./data"
    TRAIN_DIR = os.path.join(DATA_DIR, "Training")
    TEST_DIR = os.path.join(DATA_DIR, "Testing")
    
    print("==================================================")
    print("Initializing Medical Image Preprocessing Pipeline")
    print("==================================================")
    
    # 1. Load and process datasets on CPU
    X_train, y_train, class_map = load_medical_dataset(TRAIN_DIR, target_size=(128, 128))
    X_test, y_test, _ = load_medical_dataset(TEST_DIR, target_size=(128, 128))
    
    print("\n[Data Status Check]")
    print(f"Training matrices shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing matrices shape:  X={X_test.shape}, y={y_test.shape}")
    print(f"Class Map Verification:  {class_map}")
    
    # Save the clean matrices temporarily in results/ to avoid reprocessing every execution run
    os.makedirs("./results", exist_ok=True)
    np.save("./results/X_train.npy", X_train)
    np.save("./results/y_train.npy", y_train)
    np.save("./results/X_test.npy", X_test)
    np.save("./results/y_test.npy", y_test)
    print("\nPreprocessed arrays successfully cached in './results/' directory.")

if __name__ == "__main__":
    main()
