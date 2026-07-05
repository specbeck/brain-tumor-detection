import os
import cv2
import numpy as np
from skimage.exposure import equalize_adapthist

class MedicalImageProcessor:
    def __init__(self, target_size=(128, 128)):
        self.target_size = target_size

    def crop_brain_contour(self, img: np.ndarray) -> np.ndarray:
        """
        Uses Extreme Point Contouring to crop away dead background space.
        """
        # Blur to smooth noise, then threshold
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 45, 255, cv2.THRESH_BINARY)
        
        # Find external contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return img # Fallback if no contours are found
            
        # Find the largest contour (the head)
        c = max(contours, key=cv2.contourArea)
        
        # Find extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
        
        # Crop strictly to geometric boundaries
        cropped = img[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
        return cropped

    def enhance_contrast(self, img: np.ndarray) -> np.ndarray:
        """
        Applies a soft CLAHE to enhance subtle tumor textures without blowing out normal tissue.
        """
        if img.max() > 0:
            img_normalized = img / 255.0
            enhanced = equalize_adapthist(img_normalized, clip_limit=0.01)
            return (enhanced * 255).astype(np.uint8)
        return img

    def process_pipeline(self, image_path: str) -> np.ndarray:
        # Load as grayscale
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
            
        # 1. Crop empty space
        cropped = self.crop_brain_contour(img)
        
        # 2. Enhance contrast
        enhanced = self.enhance_contrast(cropped)
        
        # 3. Resize to standard dimensions
        final_img = cv2.resize(enhanced, self.target_size, interpolation=cv2.INTER_AREA)
        
        return final_img

def load_medical_dataset(base_dir: str, target_size=(128, 128)):
    """
    Walks through the dataset directory, processes images, and returns X, y arrays.
    """
    processor = MedicalImageProcessor(target_size=target_size)
    X, y = [], []
    class_mapping = {'glioma': 0, 'meningioma': 1, 'notumor': 2, 'pituitary': 3}
    
    print(f"Loading data from: {base_dir}")
    for class_name, label_idx in class_mapping.items():
        class_folder = os.path.join(base_dir, class_name)
        if not os.path.exists(class_folder):
            continue
            
        file_list = [f for f in os.listdir(class_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f" -> Processing '{class_name}': {len(file_list)} images.")
        
        for file_name in file_list:
            img_path = os.path.join(class_folder, file_name)
            try:
                processed_slice = processor.process_pipeline(img_path)
                X.append(processed_slice)
                y.append(label_idx)
            except Exception as e:
                pass # Skip corrupted files silently to maintain pipeline speed
                
    return np.array(X), np.array(y), class_mapping
