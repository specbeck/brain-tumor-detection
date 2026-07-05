import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from scipy.stats import skew, kurtosis

class FeatureExtractor:
    def __init__(self):
        # LBP parameters
        self.radius = 3
        self.n_points = 8 * self.radius
        
        # GLCM parameters
        self.distances = [1]
        self.angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] # 0, 45, 90, 135 degrees

    def extract_glcm(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts rotationally invariant texture features.
        """
        glcm = graycomatrix(image, distances=self.distances, angles=self.angles, 
                            levels=256, symmetric=True, normed=True)
        
        features = []
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
            # Average across all angles to ensure slice/orientation agnosticism
            features.append(graycoprops(glcm, prop).mean())
            
        return np.array(features)

    def extract_lbp(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts micro-pattern edge histograms.
        """
        lbp = local_binary_pattern(image, self.n_points, self.radius, method='uniform')
        
        # Calculate the histogram of LBP
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return hist

    def extract_statistics(self, image: np.ndarray) -> np.ndarray:
        """
        Extracts first-order intensity features.
        """
        pixels = image.ravel()
        features = [
            np.mean(pixels),
            np.std(pixels),
            skew(pixels),
            kurtosis(pixels)
        ]
        return np.array(features)

    def extract_grid_statistics(self, image: np.ndarray, grid_size: int = 4) -> np.ndarray:
        """
        Divides the image into a grid (e.g., 4x4) and calculates the local mean 
        and variance for each sector. Captures localized intensity abnormalities 
        (like diffuse gliomas) that get washed out in global statistics.
        """
        h, w = image.shape
        grid_h, grid_w = h // grid_size, w // grid_size
        
        grid_features = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                # Extract the local patch
                patch = image[i*grid_h : (i+1)*grid_h, j*grid_w : (j+1)*grid_w]
                
                # Calculate regional intensity signatures
                patch_mean = np.mean(patch)
                patch_std = np.std(patch)
                
                grid_features.extend([patch_mean, patch_std])
                
        return np.array(grid_features)

    
    def extract_all(self, image: np.ndarray) -> np.ndarray:
        """
        Combines all features into a single 1D vector.
        """
        glcm_feats = self.extract_glcm(image)
        lbp_feats = self.extract_lbp(image)
        stat_feats = self.extract_statistics(image)
        grid_feats = self.extract_grid_statistics(image, grid_size=8)

        return np.concatenate([glcm_feats, lbp_feats, stat_feats, grid_feats])



def transform_dataset(X_images: np.ndarray) -> np.ndarray:
    """
    Applies the FeatureExtractor to an entire dataset of images.
    """
    extractor = FeatureExtractor()
    X_features = []
    
    print(f"Extracting features for {len(X_images)} images...")
    for i, img in enumerate(X_images):
        if i > 0 and i % 500 == 0:
            print(f" -> Processed {i}/{len(X_images)} images")
        
        features = extractor.extract_all(img)
        X_features.append(features)
    
    X = np.array(X_features)
    print("Feature matrix shape:", X.shape)
    return X
