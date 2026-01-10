import os
import glob
import time
import cv2
import numpy as np
import pywt # Wavelet Library / Dalgacık Kütüphanesi
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ----------------- PROJECT SETTINGS / PROJE AYARLARI -----------------

DATASET_DIR = "DATASET" 
CLASS_FOLDERS = ["cats", "dogs", "snakes"]
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# Output ARFF file for Step 3 (includes Wavelet)
# Adım 3 için çıktı ARFF dosyası (Wavelet içerir)
OUTPUT_ARFF_FILE = "features_step3_wavelet.arff" 

# --- GLCM CONFIGURATION ---
ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# --- LBP CONFIGURATION ---
LBP_POINTS = 24 
LBP_RADIUS = 3 
LBP_METHOD = 'uniform'

# --- LCP CONFIGURATION ---
LCP_BINS = 32

# --- WAVELET CONFIGURATION / WAVELET YAPILANDIRMASI ---
# Wavelet family to use (Haar is simple and effective for textures)
# Kullanılacak dalgacık ailesi (Haar, dokular için basit ve etkilidir)
WAVELET_FAMILY = 'db1' 

# ---------------------------------------------------------------------

def find_image_paths(base_dir):
    """
    Scans the dataset directory and returns a list of (image_path, label) tuples.
    Veri seti dizinini tarar ve (resim_yolu, etiket) demetlerinin bir listesini döndürür.
    """
    print(f"[INFO] Scanning dataset directory: '{base_dir}'...")
    all_image_paths = []
    total_images = 0

    if not os.path.isdir(base_dir):
        print(f"[ERROR] Main dataset folder '{base_dir}' not found.")
        return []

    for class_label in CLASS_FOLDERS:
        class_path = os.path.join(base_dir, class_label) 
        if not os.path.isdir(class_path):
            print(f"[WARNING] Class folder '{class_path}' not found. Skipping.")
            continue
        
        print(f"[INFO] Scanning class: '{class_label}'...")
        images_in_class = 0
        for ext in IMAGE_EXTENSIONS:
            search_path = os.path.join(class_path, ext)
            found_images = glob.glob(search_path)
            for img_path in found_images:
                all_image_paths.append((img_path, class_label))
                images_in_class += 1
        
        print(f"   -> Found {images_in_class} images.")
        total_images += images_in_class

    print(f"[INFO] Scan complete. Total images found: {total_images}.\n")
    return all_image_paths

def extract_color_features(img_bgr):
    """ Calculates Mean Blue, Green, Red. """
    try:
        means = cv2.mean(img_bgr)[:3]
        return list(means)
    except Exception as e:
        print(f"[ERROR] Color extraction error: {e}")
        return None

def extract_glcm_features_averaged(img_gray):
    """ Computes GLCM features and averages them across 4 directions. """
    try:
        glcm = graycomatrix(img_gray, distances=[1], angles=ANGLES_RAD, levels=256,
                            symmetric=True, normed=True)
        feature_vector = []
        for prop in GLCM_PROPERTIES:
            val = graycoprops(glcm, prop).mean()
            feature_vector.append(val)
        return feature_vector
    except Exception as e:
        print(f"[ERROR] GLCM extraction error: {e}")
        return None

def extract_lbp_features(img_gray):
    """ Computes Local Binary Pattern (LBP) histogram. """
    try:
        lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return list(hist)
    except Exception as e:
        print(f"[ERROR] LBP extraction error: {e}")
        return None

def extract_lcp_features(img_gray):
    """ Computes Local Contrast Pattern (LCP) histogram. """
    try:
        rows, cols = img_gray.shape
        lcp_image = np.zeros((rows, cols), dtype=np.float32)
        img_float = img_gray.astype(np.float32)

        # 8-neighbor offsets
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),  (1, 0),  (1, 1)]

        for dy, dx in neighbors:
            shifted = np.roll(img_float, shift=(-dy, -dx), axis=(0, 1))
            lcp_image += np.abs(img_float - shifted)
            
        lcp_norm = cv2.normalize(lcp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist, _ = np.histogram(lcp_norm.ravel(), bins=LCP_BINS, range=(0, 256), density=True)
        
        return list(hist)
    except Exception as e:
        print(f"[ERROR] LCP extraction error: {e}")
        return None

def extract_wavelet_features(img_gray):
    """
    Performs 2D Discrete Wavelet Transform (DWT) and extracts statistics from sub-bands.
    2B Ayrık Dalgacık Dönüşümü (DWT) gerçekleştirir ve alt bantlardan istatistikleri çıkarır.
    """
    try:
        # Perform DWT
        # DWT Gerçekleştir
        # coefficients: (cA, (cH, cV, cD))
        # cA: Approximation (Yaklaşım), cH: Horizontal (Yatay), cV: Vertical (Dikey), cD: Diagonal (Çapraz)
        coeffs = pywt.dwt2(img_gray, WAVELET_FAMILY)
        cA, (cH, cV, cD) = coeffs
        
        features = []
        
        # Extract Mean and Energy from each sub-band (4 bands * 2 features = 8 features)
        # Her alt banttan Ortalama ve Enerji çıkar (4 bant * 2 özellik = 8 özellik)
        sub_bands = {'LL': cA, 'LH': cH, 'HL': cV, 'HH': cD}
        
        for name, band in sub_bands.items():
            # Mean
            # Ortalama
            mean_val = np.mean(band)
            
            # Energy (Mean of squared magnitude)
            # Enerji (Karesel büyüklüğün ortalaması)
            energy_val = np.mean(np.square(band))
            
            features.extend([mean_val, energy_val])
            
        return features
    except Exception as e:
        print(f"[ERROR] Wavelet extraction error: {e}")
        return None

def process_image(image_path):
    """ Main processing function. Coordinates all feature extractors. """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- FEATURE EXTRACTION PIPELINE ---
    
    # 1. Color (3)
    f_color = extract_color_features(img_bgr)
    
    # 2. GLCM (6)
    f_glcm = extract_glcm_features_averaged(img_gray)
    
    # 3. LBP (Histogram)
    f_lbp = extract_lbp_features(img_gray)
    
    # 4. LCP (Histogram)
    f_lcp = extract_lcp_features(img_gray)
    
    # 5. Wavelet (8 features) - NEW
    f_wavelet = extract_wavelet_features(img_gray)

    # Combine
    if f_color and f_glcm and f_lbp and f_lcp and f_wavelet:
        return f_color + f_glcm + f_lbp + f_lcp + f_wavelet
    
    return None

def write_arff(filename, data, class_labels, lbp_len, lcp_len):
    """ Writes data to Weka ARFF format. """
    print(f"\n[INFO] Writing output to '{filename}'...")
    
    header = []
    header.append(f"@RELATION animal_classification_step3_wavelet")
    header.append("")

    # Color
    header.append("@ATTRIBUTE Color_Mean_Blue NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Green NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Red NUMERIC")
    
    # GLCM
    for prop in GLCM_PROPERTIES:
        header.append(f"@ATTRIBUTE GLCM_Avg_{prop} NUMERIC")
        
    # LBP
    for i in range(lbp_len):
        header.append(f"@ATTRIBUTE LBP_Hist_{i} NUMERIC")

    # LCP
    for i in range(lcp_len):
        header.append(f"@ATTRIBUTE LCP_Hist_{i} NUMERIC")
        
    # Wavelet (8 Attributes)
    bands = ['LL', 'LH', 'HL', 'HH']
    metrics = ['Mean', 'Energy']
    for band in bands:
        for metric in metrics:
            header.append(f"@ATTRIBUTE Wavelet_{band}_{metric} NUMERIC")

    # Class
    class_str = "{" + ",".join(class_labels) + "}"
    header.append(f"@ATTRIBUTE class {class_str}")
    header.append("")
    header.append("@DATA")
    
    data_lines = []
    for features, label in data:
        feat_str = ",".join(map(str, features))
        data_lines.append(f"{feat_str},{label}")

    try:
        with open(filename, "w") as f:
            f.write("\n".join(header))
            f.write("\n")
            f.write("\n".join(data_lines))
        print(f"[SUCCESS] ARFF file created successfully with {len(data_lines)} instances.")
    except Exception as e:
        print(f"[ERROR] Could not write ARFF file: {e}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    images = find_image_paths(DATASET_DIR)
    processed_data = []
    
    if images:
        print("----------------------------------------------------")
        print(f"[INFO] Processing {len(images)} images...")
        print(f"[INFO] Features: Color + GLCM + LBP + LCP + Wavelet (New)")
        print("----------------------------------------------------")
        
        start_time = time.time()
        
        for i, (path, label) in enumerate(images):
            if (i + 1) % 100 == 0:
                print(f"[PROGRESS] Processing image {i + 1}/{len(images)}...")
            
            features = process_image(path)
            if features:
                processed_data.append((features, label))
        
        end_time = time.time()
        elapsed = end_time - start_time
        print("----------------------------------------------------")
        print(f"[INFO] Processing finished in {elapsed:.2f} seconds.")
        
        if processed_data:
            total_len = len(processed_data[0][0])
            # Color(3) + GLCM(6) + Wavelet(8) = 17 fixed
            # LCP = 32
            lcp_len = LCP_BINS
            fixed_len = 3 + 6 + 8
            lbp_len = total_len - fixed_len - lcp_len
            
            write_arff(OUTPUT_ARFF_FILE, processed_data, CLASS_FOLDERS, lbp_len, lcp_len)
        else:
            print("[ERROR] No features extracted.")
    else:
        print("[ERROR] No images found.")