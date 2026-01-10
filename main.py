import os
import glob
import time
import cv2
import numpy as np
import pywt # Wavelet Library
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ----------------- PROJECT SETTINGS / PROJE AYARLARI -----------------

DATASET_DIR = "DATASET" 
CLASS_FOLDERS = ["cats", "dogs", "snakes"]
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# Output ARFF file for Final Step (All Features + Fourier)
# Final Adımı için çıktı ARFF dosyası (Tüm Özellikler + Fourier)
OUTPUT_ARFF_FILE = "features_step5(final)_Fourier.arff" 

# --- GLCM CONFIGURATION / GLCM AYARLARI ---
# Average directions: 0, 45, 90, 135
ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# --- LBP CONFIGURATION / LBP AYARLARI ---
LBP_POINTS = 24 
LBP_RADIUS = 3 
LBP_METHOD = 'uniform'

# --- LCP CONFIGURATION / LCP AYARLARI ---
LCP_BINS = 32

# --- WAVELET CONFIGURATION / WAVELET AYARLARI ---
WAVELET_FAMILY = 'db1' 

# --- HERMITE CONFIGURATION / HERMITE AYARLARI ---
HERMITE_SIGMA = 1.0 
HERMITE_KERNEL_SIZE = 7 

# ---------------------------------------------------------------------

def find_image_paths(base_dir):
    """
    Scans the dataset directory and returns a list of (image_path, label) tuples.
    Veri seti dizinini tarar ve (resim_yolu, etiket) listesini döndürür.
    """
    print(f"[INFO] Scanning dataset directory: '{base_dir}'...")
    all_image_paths = []
    
    if not os.path.isdir(base_dir):
        print(f"[ERROR] Main dataset folder '{base_dir}' not found.")
        return []

    for class_label in CLASS_FOLDERS:
        class_path = os.path.join(base_dir, class_label) 
        if not os.path.isdir(class_path):
            continue
        
        images_in_class = 0
        for ext in IMAGE_EXTENSIONS:
            search_path = os.path.join(class_path, ext)
            found_images = glob.glob(search_path)
            for img_path in found_images:
                all_image_paths.append((img_path, class_label))
                images_in_class += 1
        
        print(f"[INFO] Class '{class_label}': {images_in_class} images found.")
        
    return all_image_paths

# --- FEATURE EXTRACTION FUNCTIONS / ÖZNİTELİK ÇIKARMA FONKSİYONLARI ---

def extract_color_features(img_bgr):
    """ Extracts Mean Blue, Green, Red values (3 Features). """
    try:
        return list(cv2.mean(img_bgr)[:3])
    except: return None

def extract_glcm_features_averaged(img_gray):
    """ Extracts GLCM features averaged across 4 directions (6 Features). """
    try:
        glcm = graycomatrix(img_gray, distances=[1], angles=ANGLES_RAD, levels=256,
                            symmetric=True, normed=True)
        feature_vector = []
        for prop in GLCM_PROPERTIES:
            val = graycoprops(glcm, prop).mean()
            feature_vector.append(val)
        return feature_vector
    except: return None

def extract_lbp_features(img_gray):
    """ Extracts Local Binary Pattern histogram. """
    try:
        lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return list(hist)
    except: return None

def extract_lcp_features(img_gray):
    """ Extracts Local Contrast Pattern histogram. """
    try:
        rows, cols = img_gray.shape
        lcp_image = np.zeros((rows, cols), dtype=np.float32)
        img_float = img_gray.astype(np.float32)
        neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        
        for dy, dx in neighbors:
            shifted = np.roll(img_float, shift=(-dy, -dx), axis=(0, 1))
            lcp_image += np.abs(img_float - shifted)
            
        lcp_norm = cv2.normalize(lcp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        hist, _ = np.histogram(lcp_norm.ravel(), bins=LCP_BINS, range=(0, 256), density=True)
        return list(hist)
    except: return None

def extract_wavelet_features(img_gray):
    """ Extracts Discrete Wavelet Transform features (8 Features). """
    try:
        coeffs = pywt.dwt2(img_gray, WAVELET_FAMILY)
        cA, (cH, cV, cD) = coeffs
        features = []
        # Extract Mean and Energy for each sub-band (LL, LH, HL, HH)
        for band in [cA, cH, cV, cD]:
            features.extend([np.mean(band), np.mean(np.square(band))])
        return features
    except: return None

def extract_hermite_features(img_gray):
    """ 
    Extracts Hermite Transform features using Gaussian Derivatives (6 Features). 
    Gauss Türevleri kullanarak Hermite Dönüşümü özelliklerini çıkarır.
    """
    try:
        ksize = HERMITE_KERNEL_SIZE
        sigma = HERMITE_SIGMA
        ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
        xx, yy = np.meshgrid(ax, ax)

        # 0th Order Gaussian
        kernel_g = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel_g /= (2 * np.pi * np.square(sigma))

        # 1st Order Derivatives
        kernel_h10 = (-xx / np.square(sigma)) * kernel_g
        kernel_h01 = (-yy / np.square(sigma)) * kernel_g

        # 2nd Order Derivatives
        kernel_h20 = ((np.square(xx) / np.power(sigma, 4)) - (1 / np.square(sigma))) * kernel_g
        kernel_h02 = ((np.square(yy) / np.power(sigma, 4)) - (1 / np.square(sigma))) * kernel_g
        kernel_h11 = ((xx * yy) / np.power(sigma, 4)) * kernel_g

        filters = [kernel_g, kernel_h10, kernel_h01, kernel_h20, kernel_h02, kernel_h11]
        hermite_features = []
        
        for kernel in filters:
            response = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
            energy = np.mean(np.square(response))
            hermite_features.append(energy)
            
        return hermite_features
    except: return None

def extract_fourier_features(img_gray):
    """
    Extracts Global Frequency features using Fourier Transform (FFT) (2 Features).
    Fourier Dönüşümü (FFT) kullanarak Küresel Frekans özelliklerini çıkarır.
    """
    try:
        # Compute 2D FFT
        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        
        # Magnitude Spectrum
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        # Mean frequency and Energy of frequency
        mean_freq = np.mean(magnitude_spectrum)
        energy_freq = np.mean(np.square(magnitude_spectrum))
        
        return [mean_freq, energy_freq]
    except Exception as e:
        print(f"[ERROR] Fourier extraction error: {e}")
        return None

def process_image(image_path):
    """ 
    Main Processing Pipeline.
    Coordinates all 7 feature extraction methods.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Color
    f_color = extract_color_features(img_bgr)
    # 2. GLCM
    f_glcm = extract_glcm_features_averaged(img_gray)
    # 3. LBP
    f_lbp = extract_lbp_features(img_gray)
    # 4. LCP
    f_lcp = extract_lcp_features(img_gray)
    # 5. Wavelet
    f_wavelet = extract_wavelet_features(img_gray)
    # 6. Hermite
    f_hermite = extract_hermite_features(img_gray)
    # 7. Fourier (NEW)
    f_fourier = extract_fourier_features(img_gray)

    # Combine all
    if f_color and f_glcm and f_lbp and f_lcp and f_wavelet and f_hermite and f_fourier:
        return f_color + f_glcm + f_lbp + f_lcp + f_wavelet + f_hermite + f_fourier
    
    return None

def write_arff(filename, data, class_labels, lbp_len, lcp_len):
    """ Writes the final ARFF file with all attributes. """
    print(f"\n[INFO] Writing output to '{filename}'...")
    
    header = []
    header.append("@RELATION animal_classification_final_full_fourier")
    header.append("")

    # --- Attribute Definitions ---
    
    # 1. Color (3)
    header.append("@ATTRIBUTE Color_Mean_Blue NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Green NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Red NUMERIC")
    
    # 2. GLCM (6)
    for prop in GLCM_PROPERTIES: 
        header.append(f"@ATTRIBUTE GLCM_Avg_{prop} NUMERIC")
        
    # 3. LBP (Variable)
    for i in range(lbp_len): 
        header.append(f"@ATTRIBUTE LBP_Hist_{i} NUMERIC")
        
    # 4. LCP (Variable)
    for i in range(lcp_len): 
        header.append(f"@ATTRIBUTE LCP_Hist_{i} NUMERIC")
        
    # 5. Wavelet (8)
    for band in ['LL', 'LH', 'HL', 'HH']:
        for metric in ['Mean', 'Energy']: 
            header.append(f"@ATTRIBUTE Wavelet_{band}_{metric} NUMERIC")
            
    # 6. Hermite (6)
    hermite_labels = ['G_00', 'H_10', 'H_01', 'H_20', 'H_02', 'H_11']
    for label in hermite_labels:
        header.append(f"@ATTRIBUTE Hermite_Energy_{label} NUMERIC")
        
    # 7. Fourier (2)
    header.append("@ATTRIBUTE Fourier_Mean NUMERIC")
    header.append("@ATTRIBUTE Fourier_Energy NUMERIC")

    # Class
    class_str = "{" + ",".join(class_labels) + "}"
    header.append(f"@ATTRIBUTE class {class_str}")
    header.append("")
    header.append("@DATA")
    
    # Write Data
    data_lines = []
    for features, label in data:
        data_lines.append(f"{','.join(map(str, features))},{label}")

    with open(filename, "w") as f:
        f.write("\n".join(header + data_lines))
    print(f"[SUCCESS] ARFF file created with {len(data_lines)} instances.")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    images = find_image_paths(DATASET_DIR)
    processed_data = []
    
    if images:
        print("----------------------------------------------------")
        print(f"[INFO] Features: Color + GLCM + LBP + LCP + Wavelet + Hermite + Fourier")
        print(f"[INFO] Target Output: {OUTPUT_ARFF_FILE}")
        print("----------------------------------------------------")
        
        start_time = time.time()
        
        for i, (path, label) in enumerate(images):
            if (i + 1) % 100 == 0: 
                print(f"[PROGRESS] Processing {i + 1}/{len(images)}...")
            
            features = process_image(path)
            if features: 
                processed_data.append((features, label))
        
        end_time = time.time()
        elapsed = end_time - start_time
        print("----------------------------------------------------")
        print(f"[INFO] Processing finished in {elapsed:.2f} seconds.")

        if processed_data:
            # Calculate dynamic lengths
            total_len = len(processed_data[0][0])
            
            # Fixed lengths:
            # Color(3) + GLCM(6) + Wavelet(8) + Hermite(6) + Fourier(2) = 25
            FIXED_ATTR_COUNT = 3 + 6 + 8 + 6 + 2
            
            lcp_len = LCP_BINS
            lbp_len = total_len - FIXED_ATTR_COUNT - lcp_len
            
            write_arff(OUTPUT_ARFF_FILE, processed_data, CLASS_FOLDERS, lbp_len, lcp_len)
        else:
            print("[ERROR] No features extracted.")
    else:
        print("[ERROR] No images found in DATASET directory.")