import os
import glob
import time
import cv2
import numpy as np
import pywt # Wavelet
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ----------------- PROJECT SETTINGS / PROJE AYARLARI -----------------

DATASET_DIR = "DATASET" 
CLASS_FOLDERS = ["cats", "dogs", "snakes"]
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# Output ARFF file for Step 4 (Hermite included)
# Adım 4 için çıktı ARFF dosyası (Hermite dahil)
OUTPUT_ARFF_FILE = "features_step4_hermite.arff" 

# --- GLCM CONFIGURATION ---
ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# --- LBP CONFIGURATION ---
LBP_POINTS = 24 
LBP_RADIUS = 3 
LBP_METHOD = 'uniform'

# --- LCP CONFIGURATION ---
LCP_BINS = 32

# --- WAVELET CONFIGURATION ---
WAVELET_FAMILY = 'db1' 

# --- HERMITE CONFIGURATION / HERMITE YAPILANDIRMASI ---
# Sigma for Gaussian window (Scale of analysis)
# Gauss penceresi için Sigma (Analiz ölçeği)
HERMITE_SIGMA = 1.0 
# Kernel size (Odd number, typically 6*sigma + 1)
# Çekirdek boyutu (Tek sayı, genellikle 6*sigma + 1)
HERMITE_KERNEL_SIZE = 7 

# ---------------------------------------------------------------------

def find_image_paths(base_dir):
    """ Scans dataset directory. """
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

# --- FEATURE EXTRACTORS ---

def extract_color_features(img_bgr):
    """ Color features (Mean B, G, R). """
    try:
        return list(cv2.mean(img_bgr)[:3])
    except: return None

def extract_glcm_features_averaged(img_gray):
    """ GLCM features (Averaged over 4 directions). """
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
    """ LBP Histogram features. """
    try:
        lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return list(hist)
    except: return None

def extract_lcp_features(img_gray):
    """ LCP Histogram features. """
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
    """ Wavelet Transform features (Mean, Energy). """
    try:
        coeffs = pywt.dwt2(img_gray, WAVELET_FAMILY)
        cA, (cH, cV, cD) = coeffs
        features = []
        for band in [cA, cH, cV, cD]:
            features.extend([np.mean(band), np.mean(np.square(band))])
        return features
    except: return None

def extract_hermite_features(img_gray):
    """
    Extracts texture features using Hermite Filters (Gaussian Derivatives).
    Simulates the Steered Hermite Transform often used in texture analysis.
    
    Hermite Filtreleri (Gauss Türevleri) kullanarak doku özniteliklerini çıkarır.
    Doku analizinde sıklıkla kullanılan Yönlendirilmiş Hermite Dönüşümünü simüle eder.
    """
    try:
        ksize = HERMITE_KERNEL_SIZE
        sigma = HERMITE_SIGMA
        
        # Create coordinate grid (x, y)
        # Koordinat ızgarası oluştur (x, y)
        ax = np.linspace(-(ksize - 1) / 2., (ksize - 1) / 2., ksize)
        xx, yy = np.meshgrid(ax, ax)

        # 0th Order: Gaussian (G)
        # 0. Derece: Gauss (G)
        kernel_g = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sigma))
        kernel_g /= (2 * np.pi * np.square(sigma)) # Normalize

        # 1st Order Derivatives (Hermite Functions)
        # 1. Derece Türevler (Hermite Fonksiyonları)
        # H10 = dG/dx = (-x/sigma^2) * G
        kernel_h10 = (-xx / np.square(sigma)) * kernel_g
        # H01 = dG/dy = (-y/sigma^2) * G
        kernel_h01 = (-yy / np.square(sigma)) * kernel_g

        # 2nd Order Derivatives
        # 2. Derece Türevler
        # H20 = d2G/dx2 = ((x^2/sigma^4) - (1/sigma^2)) * G
        kernel_h20 = ((np.square(xx) / np.power(sigma, 4)) - (1 / np.square(sigma))) * kernel_g
        # H02 = d2G/dy2 = ((y^2/sigma^4) - (1/sigma^2)) * G
        kernel_h02 = ((np.square(yy) / np.power(sigma, 4)) - (1 / np.square(sigma))) * kernel_g
        # H11 = d2G/dxdy = (xy/sigma^4) * G
        kernel_h11 = ((xx * yy) / np.power(sigma, 4)) * kernel_g

        # List of filters [G, H10, H01, H20, H02, H11]
        filters = [kernel_g, kernel_h10, kernel_h01, kernel_h20, kernel_h02, kernel_h11]
        
        hermite_features = []
        
        # Apply each filter and calculate Energy
        # Her bir filtreyi uygula ve Enerjiyi hesapla
        for kernel in filters:
            # Convolve image with kernel
            # Görüntü ile çekirdeği (kernel) konvolüsyona tabi tut
            response = cv2.filter2D(img_gray, cv2.CV_32F, kernel)
            
            # Calculate Energy (Mean of squared response)
            # Enerjiyi Hesapla (Karesel tepkinin ortalaması)
            energy = np.mean(np.square(response))
            hermite_features.append(energy)
            
        return hermite_features

    except Exception as e:
        print(f"[ERROR] Hermite extraction error: {e}")
        return None

def process_image(image_path):
    """ Main Pipeline: Color + GLCM + LBP + LCP + Wavelet + Hermite """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None: return None
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    f_color = extract_color_features(img_bgr)
    f_glcm = extract_glcm_features_averaged(img_gray)
    f_lbp = extract_lbp_features(img_gray)
    f_lcp = extract_lcp_features(img_gray)
    f_wavelet = extract_wavelet_features(img_gray)
    f_hermite = extract_hermite_features(img_gray)

    if f_color and f_glcm and f_lbp and f_lcp and f_wavelet and f_hermite:
        return f_color + f_glcm + f_lbp + f_lcp + f_wavelet + f_hermite
    return None

def write_arff(filename, data, class_labels, lbp_len, lcp_len):
    print(f"\n[INFO] Writing output to '{filename}'...")
    header = ["@RELATION animal_classification_step4_hermite", ""]

    # Color
    header.append("@ATTRIBUTE Color_Mean_Blue NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Green NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Red NUMERIC")
    # GLCM
    for prop in GLCM_PROPERTIES: header.append(f"@ATTRIBUTE GLCM_Avg_{prop} NUMERIC")
    # LBP
    for i in range(lbp_len): header.append(f"@ATTRIBUTE LBP_Hist_{i} NUMERIC")
    # LCP
    for i in range(lcp_len): header.append(f"@ATTRIBUTE LCP_Hist_{i} NUMERIC")
    # Wavelet
    for band in ['LL', 'LH', 'HL', 'HH']:
        for metric in ['Mean', 'Energy']: header.append(f"@ATTRIBUTE Wavelet_{band}_{metric} NUMERIC")
    
    # Hermite (6 Attributes)
    hermite_labels = ['G_00', 'H_10', 'H_01', 'H_20', 'H_02', 'H_11']
    for label in hermite_labels:
        header.append(f"@ATTRIBUTE Hermite_Energy_{label} NUMERIC")

    class_str = "{" + ",".join(class_labels) + "}"
    header.append(f"@ATTRIBUTE class {class_str}")
    header.append("")
    header.append("@DATA")
    
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
        print(f"[INFO] Features: Color + GLCM + LBP + LCP + Wavelet + Hermite")
        print("----------------------------------------------------")
        
        start_time = time.time()
        for i, (path, label) in enumerate(images):
            if (i + 1) % 100 == 0: print(f"[PROGRESS] Processing {i + 1}/{len(images)}...")
            features = process_image(path)
            if features: processed_data.append((features, label))
        
        end_time = time.time()  # Eksik olan kısım eklendi
        elapsed = end_time - start_time # Eksik olan kısım eklendi
        print("----------------------------------------------------")
        print(f"[INFO] Processing finished in {elapsed:.2f} seconds.") # Eksik olan kısım eklendi

        if processed_data:
            total_len = len(processed_data[0][0])
            # Fixed: Color(3)+GLCM(6)+Wavelet(8)+Hermite(6) = 23
            lcp_len = LCP_BINS
            lbp_len = total_len - 23 - lcp_len
            write_arff(OUTPUT_ARFF_FILE, processed_data, CLASS_FOLDERS, lbp_len, lcp_len)