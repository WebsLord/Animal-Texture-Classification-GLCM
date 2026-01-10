import os
import glob
import time
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern

# ----------------- PROJECT SETTINGS / PROJE AYARLARI -----------------

# Directory settings
# Dizin ayarları
DATASET_DIR = "DATASET" 
CLASS_FOLDERS = ["cats", "dogs", "snakes"]
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# Output ARFF file for Step 2 (includes LCP)
# Adım 2 için çıktı ARFF dosyası (LCP içerir)
OUTPUT_ARFF_FILE = "features_step2_lcp.arff" 

# --- GLCM CONFIGURATION / GLCM YAPILANDIRMASI ---
# Final project requires averaging features from directions 0, 45, 90, 135
# Final projesi 0, 45, 90, 135 yönlerinden gelen özelliklerin ortalamasını gerektirir
ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# --- LBP CONFIGURATION / LBP YAPILANDIRMASI ---
LBP_POINTS = 24 
LBP_RADIUS = 3 
LBP_METHOD = 'uniform'

# --- LCP CONFIGURATION / LCP YAPILANDIRMASI ---
# Number of bins for Local Contrast histogram
# Yerel Kontrast histogramı için kutu sayısı
LCP_BINS = 32

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
    """
    Calculates the mean of Blue, Green, and Red channels.
    Mavi, Yeşil ve Kırmızı kanalların ortalamasını hesaplar.
    """
    try:
        means = cv2.mean(img_bgr)[:3]
        return list(means)
    except Exception as e:
        print(f"[ERROR] Color extraction error: {e}")
        return None

def extract_glcm_features_averaged(img_gray):
    """
    Computes GLCM features and averages them across all 4 directions.
    GLCM özelliklerini hesaplar ve 4 yönün ortalamasını alır.
    """
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
    """
    Computes Local Binary Pattern (LBP) histogram.
    Yerel İkili Desen (LBP) histogramını hesaplar.
    """
    try:
        lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        return list(hist)
    except Exception as e:
        print(f"[ERROR] LBP extraction error: {e}")
        return None

def extract_lcp_features(img_gray):
    """
    Computes Local Contrast Pattern (LCP) features using a custom implementation.
    Calculates the magnitude of difference between center pixel and neighbors.
    
    Özel bir uygulama kullanarak Yerel Kontrast Deseni (LCP) özelliklerini hesaplar.
    Merkez piksel ve komşular arasındaki farkın büyüklüğünü hesaplar.
    """
    try:
        # Create an empty image for LCP output
        # LCP çıktısı için boş bir görüntü oluştur
        rows, cols = img_gray.shape
        lcp_image = np.zeros((rows, cols), dtype=np.float32)
        
        # Convert to float to avoid overflow/underflow during subtraction
        # Çıkarma sırasında taşmayı önlemek için float'a dönüştür
        img_float = img_gray.astype(np.float32)

        # Define 8 neighbors relative positions (3x3 window)
        # 8 komşu göreli konumunu tanımla (3x3 pencere)
        # (dy, dx)
        neighbors = [(-1, -1), (-1, 0), (-1, 1),
                     (0, -1),           (0, 1),
                     (1, -1),  (1, 0),  (1, 1)]

        # Calculate Local Contrast: Sum of absolute differences with neighbors
        # Yerel Kontrastı Hesapla: Komşularla mutlak farkların toplamı
        for dy, dx in neighbors:
            # Shift image to align neighbor with center using np.roll (faster than loops)
            # np.roll kullanarak komşuyu merkezle hizalamak için görüntüyü kaydır (döngülerden daha hızlı)
            shifted = np.roll(img_float, shift=(-dy, -dx), axis=(0, 1))
            
            # Calculate absolute difference
            # Mutlak farkı hesapla
            diff = np.abs(img_float - shifted)
            lcp_image += diff
            
        # Normalize LCP image to 0-255 range for histogram calculation
        # Histogram hesaplaması için LCP görüntüsünü 0-255 aralığına normalleştir
        lcp_norm = cv2.normalize(lcp_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Calculate Histogram of LCP (Texture Contrast Distribution)
        # LCP Histogramını Hesapla (Doku Kontrast Dağılımı)
        hist, _ = np.histogram(lcp_norm.ravel(), bins=LCP_BINS, range=(0, 256), density=True)
        
        return list(hist)

    except Exception as e:
        print(f"[ERROR] LCP extraction error: {e}")
        return None

def process_image(image_path):
    """
    Main processing function for a single image. Coordinates all feature extractors.
    Tek bir resim için ana işleme fonksiyonu. Tüm özellik çıkarıcıları koordine eder.
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- FEATURE EXTRACTION PIPELINE / ÖZNİTELİK ÇIKARMA HATTI ---
    
    # 1. Color (3 features) - Basic
    f_color = extract_color_features(img_bgr)
    
    # 2. GLCM (6 features) - Averaged Directions
    f_glcm = extract_glcm_features_averaged(img_gray)
    
    # 3. LBP (Histogram features) - Texture Pattern
    f_lbp = extract_lbp_features(img_gray)
    
    # 4. LCP (Histogram features) - Texture Contrast (BONUS)
    f_lcp = extract_lcp_features(img_gray)

    # Combine all features into a single vector
    # Tüm özellikleri tek bir vektörde birleştir
    if f_color and f_glcm and f_lbp and f_lcp:
        return f_color + f_glcm + f_lbp + f_lcp
    
    return None

def write_arff(filename, data, class_labels, lbp_len, lcp_len):
    """
    Writes extracted data to Weka ARFF format.
    Çıkarılan veriyi Weka ARFF formatına yazar.
    """
    print(f"\n[INFO] Writing output to '{filename}'...")
    
    header = []
    header.append(f"@RELATION animal_classification_step2_lcp")
    header.append("")

    # -- Attribute Definitions / Nitelik Tanımları --
    
    # Color Attributes
    header.append("@ATTRIBUTE Color_Mean_Blue NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Green NUMERIC")
    header.append("@ATTRIBUTE Color_Mean_Red NUMERIC")
    
    # GLCM Attributes
    for prop in GLCM_PROPERTIES:
        header.append(f"@ATTRIBUTE GLCM_Avg_{prop} NUMERIC")
        
    # LBP Attributes
    for i in range(lbp_len):
        header.append(f"@ATTRIBUTE LBP_Hist_{i} NUMERIC")

    # LCP Attributes (New)
    for i in range(lcp_len):
        header.append(f"@ATTRIBUTE LCP_Hist_{i} NUMERIC")

    # Class Attribute
    class_str = "{" + ",".join(class_labels) + "}"
    header.append(f"@ATTRIBUTE class {class_str}")
    header.append("")
    header.append("@DATA")
    
    # Write Data Rows
    # Veri Satırlarını Yaz
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

# --- MAIN EXECUTION / ANA YÜRÜTME ---
if __name__ == "__main__":
    
    images = find_image_paths(DATASET_DIR)
    processed_data = []
    
    if images:
        print("----------------------------------------------------")
        print(f"[INFO] Processing {len(images)} images...")
        print(f"[INFO] Features: Color + GLCM (Avg) + LBP + LCP (Bonus)")
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
            # Dynamic calculation of attribute lengths
            # Nitelik uzunluklarının dinamik hesaplanması
            total_len = len(processed_data[0][0])
            
            # Known fixed lengths: Color(3) + GLCM(6) = 9
            # LCP length is fixed by LCP_BINS = 32
            # Remaining is LBP
            lcp_len = LCP_BINS
            lbp_len = total_len - 3 - 6 - lcp_len
            
            write_arff(OUTPUT_ARFF_FILE, processed_data, CLASS_FOLDERS, lbp_len, lcp_len)
        else:
            print("[ERROR] No features extracted.")
    else:
        print("[ERROR] No images found. Check DATASET directory.")