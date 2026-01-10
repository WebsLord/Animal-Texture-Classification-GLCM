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

# Output ARFF file for Step 1
# Adım 1 için çıktı ARFF dosyası
OUTPUT_ARFF_FILE = "features_step1_glcm_lbp.arff" 

# --- GLCM CONFIGURATION / GLCM YAPILANDIRMASI ---
# Angles: 0, 45, 90, 135 degrees in radians
# Açılar: Radyan cinsinden 0, 45, 90, 135 derece
ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
# Properties to extract
# Çıkarılacak özellikler
GLCM_PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# --- LBP CONFIGURATION / LBP YAPILANDIRMASI ---
# Number of circularly symmetric neighbor set points
# Dairesel simetrik komşu nokta sayısı
LBP_POINTS = 24 
# Radius of circle (Spatial resolution)
# Dairenin yarıçapı (Uzamsal çözünürlük)
LBP_RADIUS = 3 
# Method: 'uniform' improves rotation invariance
# Yöntem: 'uniform' döndürme değişmezliğini artırır
LBP_METHOD = 'uniform'

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
            # Construct search pattern (e.g., DATASET/cats/*.jpg)
            # Arama desenini oluştur (örn. DATASET/cats/*.jpg)
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
        # Calculate mean for each channel (B, G, R)
        # Her kanal için ortalamayı hesapla (B, G, R)
        means = cv2.mean(img_bgr)[:3]
        return list(means)
    except Exception as e:
        print(f"[ERROR] Color extraction error: {e}")
        return None

def extract_glcm_features_averaged(img_gray):
    """
    Computes GLCM features and averages them across all 4 directions (0, 45, 90, 135).
    GLCM özelliklerini hesaplar ve 4 yönün (0, 45, 90, 135) ortalamasını alır.
    
     Requirement from Final PDF: "Obtain a single feature vector by taking the average..."
     Final PDF Gereksinimi: "...farklı yönlerden çıkarılan özelliklerin ortalamasını alarak..."
    """
    try:
        # 1. Calculate GLCM matrix
        # 1. GLCM matrisini hesapla
        glcm = graycomatrix(img_gray, distances=[1], angles=ANGLES_RAD, levels=256,
                            symmetric=True, normed=True)
        
        feature_vector = []
        
        # 2. Extract properties and average them
        # 2. Özellikleri çıkar ve ortalamasını al
        for prop in GLCM_PROPERTIES:
            # graycoprops returns values for each angle. We use .mean() to average them.
            # graycoprops her açı için değer döndürür. Ortalamasını almak için .mean() kullanıyoruz.
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
        # 1. Compute LBP image
        # 1. LBP görüntüsünü hesapla
        lbp = local_binary_pattern(img_gray, LBP_POINTS, LBP_RADIUS, LBP_METHOD)
        
        # 2. Compute Histogram of LBP
        # 2. LBP Histogramını hesapla
        # For 'uniform' LBP, the number of bins is P + 2
        # 'uniform' LBP için kutu (bin) sayısı P + 2'dir
        n_bins = int(lbp.max() + 1)
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        
        return list(hist)
    except Exception as e:
        print(f"[ERROR] LBP extraction error: {e}")
        return None

def process_image(image_path):
    """
    Main processing function for a single image.
    Tek bir resim için ana işleme fonksiyonu.
    """
    # Read image
    # Resmi oku
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        return None

    # Convert to Grayscale
    # Gri seviyeye dönüştür
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- FEATURE EXTRACTION / ÖZNİTELİK ÇIKARIMI ---
    
    # 1. Color Features (3 features)
    # 1. Renk Özellikleri (3 özellik)
    f_color = extract_color_features(img_bgr)
    
    # 2. GLCM Features (Averaged) (6 features)
    # 2. GLCM Özellikleri (Ortalaması alınmış) (6 özellik)
    f_glcm = extract_glcm_features_averaged(img_gray)
    
    # 3. LBP Features (Histogram) (26 features for P=24)
    # 3. LBP Özellikleri (Histogram) (P=24 için 26 özellik)
    f_lbp = extract_lbp_features(img_gray)

    # Combine all features
    # Tüm özellikleri birleştir
    if f_color and f_glcm and f_lbp:
        return f_color + f_glcm + f_lbp
    
    return None

def write_arff(filename, data, class_labels, lbp_len):
    """
    Writes data to Weka ARFF format.
    Veriyi Weka ARFF formatına yazar.
    """
    print(f"\n[INFO] Writing output to '{filename}'...")
    
    header = []
    header.append(f"@RELATION animal_classification_step1")
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

    # Class Attribute
    class_str = "{" + ",".join(class_labels) + "}"
    header.append(f"@ATTRIBUTE class {class_str}")
    header.append("")
    header.append("@DATA")
    
    # -- Data Rows / Veri Satırları --
    data_lines = []
    for features, label in data:
        # Convert list to comma-separated string
        # Listeyi virgülle ayrılmış dizeye dönüştür
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
    
    # 1. Find Images
    # 1. Resimleri Bul
    images = find_image_paths(DATASET_DIR)
    
    processed_data = []
    
    if images:
        print("----------------------------------------------------")
        print(f"[INFO] Processing {len(images)} images...")
        print(f"[INFO] Features: Color(3) + GLCM_Avg(6) + LBP(Histogram)")
        print("----------------------------------------------------")
        
        start_time = time.time()
        
        # 2. Extract Features Loop
        # 2. Öznitelik Çıkarma Döngüsü
        for i, (path, label) in enumerate(images):
            # Progress log every 100 images
            # Her 100 resimde bir ilerleme günlüğü
            if (i + 1) % 100 == 0:
                print(f"[PROGRESS] Processing image {i + 1}/{len(images)}...")
            
            features = process_image(path)
            if features:
                processed_data.append((features, label))
        
        end_time = time.time()
        elapsed = end_time - start_time
        print("----------------------------------------------------")
        print(f"[INFO] Processing finished in {elapsed:.2f} seconds.")
        
        # 3. Write to ARFF
        # 3. ARFF'ye Yaz
        if processed_data:
            # Determine LBP feature length from the first record to set header correctly
            # Başlığı doğru ayarlamak için ilk kayıttan LBP özellik uzunluğunu belirle
            # Total len - 3 (Color) - 6 (GLCM) = LBP Length
            lbp_length = len(processed_data[0][0]) - 3 - 6
            
            write_arff(OUTPUT_ARFF_FILE, processed_data, CLASS_FOLDERS, lbp_length)
        else:
            print("[ERROR] No features extracted. Check dataset.")
    else:
        print("[ERROR] No images found. Please check 'DATASET_DIR'.")