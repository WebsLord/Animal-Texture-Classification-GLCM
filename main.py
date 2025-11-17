import os
import glob
import time
import cv2  # OpenCV kütüphanesi
import numpy as np
from skimage.feature import graycomatrix, graycoprops # scikit-image
# 'redirect_stdout' modülünü kaldırdık

# ----------------- AYARLAR / SETTINGS -----------------

DATASET_DIR = "DATASET" 
CLASS_FOLDERS = ["cats", "dogs", "snakes"]
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]
OUTPUT_ARFF_FILE = "features.arff" # Çıktı dosyasının adı

# --- GLCM AYARLARI / GLCM SETTINGS ---
ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4] # Radyan cinsinden yönler
ANGLES_DEG = ['0', '45', '90', '135']       # ARFF başlığı için derece adları
PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# ----------------------------------------------------

def find_image_paths(base_dir):
    """
    (Adım 3'ten - Değişmedi)
    (From Step 3 - Unchanged)
    """
    print(f"Veri seti taranıyor... ({base_dir}) / Scanning dataset... ({base_dir})")
    all_image_paths = []
    total_images = 0

    if not os.path.isdir(base_dir):
        print(f"HATA: Ana veri seti klasörü '{base_dir}' bulunamadı.")
        print(f"ERROR: Main dataset folder '{base_dir}' not found.")
        return []

    for class_label in CLASS_FOLDERS:
        class_path = os.path.join(base_dir, class_label) 
        if not os.path.isdir(class_path):
            print(f"HATA: '{class_path}' klasörü bulunamadı. Lütfen kontrol et.")
            print(f"ERROR: '{class_path}' folder not found. Please check.")
            continue
        
        print(f"--- '{class_label}' sınıfı taranıyor... / Scanning class '{class_label}'...")
        images_in_class = 0
        for ext in IMAGE_EXTENSIONS:
            search_path = os.path.join(class_path, ext)
            found_images = glob.glob(search_path)
            for img_path in found_images:
                all_image_paths.append((img_path, class_label))
                images_in_class += 1
        print(f"'{class_label}' sınıfında {images_in_class} resim bulundu.")
        print(f"Found {images_in_class} images in class '{class_label}'.")
        total_images += images_in_class

    print(f"\nTarama tamamlandı. Toplam {total_images} resim bulundu.")
    print(f"Scan complete. Found a total of {total_images} images.\n")
    return all_image_paths


def extract_features_from_image(image_path):
    """
    (Adım 4'ten - Değişmedi)
    (From Step 4 - Unchanged)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            # print(f"Uyarı: '{image_path}' okunamadı. Atlanıyor.")
            return None
        
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_img, distances=[1], angles=ANGLES_RAD, levels=256,
                            symmetric=True, normed=True)
        
        image_features = []
        for prop in PROPERTIES:
            props_for_all_angles = graycoprops(glcm, prop)
            image_features.extend(props_for_all_angles[0])

        return image_features
        
    except Exception as e:
        # print(f"HATA: '{image_path}' işlenirken hata: {e}")
        return None

def write_arff_file(filename, data, class_labels):
    """
    (Adım 5'ten - Değişmedi)
    (From Step 5 - Unchanged)
    """
    print(f"\n----------------------------------------------------")
    print(f"'{filename}' dosyası oluşturuluyor...")
    print(f"Creating '{filename}' file...")
    
    header = []
    header.append("@RELATION animal_texture_features")
    header.append("")

    for prop in PROPERTIES:
        for angle in ANGLES_DEG:
            attribute_name = f"{prop}_{angle}"
            header.append(f"@ATTRIBUTE {attribute_name} NUMERIC")
            
    class_string = "{" + ",".join(class_labels) + "}"
    header.append(f"@ATTRIBUTE class {class_string}")
    header.append("")

    header.append("@DATA")
    
    data_lines = []
    for (features, label) in data:
        feature_string = ",".join(map(str, features))
        data_lines.append(f"{feature_string},{label}")

    try:
        with open(filename, "w") as f:
            f.write("\n".join(header))
            f.write("\n")
            f.write("\n".join(data_lines))
        
        print(f"'{filename}' dosyası {len(data_lines)} veri satırıyla başarıyla oluşturuldu!")
        print(f"'{filename}' file created successfully with {len(data_lines)} data rows!")
        print(f"----------------------------------------------------")
    except Exception as e:
        print(f"HATA: '{filename}' dosyası yazılırken bir hata oluştu: {e}")
        print(f"ERROR: An error occurred while writing '{filename}': {e}")


# --- Ana Program / Main Program ---
if __name__ == "__main__":
    
    # Adım 1: Tüm resim yollarını bul
    # Step 1: Find all image paths
    image_list = find_image_paths(DATASET_DIR)
    
    all_processed_data = [] # [ (features_list, label), ... ]

    if image_list:
        print("----------------------------------------------------")
        print(f"Öznitelik çıkarma işlemi başlıyor ({len(image_list)} resim)...")
        print("Feature extraction is starting...")
        print("Bu işlem ~{60-70} saniye sürecek. / This will take ~{60-70} seconds.")
        print("----------------------------------------------------")
        
        start_proc_time = time.time()
        
        # 'with' bloğunu kaldırdık
        
        # Adım 2, 3, 4: Her bir resmi işle
        # Step 2, 3, 4: Process each image
        for i, (path, label) in enumerate(image_list):
            
            # (YENİDEN EKLENDİ) Her 100 resimde bir ilerleme durumu bildir
            # (RE-ADDED) Report progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"İşleniyor... {i + 1} / {len(image_list)}")

            features = extract_features_from_image(path)
            if features:
                all_processed_data.append((features, label))

        # 3000. resim için son bir bildirim (eğer 100'e tam bölünmüyorsa)
        # Final report for the 3000th image (in case it's not a multiple of 100)
        if (i + 1) % 100 != 0:
             print(f"İşleniyor... {i + 1} / {len(image_list)}")

        end_proc_time = time.time()
        print("----------------------------------------------------")
        print(f"Tüm öznitelikler {end_proc_time - start_proc_time:.2f} saniyede çıkarıldı.")
        print(f"All features extracted in {end_proc_time - start_proc_time:.2f} seconds.")

        # Adım 5: ARFF Dosyasını Yaz
        # Step 5: Write the ARFF File
        if all_processed_data:
            write_arff_file(OUTPUT_ARFF_FILE, all_processed_data, CLASS_FOLDERS)
        else:
            print("HATA: Hiçbir resimden öznitelik çıkarılamadı.")
            print("ERROR: No features could be extracted from any image.")