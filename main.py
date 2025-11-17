import os
import glob
import time
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# ----------------- AYARLAR / SETTINGS -----------------

DATASET_DIR = "DATASET" 
CLASS_FOLDERS = ["cats", "dogs", "snakes"]
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# (YENİ) Bonus özellikli .arff dosyamızın adı
# (NEW) Our .arff file with bonus features
OUTPUT_ARFF_FILE = "features_bonus.arff" 

# --- GLCM AYARLARI / GLCM SETTINGS ---
ANGLES_RAD = [0, np.pi/4, np.pi/2, 3*np.pi/4]
ANGLES_DEG = ['0', '45', '90', '135']
PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# ----------------------------------------------------

def find_image_paths(base_dir):
    """ (Adım 3'ten - Değişmedi) """
    print(f"Veri seti taranıyor... ({base_dir}) / Scanning dataset... ({base_dir})")
    all_image_paths = []
    total_images = 0
    if not os.path.isdir(base_dir):
        print(f"HATA: Ana veri seti klasörü '{base_dir}' bulunamadı.")
        return []
    for class_label in CLASS_FOLDERS:
        class_path = os.path.join(base_dir, class_label) 
        if not os.path.isdir(class_path):
            print(f"HATA: '{class_path}' klasörü bulunamadı. Lütfen kontrol et.")
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
    (GÜNCELLENDİ - Adım 4 Bonus)
    (UPDATED - Step 4 Bonus)
    Hem GLCM (Doku) hem de Renk özniteliklerini çıkarır.
    Extracts both GLCM (Texture) and Color features.
    """
    try:
        # Görüntüyü renkli olarak oku
        # Read the image in color
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # --- BONUS: Renk Öznitelikleri (3 Öznitelik) ---
        # (YENİ) Görüntünün ortalama Mavi, Yeşil ve Kırmızı değerlerini hesapla
        # (NEW) Calculate the mean Blue, Green, and Red values of the image
        # cv2 BGR sırasıyla okur (Mavi, Yeşil, Kırmızı)
        # cv2 reads in BGR order (Blue, Green, Red)
        (mean_B, mean_G, mean_R) = cv2.mean(img)[:3]
        color_features = [mean_B, mean_G, mean_R]
        # -----------------------------------------------

        # --- DOKU: GLCM Öznitelikleri (24 Öznitelik) ---
        # Gri seviyeye dönüştür
        # Convert to grayscale
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray_img, distances=[1], angles=ANGLES_RAD, levels=256,
                            symmetric=True, normed=True)
        
        texture_features = []
        for prop in PROPERTIES:
            props_for_all_angles = graycoprops(glcm, prop)
            texture_features.extend(props_for_all_angles[0])
        # -----------------------------------------------
        
        # (YENİ) İki listeyi birleştir (3 renk + 24 doku = 27 öznitelik)
        # (NEW) Combine both lists (3 color + 24 texture = 27 features)
        all_features = color_features + texture_features
        return all_features
        
    except Exception as e:
        # print(f"HATA: '{image_path}' işlenirken hata: {e}")
        return None


def write_arff_file(filename, data, class_labels):
    """
    (GÜNCELLENDİ - Adım 5 Bonus)
    (UPDATED - Step 5 Bonus)
    Yeni renk özniteliklerini de başlığa ekler.
    Adds the new color attributes to the header.
    """
    print(f"\n----------------------------------------------------")
    print(f"'{filename}' dosyası oluşturuluyor...")
    print(f"Creating '{filename}' file...")
    
    header = []
    header.append("@RELATION animal_texture_color_features") # İlişki adını güncelledik
    header.append("")

    # --- (YENİ) Bonus Renk Öznitelikleri ---
    header.append("@ATTRIBUTE color_mean_B NUMERIC")
    header.append("@ATTRIBUTE color_mean_G NUMERIC")
    header.append("@ATTRIBUTE color_mean_R NUMERIC")
    # -----------------------------------------

    # @ATTRIBUTE GLCM tanımları (24 tane)
    for prop in PROPERTIES:
        for angle in ANGLES_DEG:
            attribute_name = f"{prop}_{angle}"
            header.append(f"@ATTRIBUTE {attribute_name} NUMERIC")
            
    # Sınıf @ATTRIBUTE tanımı
    class_string = "{" + ",".join(class_labels) + "}"
    header.append(f"@ATTRIBUTE class {class_string}")
    header.append("")
    header.append("@DATA")
    
    data_lines = []
    for (features, label) in data:
        # features listesi artık 27 elemanlı
        # feature list now has 27 elements
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
    
    image_list = find_image_paths(DATASET_DIR)
    all_processed_data = [] 

    if image_list:
        print("----------------------------------------------------")
        print(f"Öznitelik çıkarma işlemi başlıyor (BONUS VERSİYON) ({len(image_list)} resim)...")
        print("Feature extraction is starting (BONUS VERSION)...")
        print("----------------------------------------------------")
        
        start_proc_time = time.time()
        
        for i, (path, label) in enumerate(image_list):
            if (i + 1) % 100 == 0:
                print(f"İşleniyor... {i + 1} / {len(image_list)}")
            
            # (YENİ) 27 özniteliği çıkaran fonksiyon
            # (NEW) Function extracting 27 features
            features = extract_features_from_image(path)
            if features:
                all_processed_data.append((features, label))

        if (i + 1) % 100 != 0:
             print(f"İşleniyor/Processing... {i + 1} / {len(image_list)}")
        
        end_proc_time = time.time()
        print("----------------------------------------------------")
        print(f"Tüm öznitelikler {end_proc_time - start_proc_time:.2f} saniyede çıkarıldı.")
        print(f"All features extracted in {end_proc_time - start_proc_time:.2f} seconds.")

        if all_processed_data:
            # (YENİ) Yeni dosya adına yaz
            # (NEW) Write to the new file name
            write_arff_file(OUTPUT_ARFF_FILE, all_processed_data, CLASS_FOLDERS)
        else:
            print("HATA: Hiçbir resimden öznitelik çıkarılamadı.")
            print("ERROR: No features could be extracted from any image.")