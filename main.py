import os
import glob
import time

# ----------------- AYARLAR / SETTINGS -----------------

# Veri setinin bulunduğu ana klasör
# The main folder containing the dataset
DATASET_DIR = "DATASET" 

# Sınıf etiketleri olarak kullanılacak alt klasörlerin adları
# Subfolder names that will be used as class labels
CLASS_FOLDERS = ["cats", "dogs", "snakes"]

# Hangi tür resim dosyalarını arayacağız?
# Which image file extensions will we search for?
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# ----------------------------------------------------


def find_image_paths(base_dir):
    """
    Tüm sınıf klasörlerini tarar ve her bir görüntünün
    dosya yolunu ve sınıf etiketini bulur.
    
    Scans all class folders and finds the file path
    and class label for each image.
    """
    print(f"Veri seti taranıyor... ({base_dir}) / Scanning dataset... ({base_dir})")
    start_time = time.time()
    
    # (resim_yolu, sinif_etiketi) şeklinde depolayacağız
    # We will store data as (image_path, class_label) tuples
    all_image_paths = []
    total_images = 0

    if not os.path.isdir(base_dir):
        print(f"HATA: Ana veri seti klasörü '{base_dir}' bulunamadı.")
        print(f"ERROR: Main dataset folder '{base_dir}' not found.")
        return []

    # Her bir sınıf klasörü için (cats, dogs, snakes)
    # For each class folder (cats, dogs, snakes)
    for class_label in CLASS_FOLDERS:
        # Yolu birleştir: "DATASET/cats"
        # Join the path: "DATASET/cats"
        class_path = os.path.join(base_dir, class_label) 
        
        if not os.path.isdir(class_path):
            print(f"HATA: '{class_path}' klasörü bulunamadı. Lütfen kontrol et.")
            print(f"ERROR: '{class_path}' folder not found. Please check.")
            continue

        print(f"--- '{class_label}' sınıfı taranıyor... / Scanning class '{class_label}'...")
        images_in_class = 0
        
        # Tüm resim uzantılarını (jpg, png vb.) tara
        # Scan all image extensions (jpg, png, etc.)
        for ext in IMAGE_EXTENSIONS:
            # glob, joker karakterleri kullanarak dosya bulur (örn: DATASET/cats/*.jpg)
            # glob finds files using wildcards (e.g., DATASET/cats/*.jpg)
            search_path = os.path.join(class_path, ext)
            found_images = glob.glob(search_path)
            
            for img_path in found_images:
                all_image_paths.append((img_path, class_label))
                images_in_class += 1

        print(f"'{class_label}' sınıfında {images_in_class} resim bulundu.")
        print(f"Found {images_in_class} images in class '{class_label}'.")
        total_images += images_in_class

    end_time = time.time()
    print(f"\nTarama tamamlandı. Toplam {total_images} resim bulundu.")
    print(f"Scan complete. Found a total of {total_images} images.")
    print(f"Süre / Duration: {end_time - start_time:.2f} saniye / seconds")
    
    return all_image_paths

# --- Ana Program / Main Program ---
if __name__ == "__main__":
    # Fonksiyonu çalıştır ve ana veri seti klasörünü parametre olarak ver
    # Run the function and pass the main dataset folder as a parameter
    image_list = find_image_paths(DATASET_DIR)
    
    if image_list:
        print("\nBulunan ilk 5 resim örneği: / Example of first 5 images found:")
        for path, label in image_list[:5]:
            print(f"Dosya Yolu / Path: {path}, Sınıf / Class: {label}")
            
        print("...")
        print("Bulunan son 5 resim örneği: / Example of last 5 images found:")
        for path, label in image_list[-5:]:
            print(f"Dosya Yolu / Path: {path}, Sınıf / Class: {label}")