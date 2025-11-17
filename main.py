import os
import glob
import time
import cv2  # OpenCV kütüphanesi (pip install opencv-python)
import numpy as np
from skimage.feature import graycomatrix, graycoprops # (pip install scikit-image)

# ----------------- AYARLAR / SETTINGS -----------------

DATASET_DIR = "DATASET" 
CLASS_FOLDERS = ["cats", "dogs", "snakes"]
IMAGE_EXTENSIONS = ["*.jpg", "*.jpeg", "*.png"]

# --- GLCM AYARLARI / GLCM SETTINGS ---
# Örnek yönler (0, 45, 90, 135 derece)
# Example Angles (0, 45, 90, 135 degrees)
ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]

# Örnek 6 öznitelik
# The 6 properties example
PROPERTIES = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']

# ----------------------------------------------------

def find_image_paths(base_dir):
    """
    (Bu fonksiyon Adım 3'teki ile aynı, değişmedi)
    (This function is the same as in Step 3, unchanged)
    
    Tüm sınıf klasörlerini tarar ve her bir görüntünün
    dosya yolunu ve sınıf etiketini bulur.
    
    Scans all class folders and finds the file path
    and class label for each image.
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
    Tek bir görüntüyü alır, griye çevirir ve 24 GLCM özniteliğini çıkarır.
    Takes a single image, converts it to gray, and extracts 24 GLCM features.
    """
    try:
        # Görüntüyü oku / Read the image
        img = cv2.imread(image_path)
        
        # Görüntü okunamadıysa (bozuk dosya vb.) / If image can't be read (corrupt file etc.)
        if img is None:
            print(f"Uyarı: '{image_path}' okunamadı veya bozuk. Atlanıyor.")
            print(f"Warning: Could not read '{image_path}'. Skipping.")
            return None

        # Adım 2: Ön İşleme (Gri Seviyeye Dönüştürme)
        # Step 2: Preprocessing (Convert to Grayscale)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adım 3: GLCM Matrisi Oluşturma
        # Step 3: Construct GLCM Matrix
        # distance = [1] (yakın komşuluk)
        # levels=256 (0-255 gri seviyeler)
        glcm = graycomatrix(gray_img, distances=[1], angles=ANGLES, levels=256,
                            symmetric=True, normed=True)

        # Adım 4: Öznitelik Çıkarımı
        # Step 4: Feature Extraction
        
        # Her bir görüntü için 24 öznitelik (6 özellik x 4 yön) depolanacak
        # 24 features (6 props x 4 angles) will be stored for each image
        image_features = []

        for prop in PROPERTIES:
            # graycoprops, verdiğimiz 4 yön (ANGLES) için de hesap yapar
            # graycoprops calculates features for all 4 ANGLES we provided
            # Sonuç (1, 4) şeklinde bir matris olur (1 distance, 4 angle)
            # The result is a (1, 4) matrix
            props_for_all_angles = graycoprops(glcm, prop)
            
            # (1, 4) matrisini düz bir listeye (örn: [0.1, 0.2, 0.3, 0.4]) çevirip ekliyoruz
            # We flatten the (1, 4) matrix and append it to our list
            image_features.extend(props_for_all_angles[0])

        # image_features şimdi 24 elemanlı bir liste
        # image_features is now a list with 24 elements
        return image_features
        
    except Exception as e:
        print(f"HATA: '{image_path}' işlenirken bir hata oluştu: {e}")
        print(f"ERROR: An error occurred while processing '{image_path}': {e}")
        return None

# --- Ana Program / Main Program ---
if __name__ == "__main__":
    
    # 1. Aşama: Tüm resim yollarını bul
    # Phase 1: Find all image paths
    image_list = find_image_paths(DATASET_DIR)
    
    # .arff dosyasına yazacağımız tüm verileri tutan liste
    # List to hold all data we will write to the .arff file
    all_processed_data = [] # [ (features_list, label), (features_list, label), ... ]

    if image_list:
        print("----------------------------------------------------")
        print(f"Öznitelik çıkarma işlemi başlıyor ({len(image_list)} resim)...")
        print("Feature extraction is starting...")
        print("Bu işlem veri setinin büyüklüğüne göre birkaç dakika sürebilir.")
        print("This may take several minutes depending on dataset size.")
        print("----------------------------------------------------")
        
        start_proc_time = time.time()

        # 2. Aşama: Her bir resmi işle
        # Phase 2: Process each image
        for i, (path, label) in enumerate(image_list):
            
            # Her 100 resimde bir ilerleme durumu bildir
            # Report progress every 100 images
            if (i + 1) % 100 == 0:
                print(f"İşleniyor... {i + 1} / {len(image_list)}")

            # Resimden öznitelikleri çıkar
            # Extract features from the image
            features = extract_features_from_image(path)
            
            # Eğer resim bozuk değilse ve öznitelikler başarıyla çıkarıldıysa
            # If the image was not corrupt and features were extracted
            if features:
                # Öznitelik listesini ve etiketini birlikte kaydet
                # Save the feature list and its label together
                all_processed_data.append((features, label))

        end_proc_time = time.time()
        print("----------------------------------------------------")
        print("Tüm öznitelikler başarıyla çıkarıldı!")
        print("All features extracted successfully!")
        print(f"Toplam süre: {end_proc_time - start_proc_time:.2f} saniye / seconds")

        # Kanıt olarak ilk resmin verilerine bakalım
        # Let's check the data from the first image as proof
        if all_processed_data:
            first_image_data = all_processed_data[0]
            print("\nİlk resmin öznitelik örneği (Toplam 24 adet):")
            print(f"Example features from first image (24 total):")
            print(f"Etiket / Label: {first_image_data[1]}")
            print(f"Öznitelikler / Features: {first_image_data[0]}")
            print(f"Öznitelik Sayısı / Feature Count: {len(first_image_data[0])}")