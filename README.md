# Animal-Texture-Classification-GLCM (TÜRKÇE AŞŞAĞIDA)

This repository contains a project for classifying animal images based on texture and color features using Python and Weka.

## 🚀 Project Goals

The main goals of the project:
* Read and preprocess an image dataset.
* Extract texture features using **GLCM (Gray Level Co-occurrence Matrix)**.
* Additionally, extract **color features** (Mean BGR).
* Save the extracted features into an `.arff` file for analysis in Weka.
* Test classification models (J48, RandomForest) in Weka.
* Compare performance before and after **Feature Selection**.
* Interpret the results using **Kappa statistic** and Accuracy metrics.

---

## 🛠️ Tools Used

* **Python 3**
* **OpenCV (`opencv-python`):** Image reading, grayscale conversion, and color analysis.
* **Scikit-image (`scikit-image`):** Calculating GLCM matrices and texture properties.
* **Numpy:** Scientific computing.
* **Weka 3.8:** Data mining and classification analysis.

---

## 📋 Methodology and Workflow

### 1. Feature Extraction with Python (`main.py`)

The `main.py` script generated two different `.arff` files:

**a) `features.arff` (Texture Only)**
* 6 GLCM features (Contrast, ASM, etc.) were extracted for 4 angles ($0^{\circ}, 45^{\circ}, 90^{\circ}, 135^{\circ}$) for each image.
* **Result:** **24 texture features** per image.

**b) `features_bonus.arff` (Texture + Color)**
* In addition to the 24 texture features, the mean **Blue, Green, and Red (BGR)** values were added as 3 new features.
* **Result:** **27 (24 Texture + 3 Color) features** per image.

### 2. Analysis in Weka and Findings

The following tests were performed in Weka Explorer using 10-fold Cross-Validation.

#### Experiment 1: Texture Only (GLCM)
* **Model:** J48 (Decision Tree)
* **Data:** `features.arff` (24 features)
* **Accuracy:** **~58.8%**

#### Experiment 2: Texture + Color

-------------------------------------------------------------------

# Animal-Texture-Classification-GLCM

Bu repository, Python ve Weka kullanarak hayvan görsellerini doku (texture) ve renk özniteliklerine göre sınıflandıran bir proje içerir.

## 🚀 Proje Hedefleri

Projenin ana hedefleri:
* Bir görüntü veri setini okuma ve ön işleme.
* **GLCM (Gri Seviye Eş-oluşum Matrisi)** kullanarak doku öznitelikleri çıkarma.
* Ek olarak **renk öznitelikleri** (Ortalama BGR) çıkarma.
* Çıkarılan öznitelikleri Weka'da (`.arff`) analize hazır hale getirme.
* Weka'da sınıflandırma modellerini (J48, RandomForest) test etme.
* **Öznitelik Seçimi (Feature Selection)** öncesi ve sonrası performansı karşılaştırma.
* Sonuçları **Kappa istatistiği** ve doğruluk (Accuracy) metrikleri ile yorumlama.

---

## 🛠️ Kullanılan Araçlar

* **Python 3**
* **OpenCV (`opencv-python`):** Görüntü okuma, gri seviyeye dönüştürme ve renk analizi.
* **Scikit-image (`scikit-image`):** GLCM matrisleri ve doku öznitelikleri hesaplama.
* **Numpy:** Bilimsel hesaplama.
* **Weka 3.8:** Veri madenciliği ve sınıflandırma analizi.

---

## 📋 Metodoloji ve İş Akışı

### 1. Python ile Öznitelik Çıkarımı (`main.py`)

`main.py` script'i iki farklı `.arff` dosyası üretmiştir:

**a) `features.arff` (Sadece Doku)**
* Her görüntü için 4 yönde ($0^{\circ}, 45^{\circ}, 90^{\circ}, 135^{\circ}$) 6 GLCM özniteliği (Contrast, ASM, vb.) çıkarıldı.
* **Sonuç:** Görüntü başına **24 doku özniteliği**.

**b) `features_bonus.arff` (Doku + Renk)**
* Yukarıdaki 24 doku özniteliğine ek olarak, her görüntünün ortalama **Mavi, Yeşil ve Kırmızı (BGR)** değerleri 3 yeni öznitelik olarak eklendi.
* **Sonuç:** Görüntü başına **27 (24 Doku + 3 Renk) öznitelik**.

### 2. Weka ile Analiz ve Bulgular

Weka Explorer'da 10-fold Cross-Validation (Çapraz Doğrulama) metodu ile aşağıdaki testler yapılmıştır.

#### Deney 1: Sadece Doku (GLCM)
* **Model:** J48 (Karar Ağacı)
* **Veri:** `features.arff` (24 öznitelik)
* **Doğruluk:** **~58.8%**

#### Deney 2: Doku + Renk (Bonus)
* **Model:** J48 (Karar Ağacı)
* **Veri:** `features_bonus.arff` (27 öznitelik)
* **Doğruluk:** **~60.3%**
* **Bulgu:** Basit "ortalama renk" özniteliklerini eklemek, performansta ~1.5%'lik mütevazı bir artış sağlamıştır.

#### Deney 3: Öznitelik Seçiminin Etkisi
* `CfsSubsetEval` algoritması, 27 öznitelikten en önemli **11** tanesini seçti.
* **Model:** J48 (Karar Ağacı)
* **Veri:** `features_bonus.arff` (Seçilmiş 11 öznitelik)
* **Doğruluk:** **~60.3%**
* **Bulgu:** Modelin, özniteliklerin %60'ı atıldıktan sonra bile **doğruluk kaybı yaşamaması**, öznitelik seçiminin ne kadar başarılı ve verimli olduğunu kanıtlamıştır.

#### Deney 4: Gelişmiş Algoritma Testi
* **Model:** RandomForest (100 ağaçlı)
* **Veri:** `features_bonus.arff` (Normalize edilmiş, 27 öznitelik)
* **Doğruluk:** **~63.5%**
* **Kappa İstatistiği:** **~0.45**
* **Bulgu:** Daha güçlü bir algoritma olan RandomForest, en yüksek başarı oranını vermiştir. 0.45'lik Kappa değeri, modelin rastgele tahminden (%33.3) çok daha iyi olduğunu ve "orta düzeyde" bir güvenilirliğe sahip olduğunu göstermektedir.

---

## 📈 Sonuç ve Yorum

Analizler, modelin **`snakes` (yılanlar)** sınıfını (farklı pul dokusu sayesinde) yüksek doğrulukla ayırabildiğini göstermiştir. Ancak, hem **`cats` (kedi)** hem de **`dogs` (köpek)** sınıfları benzer "tüylü" doku ve renklere sahip olduğu için, modelin bu iki sınıfı ayırt etmekte zorlandığı "Confusion Matrix" (Karışıklık Matrisi) üzerinde açıkça görülmüştür.