import os
import pandas as pd
import sys
import time

# ----------------- FUSION SETTINGS / BİRLEŞTİRME AYARLARI -----------------

# Birleştirilecek dosyalar ve ön ekleri (Prefix)
FILES = [
    {"name": "features_step1_glcm_lbp.arff", "prefix": "S1"},
    {"name": "features_step2_lcp.arff", "prefix": "S2"},
    {"name": "features_step3_wavelet.arff", "prefix": "S3"},
    {"name": "features_step4_hermite.arff", "prefix": "S4"},
    {"name": "features_step5(final)_Fourier.arff", "prefix": "S5"}
]

OUTPUT_FILE = "final_fusion_model.arff"

# -------------------------------------------------------------------------

def load_arff_data(file_path):
    """
    ARFF dosyasını okur, @data kısmını DataFrame'e çevirir.
    Attribute isimlerini parse eder.
    """
    data_lines = []
    attributes = []
    data_started = False
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in lines:
            line = line.strip()
            if not line: continue
            
            # Attribute isimlerini al
            if line.lower().startswith("@attribute"):
                parts = line.split()
                if len(parts) >= 2:
                    attr_name = parts[1].strip()
                    attributes.append(attr_name)
            
            # Data başlangıcını yakala
            if line.lower().startswith("@data"):
                data_started = True
                continue
            
            # Veriyi al (yorum satırlarını atla)
            if data_started and not line.startswith("%"):
                # Satırdaki verileri virgülle ayır ve temizle
                row_data = [x.strip().replace("'", "").replace('"', "") for x in line.split(',')]
                data_lines.append(row_data)
                
        df = pd.DataFrame(data_lines, columns=attributes)
        return df
        
    except Exception as e:
        print(f"[ERROR] Dosya okunurken hata oluştu: {file_path}")
        print(f"[ERROR] Detay: {e}")
        sys.exit(1)

def main():
    start_time = time.time()
    
    print("----------------------------------------------------")
    print("[INFO] DIGITAL IMAGE PROCESSING: FEATURE FUSION SYSTEM")
    print(f"[INFO] Target Output: {OUTPUT_FILE}")
    print("----------------------------------------------------")

    combined_df = pd.DataFrame()
    final_labels = []
    total_files = len(FILES)

    # --- ADIM 1: DOSYALARI OKUMA VE BİRLEŞTİRME ---
    print(f"[INFO] Starting fusion process for {total_files} modules...")
    
    for i, file_info in enumerate(FILES):
        file_name = file_info["name"]
        prefix = file_info["prefix"]
        
        if not os.path.exists(file_name):
            print(f"[ERROR] File not found: {file_name}")
            print("[HINT] Run previous steps to generate feature files.")
            return

        print(f"[PROGRESS] Processing {i+1}/{total_files}: {file_name}...")
        
        # Veriyi yükle
        df = load_arff_data(file_name)
        
        # Label sütununu bul (Genelde 'class' veya en son sütun)
        label_col = df.columns[-1]
        
        # İlk dosya ise etiketleri sakla
        if i == 0:
            final_labels = df[label_col].values
            print(f"   -> Class Labels Detected: {len(final_labels)} instances.")
        
        # Özellikleri ayır (Label hariç)
        features_df = df.drop(columns=[label_col])
        
        # Sütun isimlerine Prefix ekle (Çakışmayı önlemek için: S1_contrast, S2_hist...)
        features_df.columns = [f"{prefix}_{col}" for col in features_df.columns]
        
        # Ana tabloya ekle
        if combined_df.empty:
            combined_df = features_df
        else:
            if len(features_df) != len(combined_df):
                print(f"[ERROR] Row mismatch! {file_name} has {len(features_df)} rows, expected {len(combined_df)}.")
                return
            combined_df = pd.concat([combined_df, features_df], axis=1)
            
        print(f"   -> Merged: {len(features_df.columns)} new features added.")

    # --- ADIM 2: DATA MATRIX HAZIRLIĞI ---
    print("----------------------------------------------------")
    print("[INFO] Finalizing Data Matrix...")
    
    # Label sütununu en sona ekle
    combined_df['class'] = final_labels

    # Sınıf isimlerini benzersiz olarak bul (J48 için gerekli)
    unique_classes = sorted(list(set(final_labels)))
    class_str = ",".join(unique_classes)
    
    print(f"[INFO] Classes: {{ {class_str} }}")
    print(f"[INFO] Total Features: {len(combined_df.columns) - 1}")
    print(f"[INFO] Total Instances: {len(combined_df)}")

    # --- MATRIX PREVIEW (GÖRSELLİK İÇİN) ---
    print("\n[INFO] --- FUSED DATA MATRIX PREVIEW (First 5 Rows) ---")
    # Pandas ayarları: Sütunları kesmesin, geniş göstersin
    pd.set_option('display.max_columns', 10) 
    pd.set_option('display.width', 1000)
    print(combined_df.head())
    print("----------------------------------------------------\n")

    # --- ADIM 3: ARFF DOSYASINI YAZMA ---
    print(f"[INFO] Writing output to '{OUTPUT_FILE}'...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"@relation 'Fusion_Model_All_Steps_Combined'\n\n")
        
        # Numeric özellikleri yaz
        for col in combined_df.columns:
            if col != 'class':
                f.write(f"@attribute {col} numeric\n")
        
        # Class attribute'unu NOMINAL formatta yaz (J48 fix)
        f.write(f"@attribute class {{{class_str}}}\n\n")
        
        f.write("@data\n")
        
        # Pandas CSV olarak yaz (Hata veren kısım düzeltildi: line_terminator -> lineterminator)
        combined_df.to_csv(f, header=False, index=False, lineterminator='\n')

    elapsed = time.time() - start_time
    print(f"[SUCCESS] Fusion completed successfully in {elapsed:.2f} seconds.")
    print(f"[SUCCESS] File saved: {os.path.abspath(OUTPUT_FILE)}")
    print("[HINT] Now open Weka -> Preprocess -> Open File -> Filter:Normalize -> Classify:J48")

if __name__ == "__main__":
    main()