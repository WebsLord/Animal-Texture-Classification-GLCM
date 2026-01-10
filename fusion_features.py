import os
import pandas as pd

# Birleştirilecek dosyaların listesi (Senin dosya isimlerinle birebir aynı)
# NOT: Bu dosyaların bu script ile aynı klasörde olduğundan emin ol.
FILES = [
    {"name": "features_step1_glcm_lbp.arff", "prefix": "S1_GLCM_LBP"},
    {"name": "features_step2_lcp.arff", "prefix": "S2_LCP"},
    {"name": "features_step3_wavelet.arff", "prefix": "S3_Wavelet"},
    {"name": "features_step4_hermite.arff", "prefix": "S4_Hermite"},
    {"name": "features_step5(final)_Fourier.arff", "prefix": "S5_Fourier"}
]

OUTPUT_FILE = "final_fusion_model.arff"

def load_arff_data(file_path):
    """
    ARFF dosyasının sadece veri (@data sonrası) kısmını okur.
    Header kısmını manuel parse eder.
    """
    data_lines = []
    attributes = []
    data_started = False
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Attribute isimlerini yakala
        if line.lower().startswith("@attribute"):
            parts = line.split()
            attr_name = parts[1]
            attributes.append(attr_name)
            
        # Data başlangıcını bul
        if line.lower().startswith("@data"):
            data_started = True
            continue
            
        # Veriyi al
        if data_started and not line.startswith("%"):
            data_lines.append(line.split(','))
            
    # DataFrame oluştur
    df = pd.DataFrame(data_lines, columns=attributes)
    return df

def main():
    print("🚀 Fusion işlemi başlıyor...")
    combined_df = pd.DataFrame()
    final_label_col = None

    for i, file_info in enumerate(FILES):
        path = file_info["name"]
        prefix = file_info["prefix"]
        
        if not os.path.exists(path):
            print(f"❌ HATA: {path} dosyası bulunamadı! Lütfen dosya ismini kontrol et.")
            return

        print(f"📂 Okunuyor: {path}...")
        df = load_arff_data(path)
        
        # Sütun isimlerini temizle (boşluk veya tırnak varsa)
        df.columns = [c.strip().replace("'", "").replace('"', "") for c in df.columns]
        
        # Son sütun (label/class) hariç diğerlerine prefix ekle
        # Label sütununu (genelde son sütundur) bulalım
        label_col_name = df.columns[-1] 
        
        # Eğer bu ilk dosya değilse, label sütununu düşür (tekrar etmesin)
        if i > 0:
            df = df.drop(columns=[label_col_name])
        else:
            # İlk dosyanın label ismini sakla
            final_label_col = label_col_name

        # Özellik isimlerini benzersiz yap (örn: contrast -> S1_GLCM_LBP_contrast)
        new_columns = []
        for col in df.columns:
            if col == label_col_name and i == 0:
                new_columns.append(col) # Label ismini değiştirme
            else:
                new_columns.append(f"{prefix}_{col}")
        
        df.columns = new_columns
        
        # Dataframe'leri yan yana (axis=1) birleştir
        if combined_df.empty:
            combined_df = df
        else:
            # Satır sayıları eşit mi kontrol et
            if len(df) != len(combined_df):
                print(f"⚠️ UYARI: Satır sayıları uyuşmuyor! ({len(combined_df)} vs {len(df)})")
            
            combined_df = pd.concat([combined_df, df], axis=1)

    print(f"✅ Tüm dosyalar birleştirildi. Toplam Özellik Sayısı: {len(combined_df.columns) - 1}")
    
    # --- ARFF OLARAK KAYDETME ---
    print(f"💾 {OUTPUT_FILE} kaydediliyor...")
    
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"@relation fusion_all_features\n\n")
        
        # Attribute satırlarını yaz
        for col in combined_df.columns:
            if col == final_label_col:
                # Sınıf etiketi için (cats,dogs,snakes)
                f.write(f"@attribute {col} {{cats,dogs,snakes}}\n")
            else:
                # Diğer tüm özellikler numeric
                f.write(f"@attribute {col} numeric\n")
                
        f.write("\n@data\n")
        
        # Veriyi yaz
        for index, row in combined_df.iterrows():
            f.write(",".join(map(str, row.values)) + "\n")

    print("🎉 İŞLEM TAMAMLANDI! Weka'da açmaya hazır.")

if __name__ == "__main__":
    main()