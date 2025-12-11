import os

# Datasetin içindeki Training klasörünün yolu
# NOT: Buradaki yolun senin bilgisayarındaki ile tam aynı olduğundan emin ol
dataset_path = "../dataset/Training"

try:
    # Klasör isimlerini al ve alfabetik sırala (Model tam olarak bunu yapıyor)
    class_names = sorted(os.listdir(dataset_path))
    
    # Gizli dosyaları (Mac'te .DS_Store gibi) temizle
    class_names = [name for name in class_names if not name.startswith('.')]

    print("\n" + "="*40)
    print("      GERÇEK SINIF SIRALAMASI")
    print("="*40)
    for index, name in enumerate(class_names):
        print(f"İndeks {index}: {name}")
    print("="*40 + "\n")
    
    print("Lütfen predict.py dosyasındaki CLASS_NAMES listesini")
    print("YUKARIDAKİ SIRAYLA AYNI olacak şekilde güncelle!")

except Exception as e:
    print(f"Hata oluştu: {e}")
    print("Lütfen 'dataset_path' değişkeninin doğru olduğundan emin ol.")