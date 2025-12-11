import tensorflow as tf
import numpy as np
import sys
import os

# --- AYARLAR ---
MODEL_PATH = '../models/brain_tumor_model.h5'
IMG_HEIGHT = 150
IMG_WIDTH = 150

# Sınıf listemiz (Küçük harfle kalabilir, aşağıda yazdırırken büyüteceğiz)
CLASS_NAMES = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

def predict_image(image_path):
    print(f"\nModel yükleniyor ve '{image_path}' inceleniyor...")
    
    if not os.path.exists(MODEL_PATH):
        print("HATA: Model dosyası bulunamadı!")
        return

    try:
        # Modeli yükle
        model = tf.keras.models.load_model(MODEL_PATH)
        
        # Resmi yükle ve hazırla
        img = tf.keras.utils.load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Batch boyutu ekle

        # Tahmin yap
        predictions = model.predict(img_array)
        score = predictions[0] 

        # SONUÇLARI YAZDIR
        print("\n" + "="*40)
        print("      YAPAY ZEKANIN TAHMİN ORANLARI")
        print("="*40)
        
        # Her sınıfın ihtimalini tek tek yaz (Buraya .upper() ekledik)
        for i in range(len(CLASS_NAMES)):
            print(f"{CLASS_NAMES[i].upper()}: \t%{100 * score[i]:.2f}")

        print("-" * 40)
        # En güçlü tahmini de büyük harfle yazdırıyoruz
        print(f"EN GÜÇLÜ TAHMİN: >> {CLASS_NAMES[np.argmax(score)].upper()} <<")
        print("="*40 + "\n")

    except Exception as e:
        print(f"Bir hata oluştu: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = "test_resmi.jpg"
    
    predict_image(img_path)