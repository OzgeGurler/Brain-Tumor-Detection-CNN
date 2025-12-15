import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# --- SAYFA AYARLARI ---
st.set_page_config(page_title="Beyin Tümörü Tespiti", page_icon="🧠")

# Başlık ve Açıklama
st.title("Beyin Tümörü Tespit Sistemi")
st.write("Bu uygulama, Derin Öğrenme (CNN) kullanarak MR görüntülerinden tümör teşhisi yapar.")
st.write("Lütfen analiz etmek istediğiniz beyin MR görüntüsünü aşağıya yükleyin.")

# --- MODELİ YÜKLEME (Bunu önbelleğe alıyoruz ki her seferinde beklemesin) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('../models/brain_tumor_model.h5')
    return model

try:
    with st.spinner('Yapay Zeka Modeli Yükleniyor...'):
        model = load_model()
except:
    st.error("HATA: Model dosyası bulunamadı! Lütfen önce train.py dosyasını çalıştırın.")
    st.stop()

# Sınıf İsimleri
class_names = ['Glioma', 'Meningioma', 'No Tumor (Sağlıklı)', 'Pituitary (Hipofiz)']

# --- RESİM YÜKLEME KISMI ---
file = st.file_uploader("Bir MR görüntüsü seçin (jpg, png, jpeg)", type=["jpg", "png", "jpeg"])

if file is not None:
    # Resmi ekrana bas
    image = Image.open(file)
    st.image(image, caption='Yüklenen Görüntü', use_column_width=True)
    
    # Resmi modele uygun hale getir (150x150 boyutuna ve array'e çevir)
    img = ImageOps.fit(image, (150, 150), Image.Resampling.LANCZOS)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Batch boyutu ekle

    # TAHMİN BUTONU
    if st.button("Analiz Et"):
        prediction = model.predict(img_array)
        score = prediction[0] # Olasılıklar
        
        # En yüksek ihtimali bul
        max_score = np.max(score)
        predicted_class = class_names[np.argmax(score)]

        # --- SONUÇ EKRANI ---
        st.write("---")
        st.subheader("🔍 Analiz Sonucu")
        
        # Sonuca göre renkli mesaj ver
        if "No Tumor" in predicted_class:
            st.success(f"Sonuç: **{predicted_class}** (Güven Oranı: %{max_score * 100:.2f})")
            st.balloons() # Ekranda balonlar uçurur :)
        else:
            st.error(f"Tespit Edilen: **{predicted_class}** (Güven Oranı: %{max_score * 100:.2f})")
            st.warning("⚠️ Lütfen uzman bir doktora başvurunuz.")

        # Detaylı Oranları Göster
        with st.expander("Detaylı Olasılık Oranlarını Gör"):
            for i in range(len(class_names)):
                st.write(f"{class_names[i]}: %{score[i]*100:.2f}")