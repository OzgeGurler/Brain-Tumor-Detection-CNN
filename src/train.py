import matplotlib.pyplot as plt
import os
from data_loader import load_data
from model_builder import create_cnn_model

# --- AYARLAR ---
# Veri setinin ana proje klasöründeki konumu (bir üst klasördeki dataset)
DATASET_PATH = "../dataset" 
IMG_HEIGHT = 150
IMG_WIDTH = 150
BATCH_SIZE = 32
EPOCHS = 25 # Eğitim tur sayısı (İstersen artırabilirsin)

def main():
    print("Veri yükleniyor...")
    # Veri setinin varlığını kontrol et
    if not os.path.exists(os.path.join(DATASET_PATH, 'Training')):
        print("HATA: 'dataset/Training' klasörü bulunamadı!")
        print("Lütfen Kaggle'dan indirdiğin veri setini 'dataset' klasörüne koyduğundan emin ol.")
        return

    train_ds, val_ds, class_names = load_data(DATASET_PATH, IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE)
    print(f"Sınıflar: {class_names}")

    print("Model oluşturuluyor...")
    model = create_cnn_model(IMG_HEIGHT, IMG_WIDTH, len(class_names))
    model.summary()

    print("Eğitim başlıyor (Bu işlem biraz sürebilir)...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS
    )

    # Modeli kaydet
    if not os.path.exists('../models'):
        os.makedirs('../models')
    model.save('../models/brain_tumor_model.h5')
    print("\nModel 'models/brain_tumor_model.h5' olarak başarıyla kaydedildi.")

    # Başarı grafiğini çiz ve kaydet
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(EPOCHS)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.savefig('egitim_sonucu_grafigi.png')
    print("Grafik kaydedildi: src/egitim_sonucu_grafigi.png")

if __name__ == "__main__":
    main()