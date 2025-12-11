from tensorflow.keras import layers, models
import tensorflow as tf

def create_cnn_model(img_height, img_width, num_classes):
    """
    Daha gelişmiş CNN mimarisi: Veri çoğaltma ve daha derin katmanlar içerir.
    """
    model = models.Sequential([
        # --- VERİ ÇOĞALTMA KATMANI (Data Augmentation) ---
        # Bu katman resimleri rastgele çevirip döndürerek modelin ezber yapmasını önler
        layers.Input(shape=(img_height, img_width, 3)), # Giriş boyutunu belirttik
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        
        # Normalizasyon
        layers.Rescaling(1./255),
        
        # --- ÖZELLİK ÇIKARMA (Feature Extraction) ---
        # Katman 1
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Katman 2
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Katman 3
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # Katman 4 (YENİ EKLEDİK - Daha derin öğrenme için)
        layers.Conv2D(256, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        
        # --- KARAR VERME ---
        layers.Flatten(),
        layers.Dense(256, activation='relu'), # Nöron sayısını artırdık
        layers.Dropout(0.5), # Ezberlemeyi önle
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    return model