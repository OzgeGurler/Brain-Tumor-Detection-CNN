import tensorflow as tf
import os

def load_data(data_path, img_height, img_width, batch_size):
    """
    Veri setini belirtilen yoldan okur ve eğitim/doğrulama setlerine ayırır.
    """
    
    # Eğitim klasörünün yolu
    train_dir = os.path.join(data_path, 'Training')

    # Eğitim verisi (%80)
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Doğrulama (Test) verisi (%20)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
    )

    # Sınıf isimlerini al
    class_names = train_ds.class_names
    
    # Performans için veriyi belleğe al (Cache ve Prefetch)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names