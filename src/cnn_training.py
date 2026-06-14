import os
import tensorflow as tf
from tensorflow.keras import layers, models

def train_cnn_pipeline():
    os.makedirs("models", exist_ok=True)
    
    train_dir = "data/train"
    test_dir = "data/test"
    
    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError(
            "🚨 Missing Kaggle data arrays! Please extract your Kaggle zip folders into 'data/train/' and 'data/test/'."
        )

    print("[INFO] Processing local Kaggle image dataset matrices...")
    # Stream images from directories securely
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(32, 32),
        batch_size=64,
        label_mode='int'
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        image_size=(32, 32),
        batch_size=64,
        label_mode='int'
    )

    # Standardize image scales (0 to 1 float bounds)
    normalization_layer = layers.Rescaling(1./255)
    train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
    val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

    print("[INFO] Building Convolutional Neural Network (CNN) Architecture...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("[INFO] Executing Model Training Pipeline (3 Epochs)...")
    model.fit(train_ds, validation_data=val_ds, epochs=3)
    
    model_path = "models/cnn_cifar10_model.h5"
    model.save(model_path)
    print(f"[SUCCESS] Core CNN image model saved safely at '{model_path}'")

if __name__ == "__main__":
    train_cnn_pipeline()