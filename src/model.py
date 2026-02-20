import tensorflow as tf
from tensorflow.keras import layers, models
from src.config import IMAGE_SIZE, NUM_CLASSES

def build_model():
    print("🧠 Building model using MobileNetV2...")

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
        include_top=False,
        weights="imagenet"
    )

    base_model.trainable = False  # Freeze pretrained layers

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax")
    ])

    print("✅ Model built successfully")
    return model
