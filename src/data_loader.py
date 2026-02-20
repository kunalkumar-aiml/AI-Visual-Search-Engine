import tensorflow as tf
from src.config import IMAGE_SIZE

def load_data():
    print("📦 Loading CIFAR10 dataset...")

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    # Resize images
    x_train = tf.image.resize(x_train, IMAGE_SIZE)
    x_test = tf.image.resize(x_test, IMAGE_SIZE)

    print("✅ Dataset loaded successfully")
    return (x_train, y_train), (x_test, y_test)
