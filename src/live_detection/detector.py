import tensorflow as tf
import numpy as np
from src.config import MODEL_SAVE_PATH, IMAGE_SIZE


class ObjectDetector:

    def __init__(self):
        print("Loading trained model...")
        self.model = tf.keras.models.load_model(MODEL_SAVE_PATH)
        print("Model loaded successfully.")

    def preprocess_frame(self, frame):
        resized = tf.image.resize(frame, IMAGE_SIZE)
        normalized = resized / 255.0
        expanded = np.expand_dims(normalized, axis=0)
        return expanded

    def predict(self, frame):
        processed_frame = self.preprocess_frame(frame)
        predictions = self.model.predict(processed_frame, verbose=0)

        class_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))

        return class_index, confidence
