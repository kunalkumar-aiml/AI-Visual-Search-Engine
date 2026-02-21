import tensorflow as tf
from src.data_loader import load_data
from src.model import build_model
from src.config import EPOCHS, MODEL_SAVE_PATH

def train_model():
    print("🚀 Starting training pipeline...")

    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train,
        epochs=EPOCHS,
        validation_data=(x_test, y_test)
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved at {MODEL_SAVE_PATH}")

    return history
