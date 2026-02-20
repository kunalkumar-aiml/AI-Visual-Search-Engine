from src.data_loader import load_data
from src.model import build_model

def main():
    print("🚀 AI Visual Search Engine Initialized")

    (x_train, y_train), (x_test, y_test) = load_data()

    model = build_model()

    model.summary()

if __name__ == "__main__":
    main()
