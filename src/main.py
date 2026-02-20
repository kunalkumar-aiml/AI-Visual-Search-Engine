from src.data_loader import load_data

def main():
    print("🚀 AI Visual Search Engine Initialized")

    (x_train, y_train), (x_test, y_test) = load_data()

    print(f"Training samples: {len(x_train)}")
    print(f"Testing samples: {len(x_test)}")

if __name__ == "__main__":
    main()
