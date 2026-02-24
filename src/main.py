from src.live_detection.detector import ObjectDetector
from src.live_detection.camera import start_camera
from src.live_detection.image_mode import detect_from_image


def main():

    print("===================================")
    print("AI Visual Search Engine")
    print("===================================")
    print("1 - Live Camera Detection")
    print("2 - Image Detection")

    choice = input("Select mode (1 or 2): ")

    detector = ObjectDetector()

    if choice == "1":
        start_camera(detector)

    elif choice == "2":
        image_path = input("Enter image path: ")
        detect_from_image(detector, image_path)

    else:
        print("Invalid selection.")


if __name__ == "__main__":
    main()
