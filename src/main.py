from src.live_detection.detector import ObjectDetector
from src.live_detection.camera import start_camera


def main():

    print("===================================")
    print("AI Visual Search Engine - Live Mode")
    print("===================================")

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck"
    ]

    detector = ObjectDetector()
    start_camera(detector, class_names)


if __name__ == "__main__":
    main()
