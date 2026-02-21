from src.live_detection.detector import ObjectDetector
from src.live_detection.camera import start_camera


def main():

    print("===================================")
    print("AI Visual Search Engine - YOLO Mode")
    print("===================================")

    detector = ObjectDetector()
    start_camera(detector)


if __name__ == "__main__":
    main()
