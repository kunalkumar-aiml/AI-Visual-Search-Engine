import torch
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks


class ObjectDetector:
    def __init__(self):
        print("Loading YOLOv8 model...")

        # Fix for PyTorch 2.6+ secure weight loading
        torch.serialization.add_safe_globals({
            "ultralytics.nn.tasks.DetectionModel": tasks.DetectionModel
        })

        self.model = YOLO("yolov8n.pt")

        print("YOLO model loaded successfully.")

    def detect(self, frame):
        results = self.model(frame)
        return results
