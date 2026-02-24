import torch
from ultralytics import YOLO

class ObjectDetector:
    def __init__(self):
        print("Loading YOLOv8 model...")

        # Torch 2.6+ compatibility fix
        torch.serialization.add_safe_globals(
            [torch.nn.modules.module.Module]
        )

        self.model = YOLO("yolov8n.pt")
        print("YOLO model loaded successfully.")

    def detect(self, frame):
        results = self.model(frame)
        return results
