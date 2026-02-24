import torch
from ultralytics import YOLO


 class YOLODetector:
    def __init__(self):
        print("Loading YOLOv8 model...")

        # Temporary override for PyTorch 2.6+ secure loading issue
        original_load = torch.load

        def custom_load(*args, **kwargs):
            kwargs["weights_only"] = False
            return original_load(*args, **kwargs)

        torch.load = custom_load

        self.model = YOLO("yolov8n.pt")

        # Restore original torch.load
        torch.load = original_load

        print("YOLO model loaded successfully.")

    def detect(self, frame):
        results = self.model(frame)
        return results
