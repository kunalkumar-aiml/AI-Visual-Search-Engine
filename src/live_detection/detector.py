from ultralytics import YOLO


class ObjectDetector:

    def __init__(self):
        print("Loading YOLOv8 model...")
        self.model = YOLO("yolov8n.pt")
        print("YOLO model loaded successfully.")

    def detect(self, frame):
        results = self.model(frame)

        detections = []

        for result in results:
            boxes = result.boxes

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                detections.append({
                    "box": (int(x1), int(y1), int(x2), int(y2)),
                    "confidence": confidence,
                    "class_id": class_id
                })

        return detections
