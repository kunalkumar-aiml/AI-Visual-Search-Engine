import cv2
import time
from src.live_detection.detector import YOLODetector
from src.emotion.emotion_detector import detect_emotion


class LiveCamera:

    def __init__(self):
        self.detector = YOLODetector()
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            print("Error: Cannot access camera.")
            exit()

    def get_dominant_color(self, image):
        # Simple average RGB method
        avg_color = image.mean(axis=0).mean(axis=0)
        r, g, b = avg_color

        if r > g and r > b:
            return "Red"
        elif g > r and g > b:
            return "Green"
        elif b > r and b > g:
            return "Blue"
        else:
            return "Mixed"

    def run(self):
        prev_time = 0

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            results = self.detector.detect(frame)

            for result in results:
                x1, y1, x2, y2, label, confidence = result

                # Crop detected region
                crop = frame[y1:y2, x1:x2]

                # Get dominant color
                color = self.get_dominant_color(crop)

                # Get emotion (only if person detected)
                emotion = "N/A"
                if label == "person":
                    emotion = detect_emotion(crop)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label
                cv2.putText(
                    frame,
                    f"{label} | {color} ({confidence:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Draw emotion
                if label == "person":
                    cv2.putText(
                        frame,
                        f"Emotion: {emotion}",
                        (x1, y1 - 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

            # FPS calculation
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 0, 0),
                2
            )

            cv2.imshow("AI Visual Search Engine - Live", frame)

            # Press q to exit
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.cap.release()
        cv2.destroyAllWindows()
