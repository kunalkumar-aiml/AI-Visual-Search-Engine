import cv2
import time
from src.config import CAMERA_INDEX, CONFIDENCE_THRESHOLD, WINDOW_NAME
from src.live_detection.color import get_dominant_color


def start_camera(detector):

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return

    print("Press 'q' to exit.")
    print("Press 's' to save snapshot.")

    prev_time = 0

    try:
        while True:
            ret, frame = cap.read()

            if not ret:
                break

            # FPS Calculation
            current_time = time.time()
            fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
            prev_time = current_time

            detections = detector.detect(frame)

            for detection in detections:
                x1, y1, x2, y2 = detection["box"]
                confidence = detection["confidence"]
                class_id = detection["class_id"]

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                label = detector.model.names[class_id]
                color_name = get_dominant_color(frame, (x1, y1, x2, y2))

                display_text = f"{label} | {color_name} ({confidence:.2f})"

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show label
                cv2.putText(
                    frame,
                    display_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Logging detection
                with open("outputs/detections.log", "a") as log_file:
                    log_file.write(
                        f"{label} | {color_name} | {confidence:.2f}\n"
                    )

            # Show FPS
            cv2.putText(
                frame,
                f"FPS: {int(fps)}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF

            # Exit
            if key == ord("q"):
                break

            # Save Snapshot
            if key == ord("s"):
                timestamp = int(time.time())
                filename = f"outputs/snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Snapshot saved: {filename}")

    finally:
        cap.release()
        cv2.destroyAllWindows()
