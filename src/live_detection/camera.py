import cv2
import time
from src.emotion.emotion_detector import detect_emotion


def start_camera(detector):
    print("Starting live camera...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.detect(frame)

        for result in results:
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    label_name = result.names[cls]

                    # ✅ Only detect emotion for person class
                    if label_name.lower() != "person":
                        continue

                    # ✅ Safe crop boundaries
                    h, w, _ = frame.shape
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)

                    cropped = frame[y1:y2, x1:x2]

                    emotion = "Unknown"

                    if cropped.size != 0:
                        emotion = detect_emotion(cropped)

                    final_label = f"{label_name} | {emotion}"

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                    # Label
                    cv2.putText(
                        frame,
                        final_label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {int(fps)}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

        cv2.imshow("AI Live Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
