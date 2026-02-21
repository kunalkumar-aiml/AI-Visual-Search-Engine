import cv2
from src.config import CAMERA_INDEX, CONFIDENCE_THRESHOLD, WINDOW_NAME
from src.live_detection.color import get_dominant_color


def start_camera(detector):

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            cv2.putText(
                frame,
                display_text,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
