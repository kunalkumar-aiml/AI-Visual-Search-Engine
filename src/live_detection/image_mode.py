import cv2
from src.live_detection.color import get_dominant_color


def detect_from_image(detector, image_path):

    print(f"Loading image: {image_path}")

    frame = cv2.imread(image_path)

    if frame is None:
        print("Error: Invalid image path.")
        return

    detections = detector.detect(frame)

    if len(detections) == 0:
        print("No objects detected.")

    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["confidence"]
        class_id = detection["class_id"]

        label = detector.model.names[class_id]
        color_name = get_dominant_color(frame, (x1, y1, x2, y2))

        display_text = f"{label} | {color_name} ({confidence:.2f})"

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            frame,
            display_text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow("Image Detection Mode", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
