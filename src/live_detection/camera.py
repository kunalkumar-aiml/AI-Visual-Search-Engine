import cv2
from src.config import CAMERA_INDEX, CONFIDENCE_THRESHOLD, WINDOW_NAME


def start_camera(detector, class_names):

    print("Opening camera...")
    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        print("Error: Unable to access camera.")
        return

    print("Press 'q' to exit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break

        class_index, confidence = detector.predict(frame)

        if confidence > CONFIDENCE_THRESHOLD:
            label = f"{class_names[class_index]} ({confidence:.2f})"

            cv2.putText(
                frame,
                label,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
