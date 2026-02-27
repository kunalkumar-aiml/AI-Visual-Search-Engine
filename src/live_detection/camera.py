import cv2
import time
from collections import deque
from src.emotion.emotion_detector import detect_emotion

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Store last 10 emotions for smoothing
emotion_history = deque(maxlen=10)


def start_camera(detector):
    print("Starting live camera...")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not access camera.")
        return

    prev_time = 0
    frame_count = 0
    current_emotion = "Detecting..."

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = detector.detect(frame)

        for result in results:
            boxes = result.boxes

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    label_name = result.names[cls]

                    # Only detect emotion for person
                    if label_name.lower() != "person":
                        continue

                    person_crop = frame[y1:y2, x1:x2]

                    if person_crop.size == 0:
                        continue

                    gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, 1.3, 5
                    )

                    for (fx, fy, fw, fh) in faces:
                        face_crop = person_crop[fy:fy+fh, fx:fx+fw]

                        # Run emotion every 3 frames (performance boost)
                        if frame_count % 3 == 0:
                            emotion = detect_emotion(face_crop)
                            emotion_history.append(emotion)

                            if len(emotion_history) > 0:
                                current_emotion = max(
                                    set(emotion_history),
                                    key=emotion_history.count
                                )

                        # Draw face box
                        cv2.rectangle(
                            person_crop,
                            (fx, fy),
                            (fx + fw, fy + fh),
                            (255, 0, 0),
                            2
                        )

                    # Draw person box
                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{label_name} | {current_emotion}",
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
