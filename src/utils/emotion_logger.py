import csv
import os
from datetime import datetime

LOG_FILE = "outputs/emotion_log.csv"

def log_emotion(emotion):
    os.makedirs("outputs", exist_ok=True)

    file_exists = os.path.isfile(LOG_FILE)

    with open(LOG_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(["Timestamp", "Emotion"])

        writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), emotion])
