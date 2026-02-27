import pandas as pd
import matplotlib.pyplot as plt
import os


LOG_FILE = "outputs/emotion_log.csv"


def show_dashboard():
    if not os.path.exists(LOG_FILE):
        print("No emotion log found.")
        return

    df = pd.read_csv(LOG_FILE)

    emotion_counts = df["Emotion"].value_counts()

    print("\nEmotion Distribution:\n")
    print(emotion_counts)

    # Bar Chart
    plt.figure()
    emotion_counts.plot(kind="bar")
    plt.title("Emotion Frequency")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.show()

    # Pie Chart
    plt.figure()
    emotion_counts.plot(kind="pie", autopct="%1.1f%%")
    plt.title("Emotion Percentage Distribution")
    plt.ylabel("")
    plt.show()
