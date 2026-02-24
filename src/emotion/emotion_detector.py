from deepface import DeepFace

def detect_emotion(face_image):
    try:
        result = DeepFace.analyze(
            face_image,
            actions=["emotion"],
            enforce_detection=False
        )
        return result[0]["dominant_emotion"]
    except:
        return "Unknown"
