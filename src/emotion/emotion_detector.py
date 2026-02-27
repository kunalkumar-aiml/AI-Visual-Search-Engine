import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

# Emotion labels (same order as training)
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load model once at startup
model = torch.load("emotion_model.pth", map_location=device)
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((48, 48)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

def detect_emotion(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        img = transform(img)
        img = img.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        emotion = class_names[predicted.item()]
        conf_score = round(confidence.item() * 100, 2)

        return f"{emotion} ({conf_score}%)"

    except Exception as e:
        print("Emotion detection error:", e)
        return "Unknown"
