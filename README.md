# AI Visual Search Engine

This project is a real-time object detection system that works using a live webcam. 
It detects multiple objects from the camera feed and displays bounding boxes with confidence scores. 
Along with object detection, it also identifies the dominant color of the detected object.

The main goal of this project was to understand how deep learning models work in real-time environments and how object detection systems are structured in production-level applications.


## What This Project Does

- Opens live camera using OpenCV
- Detects objects using YOLOv8
- Draws bounding boxes around detected objects
- Shows object name and confidence score
- Extracts dominant color from the detected region
- Uses confidence threshold filtering


## Deep Learning Part

Apart from live detection, I also implemented an image classification pipeline using TensorFlow.

- Used MobileNetV2 (Transfer Learning)
- Trained on CIFAR-10 dataset
- Achieved around 80% validation accuracy
- Saved trained model and structured training pipeline

This helped me understand model training, evaluation and deployment flow.



## Technologies Used

- Python
- TensorFlow
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Matplotlib

## How to Run

1. Install dependencies:

   pip install -r requirements.txt

2. Run the project:

   python -m src.main

Press 'q' to exit camera window.



## Project Structure

src/
 - live_detection/
 - config.py
 - model.py
 - train.py
 - main.py



## About Me

Kunal Kumar  
B.Tech CSE (AI & ML)  
SRM Institute of Science and Technology
