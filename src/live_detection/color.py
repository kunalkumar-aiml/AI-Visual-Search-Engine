import numpy as np
import cv2


def get_dominant_color(frame, box):
    x1, y1, x2, y2 = box

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return "Unknown"

    avg_color = np.mean(roi, axis=(0, 1))
    b, g, r = avg_color

    if r > g and r > b:
        return "Red"
    elif g > r and g > b:
        return "Green"
    elif b > r and b > g:
        return "Blue"
    else:
        return "Mixed"
