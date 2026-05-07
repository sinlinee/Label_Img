import cv2
import numpy as np


def read_image(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def save_image(path, image):
    success, encoded = cv2.imencode(".jpg", image)
    if success:
        encoded.tofile(path)
    return success
