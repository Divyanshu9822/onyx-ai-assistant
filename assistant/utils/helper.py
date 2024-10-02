import cv2
import time
import base64
from io import BytesIO
from PIL import Image
import numpy as np


def capture_image(camera_index=0) -> np.ndarray:
    """
    Capture an image from the camera and return it in OpenCV format.
    :param camera_index: Index of the camera to use (default: 0).
    :return: Captured image as a numpy array (OpenCV format).
    """
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise Exception("Error: Unable to access the camera.")

    time.sleep(2)
    ret, frame = camera.read()

    camera.release()

    if not ret:
        raise Exception("Error: Unable to capture the photo.")

    return frame


def encode_image(image: np.ndarray) -> str:
    """
    Encode the captured image as a base64 string.
    :param image: Image captured in OpenCV format.
    :return: Base64 encoded image string.
    """
    rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    buffered = BytesIO()
    pil_image = Image.fromarray(rgb_frame)
    pil_image.save(buffered, format="JPEG")
    encoded_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return encoded_image


def save_image(image: np.ndarray, path: str) -> None:
    """
    Save the captured image to the disk at the specified path.
    :param image: Image captured in OpenCV format.
    :param path: Path to save the image.
    """
    cv2.imwrite(path, image)
