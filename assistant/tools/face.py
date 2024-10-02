import os
from langchain_core.tools import tool
import cv2
from datetime import datetime
from deepface import DeepFace
import time
from assistant.config.settings import FACES_DB_PATH


@tool
def recognize_face() -> str:
    """
    Capture a current photo and compare it with the stored face database to recognize the person.

    :return: The name of the recognized person, or a message if no match is found.
    """
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return "Error: Unable to access the camera."

    time.sleep(2)

    ret, frame = camera.read()
    camera.release()

    if not ret:
        return "Error: Unable to capture the photo."

    temp_image_path = "temp_current_user.jpg"
    cv2.imwrite(temp_image_path, frame)

    result = DeepFace.find(img_path=temp_image_path, db_path=FACES_DB_PATH)

    os.remove(temp_image_path)

    if len(result) > 0:
        best_match = result[0]["identity"]
        person_dir = best_match.values[0]

        person_name = person_dir.split("/")[-2]

        return f"The person is {person_name}"
    else:
        return "No match found in the database."


@tool
def remember_person(person_name: str) -> str:
    """
    Capture a photo, store it in a folder named after the person, and save the photo with the current timestamp.

    :param person_name: The name of the person to remember.
    :return: A confirmation message.
    """
    person_folder = os.path.join(FACES_DB_PATH, person_name)
    if not os.path.exists(person_folder):
        os.makedirs(person_folder)

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        return "Error: Unable to access the camera."

    time.sleep(2)

    ret, frame = camera.read()
    camera.release()

    if not ret:
        return "Error: Unable to capture the photo."

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    photo_path = os.path.join(person_folder, f"{timestamp}.jpg")

    cv2.imwrite(photo_path, frame)

    return f"Memory updated. I'll remember you now, {person_name}!"
