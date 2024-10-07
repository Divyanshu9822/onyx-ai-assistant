from assistant.tools.math import exponentiate, add, subtract
from assistant.tools.face import recognize_face, remember_person
from assistant.tools.github import create_github_repo, clone_github_repo
from assistant.tools.media import (
    take_screenshot_and_query_ai,
    capture_photo_and_query_ai,
    google_lens_search,
)
from assistant.llm import llm
from assistant.assistant import Assistant
from assistant.config.settings import USER_ID
from speech_recognition import Microphone, Recognizer, UnknownValueError, RequestError
from dotenv import load_dotenv
import warnings

warnings.filterwarnings("ignore")

load_dotenv()

tools = [
    exponentiate,
    add,
    subtract,
    create_github_repo,
    clone_github_repo,
    take_screenshot_and_query_ai,
    capture_photo_and_query_ai,
    google_lens_search,
    recognize_face,
    remember_person,
]

recognizer = Recognizer()
microphone = Microphone()
assistant = Assistant(llm, tools)


def audio_callback(recognizer, audio):
    try:
        prompt = recognizer.recognize_google(audio, language="en-US")
        print(f"\nYou: {prompt}")
        assistant.answer(prompt, USER_ID)

    except UnknownValueError:
        print("There was an error processing the audio.")
    except RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")

with microphone as source:
    recognizer.adjust_for_ambient_noise(source)

stop_listening = recognizer.listen_in_background(microphone, audio_callback)
while True:
    if input("Type and Enter 'q' to quit: ") == "q": 
        break
stop_listening(wait_for_stop=False)
