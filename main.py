from assistant.tools.math import exponentiate, add, subtract
from assistant.tools.face import recognize_face, remember_person
from assistant.tools.github import create_github_repo, clone_github_repo
from assistant.tools.media import (
    take_screenshot_and_query_ai,
    capture_photo_and_query_ai,
    capture_photo_and_google_lens_image_query,
)
from assistant.llm import llm
from assistant.assistant import Assistant
from assistant.config.settings import USER_ID
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
    capture_photo_and_google_lens_image_query,
    recognize_face,
    remember_person,
]


def main():
    assistant = Assistant(llm, tools)

    print("\nHello! I am your AI assistant. How can I assist you today?")
    print("Type 'quit' anytime to exit the conversation.\n")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "quit":
            print("\nAssistant: Goodbye! Feel free to come back anytime. :)")
            break

        answer = assistant.answer(user_input, user_id=USER_ID)

        print(f"Assistant: {answer}\n")


if __name__ == "__main__":
    main()
