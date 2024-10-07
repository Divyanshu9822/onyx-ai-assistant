from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from langchain.schema.messages import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.google_lens import GoogleLensQueryRun
from langchain_community.utilities.google_lens import GoogleLensAPIWrapper
import pyautogui
import numpy as np
from assistant.llm import llm
from assistant.utils.cloudinary_utils import upload_image
from assistant.utils.helper import capture_image, encode_image


@tool
def take_screenshot_and_query_ai(query: str) -> str:
    """
    Take a screenshot, encode it as base64, and query the AI with the screenshot and a prompt.

    :param query: The text query to send to the AI.
    :return: The response from the AI.
    """
    try:
        screenshot = pyautogui.screenshot()

        screenshot_np = np.array(screenshot)

        image_base64 = encode_image(screenshot_np)

        SYSTEM_PROMPT = """
        You are an AI assistant capable of taking screenshots and responding to user queries based on it. 
        Your task is to provide helpful, concise, and natural responses to their questions about the screen. 
        Respond as a human would, without explaining your reasoning or explicitly mentioning the word 'screenshot.' 
        Simply refer to it as 'the screen.'
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                (
                    "human",
                    [
                        {"type": "text", "text": query},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"query": query, "image_base64": image_base64})
        return response

    except Exception as e:
        return str(e)


@tool
def capture_photo_and_query_ai(prompt: str) -> str:
    """
    Capture a photo using the device's camera, encode it as base64, and query the AI with the photo and a prompt.

    :param prompt: The text prompt to send to the AI.
    :return: The response from the AI.
    """
    try:
        frame = capture_image()

        image_base64 = encode_image(frame)

        SYSTEM_PROMPT = "You are an AI assistant capable of clicking images using the camera and handling user queries. Use both the text prompt and the image to provide a helpful and detailed response."

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                (
                    "human",
                    [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    ],
                ),
            ]
        )

        chain = prompt_template | llm | StrOutputParser()
        response = chain.invoke({"prompt": prompt, "image_base64": image_base64})
        return response

    except Exception as e:
        return str(e)


@tool
def google_lens_search() -> str:
    """
    Capture a photo using the device's camera and Query Google Lens with an image.

    :return: The response from Google Lens.
    """
    try:
        frame = capture_image()

        image_base64 = encode_image(frame)

        image_src = upload_image(image_base64)
        search = GoogleLensQueryRun(api_wrapper=GoogleLensAPIWrapper())
        response = search.run(image_src)
        return response

    except Exception as e:
        return str(e)
