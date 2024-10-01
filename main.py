import os
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from mem0 import Memory
import pyautogui
import base64
from io import BytesIO
from PIL import Image
import cv2
from dotenv import load_dotenv
import requests
import json
import warnings

warnings.filterwarnings("ignore")


load_dotenv()

user_id = "divyanshu"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
github_personal_token = os.getenv("GITHUB_PERSONAL_TOKEN")

qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

config = {
    "llm": {
        "provider": "litellm",
        "config": {
            "model": "gemini/gemini-1.5-flash-002",
            "temperature": 0.2,
            "max_tokens": 1024,
        },
    },
    "embedder": {
        "provider": "gemini",
        "config": {
            "model": "models/text-embedding-004",
            "embedding_dims": 768,
        },
    },
    "vector_store": {
        "provider": "qdrant",
        "config": {
            "collection_name": qdrant_collection_name,
            "url": qdrant_url,
            "api_key": qdrant_api_key,
            "embedding_model_dims": 768,
        },
    },
    "version": "v1.1",
}

memory = Memory.from_config(config_dict=config)


@tool
def exponentiate(x: float, y: float) -> float:
    """Raise 'x' to the 'y'."""
    return x**y


@tool
def add(x: float, y: float) -> float:
    """Add 'x' and 'y'."""
    return x + y


@tool
def subtract(x: float, y: float) -> float:
    """Subtract 'x' from 'y'."""
    return y - x


@tool
def create_github_repo(repo_name: str, visibility: str):
    """
    Create a GitHub repository with specified visibility.

    Args:
        repo_name (str): The name of the repository to create.
        visibility (str): Visibility of the repository, either 'public' or 'private'.

    Returns:
        dict: Response from the GitHub API.
    """
    url = "https://api.github.com/user/repos"
    headers = {
        "Authorization": f"token {github_personal_token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {"name": repo_name, "private": visibility == "private"}
    response = requests.post(url, headers=headers, data=json.dumps(data))
    return response.json()


@tool
def take_screenshot_and_query_ai(query: str) -> str:
    """
    Take a screenshot, encode it as base64, and query the AI with the screenshot and a prompt.

    :param query: The text query to send to the AI.
    :return: The response from the AI.
    """
    screenshot = pyautogui.screenshot()

    if screenshot.mode == "RGBA":
        screenshot = screenshot.convert("RGB")

    buffered = BytesIO()
    screenshot.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    SYSTEM_PROMPT = """
    You are an AI assistant capable of taking screenshots and responding to user queries based on it. Your task is to provide helpful, concise, and natural responses to their questions about the screen.

    Respond as a human would, without explaining your reasoning or explicitly mentioning the word "screenshot." Simply refer to it as "the screen."
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


@tool
def capture_photo_and_query_ai(prompt: str) -> str:
    """
    Capture a photo using the device's camera, encode it as base64, and query the AI with the photo and a prompt.

    :param prompt: The text prompt to send to the AI.
    :return: The response from the AI.
    """
    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        return "Error: Unable to access the camera."

    ret, frame = camera.read()

    camera.release()

    if not ret:
        return "Error: Unable to capture photo."

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    buffered = BytesIO()
    image = Image.fromarray(rgb_frame)
    image.save(buffered, format="JPEG")

    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

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


class Assistant:
    def __init__(self, llm, tools):
        self.agent = self._create_inference_chain(llm, tools)

    def answer(self, question: str, user_id: str) -> str:
        """
        Process a user's question and generate a response.

        :param question: The user's input question.
        :param user_id: The unique identifier for the user.
        :return: The AI-generated response to the question.
        """
        if not question:
            return

        previous_memories = memory.search(question, user_id=user_id, limit=3)

        relevant_memories_text = "\n".join(
            mem["memory"] for mem in previous_memories["results"]
        )

        prompt = f"""
        User input: {question}

        Relevant memories: 
        
        {relevant_memories_text}
        """
        response = self.agent.invoke(
            {"prompt": prompt},
            config={"configurable": {"session_id": "unused"}},
        )

        return response["output"]

    def _create_inference_chain(self, llm, tools):
        SYSTEM_PROMPT = """
        You are an AI personal assistant with context awareness, long-term memory, ability to take and interpret screenshots using along with capturing images using device camera. Your job is to assist the user, handle queries regarding the screen and the image clicked using camera, remember key details from conversations, and provide personalized support. Use past interactions to adapt responses and make future conversations more efficient. Respond naturally like a human, without explaining the reasoning behind your responses or why you chose them.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history", n_messages=3),
                (
                    "human",
                    [
                        {"type": "text", "text": "{prompt}"},
                    ],
                ),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )

        agent = create_tool_calling_agent(llm, tools, prompt=prompt_template)
        agent_executor = AgentExecutor(agent=agent, tools=tools)

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            agent_executor,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    temperature=0.2,
)

tools = [
    exponentiate,
    add,
    subtract,
    create_github_repo,
    take_screenshot_and_query_ai,
    capture_photo_and_query_ai,
]


def main():
    assistant = Assistant(llm, tools)

    while True:
        user_input = input("\nAsk me something: (or 'quit' to exit):\t")

        if user_input.lower() == "quit":
            print("Goodbye :)")
            break
        print("\nAssistant:")
        answer = assistant.answer(user_input, user_id=user_id)
        print(answer)


if __name__ == "__main__":
    main()
