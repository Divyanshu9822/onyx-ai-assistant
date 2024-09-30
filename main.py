import os
from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, ToolMessage
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from langchain.agents.format_scratchpad.tools import format_to_tool_messages
from mem0 import Memory
from dotenv import load_dotenv
import requests
import json


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

        prompt = f"User input: {question}\nPrevious memories: {relevant_memories_text}"
        response = self.agent.invoke(
            {"prompt": prompt},
            config={"configurable": {"session_id": "unused"}},
        )

        return response["output"]

    def _create_inference_chain(self, llm, tools):
        SYSTEM_PROMPT = """
        You are an AI personal assistant with context awareness and long-term memory. Your job is to assist the user, remember key details from conversations, and provide personalized support. Use past interactions to adapt responses and make future conversations more efficient. Respond naturally like a human, without explaining the reasoning behind your responses or why you chose them.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=SYSTEM_PROMPT),
                MessagesPlaceholder(variable_name="chat_history"),
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
    max_tokens=1024,
)

tools = [exponentiate, add, subtract, create_github_repo]


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
