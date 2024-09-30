import os
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_google_genai import ChatGoogleGenerativeAI
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

user_id = "divyanshu"
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

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

# # Add 5 distinct memories with metadata
# memories_to_add = [
#     ("I recently started a new job as a software engineer", {"category": "career"}),
#     ("I traveled to Japan last year for vacation", {"category": "travel"}),
#     ("I bought a new bicycle to stay active", {"category": "fitness"}),
#     ("My favorite dish is lasagna", {"category": "food"}),
#     ("I have a pet dog named Bruno", {"category": "personal"}),
# ]

# # Add each memory to the memory store with metadata and user_id
# for mem, meta in memories_to_add:
#     memory.add(mem, user_id=user_id, metadata=meta)


class Assistant:
    def __init__(self, llm):
        self.chain = self._create_inference_chain(llm)

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
        response = self.chain.invoke(
            {"prompt": prompt},
            config={"configurable": {"session_id": "unused"}},
        )

        return response

    def _create_inference_chain(self, llm):
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
            ]
        )

        chain = prompt_template | llm | StrOutputParser()

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            chain,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )


llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-002",
    temperature=0.2,
    max_tokens=1024,
)


def main():
    assistant = Assistant(llm)

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
