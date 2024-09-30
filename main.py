import os
import google.generativeai as genai
from mem0 import Memory
from dotenv import load_dotenv

load_dotenv()

user_id = "divyanshu"
os.environ["GEMINI_API_KEY"] = os.getenv("GEMINI_API_KEY")

qdrant_collection_name = os.getenv("QDRANT_COLLECTION_NAME")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")


genai.configure(api_key=os.environ["GEMINI_API_KEY"])

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


class Companion:
    """
    A class representing an AI companion that can engage in conversations and remember previous interactions.
    """

    def __init__(self):
        """
        Initialize the Companion.
        """
        self.model = genai.GenerativeModel(
            model_name="gemini-1.5-flash-002",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
                "response_mime_type": "text/plain",
            },
            system_instruction="You are an AI personal assistant with context awareness and long-term memory. Your job is to assist the user, remember key details from conversations, and provide personalized support. Use past interactions to adapt responses and make future conversations more efficient. Respond naturally and directly, like a human, without explaining the reasoning behind your responses or why you chose them. Focus on delivering helpful, concise answers without clarifications. Continuously adapt to the user's needs.",
        )
        self.chat_session = None
        self.memory = memory

    def start_chat(self):
        """
        Start a chat session.
        """
        self.chat_session = self.model.start_chat()

    def ask(self, question: str, user_id: str) -> str:
        """
        Process a user's question and generate a response.

        :param question: The user's input question.
        :param user_id: The unique identifier for the user.
        :return: The AI-generated response to the question.
        """
        if self.chat_session is None:
            self.start_chat()


        previous_memories = memory.search(question, user_id=user_id, limit=5)

        relevant_memories_text = "\n".join(
            mem["memory"] for mem in previous_memories["results"]
        )

        prompt = f"User input: {question}\nPrevious memories: {relevant_memories_text}"
        response = self.chat_session.send_message(prompt)

        return response.text

    def __enter__(self):
        """
        Enter the runtime context for the Companion.

        :return: The Companion instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit the runtime context for the Companion.

        :param exc_type: The exception type, if any.
        :param exc_val: The exception value, if any.
        :param exc_tb: The exception traceback, if any.
        """
        pass


def main():
    """
    The main function to run the AI companion interaction loop.
    """
    with Companion() as ai_companion:
        while True:
            text_input = input("\nAsk me something: (or 'quit' to exit):\t")

            if text_input.lower() == "quit":
                print("Goodbye :)")
                break
            print("\nAssistant:")
            answer = ai_companion.ask(
                text_input, user_id=user_id
            ) 
            print(answer)


if __name__ == "__main__":
    main()
