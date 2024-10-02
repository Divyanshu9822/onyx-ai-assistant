from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from assistant.memory import memory


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
