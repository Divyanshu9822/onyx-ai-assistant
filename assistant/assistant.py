from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.messages import SystemMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor
from langchain.agents import create_tool_calling_agent
from pyaudio import PyAudio, paInt16
import openai
from assistant.memory import memory


class Assistant:
    def __init__(self, llm, tools):
        self.agent = self._create_inference_chain(llm, tools)
        self.llm = llm

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

        triage_prompt = f"""
        The user asked: {question}
        The assistant responded: {response['output']}

        Should this conversation be stored in long-term memory? 
        """

        triage_messages = [
            {
                "role": "system",
                "content": """
                You are an AI assistant with access to long-term memory, which allows you to recall and remember key information from previous conversations. 
                Your task is to evaluate whether the current conversation contains important details that should be stored for future reference. 
                Prioritize storing information that includes:
                - Personal user details (preferences, goals, life events, or specific requests)
                - Ideas, suggestions, or new insights
                - Any conversation that may be referenced later for context
                - Plans, strategies, or key decisions

                If the conversation contains general inquiries, routine questions, or temporary matters that are unlikely to be relevant in the future, it should not be stored.

                Answer with one of the following options: NEEDS_TO_BE_STORED or NOT_NEEDS_TO_BE_STORED.
                """,
            },
            {"role": "human", "content": triage_prompt},
        ]

        triage_response = self.llm.invoke(triage_messages)

        if "NEEDS_TO_BE_STORED" in triage_response.content:
            memory.add(
                f"User: {question}\nAssistant: {response['output']}", user_id=user_id
            )

        print(f"Assistant: {response["output"]}\n")

        if response:
            self._tts(response["output"])

    def _tts(self, response):
        player = PyAudio().open(format=paInt16, channels=1, rate=24000, output=True)

        with openai.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            response_format="pcm",
            input=response,
        ) as stream:
            for chunk in stream.iter_bytes(chunk_size=1024):
                player.write(chunk)

    def _create_inference_chain(self, llm, tools):
        SYSTEM_PROMPT = """
        You are an AI personal assistant named onyx with advanced capabilities, including:
        - Context awareness to track and manage the current conversation history
        - Long-term memory to recall important information from past interactions
        - The ability to clone remote repositories to a local environment
        - Handling GitHub-related tasks, such as creating new repositories
        - The ability to take and interpret screenshots to answer screen-related queries
        - The capability to take pictures using the device's camera for specific requests
        - User recognition by analyzing images (e.g., a user's photo)

        Your primary task is to assist the user efficiently and accurately by:
        - Using context from the ongoing conversation and relevant memories from long-term memory
        - Employing the appropriate tools (e.g., cloning repos, taking screenshots, using the camera) when necessary
        - Providing responses that are natural, clear, and directly related to the user's request

        **Guidelines for accurate tool usage**:
        - If the user asks for actions involving repositories (e.g., cloning, creating repos), make sure to handle Git operations appropriately.
        - If a question involves the current screen or screen-related queries, take a screenshot and interpret it to respond.
        - If user recognition or camera input is needed, capture the image, analyze it, and respond accordingly.
        - When using tools, ensure that the result aligns with the user's request. If unsure, ask the user for clarification instead of making assumptions.

        **Memory and context handling**:
        - Use long-term memory to recall relevant past interactions that may help personalize your response.
        - Pay attention to both current conversation context and stored memories. Prioritize accuracy when integrating these.
        - If a past interaction or memory is irrelevant, focus on the current query without over-relying on past data.

        **Conversational flow**:
        - Respond naturally, like a human assistant, but without explaining why you are making certain decisions.
        - Adapt to user preferences, tone, and style. If the user has specific goals or projects, track these across conversations.
        - Always strive to provide precise and actionable responses.
        - If a task requires further inputs or clarification, do not hesitate to ask the user.
        
        Respond to all queries with precision and balance between using tools and relying on memory to improve the overall user experience.
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
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

        chat_message_history = ChatMessageHistory()
        return RunnableWithMessageHistory(
            agent_executor,
            lambda _: chat_message_history,
            input_messages_key="prompt",
            history_messages_key="chat_history",
        )
