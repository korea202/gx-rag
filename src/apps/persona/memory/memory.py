import json
import os

from langchain.memory import ConversationBufferMemory, FileChatMessageHistory

ROOT_DIR = os.getenv("PYTHONPATH")
HISTORY_DIR = os.path.join(ROOT_DIR, "src/apps/persona/data/chat_history")


def load_conversation_history(conversation_id: str):
    file_path = os.path.join(HISTORY_DIR, f"{conversation_id}.json")
    
    if not os.path.exists(file_path):
        with open(file_path, "w") as file:
            file.write("[]")

    history = FileChatMessageHistory(file_path=file_path)
    return history


def log_user_message(history: FileChatMessageHistory, user_message: str):
    history.add_user_message(user_message)


def log_bot_message(history: FileChatMessageHistory, bot_message: str):
    history.add_ai_message(bot_message)


def get_chat_history(conversation_id: str):
    history = load_conversation_history(conversation_id)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="user_message",
        chat_memory=history,
        return_messages=False           # False로 설정하여 문자열(buffer)로 반환
    )

    #return memory.buffer
    return memory.load_memory_variables({})['chat_history']
