import os
import sys
from typing import Dict

from dotenv import dotenv_values, load_dotenv
from pydantic import BaseModel
from fastapi import FastAPI

load_dotenv()

if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)


from src.apps.persona.chains.chains import (
    default_chain,
    parse_intent_chain,
    read_prompt_template,
    printPrompt
)

from src.apps.persona.database.database import query_db

from src.apps.persona.memory.memory import (
    get_chat_history,
    load_conversation_history,
    log_bot_message,
    log_user_message,
)

from src.apps.persona.tools.web_search import query_web_search


app = FastAPI()


class UserRequest(BaseModel):
    user_message: str


ROOT_DIR = os.getenv("PYTHONPATH")
INTENT_LIST_TXT = os.path.join(ROOT_DIR, "src/apps/persona/prompts", "intent_list.txt")

@app.post("/qna/{conversation_id}")
def gernerate_answer(req: UserRequest, conversation_id: str) -> Dict[str, str]:
    history_file = load_conversation_history(conversation_id)

    context = req.model_dump()
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST_TXT)
    context["chat_history"] = get_chat_history(conversation_id)

    intent = parse_intent_chain.invoke(context)

    print("1"*50)
    print(f"intent={intent}")
    print("1"*50)

    if intent['intent'] == "holmes":
        context["related_documents"] = query_db(context["user_message"])
        context["compressed_web_search_results"] = query_web_search(context["user_message"])

        printPrompt(context)
        answer = default_chain.invoke(context)['output']
    else:
        context["related_documents"] = ""
        context["compressed_web_search_results"] = query_web_search(context["user_message"])
        printPrompt(context)
        answer = default_chain.invoke(context)['output']

    print("*"*50)
    print("answer="+str(answer))
    print("*"*50)
    log_user_message(history_file, req.user_message)
    log_bot_message(history_file, answer)
    return {"answer": answer}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
