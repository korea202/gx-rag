import os

from src.apps.sample_chat.chains.chains import search_compression_chain, search_value_check_chain
from dotenv import load_dotenv
from langchain.tools import Tool
#from langchain.utilities import GoogleSearchAPIWrapper
from langchain_tavily import TavilySearch

load_dotenv()

""" search = GoogleSearchAPIWrapper(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    google_cse_id=os.getenv("GOOGLE_CSE_ID"),
)

search_tool = Tool(
    name="Google Search",
    description="Search Google for recent results.",
    func=search.run,
) """

# 검색 도구 생성 (top_k: 결과 개수)
search = TavilySearch(max_results=5)

search_tool = Tool(
    name="Tavily Search",
    description="Search Tavily for recent results.",
    func=search.invoke,
)

def query_web_search(user_message: str) -> str:
    context = {"user_message": user_message}
    context["related_web_search_results"] = search_tool.invoke({"query": user_message})

    has_value = search_value_check_chain.invoke(context)

    print("W"*50)
    print("search_value_check_chain.invoke="+ str(has_value))
    print("W"*50)
    if has_value["output"] == "Y":
        return search_compression_chain.invoke(context)
    else:
        return ""
