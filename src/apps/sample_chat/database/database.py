import os
from typing import List

#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

CHROMA_PERSIST_DIR = os.path.join(CUR_DIR, "database", "chroma-persist")
CHROMA_COLLECTION_NAME = "fastcampus-bot"


_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    #embedding_function=OllamaEmbeddings(model="mxbai-embed-large"),
    embedding_function=HuggingFaceEmbeddings(
                                    model_name="BAAI/bge-m3",
                                    model_kwargs={"device": "cuda"} ,
                                    encode_kwargs={"normalize_embeddings": True}),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()


def query_db(query: str, use_retriever: bool = False) -> List[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs
