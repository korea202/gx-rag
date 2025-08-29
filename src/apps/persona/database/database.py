import os
import sys
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from dotenv import dotenv_values, load_dotenv

if (python_path := dotenv_values().get('PYTHONPATH')) and python_path not in sys.path: sys.path.append(python_path)

from src.apps.persona.chains.chains import summary_chain


ROOT_DIR = os.getenv("PYTHONPATH")
FAISS_PERSIST_DIR = os.path.join(ROOT_DIR, "src/apps/persona/data", "faiss")
DOC_STORE_DIR = os.path.join(ROOT_DIR, "src/apps/persona/data", "store")


# 1. 저장된 벡터스토어 불러오기
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)  # 작은 청크
_db = FAISS.load_local(folder_path=FAISS_PERSIST_DIR, 
                               embeddings=HuggingFaceEmbeddings(
                                    model_name="BAAI/bge-m3",
                                    model_kwargs={"device": "cuda"} ,
                                    encode_kwargs={"normalize_embeddings": True}),
                                allow_dangerous_deserialization=True )

# 2. 파일 기반 독스토어 연결
fs = LocalFileStore(root_path=DOC_STORE_DIR)
store = create_kv_docstore(fs)

# 3. ParentDocumentRetriever 재생성
_retriever = ParentDocumentRetriever(
    vectorstore=_db,
    docstore=store,
    child_splitter=child_splitter
)

def query_db(query: str) -> List[str]:
    
    results = _retriever.get_relevant_documents(query)

    parents = []
    subs = []
    for doc in results:
        parents.append(summary_chain.invoke({'context': doc.page_content})['output'])
    
    results = _db.similarity_search(query)    

    for doc in results:
        subs.append(doc.page_content)

    all = parents + subs
    
    str_docs = [doc for doc in all]
    return str_docs
