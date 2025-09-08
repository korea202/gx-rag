# 🕵️‍♂️Dances With Sherlock Holmes


**RAG 파이프라인을 기반으로 페르소나 챗봇을 구현한 셜록홈즈 챗봇 입니다.**  새로운 아이디어나 다양한 기능 보다는 사용자의 질문에 대한 정확한 응답을 구현하기위한 여러가지 고민과 기능들을 구현하였습니다.


## ✨ 주요 특징 (Key Features)
* **2단계 청킹 과 글 요약 기능** :  랭체인에서 제공하는 ParentDocumentRetriever 를 사용하여 검색시에는 청크 사이즈 500 으로 구성된 RecursiveCharacterTextSplitter 으로 검색하고 검색결과는 연관된  doc 스토어 에서 보다 큰 단위로 나뉘어진 문서를 돌려받아 llm 에게 원문서의 30% 크기로 요약을 하여 rag에서 사용하게함.

* **웹검색 기능**: Tavily API를 Tool로 구성하여 셜록홈즈에 대한 보조 검색 기능이나 일반적인 주제애 대한 검색기능 제공함.

* **질문 의도 선별기능**: 사용자의 질문시 전처리 단계로  llm에게 “holmes”, “general” 의 주제별로 분류할수 있게 하여 추후 확장및 프로세스 구현에 용이함을 제공함. 


## 🔨 사전 요구사항

 * Language :  python 3.11 
 * Frontend  : Streamlit 
 * Framework : LangChain 
 * LLM : Ollama-alibayram/Qwen3-30B-A3B-Instruct-2507
 * Embedding :  BAAI/bge-m3
 * Vector DB : FAISS 
 * Search: Tavily API
 * 버전 및 이슈관리 : github 
 * 의존성 관리 : uv



## 📊 RAG 시스템 아키텍처

<img width="1024" height="1024" alt="image" src="https://github.com/user-attachments/assets/48112f78-5d9c-4c5c-b226-d4cec282dd6a" />



## ⚙️ 설치

```bash
1. git clone https://github.com/korea202/gx-rag.git

2. cd gx-rag

3. uv sync

4. uv run src/test.py

(새로 구성할시) uv init / uv venv --python 3.11

#가상환경 실행(optional) source .venv/bin/activate

#pytorch 설치(auto gpu 환경 인식, uv sync 실패시)

UV_TORCH_BACKEND=auto uv pip install torch torchvision torchaudio

#나머지 라이브러리 설치 uv pip install -r requirements.txt
```

## ⚙️ 올라마 설치
```bash
1. apt install curl
  
2. curl -fsSL https://ollama.com/install.sh | sh

3. ollama --version
  
4. nohup ollama serve &

5. ollama pull alibayram/Qwen3-30B-A3B-Instruct-2507
```

## 🚀 실행

```bash
1. .env.template 파일 카피본 생성 .env로 저장

2. .env 파일내 PYTHONPATH(프로젝트 루트디렉토리), TAVILY_API_KEY(웹검색키) 설정

3. 데이타 생성/테스트/저장: src/apps/persona/holmes.ipynb 단계별로 실행 

4. 화면 ui 실행: uv run streamlit run front_app/holmes_ui.py (4, 5번은 각각 다른 cmd 창에서 실행합니다.)

5. API단 실행: uv run src/apps/persona/holmes.py

6. http://localhost:8501
```

🚨🚨🚨 실행할때 포트관련 연결 에러가 난다면 실행전에 터미널 연결시 -L(포트포워딩) 옵션으로 아래 포트를 열어놓습니다.

```bash
ssh -i "C:\경로명\인증서명.pem" -L 11434:localhost:11434 -L 8188:localhost:8188 -L 8000:localhost:8000 -L 8501:localhost:8501 -p 연결포트주소 root@연결아이피
```
