import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain_ollama import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.tracers import ConsoleCallbackHandler

load_dotenv()


ROOT_DIR = os.getenv("PYTHONPATH")

DEFAULT_RESPONSE_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/persona/prompts", "default_response.txt"
)
INTENT_PROMPT_TEMPLATE = os.path.join(ROOT_DIR, "src/apps/persona/prompts", "parse_intent.txt")

SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/persona/prompts", "search_value_check.txt"
)
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/persona/prompts", "search_compress.txt"
)

SUMMARY_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/persona/prompts", "summary.txt"
)

llm = ChatOllama(model="alibayram/Qwen3-30B-A3B-Instruct-2507")

def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template

def printPrompt(context):
    print("fin="+ ChatPromptTemplate.from_template( template=read_prompt_template(DEFAULT_RESPONSE_PROMPT_TEMPLATE)).format(**context))
    
    
def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=False,
    )



###############################################################
#  create_chain  추후에 변경시 사용할 로직
###############################################################
def create_chain_lcel(llm, template_path, output_key):
    """
    LCEL을 사용하여 체인을 생성하고 반환합니다.
    이 체인은 LLMChain과 동일하게, 기존 입력값을 유지하며 
    지정된 output_key로 결과를 추가하는 딕셔너리를 반환합니다.
    """
    
    # 1. 기본 체인 구성 (프롬프트 | 모델 | 출력 파서)
    #    StrOutputParser()는 모델의 출력을 간단한 문자열로 변환합니다.
    base_chain = (
        ChatPromptTemplate.from_template(read_prompt_template(template_path))
        | llm
        | StrOutputParser()
    )

    # 2. 'output_key' 처리: RunnablePassthrough.assign() 사용
    #    입력 딕셔너리를 그대로 통과시키고(RunnablePassthrough),
    #    지정한 output_key에 base_chain의 결과를 할당(assign)합니다.
    lcel_chain = RunnablePassthrough.assign(**{output_key: base_chain})
    
    return lcel_chain

# 체인을 실행하는 래퍼 함수 정의
def invoke_with_debug(chain, input_data):
    """항상 디버깅 콜백과 함께 체인을 실행하는 함수"""
    print("--- Running with Debug Callbacks ---")
    return chain.invoke(
        input_data,
        config={"callbacks": [ConsoleCallbackHandler()]}
    )

parse_intent_chain = create_chain(
    llm=llm,
    template_path=INTENT_PROMPT_TEMPLATE,
    output_key="intent",
)
default_chain = create_chain(
    llm=llm, template_path=DEFAULT_RESPONSE_PROMPT_TEMPLATE, output_key="output"
)

search_value_check_chain = create_chain(
    llm=llm,
    template_path=SEARCH_VALUE_CHECK_PROMPT_TEMPLATE,
    output_key="output",
)
search_compression_chain = create_chain(
    llm=llm,
    template_path=SEARCH_COMPRESSION_PROMPT_TEMPLATE,
    output_key="output",
)

summary_chain = create_chain(
    llm=llm,
    template_path=SUMMARY_PROMPT_TEMPLATE,
    output_key="output",
)
