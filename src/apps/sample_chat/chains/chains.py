import os

from dotenv import load_dotenv
from langchain.chains import LLMChain
#from langchain.chat_models import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.prompts.chat import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.callbacks.tracers import ConsoleCallbackHandler

load_dotenv()


#CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.getenv("PYTHONPATH")
BUG_STEP1_PROMPT_TEMPLATE = os.path.join(ROOT_DIR, "src/apps/sample_chat/prompts", "bug_analyze.txt")
BUG_STEP2_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/sample_chat/prompts", "bug_solution.txt"
)
ENHANCE_STEP1_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/sample_chat/prompts", "enhancement_say_thanks.txt"
)
DEFAULT_RESPONSE_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/sample_chat/prompts", "default_response.txt"
)
INTENT_PROMPT_TEMPLATE = os.path.join(ROOT_DIR, "src/apps/sample_chat/prompts", "parse_intent.txt")

SEARCH_VALUE_CHECK_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/sample_chat/prompts", "search_value_check.txt"
)
SEARCH_COMPRESSION_PROMPT_TEMPLATE = os.path.join(
    ROOT_DIR, "src/apps/sample_chat/prompts", "search_compress.txt"
)


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as f:
        prompt_template = f.read()

    return prompt_template


#chat_file = os.path.join(CUR_DIR, "hist.json")


def create_chain(llm, template_path, output_key):
    return LLMChain(
        llm=llm,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )

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



#llm = ChatOpenAI(temperature=0.1, max_tokens=200, model="gpt-3.5-turbo")
llm = ChatOllama(model="alibayram/Qwen3-30B-A3B-Instruct-2507")

bug_step1_chain = create_chain(
    llm=llm,
    template_path=BUG_STEP1_PROMPT_TEMPLATE,
    output_key="bug_analysis",
)
bug_step2_chain = create_chain(
    llm=llm,
    template_path=BUG_STEP2_PROMPT_TEMPLATE,
    output_key="output",
)
enhance_step1_chain = create_chain(
    llm=llm,
    template_path=ENHANCE_STEP1_PROMPT_TEMPLATE,
    output_key="output",
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
