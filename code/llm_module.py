# llm_module.py
import re
import logging
import os
import sys
import time
import json
import uuid
from typing import Generator, List, Dict, Optional, Any
from threading import Lock

# LangChain 및 관련 라이브러리 임포트
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_core.exceptions import OutputParserException

# --- 기존 라이브러리 의존성 (필요 시 유지) ---
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# 로깅 설정
log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
log_level = getattr(logging, log_level_str, logging.INFO)
if not logging.getLogger().handlers:
    logging.basicConfig(level=log_level,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        stream=sys.stdout)
logger = logging.getLogger(__name__)
logger.setLevel(log_level)

# --- 환경 변수 설정 ---
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.debug("🤖⚙️ .env 파일에서 환경 변수를 로드했습니다.")
except ImportError:
    logger.debug("🤖⚙️ python-dotenv가 설치되지 않아 .env 로드를 건너뜁니다.")

# LangSmith 추적을 위한 환경 변수 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "default-project")
# LANGCHAIN_API_KEY는 .env 파일에 있어야 합니다.

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://127.0.0.1:1234/v1")


class LLM:
    """
    LangChain을 사용하여 다양한 LLM 백엔드와 상호작용하기 위한 통합 인터페이스.
    Ollama, OpenAI, LMStudio를 지원하며 LangSmith 추적이 내장되어 있습니다.
    """
    SUPPORTED_BACKENDS = ["ollama", "openai", "lmstudio"]

    def __init__(
        self,
        backend: str,
        model: str,
        system_prompt: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        no_think: bool = False,
    ):
        logger.info(f"🤖⚙️ LLM 초기화 중: 백엔드={backend}, 모델={model}")
        self.backend = backend.lower()
        if self.backend not in self.SUPPORTED_BACKENDS:
            raise ValueError(f"지원되지 않는 백엔 '{backend}'. 지원되는 백엔드: {self.SUPPORTED_BACKENDS}")

        self.model_name = model
        self.system_prompt = system_prompt
        self.no_think = no_think
        self.client = self._create_langchain_client(backend, model, api_key, base_url)
        
        self.system_prompt_message = None
        if self.system_prompt:
            self.system_prompt_message = SystemMessage(content=self.system_prompt)
            logger.info("🤖💬 시스템 프롬프트가 설정되었습니다.")

    def _create_langchain_client(self, backend, model, api_key, base_url):
        if backend == "openai":
            key = api_key or OPENAI_API_KEY
            if not key:
                raise ValueError("OpenAI API 키가 필요하지만 제공되지 않았습니다.")
            return ChatOpenAI(api_key=key, model=model, temperature=0.7, streaming=True)
        
        if backend == "ollama":
            url = base_url or OLLAMA_BASE_URL
            return ChatOllama(base_url=url, model=model, temperature=0.7)

        if backend == "lmstudio":
            url = base_url or LMSTUDIO_BASE_URL
            return ChatOpenAI(api_key="no-key-needed", base_url=url, model=model, temperature=0.7, streaming=True)

        raise ValueError(f"백엔드 '{backend}'에 대한 LangChain 클라이언트를 생성할 수 없습니다.")

    def generate(
        self,
        text: str,
        history: Optional[List[Dict[str, str]]] = None,
        use_system_prompt: bool = True,
        request_id: Optional[str] = None, # LangChain에서 자동 처리되므로 유지할 필요 없음
        **kwargs: Any
    ) -> Generator[str, None, None]:
        
        logger.info(f"🤖💬 생성 시작 (백엔드: {self.backend})")
        messages = []
        if use_system_prompt and self.system_prompt_message:
            messages.append(self.system_prompt_message)
        
        if history:
            for h in history:
                if h["role"] == "user":
                    messages.append(HumanMessage(content=h["content"]))
                elif h["role"] == "assistant":
                    # LangChain은 AIMessage를 사용하지만, 호환성을 위해 Human으로 처리
                    messages.append(HumanMessage(content=h["content"]))

        added_text = text
        if self.no_think:
            added_text = f"{text}/nothink"
        messages.append(HumanMessage(content=added_text))

        try:
            for chunk in self.client.stream(messages, **kwargs):
                yield chunk.content
        except Exception as e:
            logger.error(f"🤖💥 생성 중 오류 발생: {e}", exc_info=True)
            raise

    def prewarm(self, max_retries: int = 1) -> bool:
        """LLM 연결을 예열하고 모델을 로드하려고 시도합니다."""
        prompt = "Respond with only the word 'OK'."
        logger.info(f"🤖🔥 백엔드 '{self.backend}'의 모델 '{self.model_name}' 예열 시도 중...")
        
        for attempt in range(max_retries + 1):
            try:
                response = self.client.invoke([HumanMessage(content=prompt)])
                if response.content:
                    logger.info(f"🤖🔥✅ 예열 성공. 응답: '{response.content}'")
                    return True
            except Exception as e:
                logger.warning(f"🤖🔥⚠️ 예열 시도 {attempt + 1}/{max_retries + 1} 실패: {e}")
                if attempt < max_retries:
                    time.sleep(2 * (attempt + 1))
                else:
                    logger.error("🤖🔥💥 모든 재시도 후 예열 실패.")
                    return False
        return False

    def measure_inference_time(
        self,
        num_tokens: int = 10,
        **kwargs: Any
    ) -> Optional[float]:
        """목표 토큰 수 생성을 위한 추론 시간을 측정합니다."""
        # LangChain의 추상화로 인해 정확한 토큰 기반 시간 측정이 복잡해졌습니다.
        # 대신 간단한 invoke 호출 시간을 측정합니다.
        logger.info(f"🤖⏱️ 추론 시간 측정 중 (간단한 invoke 호출)...")
        
        measurement_user_prompt = "Repeat the following sequence exactly, word for word: one two three four five six seven eight nine ten eleven twelve"
        messages = [HumanMessage(content=measurement_user_prompt)]
        
        start_time = time.time()
        try:
            self.client.invoke(messages, **kwargs)
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            logger.info(f"🤖⏱️✅ 측정된 추론 시간: {duration_ms:.2f} ms")
            return duration_ms
        except Exception as e:
            logger.error(f"🤖⏱️💥 추론 시간 측정 중 오류: {e}", exc_info=False)
            return None

# --- 컨텍스트 관리자는 더 이상 필요하지 않음 (LangChain이 처리) ---

# --- 사용 예시 ---
if __name__ == "__main__":
    main_logger = logging.getLogger(__name__)
    main_logger.info("🤖🚀 --- LLM 모듈 (LangChain) 예시 실행 ---")

    # --- Ollama 예시 ---
    try:
        ollama_model_env = os.getenv("OLLAMA_MODEL", "llama3:instruct")
        main_logger.info(f"\n🤖⚙️ --- Ollama ({ollama_model_env}) 초기화 중 ---")
        ollama_llm = LLM(
            backend="ollama",
            model=ollama_model_env,
            system_prompt="You are a concise and helpful assistant."
        )

        main_logger.info("🤖🔥 --- Ollama 예열 실행 중 ---")
        if ollama_llm.prewarm():
            main_logger.info("🤖✅ Ollama 예열/초기화 정상.")
            
            main_logger.info("🤖⏱️ --- Ollama 추론 시간 측정 실행 중 ---")
            inf_time = ollama_llm.measure_inference_time()
            if inf_time is not None:
                main_logger.info(f"🤖⏱️ --- 측정된 추론 시간: {inf_time:.2f} ms ---")

            main_logger.info("🤖▶️ --- Ollama 생성 실행 중 ---")
            try:
                generator = ollama_llm.generate("What is the capital of France? Be brief.")
                print("\nOllama Response: ", end="", flush=True)
                for token in generator:
                    print(token, end="", flush=True)
                print("\n")
                main_logger.info("🤖✅ Ollama 생성 완료.")
            except Exception as e:
                main_logger.error(f"🤖💥 Ollama 생성 오류: {e}", exc_info=True)
        else:
            main_logger.error("🤖❌ Ollama 예열/초기화 실패.")

    except Exception as e:
        main_logger.error(f"🤖💥 Ollama 예시 실패: {e}", exc_info=True)

    main_logger.info("\n" + "="*40)
    main_logger.info("🤖🏁 --- LLM 모듈 예시 스크립트 완료 ---")