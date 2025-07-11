# speech_pipeline_manager.py
from typing import Optional, Callable, List, Dict
import threading
import logging
import time
from queue import Queue, Empty
import sys
import re

# AI-to-AI 대화 제어 설정
DISCONN_ALARM = True          # 연결 끊김 알림 활성화
MAXIMUM_AI_TO_AI = 15         # 최대 AI-to-AI 대화 횟수 (0 = 무제한)
AI_TURN_DELAY = 0.1           # AI 턴 사이 지연 시간 (초)

# LangChain 및 RAG 관련 라이브러리 임포트
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

from audio_module import AudioProcessor
from text_similarity import TextSimilarity
from text_context import TextContext
from llm_module import LLM
from colors import Colors

logger = logging.getLogger(__name__)

def load_prompt_from_file(file_path: str, default_prompt: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        logger.info(f"🗣️📄 파일에서 프롬프트를 로드했습니다: {file_path}")
        return prompt
    except FileNotFoundError:
        logger.warning(f"🗣️📄 {file_path}를 찾을 수 없습니다. 기본 프롬프트를 사용합니다.")
        return default_prompt

# 페르소나 및 오케스트레이터 프롬프트 로드
persona_ai1 = load_prompt_from_file("persona_ai1.txt", "You are a friendly conversation partner.")
persona_ai2 = load_prompt_from_file("persona_ai2.txt", "You are a helpful tutor who corrects mistakes.")
orchestrator_prompt_text = load_prompt_from_file("orchestrator.txt", "You are an orchestrator. Decide who should speak next: AI1_friend or AI2_tutor.")

# ... (기존 orpheus_prompt_addon 코드 유지)
USE_ORPHEUS_UNCENSORED = False
orpheus_prompt_addon_normal = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", and " <gasp> ".
Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()
orpheus_prompt_addon_uncensored = """
When expressing emotions, you are ONLY allowed to use the following exact tags (including the spaces):
" <moans> ", " <panting> ", " <grunting> ", " <gagging sounds> ", " <chokeing> ", " <kissing noises> ", " <laugh> ", " <chuckle> ", " <sigh> ", " <cough> ", " <sniffle> ", " <groan> ", " <yawn> ", " <gasp> ".
Do NOT create or use any other emotion tags. Do NOT remove the spaces. Use these tags exactly as shown, and only when appropriate.
""".strip()
orpheus_prompt_addon = orpheus_prompt_addon_uncensored if USE_ORPHEUS_UNCENSORED else orpheus_prompt_addon_normal


class PipelineRequest:
    def __init__(self, action: str, data: Optional[any] = None):
        self.action = action
        self.data = data
        self.timestamp = time.time()

class RunningGeneration:
    def __init__(self, id: int):
        self.id: int = id
        self.text: Optional[str] = None
        self.speaker_role: str = "assistant"
        self.timestamp = time.time()
        self.llm_generator = None
        self.llm_finished: bool = False
        self.llm_finished_event = threading.Event()
        self.llm_aborted: bool = False
        self.quick_answer: str = ""
        self.quick_answer_provided: bool = False
        self.quick_answer_first_chunk_ready: bool = False
        self.quick_answer_overhang: str = ""
        self.tts_quick_started: bool = False
        self.tts_quick_allowed_event = threading.Event()
        self.audio_chunks = Queue()
        self.total_audio_duration: float = 0.0
        self.audio_quick_finished: bool = False
        self.audio_quick_aborted: bool = False
        self.tts_quick_finished_event = threading.Event()
        self.abortion_started: bool = False
        self.tts_final_finished_event = threading.Event()
        self.tts_final_started: bool = False
        self.audio_final_aborted: bool = False
        self.audio_final_finished: bool = False
        self.final_answer: str = ""
        self.completed: bool = False

class SpeechPipelineManager:
    def __init__(
            self,
            tts_engine: str = "coqui",
            llm_provider: str = "openai",
            llm_model: str = "gpt-4o",
            no_think: bool = False,
            orpheus_model: str = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf",
        ):
        self.tts_engine = tts_engine
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.no_think = no_think
        self.orpheus_model = orpheus_model

        self.audio = AudioProcessor(engine=self.tts_engine, orpheus_model=self.orpheus_model)
        self.audio.on_first_audio_chunk_synthesize = self.on_first_audio_chunk_synthesize
        self.text_similarity = TextSimilarity(focus='end', n_words=5)
        self.text_context = TextContext()
        self.generation_counter: int = 0
        self.abort_lock = threading.Lock()
        
        self.llm = None
        self.rag_chain_ai1 = None
        self.rag_chain_ai2 = None
        self.orchestrator_chain = None
        self._setup_chains()

        self.llm.prewarm()
        self.llm_inference_time = self.llm.measure_inference_time()
        logger.debug(f"🗣️🧠🕒 LLM 추론 시간: {self.llm_inference_time:.2f}ms")

        self.history = []
        self.requests_queue = Queue()
        self.running_generation: Optional[RunningGeneration] = None
        self.shutdown_event = threading.Event()
        self.generator_ready_event = threading.Event()
        self.llm_answer_ready_event = threading.Event()
        self.stop_everything_event = threading.Event()
        self.stop_llm_request_event = threading.Event()
        self.stop_llm_finished_event = threading.Event()
        self.stop_tts_quick_request_event = threading.Event()
        self.stop_tts_quick_finished_event = threading.Event()
        self.stop_tts_final_request_event = threading.Event()
        self.stop_tts_final_finished_event = threading.Event()
        self.abort_completed_event = threading.Event()
        self.abort_block_event = threading.Event()
        self.abort_block_event.set()
        self.check_abort_lock = threading.Lock()
        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        self.previous_request = None

        self.request_processing_thread = threading.Thread(target=self._request_processing_worker, name="RequestProcessingThread", daemon=True)
        self.llm_inference_thread = threading.Thread(target=self._llm_inference_worker, name="LLMProcessingThread", daemon=True)
        self.tts_quick_inference_thread = threading.Thread(target=self._tts_quick_inference_worker, name="TTSQuickProcessingThread", daemon=True)
        self.tts_final_inference_thread = threading.Thread(target=self._tts_final_inference_worker, name="TTSFinalProcessingThread", daemon=True)
        self.ai_to_ai_worker_thread = threading.Thread(target=self._ai_to_ai_worker, name="AIToAIWorkerThread", daemon=True)
        
        self.ai_to_ai_active = False
        self.ai_to_ai_paused = False
        self.last_ai_speaker = "AI2_tutor" # AI1이 먼저 시작하도록 초기화
        self.last_ai_response = ""
        self.ai_to_ai_turn_count = 0  # AI-to-AI 대화 횟수 카운터
        self.client_connected = True  # 클라이언트 연결 상태
        self.ai_conversation_interrupted = False  # AI 대화 중단 상태
        self.ai_to_ai_session_limit_reached = False  # 세션별 대화 제한 도달 상태
        
        # 클라이언트 TTS 재생 완료 대기를 위한 콜백 참조
        self.tts_playback_finished_callback = None

        self.request_processing_thread.start()
        self.llm_inference_thread.start()
        self.tts_quick_inference_thread.start()
        self.tts_final_inference_thread.start()
        self.ai_to_ai_worker_thread.start()

        self.on_partial_assistant_text: Optional[Callable[[str], None]] = None
        self.on_final_assistant_answer: Optional[Callable[..., None]] = None
        self.on_ai_turn_start: Optional[Callable[[], None]] = None
        self.on_tts_all_complete: Optional[Callable[[], None]] = None
        self.full_output_pipeline_latency = (self.llm_inference_time or 0) + self.audio.tts_inference_time
        logger.info(f"🗣️⏱️ 전체 출력 파이프라인 지연 시간: {self.full_output_pipeline_latency:.2f}ms (LLM: {self.llm_inference_time or 0:.2f}ms, TTS: {self.audio.tts_inference_time:.2f}ms)")
        logger.info("🗣️🚀 SpeechPipelineManager가 초기화되었고 워커가 시작되었습니다.")

    def toggle_pause_ai_conversation(self):
        """AI 간 대화의 일시중지 상태를 토글합니다."""
        self.ai_to_ai_paused = not self.ai_to_ai_paused
        state = "일시중지됨" if self.ai_to_ai_paused else "재개됨"
        logger.info(f"🤖⏯️ AI 간 대화가 {state} 상태로 변경되었습니다.")
        
        return self.ai_to_ai_paused

    def set_client_connection_status(self, connected: bool):
        """클라이언트 연결 상태를 업데이트합니다."""
        self.client_connected = connected
        if not connected:
            logger.info("🔌❌ 클라이언트 연결이 끊어졌습니다. AI-to-AI 대화 중단을 위해 대기 중...")
            if self.ai_to_ai_active:
                self.ai_to_ai_active = False
                self.ai_conversation_interrupted = True
                logger.info("🤖🛑 클라이언트 연결 끊김으로 AI-to-AI 대화를 중단했습니다.")
        else:
            logger.info("🔌✅ 클라이언트가 연결되었습니다.")
            # 연결 끊김 알림 기능
            if DISCONN_ALARM and self.ai_conversation_interrupted:
                self.send_disconnection_notification()
                self.ai_conversation_interrupted = False

    def send_disconnection_notification(self):
        """연결 끊김으로 인한 AI 대화 중단 알림을 전송합니다."""
        if self.on_final_assistant_answer:
            logger.info("🔔 연결 끊김 알림 전송 중...")
            notification_message = "AI 대화가 연결 끊김으로 인해 중단되었습니다."
            
            # 히스토리에 알림 메시지 추가
            self.history.append({
                "role": "system",
                "content": notification_message
            })
            
            # 클라이언트에게 알림 전송
            self.on_final_assistant_answer(
                forced=True,
                custom_message=notification_message,
                speaker_role_override="system"
            )

    def handle_prolonged_silence(self):
        """사용자의 긴 침묵을 처리하고 AI 간 대화를 시작합니다."""
        if self.ai_to_ai_active or self.is_valid_gen():
            logger.info("🤖💬 긴 침묵 감지됨, 하지만 AI 대화가 이미 활성화되어 있거나 생성이 진행 중이므로 무시합니다.")
            return
        elif self.ai_to_ai_paused:
            logger.info("AI-to-AI 기능 일시정지 중이므로 장시간 침묵 타이머를 무시합니다.")
            return
        elif self.ai_to_ai_session_limit_reached:
            logger.info("🤖📊 이 세션에서 AI-to-AI 대화 제한에 이미 도달했으므로 침묵 타이머를 무시합니다.")
            return
        
        logger.info("🤖💬 긴 침묵으로 인해 AI 간 대화 시작 중...")
        
        # 대화 기록에서 마지막 메시지를 가져와 주제로 사용
        if self.history:
            last_message = self.history[-1]
            # 마지막 메시지의 내용을 주제/입력으로 사용
            topic = last_message.get("content", "What do you think about that?")
        else:
            # 기록이 비어있을 경우의 대체 주제
            topic = "Let's talk about something interesting."
            
        self.start_ai_to_ai_conversation(topic)

    def start_ai_to_ai_conversation(self, topic: str, manual_start: bool = False):
        logger.info(f"🤖💬 AI 간 대화 시작 (주제: {topic})")
        self.ai_to_ai_active = True
        self.ai_to_ai_paused = False # 대화 시작 시 항상 일시중지 해제
        self.ai_to_ai_turn_count = 0  # 대화 시작 시 카운터 리셋
        
        # 수동 시작의 경우 세션 제한 리셋
        if manual_start:
            self.ai_to_ai_session_limit_reached = False
            logger.info("🤖🔄 수동 시작으로 인해 세션 제한 리셋됨")
            
        self.prepare_generation(topic, is_ai_turn=True)
        
    def wait_for_client_tts_finish(self, timeout: float = 10.0) -> bool:
        """클라이언트 TTS 재생 완료를 대기합니다."""
        if self.tts_playback_finished_callback:
            return self.tts_playback_finished_callback.tts_playback_finished_event.wait(timeout)
        return False

    def _ai_to_ai_worker(self):
        logger.info("🤖💬 AI-to-AI 워커 시작 중...")
        while not self.shutdown_event.is_set():
            if not self.ai_to_ai_active:
                time.sleep(0.1)
                continue

            # --- 연결 상태 체크 ---
            if not self.client_connected:
                logger.info("🤖🔌 클라이언트 연결 끊김으로 AI-to-AI 대화 중단")
                self.ai_to_ai_active = False
                continue
            
            
            # --- 일시중지 로직 ---
            while self.ai_to_ai_paused:
                if self.shutdown_event.is_set() or not self.ai_to_ai_active:
                    break
                time.sleep(0.1)
            # --------------------

            # TTS 재생 완료 대기를 위한 콜백 확인
            if self.tts_playback_finished_callback and self.tts_playback_finished_callback.tts_playback_finished_event.wait(timeout=0.1):
                if not self.ai_to_ai_active:
                    continue
                
                logger.info("🤖💬 클라이언트 TTS 재생 완료 신호 수신됨")
                
                # 이벤트 클리어 및 다음 턴 처리
                self.tts_playback_finished_callback.tts_playback_finished_event.clear()
 
                # AI-to-AI 상태 재확인 (사용자 interrupt 방지)
                if self.ai_to_ai_active:
                    # 다음 반복 전에 일시중지 상태 다시 확인
                    if self.ai_to_ai_paused:
                        continue
                    
                    # AI 턴 카운터 증가 (이전 AI 턴이 완료되었으므로)
                    self.ai_to_ai_turn_count += 1
                    logger.info(f"🤖📊 AI-to-AI 턴 카운터: {self.ai_to_ai_turn_count}")
                    
                    # 턴 카운터 증가 후 즉시 제한 확인
                    if MAXIMUM_AI_TO_AI > 0 and self.ai_to_ai_turn_count >= MAXIMUM_AI_TO_AI:
                        logger.info(f"🤖📊 AI-to-AI 대화 횟수 제한 도달 ({self.ai_to_ai_turn_count}/{MAXIMUM_AI_TO_AI})")
                        self.ai_to_ai_active = False
                        self.ai_to_ai_session_limit_reached = True
                        
                        # 제한 도달 알림 전송
                        if self.on_final_assistant_answer:
                            limit_message = f"AI 대화가 최대 횟수 제한({MAXIMUM_AI_TO_AI}회)에 도달하여 종료되었습니다. 이 세션에서는 더 이상 AI 대화가 자동으로 시작되지 않습니다."
                            self.history.append({
                                "role": "system",
                                "content": limit_message
                            })
                            self.on_final_assistant_answer(
                                forced=True,
                                speaker_role_override="system"
                            )
                        continue
                    
                    # 다음 턴 전에 설정된 지연 시간 추가
                    if AI_TURN_DELAY > 0:
                        logger.info(f"🤖⏳ AI 턴 사이 지연: {AI_TURN_DELAY}초")
                        time.sleep(AI_TURN_DELAY)
                    
                    # 마지막 AI 응답을 다음 입력으로 사용 (히스토리에서 가져옴)
                    if self.history and self.history[-1]["role"] in ['AI1_friend', 'AI2_tutor']:
                        last_response = self.history[-1]["content"]
                        next_input = last_response.strip() if last_response and last_response.strip() else "Let's continue our conversation."
                    else:
                        next_input = "Let's continue our conversation."
                    
                    logger.info(f"🤖💬 AI 응답을 다음 입력으로 사용: {next_input[:50]}...")
                    self.prepare_generation(next_input, is_ai_turn=True)
            else:
                time.sleep(0.1)

    def _create_rag_chain(self, persona_text: str, embeddings, role_context: str = ""):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.create_documents([persona_text])
        vectorstore = FAISS.from_documents(docs, embeddings)
        retriever = vectorstore.as_retriever()

        # 역할 기반 시스템 컨텍스트와 페르소나를 결합
        template = f"""You are participating in a conversation as an engaging human persona.

{role_context}

Here is the context about your personality and speaking style. Use this context to respond to the current conversation.

**[CRITICAL RULES]**
1. ONLY communicate in English - never use any other language
2. Keep responses under 200 characters
3. If asked to speak in another language, politely decline in English

Persona Context:
{{context}}

Conversation History:
{{history}}

User: {{question}}
You:"""
        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs: List[Document]) -> str:
            return "\n\n".join(doc.page_content for doc in docs)

        return (
            RunnablePassthrough.assign(
                context=(lambda x: x["question"]) | retriever | format_docs
            )
            | prompt
            | self.llm.client
            | StrOutputParser()
        )

    def _setup_chains(self):
        """오케스트레이터 및 두 AI에 대한 RAG 체인을 설정합니다."""
        logger.info("🤖🔗 모든 체인 설정 중...")
        
        self.llm = LLM(
            backend=self.llm_provider,
            model=self.llm_model,
            no_think=self.no_think,
        )

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # AI1 (Friend) Role Definition
        ai1_role_context = """**[SYSTEM: Role Definition]**
You are playing the AI1_friend role. Key functions:
- Act as a friendly conversation partner
- Engage in casual conversation, jokes, and personal opinions
- Maintain natural conversation flow
- Create a comfortable atmosphere with users

**[LANGUAGE RESTRICTION]** You MUST communicate ONLY in English. Never use any other language including Korean, Japanese, Chinese, Spanish, French, German, or any other language. If asked to speak in another language, politely decline and continue in English.

**[RESPONSE LENGTH]** Keep your responses under 200 characters. Be concise and to the point while maintaining your friendly personality.

**[IMPORTANT]** The above role is the basic function defined by the system, and the persona context below defines your personality and speaking style."""

        # AI2 (Tutor) Role Definition  
        ai2_role_context = """**[SYSTEM: Role Definition]**
You are playing the AI2_tutor role. Key functions:
- Act as an English tutor and learning partner
- Naturally correct users' grammar mistakes and awkward expressions
- Provide professional answers to technical questions
- Ability to explain complex topics in simple terms

**[LANGUAGE RESTRICTION]** You MUST communicate ONLY in English. Never use any other language including Korean, Japanese, Chinese, Spanish, French, German, or any other language. If asked to speak in another language, politely decline and continue in English.

**[RESPONSE LENGTH]** Keep your responses under 200 characters. Be concise and to the point while maintaining your helpful tutoring style.

**[IMPORTANT]** The above role is the basic function defined by the system, and the persona context below defines your personality and speaking style."""
        
        logger.info("🤖🔗 AI1 (친구) RAG 체인 생성 중...")
        self.rag_chain_ai1 = self._create_rag_chain(persona_ai1, embeddings, ai1_role_context)
        
        logger.info("🤖🔗 AI2 (튜터) RAG 체인 생성 중...")
        self.rag_chain_ai2 = self._create_rag_chain(persona_ai2, embeddings, ai2_role_context)

        orchestrator_template = """You are an orchestrator that controls the flow of conversation. Based on the conversation history and user input below, decide who should speak next. You must respond with only "AI1_friend" or "AI2_tutor".\n\n- "AI1_friend" is a friendly conversation partner.\n- "AI2_tutor" is a smart friend who corrects grammar and expressions.\n\nConversation History:\n{history}\n\nUser: {question}\nNext speaker:"""
        orchestrator_prompt = ChatPromptTemplate.from_template(orchestrator_template)
        self.orchestrator_chain = orchestrator_prompt | self.llm.client | StrOutputParser()
        
        logger.info("🤖🔗✅ 모든 체인이 성공적으로 설정되었습니다.")

    def process_prepare_generation(self, data: dict):
        txt = data.get("text", "")
        is_ai_turn = data.get("is_ai_turn", False)

        # AI 턴이 아닐 때만 사용자 입력으로 간주하고 AI 대화 중단
        if not is_ai_turn:
            self.ai_to_ai_active = False
            # 사용자 입력 시 AI 대화 카운터 및 세션 제한 리셋
            self.ai_to_ai_turn_count = 0
            self.ai_to_ai_session_limit_reached = False
            logger.info("🤖🔄 사용자 입력으로 인해 AI 대화 카운터 및 세션 제한 리셋됨")
        
        id_in_spec = self.generation_counter + 1
        
        # AI 턴이든 사용자 턴이든, 새로운 생성을 시작하기 전에 항상 이전 생성을 중단/정리합니다.
        self.check_abort(txt, wait_for_finish=True, abort_reason=f"process_prepare_generation for new id {id_in_spec}")

        self.generation_counter += 1
        new_gen_id = self.generation_counter
        logger.info(f"🗣️✨🔄 [Gen {new_gen_id}] 새 생성 준비 중: '{txt[:50]}...' ")

        # ... (상태 리셋 로직은 동일)
        self.llm_generation_active = False
        self.tts_quick_generation_active = False
        self.tts_final_generation_active = False
        self.llm_answer_ready_event.clear()
        self.generator_ready_event.clear()
        self.stop_llm_request_event.clear()
        self.stop_llm_finished_event.clear()
        self.stop_tts_quick_request_event.clear()
        self.stop_tts_quick_finished_event.clear()
        self.stop_tts_final_request_event.clear()
        self.stop_tts_final_finished_event.clear()
        self.abort_completed_event.clear()
        self.abort_block_event.set()

        self.running_generation = RunningGeneration(id=new_gen_id)
        self.running_generation.text = txt

        if self.ai_to_ai_active and self.on_ai_turn_start:
            self.on_ai_turn_start()

        try:
            if is_ai_turn:
                # AI 간 대화: 마지막 발언자를 기반으로 다음 발언자 결정
                if self.last_ai_speaker == "AI1_friend":
                    next_speaker = "AI2_tutor"
                else:
                    next_speaker = "AI1_friend"
                logger.info(f"🤖🗣️ AI 턴 전환: {self.last_ai_speaker} -> {next_speaker}")
            else:
                # 사용자 턴: 오케스트레이터 호출
                logger.info(f"🗣️🧠🚀 [Gen {new_gen_id}] 오케스트레이터 호출 중...")
                formatted_history = "\n".join([f"{h['role']}: {h['content']}" for h in self.history])
                orchestrator_input = {"question": txt, "history": formatted_history}
                
                next_speaker = self.orchestrator_chain.invoke(orchestrator_input)
                logger.info(f"🤖🗣️ 오케스트레이터 결정: {next_speaker}")

            # It's possible that the generation was aborted while waiting for the orchestrator.
            # Check if self.running_generation is still valid and for the current generation id.
            if not self.is_valid_gen() or self.running_generation.id != new_gen_id:
                logger.warning(f"🗣️🔄 [Gen {new_gen_id}] 오케스트레이터 호출 후 생성이 중단되거나 변경되었습니다. 이 생성을 중단합니다.")
                return

            # 결정된 스피커에 따라 체인과 목소리 선택
            if "AI1_friend" in next_speaker:
                active_chain = self.rag_chain_ai1
                voice_id = "9BWtsMINqrJLrRacOk9x" # Aria
                # voice_id = "uyVNoMrnUku1dZyVEXwD" # Anna kim (테스트용 한국어 모델)
                voice = "mia"   # 오르페우스 모델용
                self.running_generation.speaker_role = "AI1_friend"
                self.last_ai_speaker = "AI1_friend"
            elif "AI2_tutor" in next_speaker:
                active_chain = self.rag_chain_ai2
                voice_id = "JBFqnCBsd6RMkjVDRZzb" # George
                # voice_id = "jB1Cifc2UQbq1gR3wnb0" # Bin (테스트용 한국어 모델)
                voice = "leo"   # 오르페우스 모델용
                self.running_generation.speaker_role = "AI2_tutor"
                self.last_ai_speaker = "AI2_tutor"
            else:
                logger.warning(f"🤖❓ 알 수 없는 스피커 결정: {next_speaker}. 기본값(AI1) 사용.")
                active_chain = self.rag_chain_ai1
                voice_id = "JBFqnCBsd6RMkjVDRZzb" # George
                voice = "mia"   # 오르페우스 모델용
                self.running_generation.speaker_role = "AI1_friend"
                self.last_ai_speaker = "AI1_friend"

            if self.tts_engine == "elabs":
                self.audio.set_voice(voice_id)
                logger.info(f"🎤🔊 ElevenLabs 목소리 변경: {voice_id}")
            elif self.tts_engine == "orpheus":
                self.audio.set_voice(voice)
                logger.info(f"🎤🔊 ElevenLabs 목소리 변경: {voice}")
            else:
                # Coqui와 같은 다른 엔진을 위한 기존 로직
                voice_path = "./voices/coqui_Daisy Studious.wav" if "AI1_friend" in next_speaker else "./voices/thsama.wav"
                self.audio.set_voice(voice_path)
                logger.info(f"🎤🔊 목소리 변경: {voice_path}")

            formatted_history = "\n".join([f"{h['role']}: {h['content']}" for h in self.history])
            input_data = {"question": txt, "history": formatted_history}
            self.running_generation.llm_generator = active_chain.stream(input_data)
            
            logger.info(f"🗣️🧠✔️ [Gen {new_gen_id}] LLM 생성기 생성됨. 생성기 준비 이벤트 설정 중.")
            self.generator_ready_event.set()
        except Exception as e:
            logger.exception(f"🗣️🧠💥 [Gen {new_gen_id}] LLM 생성기 생성 실패: {e}")
            self.running_generation = None

    # ... (나머지 코드는 이전과 거의 동일하게 유지)
    def is_valid_gen(self) -> bool:
        return self.running_generation is not None and not self.running_generation.abortion_started

    def _request_processing_worker(self):
        logger.info("🗣️🚀 요청 처리기: 시작 중...")
        while not self.shutdown_event.is_set():
            try:
                request = self.requests_queue.get(block=True, timeout=1)
                if self.previous_request:
                    if self.previous_request.data == request.data and isinstance(request.data, str):
                        if request.timestamp - self.previous_request.timestamp < 2:
                            logger.info(f"🗣️🗑️ 요청 처리기: 중복 요청 건너뛰기 - {request.action}")
                            continue
                while not self.requests_queue.empty():
                    skipped_request = self.requests_queue.get(False)
                    logger.debug(f"🗣️🗑️ 요청 처리기: 오래된 요청 건너뛰기 - {skipped_request.action}")
                    request = skipped_request
                self.abort_block_event.wait()
                logger.debug(f"🗣️🔄 요청 처리기: 가장 최근 요청 처리 중 - {request.action}")
                if request.action == "prepare":
                    self.process_prepare_generation(request.data)
                    self.previous_request = request
                elif request.action == "finish":
                     logger.info(f"🗣️🤷 요청 처리기: 'finish' 작업 수신 (현재 아무 작업 안 함).")
                     self.previous_request = request
                else:
                    logger.warning(f"🗣️❓ 요청 처리기: 알 수 없는 작업 '{request.action}'")
            except Empty:
                continue
            except Exception as e:
                logger.exception(f"🗣️💥 요청 처리기: 오류: {e}")
        logger.info("🗣️🏁 요청 처리기: 종료 중.")

    def on_first_audio_chunk_synthesize(self):
        logger.info("🗣️🎶 첫 번째 오디오 청크가 합성되었습니다. TTS 빠른 답변 허용 이벤트를 설정합니다.")
        if self.running_generation:
            self.running_generation.quick_answer_first_chunk_ready = True

    def preprocess_chunk(self, chunk: str) -> str:
        if isinstance(chunk, dict):
            chunk = chunk.get('answer', '')
        return chunk.replace("—", "-").replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'").replace("…", "...")

    def clean_quick_answer(self, text: str) -> str:
        patterns_to_remove = ["<think>", "</think>", "\n", " "]
        previous_text = None
        current_text = text
        while previous_text != current_text:
            previous_text = current_text
            for pattern in patterns_to_remove:
                while current_text.startswith(pattern):
                    current_text = current_text[len(pattern):]
        return current_text

    def _llm_inference_worker(self):
        logger.info("🗣️🧠 LLM 워커: 시작 중...")
        while not self.shutdown_event.is_set():
            ready = self.generator_ready_event.wait(timeout=1.0)
            if not ready:
                continue
            if self.stop_llm_request_event.is_set():
                logger.info("🗣️🧠❌ LLM 워커: generator_ready_event 대기 중 중단 감지됨.")
                self.stop_llm_request_event.clear()
                self.stop_llm_finished_event.set()
                self.llm_generation_active = False
                continue
            self.generator_ready_event.clear()
            self.stop_everything_event.clear()
            current_gen = self.running_generation
            if not current_gen or not current_gen.llm_generator:
                logger.warning("🗣️🧠❓ LLM 워커: 이벤트 후 유효한 생성 또는 생성기를 찾을 수 없습니다.")
                self.llm_generation_active = False
                continue
            gen_id = current_gen.id
            logger.info(f"🗣️🧠🔄 [Gen {gen_id}] LLM 워커: 생성 처리 중...")
            self.llm_generation_active = True
            self.stop_llm_finished_event.clear()
            start_time = time.time()
            token_count = 0
            try:
                for chunk in current_gen.llm_generator:
                    if self.stop_llm_request_event.is_set():
                        logger.info(f"🗣️🧠❌ [Gen {gen_id}] LLM 워커: 반복 중 중지 요청 감지됨.")
                        self.stop_llm_request_event.clear()
                        current_gen.llm_aborted = True
                        break
                    
                    processed_chunk = self.preprocess_chunk(chunk)
                    token_count += 1
                    current_gen.quick_answer += processed_chunk

                    if self.no_think:
                        current_gen.quick_answer = self.clean_quick_answer(current_gen.quick_answer)
                    if token_count == 1:
                        logger.info(f"🗣️🧠⏱️ [Gen {gen_id}] LLM 워커: TTFT: {(time.time() - start_time):.4f}s")
                    if not current_gen.quick_answer_provided:
                        context, overhang = self.text_context.get_context(current_gen.quick_answer)
                        if context:
                            logger.info(f"🗣️🧠✔️ [Gen {gen_id}] LLM 워커:  {Colors.apply('빠른 답변 발견:').magenta} {context}, 나머지: {overhang}")
                            current_gen.quick_answer = context
                            if self.on_partial_assistant_text:
                                self.on_partial_assistant_text(current_gen.quick_answer)
                            current_gen.quick_answer_overhang = overhang
                            current_gen.quick_answer_provided = True
                            self.llm_answer_ready_event.set()
                            break
                logger.info(f"🗣️🧠🏁 [Gen {gen_id}] LLM 워커: 생성기 루프 완료%s" % (" (중단됨)" if current_gen.llm_aborted else ""))
                if not current_gen.llm_aborted and not current_gen.quick_answer_provided:
                    logger.info(f"🗣️🧠✔️ [Gen {gen_id}] LLM 워커: 컨텍스트 경계를 찾지 못해 전체 응답을 빠른 답변으로 사용합니다.")
                    current_gen.quick_answer_provided = True
                    if self.on_partial_assistant_text:
                        self.on_partial_assistant_text(current_gen.quick_answer)
                    self.llm_answer_ready_event.set()
            except Exception as e:
                logger.exception(f"🗣️🧠💥 [Gen {gen_id}] LLM 워커: 생성 중 오류: {e}")
                current_gen.llm_aborted = True
            finally:
                self.llm_generation_active = False
                self.stop_llm_finished_event.set()
                if current_gen.llm_aborted:
                    logger.info(f"🗣️🧠❌ [Gen {gen_id}] LLM 중단됨, TTS 빠른/최종 중지 요청.")
                    self.stop_tts_quick_request_event.set()
                    self.stop_tts_final_request_event.set()
                    self.llm_answer_ready_event.set()
                logger.info(f"🗣️🧠🏁 [Gen {gen_id}] LLM 워커: 처리 주기 완료.")
                current_gen.llm_finished = True
                current_gen.llm_finished_event.set()

    def check_abort(self, txt: str, wait_for_finish: bool = True, abort_reason: str = "unknown") -> bool:
        with self.check_abort_lock:
            if self.running_generation:
                current_gen_id_str = f"Gen {self.running_generation.id}"
                logger.info(f"🗣️🛑❓ {current_gen_id_str} 중단 확인 요청됨 (이유: {abort_reason})")
                if self.running_generation.abortion_started:
                    logger.info(f"🗣️🛑⏳ {current_gen_id_str} 활성 생성이 이미 중단 중이므로, 완료될 때까지 대기합니다 (요청된 경우).")
                    if wait_for_finish:
                        completed = self.abort_completed_event.wait(timeout=5.0)
                        if not completed:
                             logger.error(f"🗣️🛑💥💥 {current_gen_id_str} 진행 중인 중단 완료 대기 시간 초과. 상태 불일치 가능성!")
                             self.running_generation = None
                        elif self.running_generation is not None:
                            logger.error(f"🗣️🛑💥💥 {current_gen_id_str} 중단 완료 이벤트가 설정되었지만 running_generation이 여전히 존재합니다. 상태 불일치 가능성 높음!")
                            self.running_generation = None
                        else:
                            logger.info(f"🗣️🛑✅ {current_gen_id_str} 진행 중인 중단 완료.")
                    else:
                        logger.info(f"🗣️🛑🏃 {current_gen_id_str} wait_for_finish=False이므로 진행 중인 중단을 기다리지 않습니다.")
                    return True
                else:
                    logger.info(f"🗣️🛑🤔 {current_gen_id_str} 활성 생성 발견, 텍스트 유사성 확인 중.")
                    try:
                        if self.running_generation.text is None:
                            logger.warning(f"🗣️🛑❓ {current_gen_id_str} 실행 중인 생성 텍스트가 None이므로 유사성을 비교할 수 없습니다. 다르다고 가정합니다.")
                            similarity = 0.0
                        else:
                            similarity = self.text_similarity.calculate_similarity(self.running_generation.text, txt)
                    except Exception as e:
                        logger.warning(f"🗣️🛑💥 {current_gen_id_str} 유사성 계산 오류: {e}. 다르다고 가정합니다.")
                        similarity = 0.0
                    if similarity >= 0.95:
                        logger.info(f"🗣️🛑🙅 {current_gen_id_str} 텍스트('{txt[:30]}...')가 현재 '{self.running_generation.text[:30] if self.running_generation.text else 'None'}...'와 너무 유사함({similarity:.2f}). 무시합니다.")
                        return False
                    logger.info(f"🗣️🛑🚀 {current_gen_id_str} 텍스트('{txt[:30]}...')가 '{self.running_generation.text[:30] if self.running_generation.text else 'None'}...'와 충분히 다름({similarity:.2f}). 동기적 중단 요청.")
                    start_time = time.time()
                    self.abort_generation(wait_for_completion=wait_for_finish, timeout=7.0, reason=f"check_abort가 다른 텍스트 발견 ({abort_reason})")
                    if wait_for_finish:
                        if self.running_generation is not None:
                            logger.error(f"🗣️🛑💥💥 {current_gen_id_str} !!! 중단 호출이 완료되었지만 running_generation이 여전히 None이 아닙니다. 상태 불일치 가능성 높음!")
                            self.running_generation = None
                        else:
                            logger.info(f"🗣️🛑✅ {current_gen_id_str} 동기적 중단이 {time.time() - start_time:.2f}s 내에 완료되었습니다.")
                    return True
            else:
                logger.info("🗣️🛑🤷 중단 확인 중 활성 생성을 찾을 수 없습니다.")
                return False

    def _tts_quick_inference_worker(self):
        logger.info("🗣️👄🚀 빠른 TTS 워커: 시작 중...")
        while not self.shutdown_event.is_set():
            ready = self.llm_answer_ready_event.wait(timeout=1.0)
            if not ready:
                continue
            if self.stop_tts_quick_request_event.is_set():
                logger.info("🗣️👄❌ 빠른 TTS 워커: llm_answer_ready_event 대기 중 중단 감지됨.")
                self.stop_tts_quick_request_event.clear()
                self.stop_tts_quick_finished_event.set()
                self.tts_quick_generation_active = False
                continue
            self.llm_answer_ready_event.clear()
            current_gen = self.running_generation
            if not current_gen or not current_gen.quick_answer:
                logger.warning("🗣️👄❓ 빠른 TTS 워커: 이벤트 후 유효한 생성 또는 빠른 답변을 찾을 수 없습니다.")
                self.tts_quick_generation_active = False
                continue
            if current_gen.audio_quick_aborted or current_gen.abortion_started:
                logger.info(f"🗣️👄❌ [Gen {current_gen.id}] 빠른 TTS 워커: 생성이 이미 중단됨으로 표시되었습니다. 건너뜁니다.")
                continue
            gen_id = current_gen.id
            logger.info(f"🗣️👄🔄 [Gen {gen_id}] 빠른 TTS 워커: 빠른 답변에 대한 TTS 처리 중...")
            self.tts_quick_generation_active = True
            self.stop_tts_quick_finished_event.clear()
            current_gen.tts_quick_finished_event.clear()
            current_gen.tts_quick_started = True
            allowed_to_speak = True
            try:
                if self.stop_tts_quick_request_event.is_set() or current_gen.abortion_started:
                     logger.info(f"🗣️👄❌ [Gen {gen_id}] 빠른 TTS 워커: 중지 요청 또는 중단 플래그로 인해 TTS 합성을 중단합니다.")
                     current_gen.audio_quick_aborted = True
                else:
                    logger.info(f"🗣️👄🎶 [Gen {gen_id}] 빠른 TTS 워커: 합성 중: '{current_gen.quick_answer[:50]}...' ")
                    completed = self.audio.synthesize(
                        current_gen.quick_answer,
                        current_gen.audio_chunks,
                        self.stop_tts_quick_request_event,
                        generation_obj=current_gen
                    )
                    if not completed:
                        logger.info(f"🗣️👄❌ [Gen {gen_id}] 빠른 TTS 워커: 이벤트로 합성 중지됨.")
                        current_gen.audio_quick_aborted = True
                    else:
                        logger.info(f"🗣️👄✅ [Gen {gen_id}] 빠른 TTS 워커: 합성이 성공적으로 완료되었습니다.")
            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] 빠른 TTS 워커: 합성 중 오류: {e}")
                current_gen.audio_quick_aborted = True
            finally:
                self.tts_quick_generation_active = False
                self.stop_tts_quick_finished_event.set()
                logger.info(f"🗣️👄🏁 [Gen {gen_id}] 빠른 TTS 워커: 처리 주기 완료.")
                if current_gen.audio_quick_aborted or self.stop_tts_quick_request_event.is_set():
                    logger.info(f"🗣️👄❌ [Gen {gen_id}] 빠른 TTS가 중단/미완료로 표시됨.")
                    self.stop_tts_quick_request_event.clear()
                    current_gen.audio_quick_aborted = True
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] 빠른 TTS가 성공적으로 완료됨.")
                    current_gen.tts_quick_finished_event.set()
                current_gen.audio_quick_finished = True

    def _tts_final_inference_worker(self):
        logger.info("🗣️👄🚀 최종 TTS 워커: 시작 중...")
        while not self.shutdown_event.is_set():
            current_gen = self.running_generation
            time.sleep(0.01)
            if not current_gen: continue
            if current_gen.tts_final_started: continue
            if not current_gen.tts_quick_started: continue
            if not current_gen.audio_quick_finished: continue
            gen_id = current_gen.id
            if current_gen.audio_quick_aborted:
                continue
            if not current_gen.quick_answer_provided:
                 logger.debug(f"🗣️👄🙅 [Gen {gen_id}] 최종 TTS 워커: 빠른 답변 경계를 찾지 못해 최종 TTS를 건너뜁니다 (빠른 TTS가 모든 것을 처리함).")
                 continue
            if current_gen.abortion_started:
                 logger.debug(f"🗣️👄🙅 [Gen {gen_id}] 최종 TTS 워커: 생성이 중단 중이므로 최종 TTS를 건너뜁니다.")
                 continue
            logger.info(f"🗣️👄🔄 [Gen {gen_id}] 최종 TTS 워커: 최종 TTS 처리 중...")
            def get_generator():
                if current_gen.quick_answer_overhang:
                    preprocessed_overhang = self.preprocess_chunk(current_gen.quick_answer_overhang)
                    logger.debug(f"🗣️👄< [Gen {gen_id}] 최종 TTS 생성기: 나머지 부분 생성 중: '{preprocessed_overhang[:50]}...' ")
                    current_gen.final_answer += preprocessed_overhang
                    if self.on_partial_assistant_text:
                         logger.debug(f"🗣️👄< [Gen {gen_id}] 최종 TTS 워커 on_partial_assistant_text: 나머지 부분 전송 중.")
                         try:
                            self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                         except Exception as cb_e:
                             logger.warning(f"🗣️💥 on_partial_assistant_text 콜백 오류 (나머지 부분): {cb_e}")
                    yield preprocessed_overhang
                logger.debug(f"🗣️👄< [Gen {gen_id}] 최종 TTS 생성기: 남은 LLM 청크 생성 중...")
                try:
                    for chunk in current_gen.llm_generator:
                         if self.stop_tts_final_request_event.is_set():
                             logger.info(f"🗣️👄❌ [Gen {gen_id}] 최종 TTS 생성기: LLM 반복 중 중지 요청 감지됨.")
                             current_gen.audio_final_aborted = True
                             break
                         preprocessed_chunk = self.preprocess_chunk(chunk)
                         current_gen.final_answer += preprocessed_chunk
                         if self.on_partial_assistant_text:
                            try:
                                 self.on_partial_assistant_text(current_gen.quick_answer + current_gen.final_answer)
                            except Exception as cb_e:
                                 logger.warning(f"🗣️💥 on_partial_assistant_text 콜백 오류 (최종 청크): {cb_e}")
                         yield preprocessed_chunk
                    logger.debug(f"🗣️👄< [Gen {gen_id}] 최종 TTS 생성기: LLM 청크 반복 완료.")
                except Exception as gen_e:
                     logger.exception(f"🗣️👄💥 [Gen {gen_id}] 최종 TTS 생성기: LLM 생성기 반복 중 오류: {gen_e}")
                     current_gen.audio_final_aborted = True
            self.tts_final_generation_active = True
            self.stop_tts_final_finished_event.clear()
            current_gen.tts_final_started = True
            current_gen.tts_final_finished_event.clear()
            try:
                logger.info(f"🗣️👄🎶 [Gen {gen_id}] 최종 TTS 워커: 남은 텍스트 합성 중...")
                completed = self.audio.synthesize_generator(
                    get_generator(),
                    current_gen.audio_chunks,
                    self.stop_tts_final_request_event,
                    generation_obj=current_gen
                )
                if not completed:
                     logger.info(f"🗣️👄❌ [Gen {gen_id}] 최종 TTS 워커: 이벤트로 합성 중지됨.")
                     current_gen.audio_final_aborted = True
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] 최종 TTS 워커: 합성이 성공적으로 완료되었습니다.")
            except Exception as e:
                logger.exception(f"🗣️👄💥 [Gen {gen_id}] 최종 TTS 워커: 합성 중 오류: {e}")
                current_gen.audio_final_aborted = True
            finally:
                self.tts_final_generation_active = False
                self.stop_tts_final_finished_event.set()
                logger.info(f"🗣️👄🏁 [Gen {gen_id}] 최종 TTS 워커: 처리 주기 완료.")
                if current_gen.audio_final_aborted or self.stop_tts_final_request_event.is_set():
                    logger.info(f"🗣️👄❌ [Gen {gen_id}] 최종 TTS가 중단/미완료로 표시됨.")
                    self.stop_tts_final_request_event.clear()
                    current_gen.audio_final_aborted = True
                else:
                    logger.info(f"🗣️👄✅ [Gen {gen_id}] 최종 TTS가 성공적으로 완료됨.")
                    current_gen.tts_final_finished_event.set()
                    # 모든 TTS 완료 시 콜백 호출
                    if hasattr(self, 'on_tts_all_complete') and self.on_tts_all_complete:
                        try:
                            self.on_tts_all_complete()
                        except Exception as e:
                            logger.error(f"🗣️👄💥 TTS 완료 콜백 오류: {e}")
                current_gen.audio_final_finished = True

    def prepare_generation(self, txt: str, is_ai_turn: bool = False):
        logger.info(f"🗣️📥 'prepare' 요청 대기열에 추가 중: '{txt[:50]}...' (AI 턴: {is_ai_turn})")
        self.requests_queue.put(PipelineRequest("prepare", {"text": txt, "is_ai_turn": is_ai_turn}))

    def finish_generation(self):
        logger.info(f"🗣️📥 'finish' 요청 대기열에 추가 중")
        self.requests_queue.put(PipelineRequest("finish"))

    def abort_generation(self, wait_for_completion: bool = False, timeout: float = 7.0, reason: str = ""):
        if self.shutdown_event.is_set():
            logger.warning("🗣️🔌 종료 진행 중, 중단 요청 무시.")
            return
        gen_id_str = f"Gen {self.running_generation.id}" if self.running_generation else "Gen None"
        logger.info(f"🗣️🛑🚀 'abort' 요청 중 (wait={wait_for_completion}, reason='{reason}') for {gen_id_str}")
        self.process_abort_generation()
        if wait_for_completion:
            logger.info(f"🗣️🛑⏳ 중단 완료 대기 중 (타임아웃={timeout}s)...")
            completed = self.abort_completed_event.wait(timeout=timeout)
            if completed:
                logger.info(f"🗣️🛑✅ 중단 완료 확인됨.")
            else:
                logger.warning(f"🗣️🛑⏱️ 중단 완료 이벤트 대기 시간 초과.")
            self.abort_block_event.set()

    def process_abort_generation(self):
        with self.abort_lock:
            current_gen_obj = self.running_generation
            current_gen_id_str = f"Gen {current_gen_obj.id}" if current_gen_obj else "Gen None"
            if current_gen_obj is None or current_gen_obj.abortion_started:
                if current_gen_obj is None:
                    logger.info(f"🗣️🛑🤷 {current_gen_id_str} 중단할 활성 생성이 없습니다.")
                else:
                    logger.info(f"🗣️🛑⏳ {current_gen_id_str} 중단이 이미 진행 중입니다.")
                self.abort_completed_event.set()
                self.abort_block_event.set()
                return
            logger.info(f"🗣️🛑🚀 {current_gen_id_str} 중단 프로세스 시작 중...")
            current_gen_obj.abortion_started = True
            self.abort_block_event.clear()
            self.abort_completed_event.clear()
            self.stop_everything_event.set()
            aborted_something = False
            is_llm_potentially_active = self.llm_generation_active or self.generator_ready_event.is_set()
            if is_llm_potentially_active:
                logger.info(f"🗣️🛑🧠❌ {current_gen_id_str} - LLM 중지 중...")
                self.stop_llm_request_event.set()
                self.generator_ready_event.set()
                stopped = self.stop_llm_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.info(f"🗣️🛑🧠👍 {current_gen_id_str} LLM 중지 확인 수신됨.")
                    self.stop_llm_finished_event.clear()
                else:
                    logger.warning(f"🗣️🛑🧠⏱️ {current_gen_id_str} LLM 중지 확인 대기 시간 초과.")
                if hasattr(self.llm, 'cancel_generation'):
                    logger.info(f"🗣️🛑🧠🔌 {current_gen_id_str} 외부 LLM cancel_generation 호출 중.")
                    try:
                        self.llm.cancel_generation()
                    except Exception as cancel_e:
                         logger.warning(f"🗣️🛑🧠💥 {current_gen_id_str} 외부 LLM 취소 중 오류: {cancel_e}")
                self.llm_generation_active = False
                aborted_something = True
            else:
                logger.info(f"🗣️🛑🧠📴 {current_gen_id_str} LLM이 비활성 상태로 보여 중지 필요 없음.")
            self.stop_llm_request_event.clear()
            is_tts_quick_potentially_active = self.tts_quick_generation_active or self.llm_answer_ready_event.is_set()
            if is_tts_quick_potentially_active:
                logger.info(f"🗣️🛑👄❌ {current_gen_id_str} 빠른 TTS 중지 중...")
                self.stop_tts_quick_request_event.set()
                self.llm_answer_ready_event.set()
                stopped = self.stop_tts_quick_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.info(f"🗣️🛑👄👍 {current_gen_id_str} 빠른 TTS 중지 확인 수신됨.")
                    self.stop_tts_quick_finished_event.clear()
                else:
                    logger.warning(f"🗣️🛑👄⏱️ {current_gen_id_str} 빠른 TTS 중지 확인 대기 시간 초과.")
                self.tts_quick_generation_active = False
                aborted_something = True
            else:
                logger.info(f"🗣️🛑👄📴 {current_gen_id_str} 빠른 TTS가 비활성 상태로 보여 중지 필요 없음.")
            self.stop_tts_quick_request_event.clear()
            is_tts_final_potentially_active = self.tts_final_generation_active
            if is_tts_final_potentially_active:
                logger.info(f"🗣️🛑👄❌ {current_gen_id_str} 최종 TTS 중지 중...")
                self.stop_tts_final_request_event.set()
                stopped = self.stop_tts_final_finished_event.wait(timeout=5.0)
                if stopped:
                    logger.info(f"🗣️🛑👄👍 {current_gen_id_str} 최종 TTS 중지 확인 수신됨.")
                    self.stop_tts_final_finished_event.clear()
                else:
                    logger.warning(f"🗣️🛑👄⏱️ {current_gen_id_str} 최종 TTS 중지 확인 대기 시간 초과.")
                self.tts_final_generation_active = False
                aborted_something = True
            else:
                logger.info(f"🗣️🛑👄📴 {current_gen_id_str} 최종 TTS가 비활성 상태로 보여 중지 필요 없음.")
            self.stop_tts_final_request_event.clear()
            if hasattr(self.audio, 'stop_playback'):
                logger.info(f"🗣️🛑🔊 {current_gen_id_str} 오디오 재생 중지 요청 중.")
                try:
                    self.audio.stop_playback()
                except Exception as audio_e:
                    logger.warning(f"🗣️🛑🔊💥 {current_gen_id_str} 오디오 재생 중지 오류: {audio_e}")
            if self.running_generation is not None and self.running_generation.id == current_gen_obj.id:
                logger.info(f"🗣️🛑🧹 {current_gen_id_str} 실행 중인 생성 객체 지우는 중.")
                if current_gen_obj.llm_generator and hasattr(current_gen_obj.llm_generator, 'close'):
                    try:
                        logger.info(f"🗣️🛑🧠🔌 {current_gen_id_str} LLM 생성기 스트림 닫는 중.")
                        current_gen_obj.llm_generator.close()
                    except Exception as e:
                        logger.warning(f"🗣️🛑🧠💥 {current_gen_id_str} LLM 생성기 닫기 오류: {e}")
                self.running_generation = None
            elif self.running_generation is not None and self.running_generation.id != current_gen_obj.id:
                 logger.warning(f"🗣️🛑❓ {current_gen_id_str} 불일치: 중단 중 self.running_generation이 변경됨 (현재 Gen {self.running_generation.id}). 현재 참조 지우는 중.")
                 self.running_generation = None
            elif aborted_something:
                logger.info(f"🗣️🛑🤷 {current_gen_id_str} 워커가 중단되었지만 running_generation은 이미 None이었습니다.")
            else:
                logger.info(f"🗣️🛑🤷 {current_gen_id_str} 중단할 활성 항목이 없는 것 같고, running_generation은 None입니다.")
            self.generator_ready_event.clear()
            self.llm_answer_ready_event.clear()
            logger.info(f"🗣️🛑✅ {current_gen_id_str} 중단 처리 완료. 완료 이벤트 설정 및 블록 해제 중.")
            self.abort_completed_event.set()
            self.abort_block_event.set()

    def reset(self):
        logger.info("🗣️🔄 파이프라인 상태 리셋 중...")
        self.ai_to_ai_active = False
        self.abort_generation(wait_for_completion=True, timeout=7.0, reason="reset")
        self.history = []
        logger.info("🗣️🧹 기록이 지워졌습니다. 리셋 완료.")

    def shutdown(self):
        logger.info("🗣️🔌 종료 시작 중...")
        self.shutdown_event.set()
        logger.info("🗣️🔌🛑 스레드 조인 전 최종 중단 시도 중...")
        self.abort_generation(wait_for_completion=True, timeout=3.0, reason="shutdown")
        logger.info("🗣️🔌🔔 대기 중인 스레드를 깨우기 위해 이벤트 신호 중...")
        self.generator_ready_event.set()
        self.llm_answer_ready_event.set()
        self.stop_llm_finished_event.set()
        self.stop_tts_quick_finished_event.set()
        self.stop_tts_final_finished_event.set()
        self.abort_completed_event.set()
        self.abort_block_event.set()
        threads_to_join = [
            (self.request_processing_thread, "요청 처리기"),
            (self.llm_inference_thread, "LLM 워커"),
            (self.tts_quick_inference_thread, "빠른 TTS 워커"),
            (self.tts_final_inference_thread, "최종 TTS 워커"),
        ]
        for thread, name in threads_to_join:
             if thread.is_alive():
                 logger.info(f"🗣️🔌⏳ {name} 조인 중...")
                 thread.join(timeout=5.0)
                 if thread.is_alive():
                     logger.warning(f"🗣️🔌⏱️ {name} 스레드가 깨끗하게 조인되지 않았습니다.")
             else:
                  logger.info(f"🗣️🔌👍 {name} 스레드가 이미 완료되었습니다.")
        logger.info("🗣️🔌✅ 종료 완료.")
        
        # AudioProcessor 정리
        if hasattr(self, 'audio_processor') and self.audio_processor:
            try:
                logger.info("🗣️🧹 AudioProcessor 정리 중...")
                self.audio_processor.cleanup()
            except Exception as e:
                logger.error(f"🗣️💥 AudioProcessor 정리 중 오류: {e}", exc_info=True)