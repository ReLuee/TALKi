# server.py
from queue import Queue, Empty
import logging
from logsetup import setup_logging
setup_logging(logging.INFO)
logger = logging.getLogger(__name__)
if __name__ == "__main__":
    logger.info("🖥️👋 로컬 실시간 음성 채팅에 오신 것을 환영합니다")

from upsample_overlap import UpsampleOverlap
from datetime import datetime
from colors import Colors
import uvicorn
import asyncio
import struct
import json
import time
import threading # SpeechPipelineManager 내부 및 AbortWorker를 위해 스레딩 유지
import sys
import os # 환경 변수 접근을 위해 추가

from typing import Any, Dict, Optional, Callable # docstring의 타입 힌트를 위해 추가
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import HTMLResponse, Response, FileResponse

USE_SSL = True
# TTS_START_ENGINE = "orpheus"
# TTS_START_ENGINE = "kokoro"
TTS_START_ENGINE = "coqui"
# TTS_START_ENGINE = "elabs"
# TTS_ORPHEUS_MODEL = "Orpheus_3B-1BaseGGUF/mOrpheus_3B-1Base_Q4_K_M.gguf"
TTS_ORPHEUS_MODEL = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf"

LLM_START_PROVIDER = "openai"
LLM_START_MODEL = "gpt-4o"
# LLM_START_MODEL = "qwen3:30b-a3b"
# LLM_START_MODEL = "hf.co/bartowski/huihui-ai_Mistral-Small-24B-Instruct-2501-abliterated-GGUF:Q4_K_M"
# LLM_START_PROVIDER = "lmstudio"
# LLM_START_MODEL = "Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q3_K_L.gguf"
NO_THINK = False
DIRECT_STREAM = TTS_START_ENGINE=="orpheus"

if __name__ == "__main__":
    logger.info(f"🖥️⚙️ {Colors.apply('[파라미터]').blue} 시작 엔진: {Colors.apply(TTS_START_ENGINE).blue}")
    logger.info(f"🖥️⚙️ {Colors.apply('[파라미터]').blue} 직접 스트리밍: {Colors.apply('켜짐' if DIRECT_STREAM else '꺼짐').blue}")

# 들어오는 오디오 큐에 허용되는 최대 크기 정의
try:
    MAX_AUDIO_QUEUE_SIZE = int(os.getenv("MAX_AUDIO_QUEUE_SIZE", 50))
    if __name__ == "__main__":
        logger.info(f"🖥️⚙️ {Colors.apply('[파라미터]').blue} 오디오 큐 크기 제한 설정: {Colors.apply(str(MAX_AUDIO_QUEUE_SIZE)).blue}")
except ValueError:
    if __name__ == "__main__":
        logger.warning("🖥️⚠️ 잘못된 MAX_AUDIO_QUEUE_SIZE 환경 변수. 기본값 사용: 50")
    MAX_AUDIO_QUEUE_SIZE = 50


if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

#from handlerequests import LanguageProcessor
#from audio_out import AudioOutProcessor
from audio_in import AudioInputProcessor
from speech_pipeline_manager import SpeechPipelineManager
from colors import Colors

LANGUAGE = "en"
# TTS_FINAL_TIMEOUT = 0.5 # 안정성을 위해 1.0이 필요한지 불확실
TTS_FINAL_TIMEOUT = 1.0 # 안정성을 위해 1.0이 필요한지 불확실

# --------------------------------------------------------------------
# 커스텀 캐시 없음 StaticFiles
# --------------------------------------------------------------------
class NoCacheStaticFiles(StaticFiles):
    """
    클라이언트 측 캐싱을 허용하지 않고 정적 파일을 제공합니다.

    브라우저가 정적 자산을 캐싱하는 것을 방지하는 'Cache-Control' 헤더를 추가하기 위해
    기본 Starlette StaticFiles를 오버라이드합니다. 개발에 유용합니다.
    """
    async def get_response(self, path: str, scope: Dict[str, Any]) -> Response:
        """
        요청된 경로에 대한 응답을 가져오고 캐시 없음 헤더를 추가합니다.

        Args:
            path: 요청된 정적 파일의 경로.
            scope: 요청에 대한 ASGI 스코프 딕셔너리.

        Returns:
            캐시 제어 헤더가 수정된 Starlette 응답 객체.
        """
        response: Response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        # no-store와 함께 반드시 필요한 것은 아닐 수 있지만, 만일을 대비함
        if "etag" in response.headers:
             response.headers.__delitem__("etag")
        if "last-modified" in response.headers:
             response.headers.__delitem__("last-modified")
        return response

# --------------------------------------------------------------------
# 라이프스팬 관리
# --------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    애플리케이션의 라이프스팬을 관리하며, 리소스를 초기화하고 종료합니다.

    SpeechPipelineManager, Upsampler, AudioInputProcessor와 같은 전역 컴포넌트를
    초기화하고 `app.state`에 저장합니다. 종료 시 정리를 처리합니다.

    Args:
        app: FastAPI 애플리케이션 인스턴스.
    """
    logger.info("🖥️▶️ 서버 시작 중")
    # 전역 컴포넌트 초기화, 연결별 상태 아님
    app.state.SpeechPipelineManager = SpeechPipelineManager(
        tts_engine=TTS_START_ENGINE,
        llm_provider=LLM_START_PROVIDER,
        llm_model=LLM_START_MODEL,
        no_think=NO_THINK,
        orpheus_model=TTS_ORPHEUS_MODEL,
    )

    app.state.Upsampler = UpsampleOverlap()
    app.state.AudioInputProcessor = AudioInputProcessor(
        LANGUAGE,
        is_orpheus=TTS_START_ENGINE=="orpheus",
        on_prolonged_silence_callback=app.state.SpeechPipelineManager.handle_prolonged_silence,
        pipeline_latency=app.state.SpeechPipelineManager.full_output_pipeline_latency / 1000, # 초 단위
    )
    app.state.Aborting = False # 이것을 유지할까? 제공된 스니펫에서 사용법이 명확하지 않음. 변경 최소화.

    yield

    logger.info("🖥️⏹️ 서버 종료 중")
    app.state.AudioInputProcessor.shutdown()
    app.state.SpeechPipelineManager.shutdown()


# --------------------------------------------------------------------
# FastAPI 앱 인스턴스
# --------------------------------------------------------------------
app = FastAPI(lifespan=lifespan)

# 필요한 경우 CORS 활성화
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 캐시 없이 정적 파일 마운트
app.mount("/static", NoCacheStaticFiles(directory="static"), name="static")

@app.get("/favicon.ico")
async def favicon():
    """
    favicon.ico 파일을 제공합니다.

    Returns:
        파비콘을 포함하는 FileResponse.
    """
    return FileResponse("static/favicon.ico")

@app.get("/")
async def get_index() -> HTMLResponse:
    """
    메인 index.html 페이지를 제공합니다.

    static/index.html의 내용을 읽어 HTML 응답으로 반환합니다.

    Returns:
        index.html의 내용을 포함하는 HTMLResponse.
    """
    with open("static/index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# --------------------------------------------------------------------
# 유틸리티 함수
# --------------------------------------------------------------------
def parse_json_message(text: str) -> dict:
    """
    JSON 문자열을 딕셔너리로 안전하게 파싱합니다.

    JSON이 유효하지 않으면 경고를 기록하고 빈 딕셔너리를 반환합니다.

    Args:
        text: 파싱할 JSON 문자열.

    Returns:
        파싱된 JSON을 나타내는 딕셔너리, 또는 오류 시 빈 딕셔너리.
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("🖥️⚠️ 잘못된 JSON 형식의 클라이언트 메시지를 무시합니다.")
        return {}

def format_timestamp_ns(timestamp_ns: int) -> str:
    """
    나노초 타임스탬프를 사람이 읽을 수 있는 HH:MM:SS.fff 문자열로 포맷합니다.

    Args:
        timestamp_ns: 에포크 이후의 나노초 단위 타임스탬프.

    Returns:
        시간:분:초.밀리초 형식의 문자열.
    """
    # 전체 초와 나노초 나머지로 분리
    seconds = timestamp_ns // 1_000_000_000
    remainder_ns = timestamp_ns % 1_000_000_000

    # 초 부분을 datetime 객체로 변환 (로컬 시간)
    dt = datetime.fromtimestamp(seconds)

    # 주 시간을 HH:MM:SS로 포맷
    time_str = dt.strftime("%H:%M:%S")

    # 예를 들어, 밀리초를 원하면 나머지를 1e6으로 나누고 3자리로 포맷
    milliseconds = remainder_ns // 1_000_000
    formatted_timestamp = f"{time_str}.{milliseconds:03d}"

    return formatted_timestamp

# --------------------------------------------------------------------
# 웹소켓 데이터 처리
# --------------------------------------------------------------------

async def process_incoming_data(ws: WebSocket, app: FastAPI, incoming_chunks: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    웹소켓을 통해 메시지를 수신하고, 오디오 및 텍스트 메시지를 처리합니다.

    바이너리 오디오 청크를 처리하고, 메타데이터(타임스탬프, 플래그)를 추출하여
    오디오 PCM 데이터와 메타데이터를 `incoming_chunks` 큐에 넣습니다.
    큐가 가득 차면 역압력(back-pressure)을 적용합니다.
    텍스트 메시지(JSON으로 가정)를 파싱하고 메시지 유형에 따라 작업을 트리거합니다
    (예: `callbacks`를 통해 클라이언트 TTS 상태 업데이트, 기록 지우기, 속도 설정).

    Args:
        ws: 웹소켓 연결 인스턴스.
        app: FastAPI 애플리케이션 인스턴스 (필요 시 전역 상태에 접근하기 위함).
        incoming_chunks: 처리된 오디오 메타데이터 딕셔너리를 넣을 asyncio 큐.
        callbacks: 이 연결의 상태를 관리하는 TranscriptionCallbacks 인스턴스.
    """
    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"]:
                raw = msg["bytes"]
                logger.debug(f"🖥️🔊 오디오 수신: {len(raw)} 바이트") # STT 진단 로그 1

                # 최소 8바이트 헤더 확인: 4바이트 타임스탬프(ms) + 4바이트 플래그
                if len(raw) < 8:
                    logger.warning("🖥️⚠️ 8바이트 헤더보다 짧은 패킷을 수신했습니다.")
                    continue

                # 빅엔디안 uint32 타임스탬프(ms)와 uint32 플래그 언패킹
                timestamp_ms, flags = struct.unpack("!II", raw[:8])
                client_sent_ns = timestamp_ms * 1_000_000

                # 고정 필드를 사용하여 메타데이터 구성
                metadata = {
                    "client_sent_ms":           timestamp_ms,
                    "client_sent":              client_sent_ns,
                    "client_sent_formatted":    format_timestamp_ns(client_sent_ns),
                    "isTTSPlaying":             bool(flags & 1),
                }

                # 서버 수신 시간 기록
                server_ns = time.time_ns()
                metadata["server_received"] = server_ns
                metadata["server_received_formatted"] = format_timestamp_ns(server_ns)

                # 페이로드의 나머지 부분은 원시 PCM 바이트
                metadata["pcm"] = raw[8:]

                # 데이터 넣기 전 큐 크기 확인
                current_qsize = incoming_chunks.qsize()
                if current_qsize < MAX_AUDIO_QUEUE_SIZE:
                    # 이제 메타데이터 딕셔너리(PCM 오디오 포함)만 처리 큐에 넣음.
                    await incoming_chunks.put(metadata)
                else:
                    # 큐가 가득 참, 청크를 버리고 경고 기록
                    logger.warning(
                        f"🖥️⚠️ 오디오 큐가 가득 찼습니다 ({current_qsize}/{MAX_AUDIO_QUEUE_SIZE}); 청크를 버립니다. 지연이 발생할 수 있습니다."
                    )

            elif "text" in msg and msg["text"]:
                # 텍스트 기반 메시지: JSON 파싱
                data = parse_json_message(msg["text"])
                msg_type = data.get("type")
                logger.info(Colors.apply(f"🖥️📥 ←←클라이언트: {data}").orange)


                if msg_type == "tts_start":
                    logger.info("🖥️ℹ️ 클라이언트로부터 tts_start 수신.")
                    # 콜백을 통해 연결별 상태 업데이트
                    callbacks.tts_client_playing = True
                elif msg_type == "tts_stop":
                    logger.info("🖥️ℹ️ 클라이언트로부터 tts_stop 수신.")
                    # 콜백을 통해 연결별 상태 업데이트
                    callbacks.tts_client_playing = False
                    # AI-to-AI 대화 중이면 다음 턴 시작 신호 전송
                    if app.state.SpeechPipelineManager.ai_to_ai_active:
                        callbacks.on_tts_playback_finished()
                    else:
                        # 일반 AI-사용자 대화에서는 침묵 타이머 시작
                        callbacks.start_silence_timer_after_tts_stop()
                # server.py의 handleJSONMessage 함수에 추가
                elif msg_type == "clear_history":
                    logger.info("🖥️ℹ️ 클라이언트로부터 clear_history 수신.")
                    app.state.SpeechPipelineManager.reset()
                elif msg_type == "set_speed":
                    speed_value = data.get("speed", 0)
                    speed_factor = speed_value / 100.0  # 0-100을 0.0-1.0으로 변환
                    turn_detection = app.state.AudioInputProcessor.transcriber.turn_detection
                    if turn_detection:
                        turn_detection.update_settings(speed_factor)
                        logger.info(f"🖥️⚙️ 발화 전환 감지 설정을 계수 {speed_factor:.2f}(으)로 업데이트했습니다.")
                elif msg_type == "start_ai_conversation":
                    topic = data.get("topic", "a friendly chat about their day")
                    logger.info(f"🖥️🤖 클라이언트로부터 AI 대화 시작 요청 수신 (주제: {topic})")
                    # 이 부분은 아직 구현되지 않았으므로 SpeechPipelineManager에 해당 메서드를 추가해야 합니다.
                    if hasattr(app.state.SpeechPipelineManager, "start_ai_to_ai_conversation"):
                        app.state.SpeechPipelineManager.start_ai_to_ai_conversation(topic, manual_start=True)
                    else:
                        logger.warning("SpeechPipelineManager에 'start_ai_to_ai_conversation' 메서드가 정의되지 않았습니다.")
                elif msg_type == "toggle_ai_pause":
                    logger.info("🖥️⏯️ 클라이언트로부터 AI 대화 일시중지/재개 요청 수신.")
                    callbacks.handle_toggle_ai_pause()


    except asyncio.CancelledError:
        pass # 작업 취소는 연결 끊김 시 예상됨
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('경고').red} process_incoming_data에서 연결 끊김: {repr(e)}")
    except RuntimeError as e:  # 닫힌 전송에서 자주 발생
        logger.error(f"🖥️💥 {Colors.apply('런타임 오류').red} process_incoming_data에서 발생: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('예외').red} process_incoming_data에서 발생: {repr(e)}")

async def send_text_messages(ws: WebSocket, message_queue: asyncio.Queue) -> None:
    """
    큐에서 클라이언트로 텍스트 메시지를 웹소켓을 통해 계속 전송합니다.

    `message_queue`에서 메시지를 기다리고, JSON으로 포맷하여 연결된
    웹소켓 클라이언트로 전송합니다. TTS가 아닌 메시지를 로그에 기록합니다.

    Args:
        ws: 웹소켓 연결 인스턴스.
        message_queue: JSON으로 전송될 딕셔너리를 생성하는 asyncio 큐.
    """
    try:
        while True:
            await asyncio.sleep(0.001) # 제어권 양보
            data = await message_queue.get()
            msg_type = data.get("type")
            if msg_type not in ["tts_chunk", "partial_assistant_answer", "final_assistant_answer"]:
                logger.info(Colors.apply(f"🖥️📤 →→클라이언트: {data}").orange)
            await ws.send_json(data)
    except asyncio.CancelledError:
        pass # 작업 취소는 연결 끊김 시 예상됨
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('경고').red} send_text_messages에서 연결 끊김: {repr(e)}")
    except RuntimeError as e:
        logger.error(f"🖥️💥 {Colors.apply('런타임 오류').red} send_text_messages에서 발생: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('예외').red} send_text_messages에서 발생: {repr(e)}")

async def _reset_interrupt_flag_async(app: FastAPI, callbacks: 'TranscriptionCallbacks'):
    """
    지연 후 마이크 중단 플래그를 리셋합니다 (비동기 버전).

    1초 기다린 후, AudioInputProcessor가 여전히 중단됨으로 표시되어 있는지
    확인합니다. 그렇다면 프로세서와 연결별 콜백 인스턴스 모두에서
    플래그를 리셋합니다.

    Args:
        app: FastAPI 애플리케이션 인스턴스 (AudioInputProcessor에 접근하기 위함).
        callbacks: 연결에 대한 TranscriptionCallbacks 인스턴스.
    """
    await asyncio.sleep(1)
    # AudioInputProcessor 자체의 중단 상태 확인
    if app.state.AudioInputProcessor.interrupted:
        logger.info(f"{Colors.apply('🖥️🎙️ ▶️ 마이크 계속됨 (비동기 리셋)').cyan}")
        app.state.AudioInputProcessor.interrupted = False
        # 콜백을 통해 연결별 중단 시간 리셋
        callbacks.interruption_time = 0
        logger.info(Colors.apply("🖥️🎙️ TTS 청크 후 중단 플래그 리셋됨 (비동기)").cyan)

# 더 이상 사용되지 않음: tts_stop 신호 기반으로 변경됨
# async def _start_silence_timer_after_delay_async(app: FastAPI, delay: float):
#     """
#     지정된 시간(초)만큼 기다린 후 장시간 침묵 타이머를 시작합니다.
#     """
#     await asyncio.sleep(delay)
#     logger.info(f"🖥️⏰ {delay:.2f}초 대기 후, 장시간 침묵 타이머 시작.")
#     app.state.AudioInputProcessor.transcriber.start_prolonged_silence_timer()


async def send_tts_chunks(app: FastAPI, message_queue: asyncio.Queue, callbacks: 'TranscriptionCallbacks') -> None:
    """
    SpeechPipelineManager에서 클라이언트로 TTS 오디오 청크를 계속 전송합니다.

    현재 음성 생성 상태(있는 경우)와 클라이언트 연결 상태(`callbacks`를 통해)를
    모니터링합니다. 활성 생성의 큐에서 오디오 청크를 검색하고, 업샘플링/인코딩하여
    클라이언트로 가는 `message_queue`에 넣습니다. 생성 종료 로직과 상태 리셋을
    처리합니다.

    Args:
        app: FastAPI 애플리케이션 인스턴스 (전역 컴포넌트에 접근하기 위함).
        message_queue: 나가는 TTS 청크 메시지를 넣을 asyncio 큐.
        callbacks: 이 연결의 상태를 관리하는 TranscriptionCallbacks 인스턴스.
    """
    try:
        logger.info("🖥️🔊 TTS 청크 전송기 시작")
        last_quick_answer_chunk = 0
        last_chunk_sent = 0
        prev_status = None

        while True:
            await asyncio.sleep(0.001) # 제어권 양보

            # 콜백을 통해 연결별 interruption_time 사용
            if app.state.AudioInputProcessor.interrupted and callbacks.interruption_time and time.time() - callbacks.interruption_time > 2.0:
                app.state.AudioInputProcessor.interrupted = False
                callbacks.interruption_time = 0 # 콜백을 통해 리셋
                logger.info(Colors.apply("🖥️🎙️ 2초 후 중단 플래그 리셋됨").cyan)

            is_tts_finished = app.state.SpeechPipelineManager.is_valid_gen() and app.state.SpeechPipelineManager.running_generation.audio_quick_finished

            def log_status():
                nonlocal prev_status
                last_quick_answer_chunk_decayed = (
                    last_quick_answer_chunk
                    and time.time() - last_quick_answer_chunk > TTS_FINAL_TIMEOUT
                    and time.time() - last_chunk_sent > TTS_FINAL_TIMEOUT
                )

                curr_status = (
                    # 콜백을 통해 연결별 상태 접근
                    int(callbacks.tts_to_client),
                    int(callbacks.tts_client_playing),
                    int(callbacks.tts_chunk_sent),
                    1, # 플레이스홀더?
                    int(callbacks.is_hot), # 콜백에서
                    int(callbacks.synthesis_started), # 콜백에서
                    int(app.state.SpeechPipelineManager.running_generation is not None), # 전역 관리자 상태
                    int(app.state.SpeechPipelineManager.is_valid_gen()), # 전역 관리자 상태
                    int(is_tts_finished), # 계산된 지역 변수
                    int(app.state.AudioInputProcessor.interrupted) # 입력 프로세서 상태
                )

                if curr_status != prev_status:
                    status = Colors.apply("🖥️🚦 상태 ").red
                    logger.info(
                        f"{status} ToClient {curr_status[0]}, "
                        f"ttsClientON {curr_status[1]}, " # 명확성을 위해 약간 이름 변경
                        f"ChunkSent {curr_status[2]}, "
                        f"hot {curr_status[4]}, synth {curr_status[5]}"
                        f" gen {curr_status[6]}"
                        f" valid {curr_status[7]}"
                        f" tts_q_fin {curr_status[8]}"
                        f" mic_inter {curr_status[9]}"
                    )
                    prev_status = curr_status

            # 콜백을 통해 연결별 상태 사용
            if not callbacks.tts_to_client:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if not app.state.SpeechPipelineManager.running_generation:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if app.state.SpeechPipelineManager.running_generation.abortion_started:
                await asyncio.sleep(0.001)
                log_status()
                continue

            if not app.state.SpeechPipelineManager.running_generation.audio_quick_finished:
                app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

            if not app.state.SpeechPipelineManager.running_generation.quick_answer_first_chunk_ready:
                await asyncio.sleep(0.001)
                log_status()
                continue

            chunk = None
            try:
                chunk = app.state.SpeechPipelineManager.running_generation.audio_chunks.get_nowait()
                if chunk:
                    last_quick_answer_chunk = time.time()
            except Empty:
                final_expected = app.state.SpeechPipelineManager.running_generation.quick_answer_provided
                audio_final_finished = app.state.SpeechPipelineManager.running_generation.audio_final_finished

                if not final_expected or audio_final_finished:
                    logger.info("🖥️🏁 TTS 청크 전송 및 '사용자 요청/어시스턴트 응답' 사이클 완료.")
                    callbacks.send_final_assistant_answer() # 콜백 메서드

                    # 타이머 시작 전 오디오 길이 가져오기
                    total_audio_duration = app.state.SpeechPipelineManager.running_generation.total_audio_duration
                    
                    app.state.SpeechPipelineManager.running_generation = None

                    callbacks.tts_chunk_sent = False # 콜백을 통해 리셋
                    callbacks.reset_state() # 콜백을 통해 연결 상태 리셋
                    
                    # 침묵 타이머는 이제 tts_stop 신호를 받을 때 시작됩니다.
                    # (기존 오디오 길이 기반 타이머 로직 제거됨)

                await asyncio.sleep(0.001)
                log_status()
                continue

            base64_chunk = app.state.Upsampler.get_base64_chunk(chunk)
            message_queue.put_nowait({
                "type": "tts_chunk",
                "content": base64_chunk
            })
            last_chunk_sent = time.time()

            # 콜백을 통해 연결별 상태 사용
            if not callbacks.tts_chunk_sent:
                # 스레드 대신 비동기 헬퍼 함수 사용
                asyncio.create_task(_reset_interrupt_flag_async(app, callbacks))

            callbacks.tts_chunk_sent = True # 콜백을 통해 설정

    except asyncio.CancelledError:
        pass # 작업 취소는 연결 끊김 시 예상됨
    except WebSocketDisconnect as e:
        logger.warning(f"🖥️⚠️ {Colors.apply('경고').red} send_tts_chunks에서 연결 끊김: {repr(e)}")
    except RuntimeError as e:
        logger.error(f"🖥️💥 {Colors.apply('런타임 오류').red} send_tts_chunks에서 발생: {repr(e)}")
    except Exception as e:
        logger.exception(f"🖥️💥 {Colors.apply('예외').red} send_tts_chunks에서 발생: {repr(e)}")


# --------------------------------------------------------------------
# 음성 인식 이벤트를 처리하기 위한 콜백 클래스
# --------------------------------------------------------------------
class TranscriptionCallbacks:
    """
    단일 웹소켓 연결의 음성 인식 수명 주기에 대한 상태 및 콜백을 관리합니다.

    이 클래스는 연결별 상태 플래그(예: TTS 상태, 사용자 중단)를 보유하고
    `AudioInputProcessor` 및 `SpeechPipelineManager`에 의해 트리거되는 콜백 메서드를
    구현합니다. 제공된 `message_queue`를 통해 클라이언트로 메시지를 다시 보내고
    중단 및 최종 답변 전달과 같은 상호 작용 로직을 관리합니다.
    또한 부분 음성 인식을 기반으로 중단 확인을 처리하기 위한 스레드 워커를 포함합니다.
    """
    def __init__(self, app: FastAPI, message_queue: asyncio.Queue):
        """
        웹소켓 연결을 위한 TranscriptionCallbacks 인스턴스를 초기화합니다.

        Args:
            app: FastAPI 애플리케이션 인스턴스 (전역 컴포넌트에 접근하기 위함).
            message_queue: 클라이언트로 메시지를 다시 보내기 위한 asyncio 큐.
        """
        self.app = app
        self.message_queue = message_queue
        self.final_transcription = ""
        self.abort_text = ""
        self.last_abort_text = ""

        # 여기에 연결별 상태 플래그 초기화
        self.tts_to_client: bool = False
        self.user_interrupted: bool = False
        self.tts_chunk_sent: bool = False
        self.tts_client_playing: bool = False
        self.interruption_time: float = 0.0

        # 이것들은 이미 사실상 인스턴스 변수이거나 리셋 로직이 존재했음
        self.silence_active: bool = True
        self.is_hot: bool = False
        self.user_finished_turn: bool = False
        self.synthesis_started: bool = False
        self.assistant_answer: str = ""
        self.final_assistant_answer: str = ""
        self.is_processing_potential: bool = False
        self.is_processing_final: bool = False
        self.last_inferred_transcription: str = ""
        self.final_assistant_answer_sent: bool = False
        self.partial_transcription: str = "" # 명확성을 위해 추가

        self.reset_state() # 일관성을 보장하기 위해 리셋 호출

        # AI-to-AI 대화를 위한 TTS 재생 완료 이벤트 추가
        self.tts_playback_finished_event = threading.Event()

        self.abort_request_event = threading.Event()
        self.abort_worker_thread = threading.Thread(target=self._abort_worker, name="AbortWorker", daemon=True)
        self.abort_worker_thread.start()


    def on_tts_playback_finished(self):
        """클라이언트 TTS 재생 완료 시 호출되는 콜백"""
        logger.info("🔊 클라이언트 TTS 재생 완료, AI-to-AI 다음 턴 시작 가능")
        self.tts_playback_finished_event.set()
        
    def start_silence_timer_after_tts_stop(self):
        """TTS 재생 완료 후 침묵 타이머를 시작합니다."""
        # AI-to-AI 대화가 활성화되지 않은 경우에만 침묵 타이머 시작
        if not self.app.state.SpeechPipelineManager.ai_to_ai_active:
            logger.info("🖥️⏰ TTS 재생 완료로 인한 침묵 타이머 시작")
            self.app.state.AudioInputProcessor.transcriber.start_prolonged_silence_timer()

    def handle_toggle_ai_pause(self):
        """AI 대화 일시중지/재개를 처리하고 클라이언트에 상태를 알립니다."""
        is_paused = self.app.state.SpeechPipelineManager.toggle_pause_ai_conversation()
        
        # AI 대화 상태에 따라 장시간 침묵 타이머를 제어합니다.
        if is_paused:
            logger.info("🤖⏯️ AI 대화가 일시중지되어 장시간 침묵 타이머를 중지합니다.")
            self.app.state.AudioInputProcessor.transcriber.stop_prolonged_silence_timer()
        else:
            logger.info("🤖⏯️ AI 대화가 재개되어 장시간 침묵 타이머를 시작합니다.")
            self.app.state.AudioInputProcessor.transcriber.start_prolonged_silence_timer()

        self.message_queue.put_nowait({
            "type": "ai_conversation_paused_status",
            "is_paused": is_paused
        })

    def on_ai_turn_start(self):
        """AI의 턴이 시작될 때 호출되는 콜백."""
        logger.info(f"{Colors.apply('🖥️🔊 AI 턴 시작, TTS 스트림 허용').blue}")
        self.tts_to_client = True

    def on_tts_all_complete(self):
        """모든 TTS 처리가 완료될 때 호출되는 콜백 (빠른 TTS + 최종 TTS)"""
        logger.info("🖥️🔊 모든 TTS 처리 완료, 클라이언트에 완료 신호 전송")
        self.message_queue.put_nowait({
            "type": "tts_all_complete"
        })


    def reset_state(self):
        """연결별 상태 플래그와 변수를 초기 값으로 리셋합니다."""
        # 모든 연결별 상태 플래그 리셋
        self.tts_to_client = False
        self.user_interrupted = False
        self.tts_chunk_sent = False
        # tts_client_playing은 클라이언트 상태 보고를 반영하므로 여기서 리셋하지 않음
        self.interruption_time = 0.0

        # 다른 상태 변수 리셋
        self.silence_active = True
        self.is_hot = False
        self.user_finished_turn = False
        self.synthesis_started = False
        self.assistant_answer = ""
        self.final_assistant_answer = ""
        self.is_processing_potential = False
        self.is_processing_final = False
        self.last_inferred_transcription = ""
        self.final_assistant_answer_sent = False
        self.partial_transcription = ""

        # 오디오 프로세서/파이프라인 관리자와 관련된 중단 호출 유지
        self.app.state.AudioInputProcessor.abort_generation()


    def _abort_worker(self):
        """부분 텍스트를 기반으로 중단 조건을 확인하는 백그라운드 스레드 워커."""
        while True:
            was_set = self.abort_request_event.wait(timeout=0.1) # 100ms마다 확인
            if was_set:
                self.abort_request_event.clear()
                # 텍스트가 실제로 변경된 경우에만 중단 확인 트리거
                if self.last_abort_text != self.abort_text:
                    self.last_abort_text = self.abort_text
                    logger.debug(f"🖥️🧠 부분 텍스트로 중단 확인 트리거됨: '{self.abort_text}'")
                    self.app.state.SpeechPipelineManager.check_abort(self.abort_text, False, "on_partial")

    def on_partial(self, txt: str):
        """
        부분 음성 인식 결과가 사용 가능할 때 호출되는 콜백.

        내부 상태를 업데이트하고, 부분 결과를 클라이언트로 전송하며,
        잠재적 중단을 확인하기 위해 중단 워커 스레드에 신호를 보냅니다.

        Args:
            txt: 부분 음성 인식 텍스트.
        """
        self.final_assistant_answer_sent = False # 새로운 사용자 발화는 이전 최종 답변 전송 상태를 무효화함
        self.final_transcription = "" # 이것은 부분이므로 최종 음성 인식 지우기
        self.partial_transcription = txt
        self.message_queue.put_nowait({"type": "partial_user_request", "content": txt})
        self.abort_text = txt # 중단 확인에 사용되는 텍스트 업데이트
        self.abort_request_event.set() # 중단 워커에 신호 보내기

    def safe_abort_running_syntheses(self, reason: str):
        """안전하게 합성을 중단하기 위한 플레이스홀더 (현재 아무 작업도 하지 않음)."""
        # TODO: 필요한 경우 실제 중단 로직 구현, 잠재적으로 SpeechPipelineManager와 상호 작용
        pass

    def on_tts_allowed_to_synthesize(self):
        """
        시스템이 TTS 합성을 진행할 수 있다고 판단할 때 호출되는 콜백.

        """
        # 전역 관리자 상태 접근
        if self.app.state.SpeechPipelineManager.running_generation and not self.app.state.SpeechPipelineManager.running_generation.abortion_started:
            # logger.info(f"{Colors.apply('🖥️🔊 TTS 허용됨').blue}")
            self.app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

    def on_potential_sentence(self, txt: str):
        """
        STT에 의해 잠재적으로 완전한 문장이 감지될 때 호출되는 콜백.

        이 잠재적 문장을 기반으로 음성 생성 준비를 트리거합니다.

        Args:
            txt: 잠재적 문장 텍스트.
        """
        logger.debug(f"🖥️🧠 잠재적 문장: '{txt}'")
        # 전역 관리자 상태 접근
        self.app.state.SpeechPipelineManager.prepare_generation(txt)

    def on_potential_final(self, txt: str):
        """
        잠재적인 *최종* 음성 인식이 감지될 때(hot 상태) 호출되는 콜백.

        잠재적 최종 음성 인식을 로그에 기록합니다.

        Args:
            txt: 잠재적 최종 음성 인식 텍스트.
        """
        logger.info(f"{Colors.apply('🖥️🧠 HOT: ').magenta}{txt}")

    def on_potential_abort(self):
        """STT가 사용자 발화에 기반한 잠재적 중단 필요성을 감지하면 호출되는 콜백."""
        # 플레이스홀더: 현재 아무것도 로그하지 않으며, 중단 로직을 트리거할 수 있음.
        pass

    def on_before_final(self, audio: bytes, txt: str):
        """
        사용자 차례의 최종 STT 결과가 확정되기 직전에 호출되는 콜백.

        사용자 종료 플래그를 설정하고, 보류 중인 경우 TTS를 허용하며, 마이크 입력을 중단하고,
        클라이언트로 TTS 스트림을 해제하며, 최종 사용자 요청과 보류 중인 부분
        어시스턴트 답변을 클라이언트로 전송하고, 사용자 요청을 기록에 추가합니다.

        Args:
            audio: 최종 음성 인식에 해당하는 원시 오디오 바이트. (현재 사용 안 함)
            txt: 음성 인식 텍스트 (on_final에서 약간 정제될 수 있음).
        """
        logger.info(Colors.apply('🖥️🏁 =================== 사용자 차례 종료 ===================').light_gray)
        self.user_finished_turn = True
        self.user_interrupted = False # 연결별 플래그 리셋 (사용자 종료, 중단 아님)
        # 전역 관리자 상태 접근
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            logger.info(f"{Colors.apply('🖥️🔊 TTS 허용됨 (최종 전)').blue}")
            self.app.state.SpeechPipelineManager.running_generation.tts_quick_allowed_event.set()

        # 먼저 들어오는 오디오 차단 (오디오 프로세서의 상태)
        if not self.app.state.AudioInputProcessor.interrupted:
            logger.info(f"{Colors.apply('🖥️🎙️ ⏸️ 마이크 중단됨 (차례 종료)').cyan}")
            self.app.state.AudioInputProcessor.interrupted = True
            self.interruption_time = time.time() # 연결별 플래그 설정

        logger.info(f"{Colors.apply('🖥️🔊 TTS 스트림 해제됨').blue}")
        self.tts_to_client = True # 연결별 플래그 설정

        # 최종 사용자 요청 전송 (신뢰할 수 있는 final_transcription 또는 최종이 아직 설정되지 않은 경우 현재 부분 사용)
        user_request_content = self.final_transcription if self.final_transcription else self.partial_transcription
        self.message_queue.put_nowait({
            "type": "final_user_request",
            "content": user_request_content
        })

        # 전역 관리자 상태 접근
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            # 부분 어시스턴트 답변(사용 가능한 경우)을 클라이언트로 전송
            # 연결별 user_interrupted 플래그 사용
            if self.app.state.SpeechPipelineManager.running_generation.quick_answer and not self.user_interrupted:
                self.assistant_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer
                
                # 현재 발화자 역할 가져오기
                speaker_role = self.app.state.SpeechPipelineManager.running_generation.speaker_role
                
                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": self.assistant_answer,
                    "role": speaker_role
                })

        logger.info(f"🖥️🧠 기록에 사용자 요청 추가 중: '{user_request_content}'")
        # 전역 관리자 상태 접근
        self.app.state.SpeechPipelineManager.history.append({"role": "user", "content": user_request_content})

    def on_final(self, txt: str):
        """
        사용자 차례의 최종 음성 인식 결과가 사용 가능할 때 호출되는 콜백.

        최종 음성 인식을 로그에 기록하고 저장합니다.

        Args:
            txt: 최종 음성 인식 텍스트.
        """
        logger.info(f"\n{Colors.apply('🖥️✅ 최종 사용자 요청 (STT 콜백): ').green}{txt}")
        if not self.final_transcription: # on_before_final 로직에서 아직 설정되지 않은 경우 저장
             self.final_transcription = txt

    def abort_generations(self, reason: str):
        """
        진행 중인 모든 음성 생성 프로세스의 중단을 트리거합니다.

        이유를 로그에 기록하고 SpeechPipelineManager의 중단 메서드를 호출합니다.

        Args:
            reason: 중단이 트리거된 이유를 설명하는 문자열.
        """
        logger.info(f"{Colors.apply('🖥️🛑 생성 중단:').blue} {reason}")
        # 전역 관리자 상태 접근
        self.app.state.SpeechPipelineManager.abort_generation(reason=f"server.py abort_generations: {reason}")

    def on_silence_active(self, silence_active: bool):
        """
        침묵 감지 상태가 변경될 때 호출되는 콜백.

        내부 silence_active 플래그를 업데이트합니다.

        Args:
            silence_active: 현재 침묵이 감지되면 True, 그렇지 않으면 False.
        """
        # logger.debug(f"🖥️🎙️ 침묵 활성: {silence_active}") # 선택 사항: 시끄러울 수 있음
        self.silence_active = silence_active

    def on_partial_assistant_text(self, txt: str):
        """
        어시스턴트(LLM)로부터 부분 텍스트 결과가 사용 가능할 때 호출되는 콜백.

        내부 어시스턴트 답변 상태를 업데이트하고, 사용자가 중단하지 않은 한
        부분 답변을 클라이언트로 전송합니다.

        Args:
            txt: 부분 어시스턴트 텍스트.
        """
        # logger.info(f"{Colors.apply('🖥️💬 부분 어시스턴트 답변: ').green}{txt}")
        # 연결별 user_interrupted 플래그 사용
        if not self.user_interrupted:
            self.assistant_answer = txt
            # 연결별 tts_to_client 플래그 사용
            if self.tts_to_client:
                # 현재 발화자 역할 가져오기
                speaker_role = "assistant" # 기본값
                if self.app.state.SpeechPipelineManager.is_valid_gen():
                    speaker_role = self.app.state.SpeechPipelineManager.running_generation.speaker_role

                self.message_queue.put_nowait({
                    "type": "partial_assistant_answer",
                    "content": txt,
                    "role": speaker_role
                })

    def on_recording_start(self):
        """
        오디오 입력 프로세서가 사용자 음성 녹음을 시작할 때 호출되는 콜백.

        클라이언트 측 TTS가 재생 중인 경우 중단을 트리거합니다: 서버 측
        TTS 스트리밍을 중지하고, 클라이언트에 중지/중단 메시지를 보내고, 진행 중인
        생성을 중단하고, 지금까지 생성된 최종 어시스턴트 답변을 보내고, 관련 상태를
        리셋합니다.
        """
        logger.info(f"{Colors.ORANGE}🖥️🎙️ 녹음 시작됨.{Colors.RESET} TTS 클라이언트 재생 중: {self.tts_client_playing}")
        # 연결별 tts_client_playing 플래그 사용
        if self.tts_client_playing:
            self.tts_to_client = False # 서버가 TTS 전송 중지
            self.user_interrupted = True # 연결을 사용자가 중단한 것으로 표시
            logger.info(f"{Colors.apply('🖥️❗ 녹음 시작으로 인한 TTS 중단').blue}")

            # 생성되었고 전송되지 않은 경우 최종 어시스턴트 답변 전송
            logger.info(Colors.apply("🖥️✅ 최종 어시스턴트 답변 전송 (중단 시 강제)").pink)
            self.send_final_assistant_answer(forced=True)

            # 중단을 위한 최소한의 리셋:
            self.tts_chunk_sent = False # 청크 전송 플래그 리셋
            # self.assistant_answer = "" # 선택 사항: 필요한 경우 부분 답변 지우기

            logger.info("🖥️🛑 클라이언트에 stop_tts 전송 중.")
            self.message_queue.put_nowait({
                "type": "stop_tts", # 클라이언트가 이를 처리하여 음소거/무시
                "content": ""
            })

            logger.info(f"{Colors.apply('🖥️🛑 녹음 시작으로 생성 중단').red}")
            self.abort_generations("on_recording_start, user interrupts, TTS Playing")

            logger.info("🖥️❗ 클라이언트에 tts_interruption 전송 중.")
            self.message_queue.put_nowait({ # 클라이언트에 재생 중지 및 버퍼 비우기 지시
                "type": "tts_interruption",
                "content": ""
            })

            # 이전 상태에 기반한 작업 수행 *후* 상태 리셋
            # 정확히 무엇을 리셋하고 무엇을 유지해야 하는지 주의 (tts_client_playing 등)
            # self.reset_state() # 너무 많이 지울 수 있음, 예를 들어 user_interrupted를 조기에

    def send_final_assistant_answer(self, forced=False, speaker_role_override=None):
        """
        최종 (또는 사용 가능한 최상의) 어시스턴트 답변을 클라이언트로 전송합니다.

        사용 가능한 경우 빠른 답변과 최종 답변 부분에서 전체 답변을 구성합니다.
        `forced`이고 전체 답변이 없는 경우 마지막 부분 답변을 사용합니다.
        텍스트를 정리하고 아직 전송되지 않은 경우 'final_assistant_answer'로 전송합니다.

        Args:
            forced: True인 경우, 완전한 최종 답변이 없을 때 마지막 부분 답변을
                    전송하려고 시도합니다. 기본값은 False.
            speaker_role_override: AI-to-AI 대화와 같이 역할을 명시적으로 설정해야 할 때 사용됩니다.
        """
        final_answer = ""
        # 전역 관리자 상태 접근
        if self.app.state.SpeechPipelineManager.is_valid_gen():
            final_answer = self.app.state.SpeechPipelineManager.running_generation.quick_answer + self.app.state.SpeechPipelineManager.running_generation.final_answer

        if not final_answer: # 구성된 답변이 비어 있는지 확인
            # 강제된 경우, 이 연결의 마지막으로 알려진 부분 답변 사용 시도
            if forced and self.assistant_answer:
                 final_answer = self.assistant_answer
                 logger.warning(f"🖥️⚠️ 부분 답변을 최종으로 사용 (강제): '{final_answer}'")
            else:
                logger.warning(f"🖥️⚠️ 최종 어시스턴트 답변이 비어 있어 전송하지 않습니다.")
                # if self.app.state.SpeechPipelineManager.is_valid_gen():
                #     self.app.state.SpeechPipelineManager.running_generation.turn_finished_event.set()
                return# 보낼 것이 없음

        logger.debug(f"🖥️✅ 최종 답변 전송 시도: '{final_answer}' (이전에 전송됨: {self.final_assistant_answer_sent})")

        if not self.final_assistant_answer_sent and final_answer:
            import re
            # 최종 답변 텍스트 정리
            cleaned_answer = re.sub(r'[\r\n]+', ' ', final_answer)
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()
            cleaned_answer = cleaned_answer.replace('\\n', ' ')
            cleaned_answer = re.sub(r'\s+', ' ', cleaned_answer).strip()

            if cleaned_answer: # 정리 후 비어 있지 않은지 확인
                logger.info(f"\n{Colors.apply('🖥️✅ 최종 어시스턴트 답변 (전송 중): ').green}{cleaned_answer}")
                
                # 현재 발화자 역할 가져오기
                speaker_role = speaker_role_override or "assistant" # 오버라이드 사용 또는 기본값
                if not speaker_role_override and self.app.state.SpeechPipelineManager.is_valid_gen():
                    speaker_role = self.app.state.SpeechPipelineManager.running_generation.speaker_role

                self.message_queue.put_nowait({
                    "type": "final_assistant_answer",
                    "content": cleaned_answer,
                    "role": speaker_role 
                })
                self.app.state.SpeechPipelineManager.history.append({"role": speaker_role, "content": cleaned_answer})
                self.final_assistant_answer_sent = True
                self.final_assistant_answer = cleaned_answer # 전송된 답변 저장
            else:
                logger.warning(f"🖥️⚠️ {Colors.YELLOW}최종 어시스턴트 답변이 정리 후 비어 있습니다.{Colors.RESET}")
                self.final_assistant_answer_sent = False # 전송됨으로 표시 안 함
                self.final_assistant_answer = "" # 저장된 답변 지우기
        elif forced and not final_answer: # 이전 확인으로 인해 발생해서는 안 되지만, 안전을 위해
             logger.warning(f"🖥️⚠️ {Colors.YELLOW}최종 어시스턴트 답변의 강제 전송, 하지만 비어 있었습니다.{Colors.RESET}")
             self.final_assistant_answer = "" # 저장된 답변 지우기
        
        # if self.app.state.SpeechPipelineManager.is_valid_gen():
        #     self.app.state.SpeechPipelineManager.running_generation.turn_finished_event.set()


# --------------------------------------------------------------------
# 메인 웹소켓 엔드포인트
# --------------------------------------------------------------------
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """
    실시간 음성 채팅을 위한 메인 웹소켓 연결을 처리합니다.

    연결을 수락하고, `TranscriptionCallbacks`를 통해 연결별 상태를 설정하고,
    오디오/메시지 큐를 초기화하며, 들어오는 데이터, 오디오 처리, 나가는 텍스트 메시지,
    나가는 TTS 청크를 처리하기 위한 asyncio 작업을 생성합니다.
    이러한 작업의 수명 주기를 관리하고 연결 끊김 시 정리합니다.

    Args:
        ws: FastAPI에서 제공하는 웹소켓 연결 인스턴스.
    """
    await ws.accept()
    logger.info("🖥️✅ 클라이언트가 웹소켓을 통해 연결되었습니다.")
    
    # 클라이언트 연결 상태 업데이트
    app.state.SpeechPipelineManager.set_client_connection_status(True)

    message_queue = asyncio.Queue()
    audio_chunks = asyncio.Queue()

    # 콜백 관리자 설정 - 이것이 이제 연결별 상태를 보유함
    callbacks = TranscriptionCallbacks(app, message_queue)

    # AudioInputProcessor(전역 컴포넌트)에 콜백 할당
    # callbacks 내의 이 메서드들은 이제 해당 *인스턴스* 상태에서 작동함
    app.state.AudioInputProcessor.realtime_callback = callbacks.on_partial
    app.state.AudioInputProcessor.transcriber.potential_sentence_end = callbacks.on_potential_sentence
    app.state.AudioInputProcessor.transcriber.on_tts_allowed_to_synthesize = callbacks.on_tts_allowed_to_synthesize
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_callback = callbacks.on_potential_final
    app.state.AudioInputProcessor.transcriber.potential_full_transcription_abort_callback = callbacks.on_potential_abort
    app.state.AudioInputProcessor.transcriber.full_transcription_callback = callbacks.on_final
    app.state.AudioInputProcessor.transcriber.before_final_sentence = callbacks.on_before_final
    app.state.AudioInputProcessor.recording_start_callback = callbacks.on_recording_start
    app.state.AudioInputProcessor.silence_active_callback = callbacks.on_silence_active

    # SpeechPipelineManager(전역 컴포넌트)에 콜백 할당
    app.state.SpeechPipelineManager.on_partial_assistant_text = callbacks.on_partial_assistant_text
    app.state.SpeechPipelineManager.on_final_assistant_answer = callbacks.send_final_assistant_answer
    app.state.SpeechPipelineManager.on_ai_turn_start = callbacks.on_ai_turn_start
    app.state.SpeechPipelineManager.on_tts_all_complete = callbacks.on_tts_all_complete
    
    # AI-to-AI 대화를 위한 TTS 재생 완료 콜백 연결
    app.state.SpeechPipelineManager.tts_playback_finished_callback = callbacks

    # 다른 책임을 처리하기 위한 작업 생성
    # 연결별 상태가 필요한 작업에 'callbacks' 인스턴스 전달
    tasks = [
        asyncio.create_task(process_incoming_data(ws, app, audio_chunks, callbacks)), # callbacks 전달
        asyncio.create_task(app.state.AudioInputProcessor.process_chunk_queue(audio_chunks)),
        asyncio.create_task(send_text_messages(ws, message_queue)),
        asyncio.create_task(send_tts_chunks(app, message_queue, callbacks)), # callbacks 전달
    ]

    try:
        # 작업 중 하나가 완료될 때까지 대기 (예: 클라이언트 연결 끊김)
        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            if not task.done():
                task.cancel()
        # 필요한 경우 정리할 수 있도록 취소된 작업 대기
        await asyncio.gather(*pending, return_exceptions=True)
    except Exception as e:
        logger.error(f"🖥️💥 {Colors.apply('오류').red} 웹소켓 세션에서 발생: {repr(e)}")
    finally:
        logger.info("🖥️🧹 웹소켓 작업 정리 중...")
        
        # 클라이언트 연결 상태 업데이트 (연결 끊김)
        app.state.SpeechPipelineManager.set_client_connection_status(False)
        
        for task in tasks:
            if not task.done():
                task.cancel()
        # 취소 후 모든 작업이 대기되도록 보장
        # 정리 중 첫 오류에서 gather가 멈추는 것을 방지하기 위해 return_exceptions=True 사용
        await asyncio.gather(*tasks, return_exceptions=True)
        logger.info("🖥️❌ 웹소켓 세션이 종료되었습니다.")

# --------------------------------------------------------------------
# 진입점
# --------------------------------------------------------------------
if __name__ == "__main__":

    # SSL 없이 서버 실행
    if not USE_SSL:
        logger.info("🖥️▶️ SSL 없이 서버를 시작합니다.")
        uvicorn.run("server:app", host="0.0.0.0", port=8000, log_config=None)

    else:
        logger.info("🖥️🔒 SSL로 서버를 시작하려고 시도합니다.")
        # 인증서 파일 존재 여부 확인
        cert_file = "1.pem"
        key_file = "1-key.pem"
        if not os.path.exists(cert_file) or not os.path.exists(key_file):
             logger.error(f"🖥️💥 SSL 인증서 파일({cert_file}) 또는 키 파일({key_file})을 찾을 수 없습니다.")
             logger.error("🖥️💥 mkcert를 사용하여 생성하세요:")
             logger.error("🖥️💥   choco install mkcert") # 이전 확인에 기반한 Windows 가정, 필요 시 조정
             logger.error("🖥️💥   mkcert -install")
             logger.error("🖥️💥   mkcert 127.0.0.1 YOUR_LOCAL_IP") # 필요한 경우 실제 IP로 교체하도록 사용자에게 알림
             logger.error("🖥️💥 종료합니다.")
             sys.exit(1)

        # SSL로 서버 실행
        logger.info(f"🖥️▶️ SSL로 서버를 시작합니다 (인증서: {cert_file}, 키: {key_file}).")
        uvicorn.run(
            "server:app",
            host="0.0.0.0",
            port=8000,
            log_config=None,
            ssl_certfile=cert_file,
            ssl_keyfile=key_file,
        )