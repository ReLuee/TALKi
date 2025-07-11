import asyncio
import logging
import os
import struct
import subprocess
import threading
import time
from collections import namedtuple
from queue import Queue
from typing import Callable, Generator, Optional

import numpy as np
from huggingface_hub import hf_hub_download
# RealtimeTTS가 설치되어 있고 사용 가능하다고 가정
from RealtimeTTS import (CoquiEngine, KokoroEngine, OrpheusEngine,
                         OrpheusVoice, ElevenlabsEngine, TextToAudioStream)

logger = logging.getLogger(__name__)

# 기본 설정 상수
START_ENGINE = "elabs"
Silence = namedtuple("Silence", ("comma", "sentence", "default"))
ENGINE_SILENCES = {
    "coqui":   Silence(comma=0.3, sentence=0.6, default=0.3),
    "kokoro":  Silence(comma=0.3, sentence=0.6, default=0.3),
    "orpheus": Silence(comma=0.3, sentence=0.6, default=0.3),
    "elabs": Silence(comma=0.3, sentence=0.6, default=0.3),
}
# 스트림 청크 크기는 지연 시간 대 처리량 트레이드오프에 영향을 줌
QUICK_ANSWER_STREAM_CHUNK_SIZE = 8
FINAL_ANSWER_STREAM_CHUNK_SIZE = 30

# Coqui 모델 다운로드 헬퍼 함수
def create_directory(path: str) -> None:
    """
    지정된 경로에 디렉터리가 아직 없으면 생성합니다.

    Args:
        path: 생성할 디렉터리 경로.
    """
    if not os.path.exists(path):
        os.makedirs(path)

def ensure_lasinya_models(models_root: str = "models", model_name: str = "Lasinya") -> None:
    """
    Coqui XTTS Lasinya 모델 파일이 로컬에 있는지 확인합니다.

    지정된 디렉터리 구조 내에서 필요한 모델 파일(config.json, vocab.json 등)을
    확인합니다. 파일이 누락된 경우 'KoljaB/XTTS_Lasinya' Hugging Face Hub
    리포지토리에서 다운로드합니다.

    Args:
        models_root: 모델이 저장되는 루트 디렉터리.
        model_name: 특정 모델 하위 디렉터리의 이름.
    """
    base = os.path.join(models_root, model_name)
    create_directory(base)
    files = ["config.json", "vocab.json", "speakers_xtts.pth", "model.pth"]
    for fn in files:
        local_file = os.path.join(base, fn)
        if not os.path.exists(local_file):
            # 모듈 가져오기/초기화 중에 아직 구성되지 않았을 수 있으므로 여기서는 로거를 사용하지 않음
            print(f"👄⏬ {fn}을(를) {base}(으)로 다운로드 중")
            hf_hub_download(
                repo_id="KoljaB/XTTS_Lasinya",
                filename=fn,
                local_dir=base
            )

class AudioProcessor:
    """
    RealtimeTTS를 통해 다양한 엔진을 사용하여 텍스트 음성 변환(TTS) 합성을 관리합니다.

    이 클래스는 선택된 TTS 엔진(Coqui, Kokoro 또는 Orpheus)을 초기화하고,
    스트리밍 출력을 위해 구성하며, 초기 지연 시간(TTFT)을 측정하고,
    텍스트 문자열 또는 생성기에서 오디오를 합성하여 결과 오디오 청크를
    큐에 넣는 메서드를 제공합니다. 동적 스트림 매개변수 조정을 처리하고
    첫 번째 오디오 청크 수신 시 선택적 콜백을 포함한 합성 수명 주기를 관리합니다.
    """
    def __init__(
            self,
            engine: str = START_ENGINE,
            orpheus_model: str = "orpheus-3b-0.1-ft-Q8_0-GGUF/orpheus-3b-0.1-ft-q8_0.gguf",
        ) -> None:
        """
        특정 TTS 엔진으로 AudioProcessor를 초기화합니다.

        선택한 엔진(Coqui, Kokoro, Orpheus)을 설정하고, 필요한 경우 Coqui 모델을
        다운로드하고, RealtimeTTS 스트림을 구성하고, 첫 오디오 청크까지의 시간(TTFA)을
        측정하기 위해 초기 합성을 수행합니다.

        Args:
            engine: 사용할 TTS 엔진의 이름("coqui", "kokoro", "orpheus").
            orpheus_model: Orpheus 모델 파일의 경로 또는 식별자(엔진이 "orpheus"인 경우에만 사용됨).
        """
        self.engine_name = engine
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.audio_chunks = asyncio.Queue() # 합성된 오디오 출력을 위한 큐
        self.orpheus_model = orpheus_model

        self.silence = ENGINE_SILENCES.get(engine, ENGINE_SILENCES[self.engine_name])
        self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE # 초기 청크 크기

        # 선택된 TTS 엔진을 동적으로 로드하고 구성
        if engine == "coqui":
            ensure_lasinya_models(models_root="models", model_name="Lasinya")
            self.engine = CoquiEngine(
                specific_model="Lasinya",
                local_models_path="./models",
                voice="./voices/coqui_Daisy Studious.json",
                speed=1.1,
                use_deepspeed=True,
                thread_count=6,
                stream_chunk_size=self.current_stream_chunk_size,
                overlap_wav_len=1024,
                load_balancing=True,
                load_balancing_buffer_length=0.5,
                load_balancing_cut_off=0.1,
                add_sentence_filter=True,
            )
        elif engine == "kokoro":
            self.engine = KokoroEngine(
                voice="af_heart",
                default_speed=1.26,
                trim_silence=True,
                silence_threshold=0.01,
                extra_start_ms=25,
                extra_end_ms=15,
                fade_in_ms=15,
                fade_out_ms=10,
            )
        elif engine == "orpheus":
            self.engine = OrpheusEngine(
                model=self.orpheus_model,
                temperature=0.8,
                top_p=0.95,
                repetition_penalty=1.1,
                max_tokens=1200,
            )
            voice = OrpheusVoice("mia")
            self.engine.set_voice(voice)
            
        elif engine == "elabs":
            from dotenv import load_dotenv
            load_dotenv()
            
            ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
            
            self.engine = ElevenlabsEngine(
                api_key = ELEVENLABS_API_KEY,
                voice = "George",
                id = "JBFqnCBsd6RMkjVDRZzb",
                model = "eleven_turbo_v2_5",
            )
        else:
            raise ValueError(f"지원되지 않는 엔진: {engine}")


        # RealtimeTTS 스트림 초기화
        self.stream = TextToAudioStream(
            self.engine,
            muted=True, # 오디오를 직접 재생하지 않음
            playout_chunk_size=4096, # 처리를 위한 내부 청크 크기
            on_audio_stream_stop=self.on_audio_stream_stop,
        )

        # Coqui 엔진이 빠른 청크 크기로 시작하도록 보장
        if self.engine_name == "coqui" and hasattr(self.engine, 'set_stream_chunk_size') and self.current_stream_chunk_size != QUICK_ANSWER_STREAM_CHUNK_SIZE:
            logger.info(f"👄⚙️ 초기 설정을 위해 Coqui 스트림 청크 크기를 {QUICK_ANSWER_STREAM_CHUNK_SIZE}(으)로 설정합니다.")
            self.engine.set_stream_chunk_size(QUICK_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE

        # 엔진 예열
        self.stream.feed("prewarm")
        play_kwargs = dict(
            log_synthesized_text=False, # 예열 텍스트는 로그하지 않음
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999, # 사실상 이 기능 비활성화
        )
        self.stream.play(**play_kwargs) # 예열을 위해 동기적으로 재생
        # 예열이 끝날 때까지 대기 (on_audio_stream_stop으로 표시됨)
        while self.stream.is_playing():
            time.sleep(0.01)
        self.finished_event.wait() # 중지 콜백 대기
        self.finished_event.clear()

        # 첫 오디오까지의 시간(TTFA) 측정
        start_time = time.time()
        ttfa = None
        def on_audio_chunk_ttfa(chunk: bytes):
            nonlocal ttfa
            if ttfa is None:
                ttfa = time.time() - start_time
                logger.debug(f"👄⏱️ TTFA 측정 첫 청크 도착, TTFA: {ttfa:.2f}s.")

        self.stream.feed("This is a test sentence to measure the time to first audio chunk.")
        play_kwargs_ttfa = dict(
            on_audio_chunk=on_audio_chunk_ttfa,
            log_synthesized_text=False, # 테스트 문장은 로그하지 않음
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999,
        )
        self.stream.play_async(**play_kwargs_ttfa)

        # 첫 청크가 도착하거나 스트림이 끝날 때까지 대기
        while ttfa is None and (self.stream.is_playing() or not self.finished_event.is_set()):
            time.sleep(0.01)
        self.stream.stop() # 스트림이 깨끗하게 중지되도록 보장

        # 아직 발생하지 않은 경우 중지 콜백 대기
        if not self.finished_event.is_set():
            self.finished_event.wait(timeout=2.0) # 안전을 위해 타임아웃 추가
        self.finished_event.clear()

        if ttfa is not None:
            logger.debug(f"👄⏱️ TTFA 측정이 완료되었습니다. TTFA: {ttfa:.2f}s.")
            self.tts_inference_time = ttfa * 1000  # ms로 저장
        else:
            logger.warning("👄⚠️ TTFA 측정 실패 (오디오 청크 수신되지 않음).")
            self.tts_inference_time = 0

        # 필요 시 외부에서 설정할 콜백
        self.on_first_audio_chunk_synthesize: Optional[Callable[[], None]] = None

    def set_voice(self, voice_path: str):
        """
        TTS 엔진의 목소리를 동적으로 변경합니다.

        Args:
            voice_path: 사용할 새 목소리 파일의 경로 또는 ElevenLabs의 목소리 ID.
        """
        if self.engine_name == "coqui":
            logger.info(f"👄🎤 Coqui 목소리 변경: {voice_path}")
            self.engine.set_cloning_reference(voice_path)
        elif self.engine_name == "orpheus":
            # Orpheus는 다른 방식으로 목소리를 설정할 수 있습니다 (예: OrpheusVoice 객체).
            # 이 예제에서는 간단하게 유지합니다.
            logger.warning("Orpheus 목소리의 동적 변경은 현재 구현되지 않았습니다.")
        elif self.engine_name == "elabs":
            if hasattr(self.engine, 'id') and self.engine.id != voice_path:
                logger.info(f"👄🎤 ElevenLabs 목소리 ID 변경: {voice_path}")
                self.engine.id = voice_path
        else:
            logger.warning(f"{self.engine_name} 엔진의 목소리 변경은 지원되지 않습니다.")

    def on_audio_stream_stop(self) -> None:
        """
        RealtimeTTS 오디오 스트림 처리가 중지될 때 실행되는 콜백.

        이벤트를 로그하고 완료 또는 중지를 알리기 위해 `finished_event`를 설정합니다.
        """
        logger.info("👄🛑 오디오 스트림이 중지되었습니다.")
        self.finished_event.set()

    def synthesize(
            self,
            text: str,
            audio_chunks: Queue, 
            stop_event: threading.Event,
            generation_string: str = "",
            generation_obj: Optional[any] = None,
        ) -> bool:
        """
        완전한 텍스트 문자열에서 오디오를 합성하고 청크를 큐에 넣습니다.

        전체 텍스트 문자열을 TTS 엔진에 공급합니다. 오디오 청크가 생성될 때,
        더 부드러운 스트리밍을 위해 초기에 버퍼링될 수 있으며 그 후 제공된
        큐에 들어갑니다. 합성은 stop_event를 통해 중단될 수 있습니다.
        Orpheus 엔진을 사용하는 경우 초기 무음 청크를 건너뜁니다. 첫 번째 유효한
        오디오 청크가 큐에 들어갈 때 `on_first_audio_chunk_synthesize` 콜백을
        트리거합니다.

        Args:
            text: 합성할 텍스트 문자열.
            audio_chunks: 결과 오디오 청크(바이트)를 넣을 큐.
                          일반적으로 인스턴스의 `self.audio_chunks`여야 합니다.
            stop_event: 합성 중단을 알리는 threading.Event.
                        일반적으로 인스턴스의 `self.stop_event`여야 합니다.
            generation_string: 로깅 목적을 위한 선택적 식별자 문자열.
            generation_obj: 오디오 길이를 누적하기 위한 선택적 객체.

        Returns:
            합성이 완전히 완료되면 True, stop_event에 의해 중단되면 False.
        """
        if self.engine_name == "coqui" and hasattr(self.engine, 'set_stream_chunk_size') and self.current_stream_chunk_size != QUICK_ANSWER_STREAM_CHUNK_SIZE:
            logger.info(f"👄⚙️ {generation_string} 빠른 합성을 위해 Coqui 스트림 청크 크기를 {QUICK_ANSWER_STREAM_CHUNK_SIZE}(으)로 설정합니다.")
            self.engine.set_stream_chunk_size(QUICK_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = QUICK_ANSWER_STREAM_CHUNK_SIZE

        self.stream.feed(text)
        self.finished_event.clear() # 시작하기 전에 완료 이벤트 리셋

        # 버퍼링 상태 변수
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2 # 가정된 샘플 레이트 및 샘플당 바이트 (16비트)
        start = time.time()
        self._quick_prev_chunk_time: float = 0.0 # 이전 청크 시간 추적

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            # 중단 신호 확인
            if stop_event.is_set():
                logger.info(f"👄🛑 {generation_string} 빠른 오디오 스트림이 stop_event에 의해 중단되었습니다. 텍스트: {text[:50]}...")
                # 더 이상 청크를 넣지 말고, 메인 루프가 스트림 중지를 처리하도록 함
                return

            if self.engine_name == "elabs":
                try:
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-i', 'pipe:0',
                        '-f', 's16le',
                        '-ac', '1',
                        '-ar', '24000',
                        'pipe:1'
                    ]
                    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    decoded_chunk, err = proc.communicate(input=chunk)
                    if proc.returncode != 0:
                        logger.error(f"👄💥 FFmpeg 디코딩 오류: {err.decode()}")
                        return
                    chunk = decoded_chunk
                except Exception as e:
                    logger.error(f"👄💥 ElevenLabs 오디오 청크 처리 오류: {e}", exc_info=True)
                    return

            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR # 현재 청크의 지속 시간
            if generation_obj:
                generation_obj.total_audio_duration += play_duration

            # --- Orpheus 특정: 초기 무음 건너뛰기 ---
            if on_audio_chunk.first_call and self.engine_name == "orpheus":
                if not hasattr(on_audio_chunk, "silent_chunks_count"):
                    # 무음 감지 상태 초기화
                    on_audio_chunk.silent_chunks_count = 0
                    on_audio_chunk.silent_chunks_time = 0.0
                    on_audio_chunk.silence_threshold = 200 # 무음에 대한 진폭 임계값

                try:
                    # 무음에 대한 청크 분석
                    fmt = f"{samples}h" # 16비트 부호 있는 정수에 대한 형식
                    pcm_data = struct.unpack(fmt, chunk)
                    avg_amplitude = np.abs(np.array(pcm_data)).mean()

                    if avg_amplitude < on_audio_chunk.silence_threshold:
                        on_audio_chunk.silent_chunks_count += 1
                        on_audio_chunk.silent_chunks_time += play_duration
                        logger.debug(f"👄⏭️ {generation_string} 빠른 무음 청크 {on_audio_chunk.silent_chunks_count} 건너뛰기 (평균 진폭: {avg_amplitude:.2f})")
                        return # 이 청크 건너뛰기
                    elif on_audio_chunk.silent_chunks_count > 0:
                        # 무음 후 첫 번째 비-무음 청크
                        logger.info(f"👄⏭️ {generation_string} 빠른 무음 청크 {on_audio_chunk.silent_chunks_count}개 건너뛰었으며, {on_audio_chunk.silent_chunks_time*1000:.2f}ms 절약됨")
                        # 이 비-무음 청크 처리 진행
                except Exception as e:
                    logger.warning(f"👄⚠️ {generation_string} 빠른 오디오 청크 무음 분석 오류: {e}")
                    # 오류 시 무음이 아니라고 가정하고 진행

            # --- 타이밍 및 로깅 ---
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._quick_prev_chunk_time = now
                ttfa_actual = now - start
                logger.info(f"👄🚀 {generation_string} 빠른 오디오 시작. TTFA: {ttfa_actual:.2f}s. 텍스트: {text[:50]}...")
            else:
                gap = now - self._quick_prev_chunk_time
                self._quick_prev_chunk_time = now
                if gap <= play_duration * 1.1: # 작은 허용 오차 허용
                    # logger.debug(f"👄✅ {generation_string} 빠른 청크 정상 (gap={gap:.3f}s ≤ {play_duration:.3f}s). 텍스트: {text[:50]}...")
                    good_streak += 1
                else:
                    logger.warning(f"👄❌ {generation_string} 빠른 청크 느림 (gap={gap:.3f}s > {play_duration:.3f}s). 텍스트: {text[:50]}...")
                    good_streak = 0 # 느린 청크 시 연속 기록 리셋

            put_occurred_this_call = False # 이 특정 호출에서 put이 발생했는지 추적

            # --- 버퍼링 로직 ---
            buffer.append(chunk) # 항상 수신된 청크를 먼저 추가
            buf_dur += play_duration # 버퍼 지속 시간 업데이트

            if buffering:
                # 버퍼를 비우고 버퍼링을 중지할 조건 확인
                if good_streak >= 2 or buf_dur >= 0.5: # 안정적이거나 버퍼 > 0.5s이면 비움
                    logger.info(f"👄➡️ {generation_string} 빠른 버퍼 비우기 (연속={good_streak}, 기간={buf_dur:.2f}s).")
                    for c in buffer:
                        try:
                            audio_chunks.put_nowait(c)
                            put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f"👄⚠️ {generation_string} 빠른 오디오 큐가 가득 차서 청크를 버립니다.")
                    buffer.clear()
                    buf_dur = 0.0 # 버퍼 지속 시간 리셋
                    buffering = False # 버퍼링 모드 중지
            else: # 버퍼링 중이 아님, 청크를 직접 넣기
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} 빠른 오디오 큐가 가득 차서 청크를 버립니다.")


            # --- 첫 청크 콜백 ---
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} 빠른 on_first_audio_chunk_synthesize 실행 중.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f"👄💥 {generation_string} 빠른 on_first_audio_chunk_synthesize 콜백 오류: {e}", exc_info=True)
                # 콜백이 합성 호출당 한 번만 실행되도록 보장
                on_audio_chunk.callback_fired = True

        # 이 실행을 위한 콜백 상태 초기화
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False

        play_kwargs = dict(
            log_synthesized_text=True, # 합성 중인 텍스트 로그
            on_audio_chunk=on_audio_chunk,
            muted=True, # 오디오는 큐를 통해 처리
            fast_sentence_fragment=False, # 표준 처리
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999, # 빠른 프래그먼트 강제 안 함
        )

        logger.info(f"👄▶️ {generation_string} 빠른 합성 시작. 텍스트: {text[:50]}...")
        self.stream.play_async(**play_kwargs)

        # 완료 또는 중단을 위한 대기 루프
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f"👄🛑 {generation_string} 빠른 응답 합성이 stop_event에 의해 중단되었습니다. 텍스트: {text[:50]}...")
                # 남은 버퍼 비우기? 더 빨리 멈추기 위해 안 하기로 결정.
                buffer.clear()
                # 중지 확인을 위해 잠시 대기? finished_event가 이를 처리함.
                self.finished_event.wait(timeout=1.0) # 스트림 중지 확인 대기
                return False # 중단 표시
            time.sleep(0.01)

        # # 루프가 정상적으로 종료된 경우, 버퍼에 내용이 남아 있는지 확인 (비우기 전에 스트림 종료)
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} 스트림 종료 후 남은 빠른 버퍼 비우기.")
            for c in buffer:
                 try:
                    audio_chunks.put_nowait(c)
                 except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} 최종 비우기에서 빠른 오디오 큐가 가득 차서 청크를 버립니다.")
            buffer.clear()

        logger.info(f"👄✅ {generation_string} 빠른 응답 합성이 완료되었습니다. 텍스트: {text[:50]}...")
        return True # 성공적인 완료 표시

    def synthesize_generator(
            self,
            generator: Generator[str, None, None],
            audio_chunks: Queue, # self.audio_chunks 타입과 일치해야 함
            stop_event: threading.Event,
            generation_string: str = "",
            generation_obj: Optional[any] = None,
        ) -> bool:
        """
        텍스트 청크를 생성하는 생성기에서 오디오를 합성하고 오디오를 큐에 넣습니다.

        생성기에서 생성된 텍스트 청크를 TTS 엔진에 공급합니다. 오디오 청크가
        생성될 때, 초기에 버퍼링될 수 있으며 그 후 제공된 큐에 들어갑니다.
        합성은 stop_event를 통해 중단될 수 있습니다. Orpheus 엔진을 사용하는 경우
        초기 무음 청크를 건너뜁니다. Orpheus 엔진을 사용할 때 특정 재생 매개변수를
        설정합니다. 첫 번째 유효한 오디오 청크가 큐에 들어갈 때
       `on_first_audio_chunk_synthesize` 콜백을 트리거합니다.

        Args:
            generator: 합성할 텍스트 청크(문자열)를 생성하는 생성기.
            audio_chunks: 결과 오디오 청크(바이트)를 넣을 큐.
                          일반적으로 인스턴스의 `self.audio_chunks`여야 합니다.
            stop_event: 합성 중단을 알리는 threading.Event.
                        일반적으로 인스턴스의 `self.stop_event`여야 합니다.
            generation_string: 로깅 목적을 위한 선택적 식별자 문자열.
            generation_obj: 오디오 길이를 누적하기 위한 선택적 객체.

        Returns:
            합성이 완전히 완료되면 True, stop_event에 의해 중단되면 False.
        """
        if self.engine_name == "coqui" and hasattr(self.engine, 'set_stream_chunk_size') and self.current_stream_chunk_size != FINAL_ANSWER_STREAM_CHUNK_SIZE:
            logger.info(f"👄⚙️ {generation_string} 생성기 합성을 위해 Coqui 스트림 청크 크기를 {FINAL_ANSWER_STREAM_CHUNK_SIZE}(으)로 설정합니다.")
            self.engine.set_stream_chunk_size(FINAL_ANSWER_STREAM_CHUNK_SIZE)
            self.current_stream_chunk_size = FINAL_ANSWER_STREAM_CHUNK_SIZE

        # 생성기를 스트림에 공급
        self.stream.feed(generator)
        self.finished_event.clear() # 완료 이벤트 리셋

        # 버퍼링 상태 변수
        buffer: list[bytes] = []
        good_streak: int = 0
        buffering: bool = True
        buf_dur: float = 0.0
        SR, BPS = 24000, 2 # 가정된 샘플 레이트 및 샘플당 바이트
        start = time.time()
        self._final_prev_chunk_time: float = 0.0 # 생성기 합성을 위한 별도 타이머

        def on_audio_chunk(chunk: bytes):
            nonlocal buffer, good_streak, buffering, buf_dur, start
            if stop_event.is_set():
                logger.info(f"👄🛑 {generation_string} 최종 오디오 스트림이 stop_event에 의해 중단되었습니다.")
                return

            if self.engine_name == "elabs":
                try:
                    ffmpeg_cmd = [
                        'ffmpeg',
                        '-i', 'pipe:0',
                        '-f', 's16le',
                        '-ac', '1',
                        '-ar', '24000',
                        'pipe:1'
                    ]
                    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    decoded_chunk, err = proc.communicate(input=chunk)
                    if proc.returncode != 0:
                        logger.error(f"👄💥 FFmpeg 디코딩 오류 (생성기): {err.decode()}")
                        return
                    chunk = decoded_chunk
                except Exception as e:
                    logger.error(f"👄💥 ElevenLabs 오디오 청크 처리 오류 (생성기): {e}", exc_info=True)
                    return

            now = time.time()
            samples = len(chunk) // BPS
            play_duration = samples / SR
            if generation_obj:
                generation_obj.total_audio_duration += play_duration

            # --- Orpheus 특정: 초기 무음 건너뛰기 ---
            if on_audio_chunk.first_call and self.engine_name == "orpheus":
                if not hasattr(on_audio_chunk, "silent_chunks_count"):
                    on_audio_chunk.silent_chunks_count = 0
                    on_audio_chunk.silent_chunks_time = 0.0
                    # 최종 답변에 대해 잠재적으로 더 낮은 임계값? 또는 일관성 유지? 원본 코드처럼 100 사용.
                    on_audio_chunk.silence_threshold = 100

                try:
                    fmt = f"{samples}h"
                    pcm_data = struct.unpack(fmt, chunk)
                    avg_amplitude = np.abs(np.array(pcm_data)).mean()

                    if avg_amplitude < on_audio_chunk.silence_threshold:
                        on_audio_chunk.silent_chunks_count += 1
                        on_audio_chunk.silent_chunks_time += play_duration
                        logger.debug(f"👄⏭️ {generation_string} 최종 무음 청크 {on_audio_chunk.silent_chunks_count} 건너뛰기 (평균 진폭: {avg_amplitude:.2f})")
                        return # 건너뛰기
                    elif on_audio_chunk.silent_chunks_count > 0:
                        logger.info(f"👄⏭️ {generation_string} 최종 무음 청크 {on_audio_chunk.silent_chunks_count}개 건너뛰었으며, {on_audio_chunk.silent_chunks_time*1000:.2f}ms 절약됨")
                except Exception as e:
                    logger.warning(f"👄⚠️ {generation_string} 최종 오디오 청크 무음 분석 오류: {e}")

            # --- 타이밍 및 로깅 ---
            if on_audio_chunk.first_call:
                on_audio_chunk.first_call = False
                self._final_prev_chunk_time = now
                ttfa_actual = now-start
                logger.info(f"👄🚀 {generation_string} 최종 오디오 시작. TTFA: {ttfa_actual:.2f}s.")
            else:
                gap = now - self._final_prev_chunk_time
                self._final_prev_chunk_time = now
                if gap <= play_duration * 1.1:
                    # logger.debug(f"👄✅ {generation_string} 최종 청크 정상 (gap={gap:.3f}s ≤ {play_duration:.3f}s).")
                    good_streak += 1
                else:
                    logger.warning(f"👄❌ {generation_string} 최종 청크 느림 (gap={gap:.3f}s > {play_duration:.3f}s).")
                    good_streak = 0

            put_occurred_this_call = False

            # --- 버퍼링 로직 ---
            buffer.append(chunk)
            buf_dur += play_duration
            if buffering:
                if good_streak >= 2 or buf_dur >= 0.5: # 합성 로직과 동일한 비우기 로직
                    logger.info(f"👄➡️ {generation_string} 최종 버퍼 비우기 (연속={good_streak}, 기간={buf_dur:.2f}s).")
                    for c in buffer:
                        try:
                           audio_chunks.put_nowait(c)
                           put_occurred_this_call = True
                        except asyncio.QueueFull:
                            logger.warning(f"👄⚠️ {generation_string} 최종 오디오 큐가 가득 차서 청크를 버립니다.")
                    buffer.clear()
                    buf_dur = 0.0
                    buffering = False
            else: # 버퍼링 중 아님
                try:
                    audio_chunks.put_nowait(chunk)
                    put_occurred_this_call = True
                except asyncio.QueueFull:
                    logger.warning(f"👄⚠️ {generation_string} 최종 오디오 큐가 가득 차서 청크를 버립니다.")


            # --- 첫 청크 콜백 --- (합성과 동일한 콜백 사용)
            if put_occurred_this_call and not on_audio_chunk.callback_fired:
                if self.on_first_audio_chunk_synthesize:
                    try:
                        logger.info(f"👄🚀 {generation_string} 최종 on_first_audio_chunk_synthesize 실행 중.")
                        self.on_first_audio_chunk_synthesize()
                    except Exception as e:
                        logger.error(f"👄💥 {generation_string} 최종 on_first_audio_chunk_synthesize 콜백 오류: {e}", exc_info=True)
                on_audio_chunk.callback_fired = True

        # 콜백 상태 초기화
        on_audio_chunk.first_call = True
        on_audio_chunk.callback_fired = False

        play_kwargs = dict(
            log_synthesized_text=True, # 생성기에서 텍스트 로그
            on_audio_chunk=on_audio_chunk,
            muted=True,
            fast_sentence_fragment=False,
            comma_silence_duration=self.silence.comma,
            sentence_silence_duration=self.silence.sentence,
            default_silence_duration=self.silence.default,
            force_first_fragment_after_words=999999
        )

        # 생성기 스트리밍을 위한 Orpheus 특정 매개변수 추가
        if self.engine_name == "orpheus":
            # 이들은 합성 전에 더 많은 텍스트를 기다리도록 장려하며, 생성기에 더 좋을 수 있음
            play_kwargs["minimum_sentence_length"] = 200
            play_kwargs["minimum_first_fragment_length"] = 200

        logger.info(f"👄▶️ {generation_string} 생성기에서 최종 합성 시작.")
        self.stream.play_async(**play_kwargs)

        # 완료 또는 중단을 위한 대기 루프
        while self.stream.is_playing() or not self.finished_event.is_set():
            if stop_event.is_set():
                self.stream.stop()
                logger.info(f"👄🛑 {generation_string} 최종 응답 합성이 stop_event에 의해 중단되었습니다.")
                buffer.clear()
                self.finished_event.wait(timeout=1.0) # 스트림 중지 확인 대기
                return False # 중단 표시
            time.sleep(0.01)

        # 비우기 조건이 충족되기 전에 스트림이 종료된 경우 남은 버퍼 비우기
        if buffering and buffer and not stop_event.is_set():
            logger.info(f"👄➡️ {generation_string} 스트림 종료 후 남은 최종 버퍼 비우기.")
            for c in buffer:
                try:
                   audio_chunks.put_nowait(c)
                except asyncio.QueueFull:
                   logger.warning(f"👄⚠️ {generation_string} 최종 비우기에서 최종 오디오 큐가 가득 차서 청크를 버립니다.")
            buffer.clear()

        logger.info(f"👄✅ {generation_string} 최종 응답 합성이 완료되었습니다.")
        return True # 성공적인 완료 표시
