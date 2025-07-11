import logging
logger = logging.getLogger(__name__)

from turndetect import strip_ending_punctuation
from difflib import SequenceMatcher
from colors import Colors
from text_similarity import TextSimilarity
from scipy import signal
import numpy as np
import threading
import textwrap
import torch
import json
import copy
import time
import re
from typing import Optional, Callable, Any, Dict, List

# --- 설정 플래그 ---
USE_TURN_DETECTION = True
START_STT_SERVER = False # RealtimeSTT의 클라이언트/서버 버전을 사용하려면 True로 설정

# --- 레코더 설정 (명확성을 위해 여기로 이동, 외부화 가능) ---
# 생성자에 아무것도 제공되지 않을 경우의 기본 설정
DEFAULT_RECORDER_CONFIG: Dict[str, Any] = {
    "use_microphone": False,
    "spinner": False,
    "model": "large-v3",
    "realtime_model_type": "large-v3",
    "use_main_model_for_realtime": False,
    "language": "en", # 기본값, __init__에서 source_language에 의해 덮어쓰여짐
    "silero_sensitivity": 0.05,
    "webrtc_sensitivity": 3,
    "post_speech_silence_duration": 0.7,
    "min_length_of_recording": 0.5,
    "min_gap_between_recordings": 0,
    "enable_realtime_transcription": True,
    "realtime_processing_pause": 0.03,
    "silero_use_onnx": True,
    "silero_deactivity_detection": True,
    "early_transcription_on_silence": 0,
    "beam_size": 3,
    "beam_size_realtime": 3,
    "no_log_file": True,
    "wake_words": "jarvis",
    "wakeword_backend": "pvporcupine",
    "allowed_latency_limit": 500,
    # 콜백은 _create_recorder에서 동적으로 추가됨
    "debug_mode": True,
    "initial_prompt_realtime": "The sky is blue. When the sky... She walked home. Because he... Today is sunny. If only I...",
    "faster_whisper_vad_filter": False,
}


if START_STT_SERVER:
    from RealtimeSTT import AudioToTextRecorderClient
else:
    from RealtimeSTT import AudioToTextRecorder

if USE_TURN_DETECTION:
    from turndetect import TurnDetection


INT16_MAX_ABS_VALUE: float = 32768.0
SAMPLE_RATE: int = 16000


class TranscriptionProcessor:
    """
    RealtimeSTT를 사용하여 오디오 음성 인식을 관리하며, 실시간 및 최종
    음성 인식 콜백, 침묵 감지, 발화자 전환 감지(선택 사항),
    그리고 잠재적인 문장 끝 감지를 처리합니다.

    이 클래스는 원시 오디오 입력과 음성 인식 결과 사이의 다리 역할을 하며,
    RealtimeSTT 레코더를 조정하고, 콜백을 처리하며, 침묵, 잠재적 문장,
    그리고 발화자 전환 타이밍과 관련된 내부 상태를 관리합니다.
    """
    # --- 침묵 모니터 로직을 위한 상수 ---
    # 파이프라인이 너무 일찍 또는 늦게 시작되지 않도록 보장하기 위한 예약 시간
    _PIPELINE_RESERVE_TIME_MS: float = 0.02 # 20 ms
    # silence_waiting_time의 끝에서 "hot" 상태를 고려하기 시작하는 오프셋
    _HOT_THRESHOLD_OFFSET_S: float = 0.35
    # "hot" 조건이 의미를 갖기 위한 최소 지속 시간
    _MIN_HOT_CONDITION_DURATION_S: float = 0.15
    # 전체 침묵 지속 시간 전에 TTS 합성이 허용될 수 있는 시간
    _TTS_ALLOWANCE_OFFSET_S: float = 0.25
    # 침묵 시작에 대한 잠재적 문장 끝 감지를 위한 최소 시간
    _MIN_POTENTIAL_END_DETECTION_TIME_MS: float = 0.02 # 20 ms
    # 캐시된 문장 끝 타임스탬프의 최대 유효 기간 (ms)
    _SENTENCE_CACHE_MAX_AGE_MS: float = 0.2
    # 잠재적 문장 끝을 트리거하기 위해 캐시 유효 기간 내에 필요한 감지 횟수
    _SENTENCE_CACHE_TRIGGER_COUNT: int = 3


    def __init__(
            self,
            source_language: str = "en",
            realtime_transcription_callback: Optional[Callable[[str], None]] = None,
            full_transcription_callback: Optional[Callable[[str], None]] = None,
            potential_full_transcription_callback: Optional[Callable[[str], None]] = None,
            potential_full_transcription_abort_callback: Optional[Callable[[], None]] = None,
            potential_sentence_end: Optional[Callable[[str], None]] = None,
            before_final_sentence: Optional[Callable[[Optional[np.ndarray], Optional[str]], bool]] = None,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            on_recording_start_callback: Optional[Callable[[], None]] = None,
            on_prolonged_silence_callback: Optional[Callable[[], None]] = None,
            is_orpheus: bool = False,
            local: bool = True,
            tts_allowed_event: Optional[threading.Event] = None, # 참고: 제공된 원본 코드에서는 사용되지 않는 것으로 보임
            pipeline_latency: float = 0.5,
            recorder_config: Optional[Dict[str, Any]] = None, # 사용자 정의 설정 전달 허용
    ) -> None:
        """
        TranscriptionProcessor를 초기화합니다.

        Args:
            source_language: 음성 인식을 위한 언어 코드 (예: "en").
            realtime_transcription_callback: 실시간 음성 인식 업데이트를 위한 콜백. 부분 텍스트를 받습니다.
            full_transcription_callback: 최종 음성 인식 결과를 위한 콜백. 최종 텍스트를 받습니다.
            potential_full_transcription_callback: 전체 음성 인식이 임박했을 때("hot" 상태) 트리거되는 콜백. 현재 실시간 텍스트를 받습니다.
            potential_full_transcription_abort_callback: 최종 음성 인식 전에 "hot" 상태가 종료될 때 트리거되는 콜백.
            potential_sentence_end: 잠재적인 문장 끝이 감지될 때 트리거되는 콜백. 잠재적으로 완성된 문장 텍스트를 받습니다.
            before_final_sentence: 레코더가 음성 인식을 완료하기 직전에 트리거되는 콜백. 오디오 사본과 현재 실시간 텍스트를 받습니다. True를 반환하면 레코더 동작에 영향을 줄 수 있습니다(지원되는 경우).
            silence_active_callback: 침묵 감지 상태가 변경될 때 트리거되는 콜백. 불리언 값(침묵 활성 시 True)을 받습니다.
            on_recording_start_callback: 침묵 또는 깨우기 단어 후 레코더가 녹음을 시작할 때 트리거되는 콜백.
            on_prolonged_silence_callback: 사용자의 침묵이 5초 이상 지속될 때 트리거되는 콜백.
            is_orpheus: 'Orpheus' 모드에 대한 특정 타이밍 조정을 사용해야 하는지를 나타내는 플래그.
            local: TurnDetection(활성화된 경우)이 로컬 처리 컨텍스트인지 원격 처리 컨텍스트인지를 나타내는 데 사용되는 플래그.
            tts_allowed_event: TTS 합성이 허용될 때 설정될 수 있는 이벤트(현재 제공된 로직에서는 사용되지 않음).
            pipeline_latency: 다운스트림 처리 파이프라인의 예상 지연 시간(초). 타이밍 계산에 사용됩니다.
            recorder_config: 기본 RealtimeSTT 레코더 설정을 덮어쓰기 위한 선택적 딕셔너리.
        """
        self.source_language = source_language
        self.realtime_transcription_callback = realtime_transcription_callback
        self.full_transcription_callback = full_transcription_callback
        self.potential_full_transcription_callback = potential_full_transcription_callback
        self.potential_full_transcription_abort_callback = potential_full_transcription_abort_callback
        self.potential_sentence_end = potential_sentence_end
        self.before_final_sentence = before_final_sentence
        self.silence_active_callback = silence_active_callback
        self.on_recording_start_callback = on_recording_start_callback
        self.on_prolonged_silence_callback = on_prolonged_silence_callback
        self.is_orpheus = is_orpheus
        self.pipeline_latency = pipeline_latency
        self.recorder: Optional[AudioToTextRecorder | AudioToTextRecorderClient] = None
        self.is_silero_speech_active: bool = False # 참고: 사용되지 않는 것으로 보임
        self.silero_working: bool = False         # 참고: 사용되지 않는 것으로 보임
        self.on_wakeword_detection_start: Optional[Callable] = None # 참고: 사용되지 않는 것으로 보임
        self.on_wakeword_detection_end: Optional[Callable] = None   # 참고: 사용되지 않는 것으로 보임
        self.realtime_text: Optional[str] = None
        self.sentence_end_cache: List[Dict[str, Any]] = []
        self.potential_sentences_yielded: List[Dict[str, Any]] = []
        self.stripped_partial_user_text: str = ""
        self.final_transcription: Optional[str] = None
        self.shutdown_performed: bool = False
        self.silence_time: float = 0.0
        self.silence_active: bool = False
        self.last_audio_copy: Optional[np.ndarray] = None
        self.prolonged_silence_triggered: bool = False
        self.prolonged_silence_start_time: float = 0.0

        self.on_tts_allowed_to_synthesize: Optional[Callable] = None # 참고: 사용되지 않는 것으로 보임

        self.text_similarity = TextSimilarity(focus='end', n_words=5)

        # 제공된 설정 또는 기본값 사용
        self.recorder_config = copy.deepcopy(recorder_config if recorder_config else DEFAULT_RECORDER_CONFIG)
        self.recorder_config['language'] = self.source_language # 언어가 설정되었는지 확인

        if USE_TURN_DETECTION:
            logger.info(f"👂🔄 {Colors.YELLOW}발화자 전환 감지 활성화됨{Colors.RESET}")
            self.turn_detection = TurnDetection(
                on_new_waiting_time=self.on_new_waiting_time,
                local=local,
                pipeline_latency=pipeline_latency
            )

        self._create_recorder()
        self._start_silence_monitor()

    # --- 레코더 파라미터 추상화 ---

    def _get_recorder_param(self, param_name: str, default: Any = None) -> Any:
        """
        레코더 인스턴스에서 파라미터를 가져오는 내부 헬퍼.
        클라이언트/서버 차이를 추상화합니다.

        Args:
            param_name: 검색할 파라미터의 이름.
            default: 레코더가 초기화되지 않았거나 파라미터가 없을 때 반환할 값.

        Returns:
            레코더 파라미터의 값 또는 기본값.
        """
        if not self.recorder:
            return default
        if START_STT_SERVER:
            # get_parameter가 존재하고 잠재적 오류를 처리한다고 가정
            return self.recorder.get_parameter(param_name) # type: ignore
        else:
            # 로컬 버전의 경우 속성에 직접 접근
            return getattr(self.recorder, param_name, default)

    def _set_recorder_param(self, param_name: str, value: Any) -> None:
        """
        레코더 인스턴스에 파라미터를 설정하는 내부 헬퍼.
        클라이언트/서버 차이를 추상화합니다.

        Args:
            param_name: 설정할 파라미터의 이름.
            value: 파라미터에 설정할 값.
        """
        if not self.recorder:
            return
        if START_STT_SERVER:
            # set_parameter가 존재하고 잠재적 오류를 처리한다고 가정
            self.recorder.set_parameter(param_name, value) # type: ignore
        else:
            # 로컬 버전의 경우 속성을 직접 설정
            setattr(self.recorder, param_name, value)

    def _is_recorder_recording(self) -> bool:
        """
        레코더가 현재 녹음 중인지 확인하는 내부 헬퍼.
        클라이언트/서버 차이를 추상화합니다.

        Returns:
            레코더가 활성화되어 녹음 중이면 True, 그렇지 않으면 False.
        """
        if not self.recorder:
            return False
        if START_STT_SERVER:
            return self.recorder.get_parameter("is_recording") # type: ignore
        else:
            # 접근하기 전에 속성이 존재하는지 확인
            return getattr(self.recorder, "is_recording", False)

    # --- 침묵 모니터 ---
    def _start_silence_monitor(self) -> None:
        """
        침묵 지속 시간을 모니터링하고 잠재적 문장 끝 감지, TTS 합성 허용,
        그리고 잠재적 전체 음성 인식("hot") 상태 변경과 같은 이벤트를 트리거하는
        백그라운드 스레드를 시작합니다.
        """
        def monitor():
            hot = False
            # 추상화된 getter를 사용하여 silence_time 초기화
            self.silence_time = self._get_recorder_param("speech_end_silence_start", 0.0)

            while not self.shutdown_performed:
                # 사용자 턴 관리 로직
                speech_end_silence_start = self.silence_time
                if self.recorder and speech_end_silence_start is not None and speech_end_silence_start != 0:
                    silence_waiting_time = self._get_recorder_param("post_speech_silence_duration", 0.0)
                    time_since_silence = time.time() - speech_end_silence_start
                    latest_pipe_start_time = silence_waiting_time - self.pipeline_latency - self._PIPELINE_RESERVE_TIME_MS
                    potential_sentence_end_time = max(latest_pipe_start_time, self._MIN_POTENTIAL_END_DETECTION_TIME_MS)
                    start_hot_condition_time = max(silence_waiting_time - self._HOT_THRESHOLD_OFFSET_S, self._MIN_HOT_CONDITION_DURATION_S)

                    if self.is_orpheus:
                        orpheus_potential_end_time = silence_waiting_time - self._HOT_THRESHOLD_OFFSET_S
                        potential_sentence_end_time = max(potential_sentence_end_time, orpheus_potential_end_time)

                    if time_since_silence > potential_sentence_end_time:
                        current_text = self.realtime_text or ""
                        # logger.info(f"👂🔚 {Colors.YELLOW}잠재적 문장 끝 감지됨 (시간 초과){Colors.RESET}: {current_text}")
                        self.detect_potential_sentence_end(current_text, force_yield=True, force_ellipses=True)

                    tts_allowance_time = silence_waiting_time - self._TTS_ALLOWANCE_OFFSET_S
                    if time_since_silence > tts_allowance_time and self.on_tts_allowed_to_synthesize:
                        self.on_tts_allowed_to_synthesize()

                    hot_condition_met = time_since_silence > start_hot_condition_time
                    if hot_condition_met and not hot:
                        hot = True
                        print(f"{Colors.MAGENTA}HOT{Colors.RESET}")
                        if self.potential_full_transcription_callback:
                            self.potential_full_transcription_callback(self.realtime_text)
                    elif not hot_condition_met and hot and self._is_recorder_recording():
                        print(f"{Colors.CYAN}COLD (침묵 중){Colors.RESET}")
                        if self.potential_full_transcription_abort_callback:
                            self.potential_full_transcription_abort_callback()
                        hot = False
                
                elif hot and self._is_recorder_recording():
                    print(f"{Colors.CYAN}COLD (침묵 종료){Colors.RESET}")
                    if self.potential_full_transcription_abort_callback:
                        self.potential_full_transcription_abort_callback()
                    hot = False

                # 장시간 시스템 유휴 상태 감지 로직
                if self.prolonged_silence_start_time != 0.0:
                    time_since_idle = time.time() - self.prolonged_silence_start_time
                    if time_since_idle > 5.0 and not self.prolonged_silence_triggered:
                        logger.info("👂💤 5초 이상 시스템 유휴 상태 감지됨. AI 대화 시작.")
                        self.prolonged_silence_triggered = True
                        if self.on_prolonged_silence_callback:
                            self.on_prolonged_silence_callback()
                        self.stop_prolonged_silence_timer()

                time.sleep(0.01)

        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()

    def start_prolonged_silence_timer(self):
        """장시간 침묵 감지를 위한 타이머를 시작합니다."""
        if self.prolonged_silence_start_time == 0.0:
            logger.info("👂💤 장시간 침묵 타이머 시작됨.")
            self.prolonged_silence_start_time = time.time()
            self.prolonged_silence_triggered = False

    def stop_prolonged_silence_timer(self):
        """장시간 침묵 감지를 위한 타이머를 중지합니다."""
        if self.prolonged_silence_start_time != 0.0:
            logger.info("👂💤 장시간 침묵 타이머 중지됨.")
            self.prolonged_silence_start_time = 0.0

    def on_new_waiting_time(
            self,
            waiting_time: float,
            text: Optional[str] = None,
        ) -> None:
        """
        TurnDetection이 새로운 대기 시간을 계산했을 때의 콜백 핸들러.
        레코더의 post_speech_silence_duration 파라미터를 업데이트합니다.

        Args:
            waiting_time: 새로 계산된 침묵 지속 시간 (초).
            text: TurnDetection이 대기 시간을 계산하는 데 사용한 텍스트 (로깅용).
        """
        if self.recorder:
            current_duration = self._get_recorder_param("post_speech_silence_duration")
            if current_duration != waiting_time:
                log_text = text if text else "(제공된 텍스트 없음)"
                logger.info(f"👂⏳ {Colors.GRAY}새로운 대기 시간: {Colors.RESET}{Colors.YELLOW}{waiting_time:.2f}{Colors.RESET}{Colors.GRAY} (텍스트: {log_text}){Colors.RESET}")
                self._set_recorder_param("post_speech_silence_duration", waiting_time)
        else:
            logger.warning("👂⚠️ 레코더가 초기화되지 않아 새 대기 시간을 설정할 수 없습니다.")

    def transcribe_loop(self) -> None:
        """
        레코더와 함께 최종 음성 인식 콜백 메커니즘을 설정합니다.

        이 메서드는 완전한 발화 음성 인식이 가능할 때 레코더에 의해 호출될
        `on_final` 콜백을 정의합니다. 그런 다음 이 콜백을 레코더 인스턴스에
        등록합니다.
        """
        def on_final(text: Optional[str]):
            if text is None or text == "":
                logger.warning("👂❓ 최종 음성 인식 결과로 None 또는 빈 문자열을 받았습니다.")
                return

            self.final_transcription = text
            logger.info(f"👂✅ {Colors.apply('최종 사용자 발화: ').green} {Colors.apply(text).yellow}")
            self.sentence_end_cache.clear()
            self.potential_sentences_yielded.clear()

            if USE_TURN_DETECTION and hasattr(self, 'turn_detection'):
                self.turn_detection.reset()
            if self.full_transcription_callback:
                self.full_transcription_callback(text)

        if self.recorder:
            # 특정 메서드는 클라이언트/로컬 STT 버전에 따라 다를 수 있음
            # 공통 'text' 메서드가 존재하거나 적용된다고 가정
            if hasattr(self.recorder, 'text'):
                self.recorder.text(on_final) # type: ignore # 메서드가 존재한다고 가정
            elif START_STT_SERVER:
                 logger.warning("👂⚠️ 레코더 클라이언트에 'text' 메서드가 없습니다. 'on_final_transcription' 파라미터 설정을 시도합니다.")
                 # 클라이언트를 위해 파라미터를 통해 설정 시도, 정확한 API가 아닐 수 있음
                 try:
                     self._set_recorder_param('on_final_transcription', on_final)
                 except Exception as e:
                     logger.error(f"👂💥 클라이언트의 최종 음성 인식 콜백 파라미터 설정 실패: {e}")
            else:
                logger.warning("👂⚠️ 로컬 레코더 객체에 최종 콜백을 위한 'text' 메서드가 없습니다.")
        else:
            logger.error("👂❌ 최종 콜백을 설정할 수 없음: 레코더가 초기화되지 않았습니다.")


    def abort_generation(self) -> None:
        """
        잠재적으로 생성된 문장들의 캐시를 지웁니다.

        이는 이전에 감지된 잠재적 문장 끝에 기반하여 트리거될 수 있는
        추가적인 동작들을 효과적으로 중지시키며, 처리를 리셋하거나
        외부에서 중단해야 할 때 유용합니다.
        """
        self.potential_sentences_yielded.clear()
        logger.info("👂⏹️ 잠재적 문장 생성 캐시가 비워졌습니다 (생성 중단).")

    def perform_final(self, audio_bytes: Optional[bytes] = None) -> None:
        """
        마지막으로 알려진 실시간 텍스트를 사용하여 최종 음성 인식 프로세스를
        수동으로 트리거합니다.

        이는 레코더의 자연스러운 발화 종료 감지를 우회하고 즉시
        `self.realtime_text` 내용으로 최종 음성 인식 콜백을 호출합니다.
        외부에서 음성 인식을 완료해야 하는 시나리오에 유용합니다.

        Args:
            audio_bytes: 선택적 오디오 데이터 (현재 이 메서드의 로직에서는
                         사용되지 않지만, 향후 사용 또는 API 일관성을 위해 유지됨).
        """
        if self.recorder: # 레코더 존재 여부를 주로 게이트키퍼로 확인
            if self.realtime_text is None:
                logger.warning(f"👂❓ {Colors.RED}강제로 최종 음성 인식을 수행하지만 realtime_text가 None입니다. 빈 문자열을 사용합니다.{Colors.RESET}")
                current_text = ""
            else:
                current_text = self.realtime_text

            self.final_transcription = current_text # 내부 상태 업데이트
            logger.info(f"👂❗ {Colors.apply('강제 최종 사용자 발화: ').green} {Colors.apply(current_text).yellow}")
            self.sentence_end_cache.clear()
            self.potential_sentences_yielded.clear()

            if USE_TURN_DETECTION and hasattr(self, 'turn_detection'):
                self.turn_detection.reset()
            if self.full_transcription_callback:
                self.full_transcription_callback(current_text)
        else:
            logger.warning("👂⚠️ 최종 작업을 수행할 수 없음: 레코더가 초기화되지 않았습니다.")


    def _normalize_text(self, text: str) -> str:
        """
        비교 목적으로 텍스트를 정규화하는 내부 헬퍼.
        소문자로 변환하고, 알파벳/숫자가 아닌 문자(공백 제외)를 제거하고,
        여분의 공백을 축소합니다.

        Args:
            text: 정규화할 입력 문자열.

        Returns:
            정규화된 문자열.
        """
        text = text.lower()
        # 모든 알파벳/숫자가 아닌 문자 제거 (공백 유지)
        text = re.sub(r'[^a-z0-9\s]', '', text) # SequenceMatcher를 위해 공백 유지 / 영어 전용
        # text = re.sub(r'[^\w\s]', '', text, flags=re.UNICODE) # 다국어용 설정, 적용 시 language 설정 모두 변경 필요
        # 여분의 공백 제거 및 양쪽 끝 공백 제거
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def is_basically_the_same(
            self,
            text1: str,
            text2: str,
            similarity_threshold: float = 0.96, # 설정 가능하게 만드는 것을 고려
        ) -> bool:
        """
        두 텍스트 문자열이 매우 유사한지 확인하며, 끝 단어에 초점을 맞춥니다.
        내부 TextSimilarity 인스턴스를 사용합니다.

        Args:
            text1: 첫 번째 텍스트 문자열.
            text2: 두 번째 텍스트 문자열.
            similarity_threshold: 텍스트를 동일하다고 간주할 최소 유사도 점수 (0에서 1).

        Returns:
            유사도 점수가 임계값을 초과하면 True, 그렇지 않으면 False.
        """
        # 전용 TextSimilarity 클래스 인스턴스 사용
        similarity = self.text_similarity.calculate_similarity(text1, text2)
        return similarity > similarity_threshold

    def detect_potential_sentence_end(self, text: Optional[str], force_yield: bool = False, force_ellipses: bool = False) -> None:
        """
        끝나는 구두점과 시간적 안정성을 기반으로 잠재적인 문장 끝을 감지합니다.

        제공된 텍스트가 문장 끝 구두점(., !, ?)으로 끝나는지 확인합니다.
        그렇다면, 정규화된 텍스트와 타임스탬프를 캐시합니다. 동일한 정규화된
        텍스트 끝이 짧은 시간 내에 자주 나타나거나 `force_yield`가 True인 경우
        (예: 침묵 시간 초과로 인해), `potential_sentence_end` 콜백을 트리거하여
        동일한 문장에 대한 중복 트리거를 피합니다.

        Args:
            text: 확인할 실시간 음성 인식 텍스트.
            force_yield: True인 경우, 구두점 및 타이밍 확인을 우회하고 콜백 트리거를
                         강제합니다 (텍스트가 유효하고 아직 생성되지 않은 경우).
            force_ellipses: True인 경우 (`force_yield`와 함께 사용), "..."를
                            문장 끝으로 간주하도록 허용합니다.
        """
        if not text:
            return

        stripped_text_raw = text.strip() # 콜백을 위해 원본 유지
        if not stripped_text_raw:
            return

        # 강제되지 않는 한 말줄임표는 문장 끝으로 간주하지 않음
        if stripped_text_raw.endswith("...") and not force_ellipses:
            return

        end_punctuations = [".", "!", "?"]
        now = time.time()

        # 텍스트가 표준 구두점으로 끝나거나 강제된 경우에만 진행
        ends_with_punctuation = any(stripped_text_raw.endswith(p) for p in end_punctuations)
        if not ends_with_punctuation and not force_yield:
            return

        normalized_text = self._normalize_text(stripped_text_raw)
        if not normalized_text: # 정규화 후 빈 문자열이 되는 경우 처리
            return

        # --- 캐시 관리 ---
        entry_found = None
        for entry in self.sentence_end_cache:
            # 정규화된 텍스트가 캐시된 항목과 일치하는지 확인
            if self.is_basically_the_same(entry['text'], normalized_text):
                entry_found = entry
                break

        if entry_found:
            entry_found['timestamps'].append(now)
            # 최근 타임스탬프만 유지
            entry_found['timestamps'] = [t for t in entry_found['timestamps'] if now - t <= self._SENTENCE_CACHE_MAX_AGE_MS]
        else:
            # 새 항목 추가
            entry_found = {'text': normalized_text, 'timestamps': [now]}
            self.sentence_end_cache.append(entry_found)
            # 선택 사항: 시간이 지남에 따라 캐시 크기가 너무 커지면 제한
            # MAX_CACHE_SIZE = 50
            # if len(self.sentence_end_cache) > MAX_CACHE_SIZE:
            #     self.sentence_end_cache.pop(0) # 가장 오래된 것 제거

        # --- 생성(Yielding) 로직 ---
        should_yield = False
        if force_yield:
            should_yield = True
        # 동일한 문장 끝이 최근에 여러 번 나타난 경우 생성
        elif ends_with_punctuation and len(entry_found['timestamps']) >= self._SENTENCE_CACHE_TRIGGER_COUNT:
             should_yield = True


        if should_yield:
            # 이 *정확한* 정규화된 텍스트가 최근에 이미 생성되었는지 확인
            already_yielded = False
            for yielded_entry in self.potential_sentences_yielded:
                # 생성된 텍스트에 대해 동일한 유사성 검사 사용
                if self.is_basically_the_same(yielded_entry['text'], normalized_text):
                    already_yielded = True
                    break

            if not already_yielded:
                # 생성된 목록에 추가 (비교를 위해 정규화된 텍스트 사용, 타임스탬프 유지)
                self.potential_sentences_yielded.append({'text': normalized_text, 'timestamp': now})
                # 선택 사항: 생성된 목록 크기 제한
                # MAX_YIELDED_SIZE = 20
                # if len(self.potential_sentences_yielded) > MAX_YIELDED_SIZE:
                #    self.potential_sentences_yielded.pop(0)

                logger.info(f"👂➡️ 잠재적 문장 끝 생성: {stripped_text_raw}")
                if self.potential_sentence_end:
                    self.potential_sentence_end(stripped_text_raw) # 원본 구두점으로 콜백
            # else: # 매번 이것을 로깅할 필요는 없음, 시끄러울 수 있음
                 # logger.debug(f"👂➡️ 문장 '{normalized_text}'가 생성된 '{yielded_entry.get('text', '')}'와 일치하여 다시 생성하지 않음.")


    def set_silence(self, silence_active: bool) -> None:
        """
        내부 침묵 상태를 업데이트하고 silence_active_callback을 트리거합니다.

        Args:
            silence_active: 새로운 침묵 상태 (침묵이 활성화되면 True).
        """
        if self.silence_active != silence_active:
            self.silence_active = silence_active
            logger.info(f"👂🤫 침묵 상태 변경: {'활성' if silence_active else '비활성'}")
            if self.silence_active_callback:
                self.silence_active_callback(silence_active)

    def get_last_audio_copy(self) -> Optional[np.ndarray]:
        """
        마지막으로 성공적으로 캡처된 오디오 버퍼를 float32 NumPy 배열로 반환합니다.

        먼저 현재 오디오 버퍼를 가져오려고 시도합니다. 성공하면 내부 캐시(`last_audio_copy`)를
        업데이트하고 새 사본을 반환합니다. 현재 버퍼를 가져오는 데 실패하면(예: 레코더가
        준비되지 않았거나 버퍼가 비어 있음), 이전에 캐시된 버퍼를 반환합니다.

        Returns:
            오디오를 나타내는 float32 NumPy 배열( [-1.0, 1.0]으로 정규화됨),
            또는 아직 성공적으로 캡처된 오디오가 없으면 None.
        """
        # 먼저 현재 오디오를 가져오려고 시도
        audio_copy = self.get_audio_copy()

        if audio_copy is not None and len(audio_copy) > 0:
            # 성공하면 새 사본을 업데이트하고 반환
            # self.last_audio_copy는 성공 시 get_audio_copy() 내에서 이미 업데이트됨
            return audio_copy
        else:
            # 현재 오디오 가져오기에 실패하면 마지막으로 알려진 양호한 사본을 반환
            logger.debug("👂💾 현재 가져오기가 실패했거나 비어 있으므로 마지막으로 알려진 오디오 사본을 반환합니다.")
            return self.last_audio_copy

    def get_audio_copy(self) -> Optional[np.ndarray]:
        """
        레코더의 프레임에서 현재 오디오 버퍼를 복사합니다.

        원시 오디오 프레임을 검색하고, 연결하고, float32 NumPy 배열로 변환하여
        [-1.0, 1.0]으로 정규화하고, 깊은 복사본을 반환합니다. 성공하면
        `self.last_audio_copy`를 업데이트합니다. 레코더를 사용할 수 없거나,
        프레임이 비어 있거나, 오류가 발생하면 `last_audio_copy`를 반환합니다.

        Returns:
            현재 오디오 버퍼의 깊은 복사본인 float32 NumPy 배열,
            또는 현재 가져오기가 실패한 경우 마지막으로 알려진 양호한 사본, 또는
            성공적으로 캡처된 오디오가 없는 경우 None.
        """
        if not self.recorder:
             logger.warning("👂⚠️ 오디오 사본을 가져올 수 없음: 레코더가 초기화되지 않았습니다.")
             return self.last_audio_copy # 사용 가능한 경우 마지막으로 알려진 양호한 사본 반환
        if not hasattr(self.recorder, 'frames'):
             logger.warning("👂⚠️ 오디오 사본을 가져올 수 없음: 레코더에 'frames' 속성이 없습니다.")
             return self.last_audio_copy

        try:
             # 프레임에 안전하게 접근
             # 동시에 접근하는 경우 프레임이 스레드로부터 안전한지 확인
             with self.recorder.frames_lock if hasattr(self.recorder, 'frames_lock') else threading.Lock(): # 레코더의 잠금을 사용할 수 있는 경우 사용
                 frames_data = list(self.recorder.frames) # 데크 항목의 사본 생성

             if not frames_data:
                 logger.debug("👂💾 레코더 프레임 버퍼가 현재 비어 있습니다.")
                 return self.last_audio_copy # 현재가 비어 있으면 마지막으로 알려진 것 반환

             # 오디오 버퍼 처리
             full_audio_array = np.frombuffer(b''.join(frames_data), dtype=np.int16)
             if full_audio_array.size == 0:
                 logger.debug("👂💾 레코더 프레임 버퍼를 합친 후 빈 배열이 되었습니다.")
                 return self.last_audio_copy # 합친 후 버퍼가 비어 있으면 마지막으로 알려진 것 반환

             full_audio = full_audio_array.astype(np.float32) / INT16_MAX_ABS_VALUE
             # 여기서 deepcopy는 필요 없음. full_audio는 버퍼에서 파생된 새 배열임
             audio_copy = full_audio

             # 새 사본이 유효하고 데이터가 있는 경우에만 last_audio_copy 업데이트
             if audio_copy is not None and len(audio_copy) > 0:
                 self.last_audio_copy = audio_copy
                 logger.debug(f"👂💾 오디오 사본을 성공적으로 가져왔습니다 (길이: {len(audio_copy)} 샘플).")


             return audio_copy
        except Exception as e:
             logger.error(f"👂💥 오디오 사본 가져오기 오류: {e}", exc_info=True)
             return self.last_audio_copy # 오류 시 마지막으로 알려진 것 반환

    def _create_recorder(self) -> None:
        """
        지정된 설정과 콜백으로 RealtimeSTT 레코더 인스턴스(로컬 또는 클라이언트)를
        초기화하는 내부 헬퍼.
        """

        # `self`를 캡처하기 위해 콜백을 로컬로 정의
        def start_silence_detection():
            """레코더가 침묵 시작(발화 종료)을 감지할 때 트리거되는 콜백."""
            self.set_silence(True)
            # 침묵 시작 시간을 즉시 캡처. 가능한 경우 레코더의 시간 사용.
            recorder_silence_start = self._get_recorder_param("speech_end_silence_start", None)
            self.silence_time = recorder_silence_start if recorder_silence_start else time.time()
            logger.debug(f"👂🤫 침묵 감지됨 (start_silence_detection 호출됨). 침묵 시간 설정: {self.silence_time}")


        def stop_silence_detection():
            """레코더가 침묵 종료(발화 시작)를 감지할 때 트리거되는 콜백."""
            self.set_silence(False)
            self.silence_time = 0.0 # 침묵 시간 리셋
            self.stop_prolonged_silence_timer()
            logger.debug("👂🗣️ 발화 감지됨 (stop_silence_detection 호출됨). 침묵 시간 리셋됨.")


        def start_recording():
            """레코더가 새 녹음 세그먼트를 시작할 때 트리거되는 콜백."""
            logger.info("👂▶️ 녹음 시작됨.")
            self.set_silence(False) # 침묵이 비활성으로 표시되도록 보장
            self.silence_time = 0.0   # 침묵 타이머가 리셋되도록 보장
            self.stop_prolonged_silence_timer()
            if self.on_recording_start_callback:
                self.on_recording_start_callback()

        def stop_recording() -> bool:
            """
            레코더가 녹음 세그먼트를 중지할 때, 최종 음성 인식이 생성되기
            직전에 트리거되는 콜백.
            """
            logger.info("👂⏹️ 녹음 중지됨.")
            # 레코더가 최종 처리를 위해 오디오를 지우기 *전에* 오디오 가져오기
            audio_copy = self.get_last_audio_copy() # 견고성을 위해 get_last_audio_copy 사용
            if self.before_final_sentence:
                logger.debug("👂➡️ before_final_sentence 콜백 호출 중...")
                # 오디오와 *현재* 실시간 텍스트 전달
                try:
                    # 반환 값이 레코더에 영향을 줄 수 있으므로 그대로 전달.
                    # 콜백이 None을 반환하거나 오류를 발생시키면 기본값 False
                    result = self.before_final_sentence(audio_copy, self.realtime_text)
                    return result if isinstance(result, bool) else False
                except Exception as e:
                    logger.error(f"👂💥 before_final_sentence 콜백 오류: {e}", exc_info=True)
                    return False # 오류 시 False가 반환되도록 보장
            return False # 콜백이 없거나 True를 반환하지 않으면 아무 조치도 취하지 않았음을 나타냄

        def on_partial(text: Optional[str]):
            """실시간 음성 인식 업데이트를 위해 트리거되는 콜백."""
            if text is None:
                # logger.warning(f"👂❓ {Colors.RED}부분 텍스트로 None을 받음{Colors.RESET}") # 시끄러울 수 있음
                return
            self.realtime_text = text # 최신 실시간 텍스트 업데이트

            # 구두점 안정성을 기반으로 잠재적 문장 끝 감지
            self.detect_potential_sentence_end(text)

            # 부분 음성 인식 콜백 및 발화자 전환 감지를 위한 처리
            stripped_partial_user_text_new = strip_ending_punctuation(text)
            # 디버그 수준에 따라 중요한 변경 사항만 로깅하거나 모든 부분 로깅
            if stripped_partial_user_text_new != self.stripped_partial_user_text:
                self.stripped_partial_user_text = stripped_partial_user_text_new
                logger.info(f"👂📝 부분 음성 인식: {Colors.CYAN}{text}{Colors.RESET}")
                if self.realtime_transcription_callback:
                    self.realtime_transcription_callback(text)
                if USE_TURN_DETECTION and hasattr(self, 'turn_detection'):
                    self.turn_detection.calculate_waiting_time(text=text)
            else: # 덜 중요한 업데이트는 다르게 로깅 (선택 사항, 필요 시 주석 해제)
                 logger.debug(f"👂📝 부분 음성 인식 (제거 후 변경 없음): {Colors.GRAY}{text}{Colors.RESET}")


        # --- 레코더 설정 준비 ---
        # 인스턴스의 설정(기본 또는 사용자 제공)으로 시작
        active_config = self.recorder_config.copy()

        # AudioToTextRecorder의 올바른 키를 사용하여 동적으로 할당된 콜백 추가
        active_config["on_realtime_transcription_update"] = on_partial
        # *** 수정된 매핑 ***
        active_config["on_turn_detection_start"] = start_silence_detection # 침묵이 시작될 때(발화 종료) 트리거됨
        active_config["on_turn_detection_stop"] = stop_silence_detection  # 침묵이 멈출 때(발화 시작) 트리거됨
        # *** 수정 끝 ***
        active_config["on_recording_start"] = start_recording
        active_config["on_recording_stop"] = stop_recording # 이 콜백은 최종 텍스트 전에 발생

        # 사용 중인 설정 로깅
        def _pretty(v, max_len=60):
            if callable(v): return f"[콜백: {v.__name__}]"
            if isinstance(v, str):
                one_line = v.replace("\n", " ")
                return (one_line[:max_len].rstrip() + " [...]") if len(one_line) > max_len else one_line
            return v

        pretty_cfg = {k: _pretty(v) for k, v in active_config.items()}
        # 필요한 경우 민감하거나 너무 긴 항목 처리
        # 예: if 'api_key' in pretty_cfg: pretty_cfg['api_key'] = '********'
        padded_cfg = textwrap.indent(json.dumps(pretty_cfg, indent=2), "    ")

        recorder_type = "AudioToTextRecorderClient" if START_STT_SERVER else "AudioToTextRecorder"
        logger.info(f"👂⚙️ 다음 파라미터로 {recorder_type} 생성 중:")
        print(Colors.apply(padded_cfg).blue) # 로거가 망가뜨릴 수 있으므로 포맷된 JSON에는 print 사용


        # --- 레코더 인스턴스화 ---
        try:
            if START_STT_SERVER:
                # 참고: 클라이언트는 다른 콜백 이름을 사용할 수 있으므로 필요 시 조정
                # 지금은 동일한 이름을 수락하거나 내부적으로 처리한다고 가정
                self.recorder = AudioToTextRecorderClient(**active_config)
                # 필요한 경우 깨우기 단어 비활성화 (설정 딕셔너리를 통해서도 가능)
                self._set_recorder_param("use_wake_words", False)
            else:
                # 수정된 active_config로 로컬 레코더 인스턴스화
                self.recorder = AudioToTextRecorder(**active_config)
                # 필요한 경우 깨우기 단어 비활성화 (파라미터 설정을 통해 이중 확인)
                self._set_recorder_param("use_wake_words", False) # 헬퍼 메서드 사용

            logger.info(f"👂✅ {recorder_type} 인스턴스가 성공적으로 생성되었습니다.")

        except Exception as e:
            # 상세 디버깅을 위해 트레이스백과 함께 예외 로깅
            logger.exception(f"👂🔥 레코더 생성 실패: {e}")
            self.recorder = None # 생성 실패 시 레코더가 None인지 확인

    def feed_audio(self, chunk: bytes, audio_meta_data: Optional[Dict[str, Any]] = None) -> None:
        """
        처리를 위해 오디오 청크를 기본 레코더 인스턴스에 공급합니다.

        Args:
            chunk: 원시 오디오 데이터 청크를 포함하는 바이트 객체.
            audio_meta_data: 레코더에 필요한 경우 오디오에 대한 메타데이터
                             (예: 샘플 레이트, 채널)를 포함하는 선택적 딕셔너리.
        """
        if self.recorder and not self.shutdown_performed:
            try:
                # feed_audio가 메타데이터를 예상하는지 확인하고 가능한 경우 제공
                if START_STT_SERVER:
                     # 클라이언트는 특정 형식의 메타데이터를 요구할 수 있음
                     self.recorder.feed_audio(chunk) # 클라이언트가 내부적으로 메타데이터를 처리하거나 청크당 필요하지 않다고 가정
                else:
                     # 로컬 레코더는 제공된 경우 메타데이터를 사용할 수 있음
                     self.recorder.feed_audio(chunk) # 로컬도 현재로서는 유사하게 처리한다고 가정

                logger.debug(f"👂🔊 크기 {len(chunk)} 바이트의 오디오 청크를 레코더에 공급했습니다.")
            except Exception as e:
                logger.error(f"👂💥 오디오를 레코더에 공급하는 중 오류 발생: {e}")
        elif not self.recorder:
            logger.warning("👂⚠️ 오디오를 공급할 수 없음: 레코더가 초기화되지 않았습니다.")
        elif self.shutdown_performed:
            logger.debug("👂🚫 오디오를 공급할 수 없음: 이미 종료 작업이 수행되었습니다.")
        # shutdown_performed가 True인 경우 예상된 동작이므로 경고 없음

    def shutdown(self) -> None:
        """
        레코더 인스턴스를 종료하고, 리소스를 정리하며, 추가 처리를 방지합니다.
        `shutdown_performed` 플래그를 설정합니다.
        """
        if not self.shutdown_performed:
            logger.info("👂🔌 TranscriptionProcessor 종료 중...")
            self.shutdown_performed = True # 루프/스레드를 중지하기 위해 플래그를 일찍 설정

            if self.recorder:
                logger.info("👂🔌 레코더 shutdown() 호출 중...")
                try:
                    self.recorder.shutdown()
                    logger.info("👂🔌 레코더 shutdown() 메서드가 완료되었습니다.")
                except Exception as e:
                    logger.error(f"👂💥 레코더 종료 중 오류 발생: {e}", exc_info=True)
                finally:
                    self.recorder = None
            else:
                logger.info("👂🔌 종료할 활성 레코더 인스턴스가 없습니다.")

            # 필요한 경우 다른 리소스 정리 (예: 발화자 전환 감지?)
            if USE_TURN_DETECTION and hasattr(self, 'turn_detection') and hasattr(self.turn_detection, 'shutdown'):
                logger.info("👂🔌 TurnDetection 종료 중...")
                try:
                    self.turn_detection.shutdown() # 예: TurnDetection에 shutdown 메서드가 있다고 가정
                except Exception as e:
                     logger.error(f"👂💥 TurnDetection 종료 중 오류 발생: {e}", exc_info=True)

            logger.info("👂🔌 TranscriptionProcessor 종료 프로세스가 완료되었습니다.")
        else:
            logger.info("👂ℹ️ 이미 종료 작업이 수행되었습니다.")
