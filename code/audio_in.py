import asyncio
import logging
from typing import Optional, Callable
import numpy as np
from scipy.signal import resample_poly
from transcribe import TranscriptionProcessor

logger = logging.getLogger(__name__)


class AudioInputProcessor:
    """
    오디오 입력을 관리하고, 음성 인식을 위해 처리하며, 관련된 콜백을 처리합니다.

    이 클래스는 원시 오디오 청크를 수신하여 필요한 형식(16kHz)으로 리샘플링하고,
    이를 내부 `TranscriptionProcessor`에 공급하며, 실시간 음성 인식 업데이트,
    녹음 시작 이벤트 및 침묵 감지에 대한 콜백을 관리합니다.
    또한 음성 인식 프로세스를 백그라운드 작업으로 실행합니다.
    """

    _RESAMPLE_RATIO = 3  # 48kHz(입력으로 가정)에서 16kHz로의 리샘플링 비율.

    def __init__(
            self,
            language: str = "en",
            is_orpheus: bool = False,
            silence_active_callback: Optional[Callable[[bool], None]] = None,
            on_prolonged_silence_callback: Optional[Callable[[], None]] = None,
            pipeline_latency: float = 0.5,
        ) -> None:
        """
        AudioInputProcessor를 초기화합니다.

        Args:
            language: 음성 인식을 위한 대상 언어 코드 (예: "en").
            is_orpheus: 특정 모델 변형을 사용해야 하는지를 나타내는 플래그.
            silence_active_callback: 침묵 상태가 변경될 때 호출되는 선택적 콜백 함수.
                                     불리언 인수(침묵이 활성화된 경우 True)를 받습니다.
            on_prolonged_silence_callback: 사용자의 침묵이 5초 이상 지속될 때 트리거되는 콜백.
            pipeline_latency: 처리 파이프라인의 예상 지연 시간(초).
        """
        self.last_partial_text: Optional[str] = None
        self.transcriber = TranscriptionProcessor(
            language,
            on_recording_start_callback=self._on_recording_start,
            silence_active_callback=self._silence_active_callback,
            on_prolonged_silence_callback=on_prolonged_silence_callback,
            is_orpheus=is_orpheus,
            pipeline_latency=pipeline_latency,
        )
        # 음성 인식 루프가 치명적으로 실패했는지를 나타내는 플래그
        self._transcription_failed = False
        self.transcription_task = asyncio.create_task(self._run_transcription_loop())


        self.realtime_callback: Optional[Callable[[str], None]] = None
        self.recording_start_callback: Optional[Callable[[None], None]] = None # 타입 조정
        self.silence_active_callback: Optional[Callable[[bool], None]] = silence_active_callback
        self.interrupted = False # TODO: 사용법 명확화 또는 이름 변경 고려 (사용자 발화에 의해 중단되었는가?)

        self._setup_callbacks()
        logger.info("👂🚀 AudioInputProcessor가 초기화되었습니다.")

    def _silence_active_callback(self, is_active: bool) -> None:
        """침묵 감지 상태를 위한 내부 콜백 릴레이."""
        if self.silence_active_callback:
            self.silence_active_callback(is_active)

    def _on_recording_start(self) -> None:
        """음성 인식기가 녹음을 시작할 때 트리거되는 내부 콜백 릴레이."""
        if self.recording_start_callback:
            self.recording_start_callback()

    def abort_generation(self) -> None:
        """진행 중인 모든 생성 프로세스를 중단하도록 내부 음성 인식기에 신호를 보냅니다."""
        logger.info("👂🛑 생성 중단이 요청되었습니다.")
        self.transcriber.abort_generation()

    def _setup_callbacks(self) -> None:
        """TranscriptionProcessor 인스턴스를 위한 내부 콜백을 설정합니다."""
        def partial_transcript_callback(text: str) -> None:
            """음성 인식기로부터의 부분적인 음성 인식 결과를 처리합니다."""
            if text != self.last_partial_text:
                self.last_partial_text = text
                if self.realtime_callback:
                    self.realtime_callback(text)

        self.transcriber.realtime_transcription_callback = partial_transcript_callback

    async def _run_transcription_loop(self) -> None:
        """
        백그라운드 asyncio 작업에서 음성 인식 루프를 계속 실행합니다.

        내부 `transcribe_loop`를 반복적으로 호출합니다. `transcribe_loop`가
        정상적으로 완료되면(한 사이클 완료), 이 루프는 다시 호출합니다.
        `transcribe_loop`에서 예외가 발생하면 치명적인 오류로 처리하고,
        플래그를 설정한 후 이 루프는 종료됩니다. CancelledError는 별도로 처리합니다.
        """
        task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
        logger.info(f"👂▶️ 백그라운드 음성 인식 작업을 시작합니다 ({task_name}).")
        while True: # transcribe_loop를 계속 호출하기 위해 루프 복원
            try:
                # 내부 블로킹 루프의 한 사이클 실행
                await asyncio.to_thread(self.transcriber.transcribe_loop)
                # transcribe_loop가 오류 없이 반환되면 한 사이클이 완료된 것임.
                # `while True`는 다시 호출될 것을 보장함.
                logger.debug("👂✅ TranscriptionProcessor.transcribe_loop가 한 사이클 완료했습니다.")
                # transcribe_loop가 즉시 반환될 경우의 타이트 루프를 방지하기 위해 작은 sleep 추가
                await asyncio.sleep(0.01)
            except asyncio.CancelledError:
                logger.info(f"👂🚫 음성 인식 루프({task_name})가 취소되었습니다.")
                # 취소 시 실패 플래그 설정 안 함
                break # while 루프 종료
            except Exception as e:
                # transcribe_loop 내에서 실제 오류 발생
                logger.error(f"👂💥 음성 인식 루프({task_name})에서 치명적인 오류가 발생했습니다: {e}. 루프가 종료됩니다.", exc_info=True)
                self._transcription_failed = True # 실패 플래그 설정
                break # while 루프 종료, 재시도 중단

        logger.info(f"👂⏹️ 백그라운드 음성 인식 작업({task_name})이 종료되었습니다.")


    def process_audio_chunk(self, raw_bytes: bytes) -> np.ndarray:
        """
        원시 오디오 바이트(int16)를 16kHz 16비트 PCM numpy 배열로 반환합니다.

        오디오는 정확한 리샘플링을 위해 float32로 변환된 후 다시 int16으로
        변환되며, 유효 범위를 벗어나는 값은 클리핑됩니다.

        Args:
            raw_bytes: int16 형식으로 가정되는 원시 오디오 데이터.

        Returns:
            16kHz에서 int16 형식으로 리샘플링된 오디오를 포함하는 numpy 배열.
            입력이 조용하면 0으로 채워진 배열을 반환합니다.
        """
        raw_audio = np.frombuffer(raw_bytes, dtype=np.int16)

        if np.max(np.abs(raw_audio)) == 0:
            # 침묵에 대한 리샘플링 후 예상 길이 계산
            expected_len = int(np.ceil(len(raw_audio) / self._RESAMPLE_RATIO))
            return np.zeros(expected_len, dtype=np.int16)

        # 리샘플링 정밀도를 위해 float32로 변환
        audio_float32 = raw_audio.astype(np.float32)

        # float32 데이터를 사용하여 리샘플링
        resampled_float = resample_poly(audio_float32, 1, self._RESAMPLE_RATIO)

        # 유효성을 보장하기 위해 클리핑하여 int16으로 다시 변환
        resampled_int16 = np.clip(resampled_float, -32768, 32767).astype(np.int16)

        return resampled_int16


    async def process_chunk_queue(self, audio_queue: asyncio.Queue) -> None:
        """
        asyncio 큐에서 수신된 오디오 청크를 계속 처리합니다.

        오디오 데이터를 검색하고, `process_audio_chunk`를 사용하여 처리하며,
        중단되거나 음성 인식 작업이 실패하지 않는 한 그 결과를 음성 인식기에
        공급합니다. 큐에서 `None`을 수신하거나 오류 발생 시 중지됩니다.

        Args:
            audio_queue: 'pcm'(원시 오디오 바이트)을 포함하는 딕셔너리 또는
                         종료를 위한 None을 생성할 것으로 예상되는 asyncio 큐.
        """
        logger.info("👂▶️ 오디오 청크 처리 루프를 시작합니다.")
        while True:
            try:
                # 항목을 가져오기 *전에* 음성 인식 작업이 영구적으로 실패했는지 확인
                if self._transcription_failed:
                    logger.error("👂🛑 이전에 음성 인식 작업이 실패했습니다. 오디오 처리를 중지합니다.")
                    break # 음성 인식 백엔드가 다운된 경우 처리 중지

                # 작업이 예기치 않게 종료되었는지 확인 (예: 실패하지 않고 취소됨)
                # 종료 중 self.transcription_task가 None일 수 있으므로 존재 여부 확인 필요
                if self.transcription_task and self.transcription_task.done() and not self._transcription_failed:
                     # 작업이 완료된 경우 예외 상태 확인 시도
                    task_exception = self.transcription_task.exception()
                    if task_exception and not isinstance(task_exception, asyncio.CancelledError):
                        # CancelledError 이외의 예외가 있었으면 실패로 처리.
                        logger.error(f"👂🛑 음성 인식 작업이 예기치 않은 오류로 종료되었습니다: {task_exception}. 오디오 처리를 중지합니다.", exc_info=task_exception)
                        self._transcription_failed = True # 실패로 표시
                        break
                    else:
                         # 깨끗하게 종료되었거나 취소됨
                        logger.warning("👂⏹️ 음성 인식 작업이 더 이상 실행되지 않습니다 (완료 또는 취소됨). 오디오 처리를 중지합니다.")
                        break # 처리 중지

                audio_data = await audio_queue.get()
                if audio_data is None:
                    logger.info("👂🔌 오디오 처리를 위한 종료 신호를 수신했습니다.")
                    break  # 종료 신호

                pcm_data = audio_data.pop("pcm")

                # 오디오 청크 처리 (리샘플링은 float32를 통해 일관되게 발생)
                processed = self.process_audio_chunk(pcm_data)
                if processed.size == 0:
                    continue # 빈 청크 건너뛰기

                # 중단되지 않았고 음성 인식기가 실행 중이어야 할 때만 오디오 공급
                if not self.interrupted:
                    # queue.get과 여기 사이에 설정되었을 수 있으므로 실패 플래그 다시 확인
                     if not self._transcription_failed:
                        # 내부 프로세서에 오디오 공급
                        logger.debug(f"👂🎤 STT 엔진에 오디오 공급: {len(processed)} 샘플") # STT 진단 로그 2
                        self.transcriber.feed_audio(processed.tobytes(), audio_data)
                     # 루프 시작 부분의 확인이 종료를 처리하므로 여기서 'else'는 필요 없음

            except asyncio.CancelledError:
                logger.info("👂🚫 오디오 처리 작업이 취소되었습니다.")
                break
            except Exception as e:
                # 오디오 청크 처리 중 일반 오류 기록
                logger.error(f"👂💥 큐 루프에서 오디오 처리 오류: {e}", exc_info=True)
                # 오류 기록 후 후속 청크 처리 계속.
                # 오류가 지속될 경우 중단하는 로직 추가 고려.
        logger.info("👂⏹️ 오디오 청크 처리 루프가 종료되었습니다.")


    def shutdown(self) -> None:
        """
        오디오 프로세서 및 음성 인식기에 대한 종료 절차를 시작합니다.

        음성 인식기에 종료 신호를 보내고 백그라운드 음성 인식 작업을 취소합니다.
        """
        logger.info("👂🛑 AudioInputProcessor를 종료합니다...")
        # 루프에 신호를 보내기 위해 음성 인식기 종료가 먼저 호출되도록 보장
        if hasattr(self.transcriber, 'shutdown'):
             logger.info("👂🛑 TranscriptionProcessor에 종료 신호를 보냅니다.")
             self.transcriber.shutdown()
        else:
             logger.warning("👂⚠️ TranscriptionProcessor에 shutdown 메서드가 없습니다.")

        if self.transcription_task and not self.transcription_task.done():
            task_name = self.transcription_task.get_name() if hasattr(self.transcription_task, 'get_name') else 'TranscriptionTask'
            logger.info(f"👂🚫 백그라운드 음성 인식 작업({task_name})을 취소합니다...")
            self.transcription_task.cancel()
            # 선택 사항: 비동기 종료 컨텍스트에서 여기에 타임아웃과 함께 await 추가
            # try:
            #     await asyncio.wait_for(self.transcription_task, timeout=5.0)
            # except (asyncio.TimeoutError, asyncio.CancelledError, Exception) as e:
            #     logger.warning(f"👂⚠️ 음성 인식 작업 {task_name} 취소 대기 중 오류/타임아웃: {e}")
        else:
            logger.info("👂✅ 음성 인식 작업이 종료 중 이미 완료되었거나 실행 중이 아닙니다.")

        logger.info("👂👋 AudioInputProcessor 종료 시퀀스가 시작되었습니다.")