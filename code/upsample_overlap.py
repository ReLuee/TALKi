import base64
import numpy as np
from scipy.signal import resample_poly
from typing import Optional

class UpsampleOverlap:
    """
    겹침 처리를 포함한 청크 단위 오디오 업샘플링을 관리합니다.

    이 클래스는 순차적인 오디오 청크를 처리하고, `scipy.signal.resample_poly`를
    사용하여 24kHz에서 48kHz로 업샘플링하며, 경계 아티팩트를 완화하기 위해
    청크 간의 겹침을 관리합니다. 처리된 업샘플링된 오디오 세그먼트는
    Base64 인코딩된 문자열로 반환됩니다. 호출 간에 겹침을 올바르게 처리하기
    위해 내부 상태를 유지합니다.
    """
    def __init__(self):
        """
        UpsampleOverlap 프로세서를 초기화합니다.

        처리 중 겹침을 처리하기 위해 이전 오디오 청크와 그 리샘플링된 버전을
        추적하는 데 필요한 내부 상태를 설정합니다.
        """
        self.previous_chunk: Optional[np.ndarray] = None
        self.resampled_previous_chunk: Optional[np.ndarray] = None

    def get_base64_chunk(self, chunk: bytes) -> str:
        """
        들어오는 오디오 청크를 처리하고 업샘플링하여 관련 세그먼트를 Base64로 반환합니다.

        원시 PCM 바이트(16비트 부호 있는 정수로 가정) 청크를 float32 numpy 배열로
        변환하고, 정규화하고, 24kHz에서 48kHz로 업샘플링합니다. 이전 청크의 데이터를
        사용하여 겹침을 만들고, 결합된 오디오를 리샘플링하며, 현재 청크에 주로
        해당하는 중앙 부분을 추출하여 전환을 부드럽게 합니다. 다음 호출을 위해
        상태가 업데이트됩니다. 추출된 오디오 세그먼트는 다시 16비트 PCM 바이트로
        변환되어 Base64 인코딩된 문자열로 반환됩니다.

        Args:
            chunk: 원시 오디오 데이터 바이트 (PCM 16비트 부호 있는 정수 형식 예상).

        Returns:
            입력 청크에 해당하는 업샘플링된 오디오 세그먼트를 나타내는 Base64
            인코딩된 문자열 (겹침에 맞게 조정됨). 입력 청크가 비어 있으면 빈
            문자열을 반환합니다.
        """
        audio_int16 = np.frombuffer(chunk, dtype=np.int16)
        # 잠재적으로 비어 있는 청크를 부드럽게 처리
        if audio_int16.size == 0:
             return "" # 빈 입력 청크에 대해 빈 문자열 반환

        audio_float = audio_int16.astype(np.float32) / 32768.0

        # 상태 및 첫 번째 청크 로직에 필요하므로 먼저 현재 청크를 독립적으로 업샘플링
        upsampled_current_chunk = resample_poly(audio_float, 48000, 24000)

        if self.previous_chunk is None:
            # 첫 번째 청크: 업샘플링된 버전의 전반부 출력
            half = len(upsampled_current_chunk) // 2
            part = upsampled_current_chunk[:half]
        else:
            # 후속 청크: 이전 float 청크와 현재 float 청크 결합
            combined = np.concatenate((self.previous_chunk, audio_float))
            # 결합된 청크 업샘플링
            up = resample_poly(combined, 48000, 24000)

            # 중간 부분을 추출하기 위한 길이 및 인덱스 계산
            # self.resampled_previous_chunk가 None이 아닌지 확인 (외부 if 때문에 여기서 발생해서는 안 됨)
            assert self.resampled_previous_chunk is not None
            prev_len = len(self.resampled_previous_chunk) # *업샘플링된* 이전 청크의 길이
            h_prev = prev_len // 2 # *업샘플링된* 이전 청크의 중간점 인덱스

            # *** 수정된 인덱스 계산 (원본으로 되돌림) ***
            # 현재 청크의 주요 기여에 해당하는 부분의 끝 인덱스 계산
            # 이 인덱스는 결합된 'up' 배열 내에서 *현재* 청크 기여의 중간점을 나타냄
            h_cur = (len(up) - prev_len) // 2 + prev_len

            part = up[h_prev:h_cur]

        # 다음 반복을 위해 상태 업데이트
        self.previous_chunk = audio_float
        self.resampled_previous_chunk = upsampled_current_chunk # *다음* 겹침을 위해 *현재* 청크의 업샘플링된 버전 저장

        # 추출된 부분을 다시 PCM16 바이트로 변환하고 인코딩
        pcm = (part * 32767).astype(np.int16).tobytes()
        return base64.b64encode(pcm).decode('utf-8')

    def flush_base64_chunk(self) -> Optional[str]:
        """
        모든 청크가 처리된 후 마지막 남은 업샘플링된 오디오 세그먼트를 반환합니다.

        `get_base64_chunk`의 마지막 호출 후, 상태는 맨 마지막 입력 청크의 업샘플링된
        버전(`self.resampled_previous_chunk`)을 보유합니다. 이 메서드는 그 *전체*
        최종 업샘플링된 청크를 16비트 PCM 바이트로 변환하고 Base64로 인코딩하여
        반환합니다. 그런 다음 내부 상태를 지웁니다. 모든 입력 청크가 `get_base64_chunk`에
        전달된 후 한 번 호출되어야 합니다.

        Returns:
            최종 업샘플링된 오디오 청크를 포함하는 Base64 인코딩된 문자열,
            또는 처리된 청크가 없거나 flush가 이미 호출된 경우 None.
        """
        # *** 수정된 FLUSH 로직 (원본으로 되돌림) ***
        if self.resampled_previous_chunk is not None:
            # 원본 로직에 따라 전체 마지막 업샘플링된 청크 반환
            pcm = (self.resampled_previous_chunk * 32767).astype(np.int16).tobytes()

            # 플러시 후 상태 지우기
            self.previous_chunk = None
            self.resampled_previous_chunk = None
            return base64.b64encode(pcm).decode('utf-8')
        return None # 플러시할 것이 없으면 None 반환
