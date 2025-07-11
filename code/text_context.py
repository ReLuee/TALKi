import logging
from typing import Optional, Set, Tuple, Dict, Union # 타입 힌트를 위해 추가

logger = logging.getLogger(__name__)
from colors import Colors # 외부에서 필요하다고 가정

class TextContext:
    """
    주어진 문자열에서 의미 있는 텍스트 세그먼트(컨텍스트)를 추출합니다.

    이 클래스는 미리 정의된 분리 토큰으로 끝나고, 지정된 길이 제약 조건을 준수하며,
    최소한의 영숫자 문자를 포함하는 부분 문자열을 식별합니다.
    """
    def __init__(self, split_tokens: Optional[Set[str]] = None) -> None:
        """
        TextContext 프로세서를 초기화합니다.

        유효한 컨텍스트 경계를 결정하는 데 사용되는 문자를 설정합니다.

        Args:
            split_tokens: 선택적인 문자열 집합. 각 문자열은 잠재적인
                          컨텍스트 끝 마커로 처리됩니다. None인 경우, 기본
                          구두점 및 공백 문자 집합이 사용됩니다.
        """
        if split_tokens is None:
            # 명확성을 위해 내부적으로 더 명시적인 변수 이름 사용
            default_splits: Set[str] = {".", "!", "?", ",", ";", ":", "\n", "-", "。", "、"}
            self.split_tokens: Set[str] = default_splits
        else:
            self.split_tokens: Set[str] = set(split_tokens)

    def get_context(self, txt: str, min_len: int = 6, max_len: int = 120, min_alnum_count: int = 10) -> Tuple[Optional[str], Optional[str]]:
        """
        입력 텍스트의 시작 부분에서 가장 짧은 유효한 컨텍스트를 찾습니다.

        `txt` 텍스트를 시작부터 `max_len` 문자까지 스캔합니다. `self.split_tokens`의
        문자가 처음 나타나는 위치를 찾습니다. 발견되면, 해당 토큰에서 끝나는
        부분 문자열이 `min_len`(전체 길이) 및 `min_alnum_count`(영숫자 문자 수)
        기준을 충족하는지 확인합니다.

        Args:
            txt: 컨텍스트를 추출할 입력 문자열.
            min_len: 추출된 컨텍스트 부분 문자열에 허용되는 최소 전체 길이.
            max_len: 추출된 컨텍스트 부분 문자열에 허용되는 최대 전체 길이.
                     이 문자 수를 검사한 후 검색이 중지됩니다.
            min_alnum_count: 추출된 컨텍스트 부분 문자열 내에 필요한 최소
                             영숫자 문자 수.

        Returns:
            다음을 포함하는 튜플:
            - 발견된 경우 추출된 컨텍스트 문자열, 그렇지 않으면 None.
            - 컨텍스트 이후의 입력 문자열의 나머지 부분, 그렇지 않으면 None.
            제약 조건 내에서 적합한 컨텍스트를 찾지 못하면 (None, None)을 반환합니다.
        """
        alnum_count = 0

        for i in range(1, min(len(txt), max_len) + 1):
            char = txt[i - 1]
            if char.isalnum():
                alnum_count += 1

            # 현재 문자가 잠재적인 컨텍스트 끝인지 확인
            if char in self.split_tokens:
                # 길이 및 영숫자 수 기준이 충족되는지 확인
                if i >= min_len and alnum_count >= min_alnum_count:
                    context_str = txt[:i]
                    remaining_str = txt[i:]
                    logger.info(f"🧠 {Colors.MAGENTA}문자 번호 {i} 뒤에서 컨텍스트 발견, 컨텍스트: {context_str}")
                    return context_str, remaining_str

        # max_len 제한 내에서 적합한 컨텍스트를 찾지 못함
        return None, None
