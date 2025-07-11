import logging
import time
from colors import Colors # 'colors' 라이브러리가 설치되어 있다고 가정 (pip install ansicolors) 또는 사용자 정의 Colors 클래스
from typing import Optional # 다른 곳에서 필요할 경우 타입 힌트 일관성을 위해 추가되었지만, 현재 인수/반환에서는 엄격하게 사용되지 않음

# --- 시간을 로컬로 처리하기 위한 커스텀 포매터 정의 ---
class CustomTimeFormatter(logging.Formatter):
    """
    타임스탬프를 로컬 시간을 사용하여 MM:SS.cs로 표시하는 로깅 포매터입니다.

    `logging.Formatter`를 상속하고 `formatTime` 메서드를 오버라이드하여
    로그 레코드의 생성 시간을 기반으로 한 로컬 시간대를 사용하여 콘솔 출력에
    적합한 특정하고 간결한 타임스탬프 형식을 제공합니다.
    """

    def formatTime(self, record: logging.LogRecord, datefmt: Optional[str] = None) -> str:
        """
        로그 레코드의 생성 시간을 MM:SS.cs 형식으로 포맷합니다.

        `time.localtime`을 사용하여 레코드의 생성 타임스탬프를 변환하고
        분, 초, 센티초(1/100초)로 포맷합니다. 기본 클래스에서 제공하는 `datefmt` 인수는
        이 커스텀 구현에서는 무시됩니다.

        Args:
            record: 생성 시간을 포맷해야 하는 로그 레코드.
            datefmt: 선택적 날짜 형식 문자열 (이 메서드에서는 무시됨).

        Returns:
            포맷된 시간을 나타내는 문자열 (예: "59:23.18").
        """
        # 원래 의도대로 localtime을 사용하되, 이 포매터 내에서 로컬로 사용
        now = time.localtime(record.created)
        cs = int((record.created % 1) * 100)  # 센티초
        # 시간 문자열을 필요한 형식으로 포맷
        s = time.strftime("%M:%S", now) + f".{cs:02d}"
        return s

def setup_logging(level: int = logging.INFO) -> None:
    """
    루트 로거를 커스텀 포맷과 레벨로 콘솔 출력용으로 구성합니다.

    핸들러가 아직 없는 경우 루트 로거에 `StreamHandler`를 설정합니다.
    타임스탬프를 MM:SS.cs로 표시하기 위해 `CustomTimeFormatter`를 적용하고
    다른 로그 메시지 부분(타임스탬프, 로거 이름, 레벨, 메시지)에 ANSI 색상을 사용합니다.
    이 설정은 레코드 팩토리나 전역 포매터 변환기와 같은 전역 로깅 상태를
    수정하지 않습니다.

    Args:
        level: 루트 로거와 콘솔 핸들러의 최소 로깅 레벨
               (예: `logging.DEBUG`, `logging.INFO`). 기본값은 `logging.INFO`.
    """
    # 루트 로거에 이미 핸들러가 있는지 확인하여 여러 번 추가하는 것을 방지
    root_logger = logging.getLogger()
    if not root_logger.hasHandlers():
        # 로거 자체에 레벨 설정
        root_logger.setLevel(level)

        # --- 포맷 문자열 정의 ---
        # 참고: 이제 표준 '%(asctime)s' 플레이스홀더를 사용할 것입니다.
        # 왜냐하면 우리의 CustomTimeFormatter가 이를 올바르게 포맷할 것이기 때문입니다.
        prefix = Colors.apply("🖥️").gray
        # 표준 %(asctime)s 사용 - 우리의 커스텀 포매터가 외관을 처리할 것임
        timestamp = Colors.apply("%(asctime)s").blue
        levelname = Colors.apply("%(levelname)-4.4s").green.bold
        message = Colors.apply("%(message)s")
        logger_name = Colors.apply("%(name)-10.10s").gray

        log_format = f"{timestamp} {logger_name} {levelname} {message}"
        # --- 포맷 문자열 정의 완료 ---

        # --- 명시적 핸들러/포매터를 사용하여 로깅 구성 ---
        # basicConfig 및 전역 상태 수정을 피함

        # 1. 커스텀 포매터 인스턴스 생성
        # 원하는 로그 포맷 문자열을 전달합니다. formatTime이 오버라이드되었으므로 datefmt는 필요 없음.
        formatter = CustomTimeFormatter(log_format)

        # 2. 핸들러 생성 (예: 콘솔에 로그하기 위한 StreamHandler)
        handler = logging.StreamHandler()

        # 3. 핸들러에 커스텀 포매터 설정
        handler.setFormatter(formatter)

        # 4. 핸들러에 레벨 설정 (선택 사항이지만 좋은 습관,
        #    basicConfig는 이를 암시적으로 수행). 이 핸들러가 처리하는 메시지를 제어.
        handler.setLevel(level)

        # 5. 구성된 핸들러를 루트 로거에 추가
        root_logger.addHandler(handler)

