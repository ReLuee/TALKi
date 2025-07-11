class Colors:
    # 전경색 (표준)
    BLACK = "\033[30m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    GRAY = "\033[90m"

    # 배경색 (표준)
    BG_BLACK = "\033[40m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"
    BG_BLUE = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN = "\033[46m"
    BG_WHITE = "\033[47m"

    # 밝은 배경색
    BG_GRAY = "\033[100m"
    BG_LIGHT_RED = "\033[101m"
    BG_LIGHT_GREEN = "\033[102m"
    BG_LIGHT_YELLOW = "\033[103m"
    BG_LIGHT_BLUE = "\033[104m"
    BG_LIGHT_MAGENTA = "\033[105m"
    BG_LIGHT_CYAN = "\033[106m"
    BG_LIGHT_WHITE = "\033[107m"

    # 텍스트 스타일
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    UNDERLINE = "\033[4m"
    BLINK = "\033[5m"
    REVERSE = "\033[7m"
    HIDDEN = "\033[8m"

    # 추가 256색 전경색 (자주 사용하는 색상)
    ORANGE = "\033[38;5;208m"
    PINK = "\033[38;5;213m"
    TEAL = "\033[38;5;30m"
    VIOLET = "\033[38;5;129m"
    BROWN = "\033[38;5;94m"
    LIGHT_GRAY = "\033[38;5;250m"
    DARK_GRAY = "\033[38;5;238m"

    # 추가 256색 배경색
    BG_ORANGE = "\033[48;5;208m"
    BG_PINK = "\033[48;5;213m"
    BG_TEAL = "\033[48;5;30m"
    BG_VIOLET = "\033[48;5;129m"
    BG_BROWN = "\033[48;5;94m"
    BG_LIGHT_GRAY = "\033[48;5;250m"
    BG_DARK_GRAY = "\033[48;5;238m"

    class Formatter:
        def __init__(self, text: str):
            self.text = text
            self.effects = []

        def __getattr__(self, name: str):
            # 속성 접근을 해당 ANSI 코드로 매핑합니다.
            # 이를 통해 .red 또는 .bold와 같이 호출할 수 있습니다.
            code = getattr(Colors, name.upper(), None)
            if code is None:
                raise AttributeError(f"잘못된 스타일 '{name}'.")
            self.effects.append(code)
            return self

        def __str__(self) -> str:
            # 모든 효과를 연결한 다음 텍스트를 붙이고 마지막으로 리셋합니다.
            return "".join(self.effects) + self.text + Colors.RESET

    @classmethod
    def apply(cls, text: str) -> "Colors.Formatter":
        """
        주어진 텍스트에 대한 포매팅 체인을 시작합니다.
        
        사용법:
            print(Colors.apply("Hello").red.bold)
        """
        return cls.Formatter(text)


# 사용 예시:
if __name__ == "__main__":
    print(Colors.apply("이 텍스트는 빨간색입니다.").red)
    print(Colors.apply("굵고 밑줄 친 파란색 텍스트입니다.").blue.bold.underline)
    print(Colors.apply("노란색 배경에 빨간색 밑줄 친 굵은 텍스트입니다.").red.bg_yellow.underline.bold)
