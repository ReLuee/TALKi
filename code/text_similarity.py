import re
from difflib import SequenceMatcher
import logging

logger = logging.getLogger(__name__)

class TextSimilarity:
    """
    두 텍스트 문자열을 비교하고 유사도 비율을 계산합니다.

    이 클래스는 `difflib.SequenceMatcher`를 사용하여 두 텍스트 간의 유사도를 계산하는
    메서드를 제공합니다. 전체 텍스트 비교, 마지막 몇 단어에만 집중하는 비교,
    또는 전체 유사도와 끝 부분 유사도의 가중 평균을 사용하는 등 다양한 비교 전략을
    지원합니다. 비교 전에 텍스트는 정규화(소문자 변환, 구두점 제거)됩니다.

    속성:
        similarity_threshold (float): `are_texts_similar` 메서드에서 텍스트가
                                      유사하다고 판단하는 최소 유사도 비율 (0.0에서 1.0 사이).
        n_words (int): 'end' 또는 'weighted' 포커스 모드를 사용할 때 각 텍스트의
                       끝에서부터 고려할 단어의 수.
        focus (str): 비교 전략 ('overall', 'end', 또는 'weighted').
        end_weight (float): `focus`가 'weighted'일 때 끝 부분 유사도에 할당되는
                            가중치 (0.0에서 1.0 사이). 전체 유사도는
                            `1.0 - end_weight`의 가중치를 받습니다.
    """
    def __init__(self,
                 similarity_threshold: float = 0.96,
                 n_words: int = 5,
                 focus: str = 'weighted', # 기본값으로 가중치 접근 방식 사용
                 end_weight: float = 0.7): # 기본값: 끝 부분 유사도에 70% 가중치
        """
        TextSimilarity 비교기를 초기화합니다.

        Args:
            similarity_threshold: `are_texts_similar`를 위한 비율 임계값.
                                  0.0과 1.0 사이여야 합니다.
            n_words: 포커스 비교 모드에서 끝 부분에서 추출할 단어의 수.
                     양의 정수여야 합니다.
            focus: 비교 전략. 'overall', 'end', 'weighted' 중 하나여야 합니다.
            end_weight: 'weighted' 모드에서 끝 부분 유사도에 대한 가중치.
                        0.0과 1.0 사이여야 합니다. 다른 모드에서는 무시됩니다.

        Raises:
            ValueError: 인수가 유효한 범위나 타입을 벗어나는 경우.
        """
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError("similarity_threshold는 0.0과 1.0 사이여야 합니다.")
        if not isinstance(n_words, int) or n_words < 1:
            raise ValueError("n_words는 양의 정수여야 합니다.")
        if focus not in ['end', 'weighted', 'overall']:
            raise ValueError("focus는 'end', 'weighted', 'overall' 중 하나여야 합니다.")
        if not 0.0 <= end_weight <= 1.0:
            raise ValueError("end_weight는 0.0과 1.0 사이여야 합니다.")

        self.similarity_threshold = similarity_threshold
        self.n_words = n_words
        self.focus = focus
        # end_weight는 focus가 'weighted'일 때만 의미가 있도록 보장
        self.end_weight = end_weight if focus == 'weighted' else 0.0

        # 효율성을 위해 정규식 미리 컴파일
        self._punctuation_regex = re.compile(r'[^\w\s]')
        self._whitespace_regex = re.compile(r'\s+')

    def _normalize_text(self, text: str) -> str:
        """
        비교를 위해 텍스트를 단순화하여 준비합니다.

        입력 텍스트를 소문자로 변환하고, 알파벳/숫자나 공백이 아닌 모든 문자를 제거하며,
        연속된 여러 공백을 단일 공백으로 축소하고, 앞뒤 공백을 제거합니다.
        문자열이 아닌 입력은 경고를 기록하고 빈 문자열을 반환하여 처리합니다.

        Args:
            text: 정규화할 원본 텍스트 문자열.

        Returns:
            정규화된 텍스트 문자열. 입력이 문자열이 아니거나 정규화 후 비어 있으면
            빈 문자열을 반환합니다.
        """
        if not isinstance(text, str):
             # 문자열이 아닌 입력을 부드럽게 처리
             logger.warning(f"⚠️ 입력이 문자열이 아닙니다: {type(text)}. 빈 문자열로 변환합니다.")
             text = ""
        text = text.lower()
        text = self._punctuation_regex.sub('', text)
        text = self._whitespace_regex.sub(' ', text).strip()
        return text

    def _get_last_n_words_text(self, normalized_text: str) -> str:
        """
        정규화된 텍스트 문자열에서 마지막 `n_words`개의 단어를 추출합니다.

        텍스트를 공백으로 분리하고 마지막 `n_words`개의 단어를 다시 합칩니다.
        텍스트의 단어 수가 `n_words`보다 적으면 전체 텍스트가 반환됩니다.

        Args:
            normalized_text: `_normalize_text`에 의해 처리된 텍스트 문자열.

        Returns:
            입력의 마지막 `n_words`개 단어를 포함하는 문자열 (공백으로 연결됨).
            입력이 비어 있으면 빈 문자열을 반환합니다.
        """
        words = normalized_text.split()
        # 텍스트의 단어 수가 n_words보다 적은 경우를 자동으로 처리
        last_words_segment = words[-self.n_words:]
        return ' '.join(last_words_segment)

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        설정에 따라 두 텍스트 간의 유사도 비율을 계산합니다.

        두 입력 텍스트를 모두 정규화한 다음, `focus` 전략('overall', 'end', 'weighted')에
        따라 `difflib.SequenceMatcher`를 사용하여 유사도를 계산합니다.
        정규화 후 빈 문자열을 적절히 처리합니다.

        Args:
            text1: 비교할 첫 번째 텍스트 문자열.
            text2: 비교할 두 번째 텍스트 문자열.

        Returns:
            계산된 유사도 비율을 나타내는 0.0과 1.0 사이의 float 값.
            1.0은 (정규화 및 포커싱 후) 동일한 시퀀스를, 0.0은 유사성 없음을 나타냅니다.

        Raises:
            RuntimeError: 인스턴스의 `focus` 속성에 잘못된 값이 있는 경우
                          (__init__ 유효성 검사로 인해 발생해서는 안 됨).
        """
        norm_text1 = self._normalize_text(text1)
        norm_text2 = self._normalize_text(text2)

        # 엣지 케이스 처리: 두 정규화된 텍스트가 모두 비어 있으면 -> 완벽 일치
        if not norm_text1 and not norm_text2:
            return 1.0
        # 참고: SequenceMatcher는 ""와의 비교를 올바르게 처리하므로(다른 쪽이 비어있지 않으면 비율 0.0),
        # 한 쪽만 비어 있는 경우에 대한 명시적인 확인이 필요 없습니다.

        # matcher를 한 번 초기화하고, 시퀀스를 설정하여 재사용
        # autojunk=False는 상세 비교를 강제하며, 잠재적으로 느릴 수 있지만 휴리스틱을 피합니다.
        matcher = SequenceMatcher(isjunk=None, a=None, b=None, autojunk=False)

        if self.focus == 'overall':
            matcher.set_seqs(norm_text1, norm_text2)
            return matcher.ratio()

        elif self.focus == 'end':
            end_text1 = self._get_last_n_words_text(norm_text1)
            end_text2 = self._get_last_n_words_text(norm_text2)
            # SequenceMatcher는 빈 문자열을 올바르게 처리합니다 (("", "") -> 1.0, ("abc", "") -> 0.0)
            matcher.set_seqs(end_text1, end_text2)
            return matcher.ratio()

        elif self.focus == 'weighted':
            # 전체 유사도 계산
            matcher.set_seqs(norm_text1, norm_text2)
            sim_overall = matcher.ratio()

            # 끝 부분 유사도 계산
            end_text1 = self._get_last_n_words_text(norm_text1)
            end_text2 = self._get_last_n_words_text(norm_text2)

            # matcher를 재사용하고 SequenceMatcher가 빈 끝 세그먼트를 처리하도록 함
            # SequenceMatcher는 빈 문자열을 올바르게 처리합니다 (("", "") -> 1.0, ("abc", "") -> 0.0)
            matcher.set_seqs(end_text1, end_text2)
            sim_end = matcher.ratio()

            # 가중 평균 계산
            weighted_sim = (1 - self.end_weight) * sim_overall + self.end_weight * sim_end
            return weighted_sim

        else:
            # __init__ 유효성 검사로 인해 이런 일이 발생해서는 안 되지만, 안전장치로:
            # 일관성을 위해 아이콘을 추가하지만, 이 로그는 내부 오류를 나타냅니다.
            # ❌는 예상치 못한 실패를 암시합니다.
            logger.error(f"❌ 계산 중 잘못된 포커스 모드를 만났습니다: {self.focus}")
            raise RuntimeError("계산 중 잘못된 포커스 모드를 만났습니다.")


    def are_texts_similar(self, text1: str, text2: str) -> bool:
        """
        두 텍스트가 유사도 임계값을 충족하는지 확인합니다.

        설정된 메서드(`calculate_similarity`)를 사용하여 `text1`과 `text2` 사이의
        유사도를 계산하고 그 결과를 인스턴스의 `similarity_threshold`와 비교합니다.

        Args:
            text1: 첫 번째 텍스트 문자열.
            text2: 두 번째 텍스트 문자열.

        Returns:
            계산된 유사도 비율이 `self.similarity_threshold`보다 크거나 같으면
            True, 그렇지 않으면 False를 반환합니다.
        """
        similarity = self.calculate_similarity(text1, text2)
        return similarity >= self.similarity_threshold

if __name__ == "__main__":
    # 예제 출력을 위한 기본 로깅 설정
    logging.basicConfig(level=logging.INFO)

    # --- 예제 사용법 ---
    text_long1 = "This is a very long text that goes on and on, providing lots of context, but the important part is how it concludes in the final sentence."
    text_long2 = "This is a very long text that goes on and on, providing lots of context, but the important part is how it concludes in the final sentence."
    text_long_diff_end = "This is a very long text that goes on and on, providing lots of context, but the important part is how it finishes in the last words."
    text_short1 = "Check the end."
    text_short2 = "Check the ending."
    text_short_very = "End."
    text_empty = ""
    text_punct = "!!!"
    text_non_string = 12345

    print("--- 표준 유사도 (전체 포커스) ---")
    sim_overall = TextSimilarity(focus='overall')
    print(f"긴 텍스트 동일: {sim_overall.calculate_similarity(text_long1, text_long2):.4f}") # 예상: 1.0000
    print(f"긴 텍스트 끝 다름: {sim_overall.calculate_similarity(text_long1, text_long_diff_end):.4f}") # 예상: 높음 (예: > 0.9)
    print(f"짧은 텍스트 끝 다름: {sim_overall.calculate_similarity(text_short1, text_short2):.4f}") # 예상: 중간/높음
    print(f"짧은 텍스트 vs 매우 짧은 텍스트: {sim_overall.calculate_similarity(text_short1, text_short_very):.4f}") # 예상: 낮음/중간
    print(f"빈 텍스트 vs 구두점: {sim_overall.calculate_similarity(text_empty, text_punct):.4f}") # 예상: 1.0000 (둘 다 ""로 정규화됨)
    print(f"짧은 텍스트 vs 빈 텍스트: {sim_overall.calculate_similarity(text_short1, text_empty):.4f}") # 예상: 0.0000
    print(f"문자열 아닌 값 vs 빈 텍스트: {sim_overall.calculate_similarity(text_non_string, text_empty):.4f}") # 예상: 1.0000 (둘 다 ""로 정규화됨)


    print("\n--- 끝 부분 포커스 (마지막 5 단어) ---")
    sim_end_only = TextSimilarity(focus='end', n_words=5)
    print(f"긴 텍스트 동일: {sim_end_only.calculate_similarity(text_long1, text_long2):.4f}") # 예상: 1.0000
    print(f"긴 텍스트 끝 다름: {sim_end_only.calculate_similarity(text_long1, text_long_diff_end):.4f}") # 예상: 낮음 (마지막 5 단어에 따라 다름)
    print(f"짧은 텍스트 끝 다름: {sim_end_only.calculate_similarity(text_short1, text_short2):.4f}") # 'check the end' vs 'check the ending' 비교 -> 중간/높음
    print(f"짧은 텍스트 vs 매우 짧은 텍스트: {sim_end_only.calculate_similarity(text_short1, text_short_very):.4f}") # 'check the end' vs 'end' 비교 -> 낮음
    print(f"빈 텍스트 vs 구두점: {sim_end_only.calculate_similarity(text_empty, text_punct):.4f}") # 예상: 1.0000 (둘 다 ""로 정규화됨)
    print(f"짧은 텍스트 vs 빈 텍스트: {sim_end_only.calculate_similarity(text_short1, text_empty):.4f}") # 예상: 0.0000


    print("\n--- 가중치 포커스 (끝 70%, 마지막 5 단어) ---")
    sim_weighted = TextSimilarity(focus='weighted', n_words=5, end_weight=0.7)
    print(f"긴 텍스트 동일: {sim_weighted.calculate_similarity(text_long1, text_long2):.4f}") # 예상: 1.0000
    print(f"긴 텍스트 끝 다름: {sim_weighted.calculate_similarity(text_long1, text_long_diff_end):.4f}") # 예상: 전체 포커스보다 낮고, 끝 포커스보다 높음
    print(f"짧은 텍스트 끝 다름: {sim_weighted.calculate_similarity(text_short1, text_short2):.4f}") # 예상: 가중치 적용 결과
    print(f"짧은 텍스트 vs 매우 짧은 텍스트: {sim_weighted.calculate_similarity(text_short1, text_short_very):.4f}") # 예상: 가중치 적용 결과
    print(f"빈 텍스트 vs 구두점: {sim_weighted.calculate_similarity(text_empty, text_punct):.4f}") # 예상: 1.0000 (둘 다 ""로 정규화됨)
    print(f"짧은 텍스트 vs 빈 텍스트: {sim_weighted.calculate_similarity(text_short1, text_empty):.4f}") # 예상: 0.0000


    print("\n--- 끝 포커스로 짧은 텍스트 처리 (n_words=5) ---")
    short1 = "one two three"
    short2 = "one two four"
    short3 = "one"
    sim_end_short = TextSimilarity(focus='end', n_words=5)
    print(f"'{short1}' vs '{short2}': {sim_end_short.calculate_similarity(short1, short2):.4f}") # 5 단어 미만이므로 전체 텍스트 비교
    print(f"'{short1}' vs '{short3}': {sim_end_short.calculate_similarity(short1, short3):.4f}") # 5 단어 미만이므로 전체 텍스트 비교


    print("\n--- 임계값 확인 ---")
    threshold_checker = TextSimilarity(similarity_threshold=0.90, focus='overall')
    print(f"'{text_long1}' vs '{text_long_diff_end}' 유사한가? {threshold_checker.are_texts_similar(text_long1, text_long_diff_end)}")
    print(f"'{text_short1}' vs '{text_short2}' 유사한가? {threshold_checker.are_texts_similar(text_short1, text_short2)}")
