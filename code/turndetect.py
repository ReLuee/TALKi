import logging
logger = logging.getLogger(__name__)

import transformers
import collections
import threading
import queue
import torch
import time
import re

# 설정 상수
model_dir_local = "KoljaB/SentenceFinishedClassification"
model_dir_cloud = "/root/models/sentenceclassification/"
sentence_end_marks = ['.', '!', '?', '。'] # 문장 끝으로 간주되는 문자들

# 확률-정지 시간 보간을 위한 앵커 포인트
anchor_points = [
    (0.0, 1.0), # 확률 0.0은 정지 시간 1.0에 매핑
    (1.0, 0.0)  # 확률 1.0은 정지 시간 0.0에 매핑
]

def ends_with_string(text: str, s: str) -> bool:
    """
    문자열이 특정 부분 문자열로 끝나는지 확인하며, 뒤에 한 글자 정도의 여유를 허용합니다.

    뒤에 공백이 있는 경우에도 구두점을 확인하는 데 유용합니다.

    Args:
        text: 확인할 입력 문자열.
        s: 끝에서 찾을 부분 문자열.

    Returns:
        텍스트가 s로 끝나거나, text[:-1]이 s로 끝나면 True. 그렇지 않으면 False.
    """
    if text.endswith(s):
        return True
    # 마지막 문자를 무시했을 때 끝나는지 확인 (예: 뒤따르는 공백)
    if len(text) > 1 and text[:-1].endswith(s):
        return True
    return False

def preprocess_text(text: str) -> str:
    """
    텍스트 문자열의 시작 부분을 정리하고 정규화합니다.

    다음 단계를 적용합니다:
    1. 앞쪽 공백을 제거합니다.
    2. 앞쪽 말줄임표("...")가 있으면 제거합니다.
    3. (말줄임표 제거 후) 다시 앞쪽 공백을 제거합니다.
    4. 남은 텍스트의 첫 글자를 대문자로 만듭니다.

    Args:
        text: 입력 텍스트 문자열.

    Returns:
        전처리된 텍스트 문자열.
    """
    text = text.lstrip() # 앞쪽 공백 제거
    if text.startswith("..."): # 시작하는 말줄임표가 있으면 제거
        text = text[3:]
    text = text.lstrip() # 말줄임표 제거 후 다시 앞쪽 공백 제거
    if text:
        text = text[0].upper() + text[1:] # 첫 글자 대문자화

    return text

def strip_ending_punctuation(text: str) -> str:
    """
    `sentence_end_marks`에 정의된 뒤쪽 구두점을 제거합니다.

    먼저 뒤쪽 공백을 제거한 다음, 문자열 끝에서 발견되는 `sentence_end_marks`의
    문자들을 반복적으로 제거합니다.

    Args:
        text: 입력 텍스트 문자열.

    Returns:
        지정된 뒤쪽 구두점이 제거된 텍스트 문자열.
    """
    text = text.rstrip()
    for char in sentence_end_marks:
        # 여러 개일 경우(예: "!!")를 대비해 각 구두점을 반복적으로 제거
        while text.endswith(char):
             text = text.rstrip(char)
    return text # 제거된 텍스트 반환

def find_matching_texts(texts_without_punctuation: collections.deque) -> list[tuple[str, str]]:
    """
    구두점이 제거된 텍스트가 동일한 최근 연속 항목들을 찾습니다.

    (original_text, stripped_text) 튜플의 데크를 뒤에서부터 반복합니다.
    *마지막* 항목의 제거된 텍스트와 일치하는 모든 항목을 수집하고,
    일치하지 않는 제거된 텍스트를 만나면 중단합니다.

    Args:
        texts_without_punctuation: 각 튜플이 (original_text, stripped_text)인
                                   튜플의 데크.

    Returns:
        일치하는 (original_text, stripped_text) 튜플의 리스트 (가장 오래된 일치 항목이 먼저).
        입력 데크가 비어 있으면 빈 리스트를 반환합니다.
    """
    if not texts_without_punctuation:
        return []

    # 마지막 항목에서 제거된 텍스트 가져오기
    last_stripped_text = texts_without_punctuation[-1][1]

    matching_entries = []

    # 데크를 뒤에서부터 반복
    for entry in reversed(texts_without_punctuation):
        original_text, stripped_text = entry

        # 불일치를 찾으면 수집 중단
        if stripped_text != last_stripped_text:
            break

        # 일치하는 항목 추가 (나중에 뒤집을 것)
        matching_entries.append(entry)

    # 원래 삽입 순서를 유지하기 위해 결과 뒤집기
    matching_entries.reverse()

    return matching_entries

def interpolate_detection(prob: float) -> float:
    """
    미리 정의된 앵커 포인트를 사용하여 확률에 기반한 값을 선형 보간합니다.

    입력 확률 `prob`(0.0과 1.0 사이로 제한됨)을 `anchor_points` 리스트에
    기반한 출력 값으로 매핑합니다. `prob`이 속하는 `anchor_points` 내의
    세그먼트를 찾아 선형 보간을 수행합니다.

    Args:
        prob: 입력 확률, 0.0과 1.0 사이로 예상됨.

    Returns:
        보간된 값. 보간에 실패하면 대체 값으로 4.0을 반환합니다
        (앵커 포인트가 [0,1] 범위를 포함하면 발생하지 않아야 함).
    """
    # 만약을 위해 확률을 0.0과 1.0 사이로 제한
    p = max(0.0, min(prob, 1.0))

    # 확률이 앵커 포인트와 정확히 일치하는지 확인
    for ap_p, ap_val in anchor_points:
        if abs(ap_p - p) < 1e-9: # 부동 소수점 비교를 위해 허용 오차 사용
            return ap_val

    # p가 속하는 세그먼트 [p1, p2] 찾기
    for i in range(len(anchor_points) - 1):
        p1, v1 = anchor_points[i]
        p2, v2 = anchor_points[i+1]
        # 이 로직이 작동하려면 앵커 포인트가 확률 순으로 정렬되어야 함
        if p1 <= p <= p2:
            # 앵커 포인트가 동일한 확률을 가질 경우 0으로 나누는 것을 방지
            if abs(p2 - p1) < 1e-9:
                return v1 # 또는 v2, p1=p2이면 비슷해야 함
            # 선형 보간 공식
            ratio = (p - p1) / (p2 - p1)
            return v1 + ratio * (v2 - v1)

    # 대체: anchor_points가 [0,1]을 제대로 포함하면 도달해서는 안 됨.
    logger.warning(f"🎤⚠️ 확률 {p}가 정의된 앵커 포인트 {anchor_points}를 벗어났습니다. 대체 값을 반환합니다.")
    return 4.0

class TurnDetection:
    """
    텍스트 입력과 문장 완성 모델을 기반으로 발화자 전환 감지 로직을 관리합니다.

    이 클래스는 텍스트 세그먼트를 받아 트랜스포머 모델을 사용하여 문장 완성
    확률을 예측하고, 구두점을 고려하여 다음 발화자가 시작하기 전에 제안된
    대기 시간(정지 기간)을 계산합니다. 처리를 위해 백그라운드 스레드를 사용하고
    새로운 대기 시간 제안에 대한 콜백을 제공합니다. 또한 최근 텍스트의 이력을
    유지하고 모델 예측을 위해 캐싱을 사용합니다.
    """

    def __init__(
        self,
        on_new_waiting_time: callable,
        local: bool = False,
        pipeline_latency: float = 0.5,
        pipeline_latency_overhead: float = 0.1,
    ) -> None:
        """
        TurnDetection 인스턴스를 초기화합니다.

        문장 분류 모델과 토크나이저를 로드하고, 내부 상태(데크, 캐시)를 설정하고,
        백그라운드 처리 스레드를 시작하고, 모델 워밍업을 수행합니다.

        Args:
            on_new_waiting_time: 새로운 대기 시간이 계산될 때 호출되는 콜백 함수.
                                 `(time: float, text: str)`을 받습니다.
            local: True이면 `model_dir_local`에서 모델을 로드하고, 그렇지 않으면 `model_dir_cloud`에서 로드합니다.
            pipeline_latency: STT/처리 파이프라인의 예상 기본 지연 시간(초).
            pipeline_latency_overhead: 파이프라인 지연 시간에 추가되는 버퍼.
        """
        model_dir = model_dir_local if local else model_dir_cloud

        self.on_new_waiting_time = on_new_waiting_time

        self.current_waiting_time: float = -1 # 마지막으로 제안된 시간을 추적
        # 효율적이고 제한된 이력 저장을 위해 maxlen을 가진 데크 사용
        self.text_time_deque: collections.deque[tuple[float, str]] = collections.deque(maxlen=100)
        self.texts_without_punctuation: collections.deque[tuple[str, str]] = collections.deque(maxlen=20)

        self.text_queue: queue.Queue[str] = queue.Queue() # 들어오는 텍스트를 위한 큐
        self.text_worker = threading.Thread(
            target=self._text_worker,
            daemon=True # 이 스레드가 실행 중이더라도 프로그램이 종료될 수 있도록 허용
        )
        self.text_worker.start()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🎤🔌 사용 장치: {self.device}")
        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(model_dir)
        self.classification_model = transformers.DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.classification_model.to(self.device)
        self.classification_model.eval() # 모델을 평가 모드로 설정
        self.max_length: int = 128 # 모델의 최대 시퀀스 길이
        self.pipeline_latency: float = pipeline_latency
        self.pipeline_latency_overhead: float = pipeline_latency_overhead

        # LRU 동작을 위해 OrderedDict로 완료 확률 캐시 초기화
        self._completion_probability_cache: collections.OrderedDict[str, float] = collections.OrderedDict()
        self._completion_probability_cache_max_size: int = 256 # LRU 캐시의 최대 크기

        # 빠른 초기 예측을 위해 분류 모델 워밍업
        logger.info("🎤🔥 분류 모델 워밍업 중...")
        with torch.no_grad():
            warmup_text = "This is a warmup sentence."
            inputs = self.tokenizer(
                warmup_text,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            _ = self.classification_model(**inputs) # 예측 한 번 실행
        logger.info("🎤✅ 분류 모델 워밍업 완료.")

        # 기본 동적 정지 설정 (speed_factor=0.0에 대해 초기화됨)
        self.detection_speed: float = 0.5
        self.ellipsis_pause: float = 2.3
        self.punctuation_pause: float = 0.39
        self.exclamation_pause: float = 0.35
        self.question_pause: float = 0.33
        self.unknown_sentence_detection_pause: float = 1.25
        # 초기 설정 적용 (나중에 다시 호출 가능)
        self.update_settings(speed_factor=0.0)

    def update_settings(self, speed_factor: float) -> None:
        """
        속도 계수에 따라 동적 정지 파라미터를 조정합니다.

        계산에 사용되는 다양한 정지 기간에 대해 '빠름'(speed_factor=0.0)과
        '매우 느림'(speed_factor=1.0) 설정 사이를 선형 보간합니다.
        speed_factor를 0.0과 1.0 사이로 제한합니다.

        Args:
            speed_factor: 미리 정의된 설정 사이의 보간을 제어하는 0.0(가장 빠름)과
                          1.0(가장 느림) 사이의 부동 소수점.
        """
        speed_factor = max(0.0, min(speed_factor, 1.0)) # 계수 제한

        # 기본 '빠름' 설정 (speed_factor = 0.0)
        fast = {
            'detection_speed': 0.5,
            'ellipsis_pause': 2.3,
            'punctuation_pause': 0.39,
            'exclamation_pause': 0.35,
            'question_pause': 0.33,
            'unknown_sentence_detection_pause': 1.25
        }

        # 목표 '매우 느림' 설정 (speed_factor = 1.0)
        very_slow = {
            'detection_speed': 1.7,
            'ellipsis_pause': 3.0,
            'punctuation_pause': 0.9,
            'exclamation_pause': 0.8,
            'question_pause': 0.8,
            'unknown_sentence_detection_pause': 1.9
        }

        # 각 파라미터에 대한 선형 보간
        self.detection_speed = fast['detection_speed'] + speed_factor * (very_slow['detection_speed'] - fast['detection_speed'])
        self.ellipsis_pause = fast['ellipsis_pause'] + speed_factor * (very_slow['ellipsis_pause'] - fast['ellipsis_pause'])
        self.punctuation_pause = fast['punctuation_pause'] + speed_factor * (very_slow['punctuation_pause'] - fast['punctuation_pause'])
        self.exclamation_pause = fast['exclamation_pause'] + speed_factor * (very_slow['exclamation_pause'] - fast['exclamation_pause'])
        self.question_pause = fast['question_pause'] + speed_factor * (very_slow['question_pause'] - fast['question_pause'])
        self.unknown_sentence_detection_pause = fast['unknown_sentence_detection_pause'] + speed_factor * (very_slow['unknown_sentence_detection_pause'] - fast['unknown_sentence_detection_pause'])
        logger.info(f"🎤⚙️ speed_factor={speed_factor:.2f}로 발화자 전환 감지 설정 업데이트됨")


    def suggest_time(
            self,
            time_val: float, # 내장 함수와 겹치지 않도록 'time'에서 이름 변경
            text: str = None
        ) -> None:
        """
        제안된 정지 기간으로 `on_new_waiting_time` 콜백을 호출합니다.

        중복 호출을 피하기 위해 새로운 제안 시간이 현재 저장된
        `current_waiting_time`과 다른 경우에만 콜백을 트리거합니다.

        Args:
            time_val: 계산된 제안 대기 시간(초).
            text: 이 대기 시간 계산과 관련된 텍스트 세그먼트.
        """
        if time_val == self.current_waiting_time:
            return # 변경 없음, 아무것도 하지 않음

        self.current_waiting_time = time_val

        if self.on_new_waiting_time:
            self.on_new_waiting_time(time_val, text)

    def get_completion_probability(
        self,
        sentence: str
    ) -> float:
        """
        ML 모델을 사용하여 주어진 문장이 완전할 확률을 계산합니다.

        성능 향상을 위해 내부 LRU 캐시(`_completion_probability_cache`)를 사용하여
        이전에 본 문장에 대한 결과를 저장하고 검색합니다.

        Args:
            sentence: 분석할 입력 문장 문자열.

        Returns:
            모델에 의해 문장이 완전하다고 간주될 확률(0.0과 1.0 사이)을
            나타내는 부동 소수점.
        """
        # 먼저 캐시 확인
        if sentence in self._completion_probability_cache:
            self._completion_probability_cache.move_to_end(sentence) # 최근 사용으로 표시
            return self._completion_probability_cache[sentence]

        # 캐시에 없으면 모델 예측 실행
        import torch
        import torch.nn.functional as F

        inputs = self.tokenizer(
            sentence,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_length
        )
        # 입력 텐서를 올바른 장치(CPU 또는 GPU)로 이동
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad(): # 추론을 위해 그래디언트 계산 비활성화
            outputs = self.classification_model(**inputs)

        logits = outputs.logits
        # softmax를 적용하여 확률 [prob_incomplete, prob_complete] 얻기
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()
        prob_complete = probabilities[1] # 인덱스 1은 'complete' 레이블에 해당

        # 결과를 캐시에 저장
        self._completion_probability_cache[sentence] = prob_complete
        self._completion_probability_cache.move_to_end(sentence) # 최근 사용으로 표시

        # 캐시 크기 유지 (LRU 축출)
        if len(self._completion_probability_cache) > self._completion_probability_cache_max_size:
            self._completion_probability_cache.popitem(last=False) # 가장 최근에 사용되지 않은 항목 제거

        return prob_complete

    def get_suggested_whisper_pause(self, text: str) -> float:
        """
        텍스트의 끝 구두점에 따라 기본 정지 기간을 결정합니다.

        특정 끝 패턴('...', '.', '!', '?')을 확인하고 인스턴스의 설정에
        정의된 해당 정지 기간(예: `self.ellipsis_pause`)을 반환합니다.
        특정 구두점이 일치하지 않으면 `self.unknown_sentence_detection_pause`를
        반환합니다.

        Args:
            text: 입력 텍스트 문자열.

        Returns:
            끝 구두점에 따른 제안된 정지 기간(초).
        """
        if ends_with_string(text, "..."):
            return self.ellipsis_pause
        elif ends_with_string(text, "."):
            return self.punctuation_pause
        elif ends_with_string(text, "!"):
            return self.exclamation_pause
        elif ends_with_string(text, "?"):
            return self.question_pause
        else:
            # 특정 끝이 감지되지 않음, 알 수 없는 끝에 대한 일반 정지 사용
            return self.unknown_sentence_detection_pause

    def _text_worker(
        self
    ) -> None:
        """
        큐에서 텍스트를 처리하여 발화자 전환을 감지하는 백그라운드 워커 스레드.

        `self.text_queue`에서 텍스트 항목을 계속 검색합니다. 각 항목에 대해:
        1. 텍스트를 전처리합니다.
        2. 텍스트 이력 데크를 업데이트합니다.
        3. 구두점 일관성을 분석하기 위해 최근 일치하는 텍스트 세그먼트를 찾습니다.
        4. 일치 항목에서 관찰된 구두점을 기반으로 평균 정지를 계산합니다.
        5. 문장 완성 모델을 위해 텍스트를 추가로 정리합니다.
        6. 모델에서 완성 확률을 얻습니다(캐시 사용).
        7. 모델의 확률을 다른 정지 값으로 보간합니다.
        8. 가중치를 사용하여 구두점 기반 정지와 모델 기반 정지를 결합합니다.
        9. 속도 계수 및 조정(예: 말줄임표)을 적용합니다.
        10. 최종 정지가 최소 파이프라인 지연 시간 요구 사항을 충족하는지 확인합니다.
        11. 최종 계산된 정지 기간으로 `suggest_time`을 호출합니다.
        잠재적인 종료를 허용하기 위해 큐 타임아웃을 부드럽게 처리합니다.
        """
        while True:
            try:
                # 큐에서 텍스트를 기다리되, 영원히 차단되지 않도록 타임아웃 설정
                text = self.text_queue.get(block=True, timeout=0.1)
            except queue.Empty:
                # 타임아웃 내에 텍스트를 받지 못함, 다시 반복
                time.sleep(0.01) # 유휴 상태일 때 CPU를 양보하기 위한 작은 sleep
                continue

            # --- 텍스트를 받으면 처리 시작 ---
            logger.info(f"🎤⚙️ \"{text}\"에 대한 정지 계산 시작")
            
            processed_text = preprocess_text(text) # 초기 정리 적용

            # 이력 데크 업데이트
            current_time = time.time()
            self.text_time_deque.append((current_time, processed_text))
            text_without_punctuation = strip_ending_punctuation(processed_text)
            self.texts_without_punctuation.append((processed_text, text_without_punctuation))

            # 일관된 구두점 정지를 위해 최근 일치하는 텍스트 분석
            matches = find_matching_texts(self.texts_without_punctuation)

            added_pauses = 0
            contains_ellipses = False
            if matches: # matches가 비어 있을 때 0으로 나누는 것을 방지
                for i, match in enumerate(matches):
                    same_text, _ = match # 여기서는 원본 텍스트만 필요
                    whisper_suggested_pause_match = self.get_suggested_whisper_pause(same_text)
                    added_pauses += whisper_suggested_pause_match
                    if ends_with_string(same_text, "..."):
                        contains_ellipses = True
                # 최근 일관된 세그먼트를 기반으로 평균 정지 계산
                avg_pause = added_pauses / len(matches)
            else:
                # 일치하는 것이 없으면 현재 텍스트가 제안하는 정지를 직접 사용
                 avg_pause = self.get_suggested_whisper_pause(processed_text)
                 if ends_with_string(processed_text, "..."):
                    contains_ellipses = True

            whisper_suggested_pause = avg_pause # 평균된 정지 사용

            # 문장 완성 모델을 위해 텍스트 준비 (모든 구두점 제거)
            import string
            transtext = processed_text.translate(str.maketrans('', '', string.punctuation))
            # 끝에 남아있을 수 있는 알파벳/숫자가 아닌 문자 추가 정리
            cleaned_for_model = re.sub(r'[^a-zA-Z\s]+$', '', transtext).rstrip() # 뒤따르는 공백도 제거

            # 문장 완성 확률 얻기
            prob_complete = self.get_completion_probability(cleaned_for_model)

            # 확률을 정지 기간으로 보간
            sentence_finished_model_pause = interpolate_detection(prob_complete)

            # 정지 결합: 구두점 정지에 더 많은 중요도를 두는 가중 평균
            weight_towards_whisper = 0.65
            weighted_pause = (weight_towards_whisper * whisper_suggested_pause +
                             (1 - weight_towards_whisper) * sentence_finished_model_pause)

            # 전체 속도 계수 적용
            final_pause = weighted_pause * self.detection_speed

            # 최근에 말줄임표가 감지된 경우 약간의 추가 정지 추가
            if contains_ellipses:
                final_pause += 0.2

            logger.info(f"🎤📊 계산된 정지: 구두점={whisper_suggested_pause:.2f}, 모델={sentence_finished_model_pause:.2f}, 가중치={weighted_pause:.2f}, 최종={final_pause:.2f} (\"{processed_text}\", 확률={prob_complete:.2f})")


            # 최종 정지가 파이프라인 지연 시간 오버헤드보다 작지 않은지 확인
            min_pause = self.pipeline_latency + self.pipeline_latency_overhead
            if final_pause < min_pause:
                logger.info(f"🎤⚠️ 최종 정지({final_pause:.2f}s)가 최소값({min_pause:.2f}s)보다 작습니다. 최소값을 사용합니다.")
                final_pause = min_pause
            
            # 콜백을 통해 계산된 시간 제안
            self.suggest_time(final_pause, processed_text) # 컨텍스트를 위해 processed_text 사용

            # 큐에 대해 작업 완료 표시 (queue.join() 사용 시 중요)
            self.text_queue.task_done()

    def calculate_waiting_time(
            self,
            text: str) -> None:
        """
        대기 시간 계산을 위해 텍스트 세그먼트를 처리 큐에 추가합니다.

        이것은 텍스트를 발화자 전환 감지 시스템에 공급하는 진입점입니다.
        실제 계산은 백그라운드 워커 스레드에서 비동기적으로 발생합니다.

        Args:
            text: 처리할 텍스트 세그먼트 (예: STT에서).
        """
        logger.info(f"🎤📥 정지 계산을 위해 텍스트 큐에 추가: \"{text}\"")
        self.text_queue.put(text)

    def reset(self) -> None:
        """
        TurnDetection 인스턴스의 내부 상태를 리셋합니다.

        텍스트 이력 데크, 모델 예측 캐시를 지우고 현재 대기 시간 추적기를
        리셋합니다. 새로운 대화나 상호작용 컨텍스트를 시작하는 데 유용합니다.
        """
        logger.info("🎤🔄 TurnDetection 상태 리셋 중.")
        # 이력 데크 지우기
        self.text_time_deque.clear()
        self.texts_without_punctuation.clear()
        # 마지막으로 제안된 시간 리셋
        self.current_waiting_time = -1
        # 예측 캐시 지우기
        if hasattr(self, "_completion_probability_cache"):
            self._completion_probability_cache.clear()
        # 처리 큐 지우기 (선택 사항, 처리되지 않은 항목을 버릴 수 있음)
        # while not self.text_queue.empty():
        #     try:
        #         self.text_queue.get_nowait()
        #         self.text_queue.task_done()
        #     except queue.Empty:
        #         break
