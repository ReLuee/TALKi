# Unity Voice Chat Integration

이 폴더는 TALKi 실시간 AI 음성 채팅 시스템의 Unity 클라이언트 구현을 포함합니다.

## 파일 구조

### 필수 스크립트
- **WebSocketClient.cs**: WebSocket 통신 및 메시지 처리
- **AudioManager.cs**: 마이크 입력 및 TTS 오디오 출력 관리
- **VoiceChatUI.cs**: 사용자 인터페이스 및 채팅 화면
- **VoiceChatManager.cs**: 전체 시스템 관리자

## 설치 및 설정

### 1. 필수 패키지 설치

Unity Package Manager에서 다음 패키지들을 설치하세요:

```
- TextMeshPro
- WebSocketSharp (NuGet 또는 Asset Store)
- Newtonsoft.Json (Unity 내장 또는 별도 설치)
```

### 2. 프로젝트 설정

1. **Player Settings**에서 다음을 확인:
   - API Compatibility Level: .NET Framework (또는 .NET Standard 2.1)
   - Scripting Backend: Mono (IL2CPP도 지원)

2. **Microphone Permission** (모바일 플랫폼):
   - Android: `android.permission.RECORD_AUDIO`
   - iOS: `NSMicrophoneUsageDescription`

### 3. 씬 설정

1. 빈 GameObject 생성 후 이름을 "VoiceChatSystem"으로 변경
2. 다음 컴포넌트들을 추가:
   - `VoiceChatManager`
   - `WebSocketClient`
   - `AudioManager`
   - `AudioSource` (AudioManager에서 자동으로 요구됨)

3. UI Canvas 생성 후 `VoiceChatUI` 컴포넌트 추가

## UI 구성

### 필수 UI 요소들

Canvas 하위에 다음 UI 요소들을 생성하고 VoiceChatUI에 연결하세요:

#### 버튼들
- Connect Button
- Disconnect Button  
- Microphone Toggle Button
- Clear History Button
- AI Talk Button
- AI Pause Button
- Copy Button

#### 상태 표시
- Status Text (TextMeshPro)
- Connection Status Indicator (Image)
- Microphone Status Indicator (Image)
- TTS Status Indicator (Image)

#### 채팅 영역
- Chat Scroll Rect
- Chat Content (Vertical Layout Group)
- User Message Prefab
- Assistant Message Prefab

#### 설정 슬라이더
- Speed Slider
- Microphone Gain Slider
- TTS Volume Slider
- Microphone Selection Dropdown

#### 오디오 시각화
- Audio Level Bar (Image with Fill)

## 사용법

### 기본 연결

1. 서버 URL 설정 (기본값: `ws://localhost:8000/ws`)
2. Connect 버튼 클릭
3. 마이크 권한 허용
4. 음성으로 대화 시작

### 프로그래밍 방식 사용

```csharp
// 연결
voiceChatManager.webSocketClient.Connect();

// 메시지 전송
voiceChatManager.webSocketClient.SendMessage(new { 
    type = "clear_history" 
});

// AI 대화 시작
voiceChatManager.webSocketClient.StartAIConversation("주제를 입력하세요");

// 마이크 토글
voiceChatManager.audioManager.ToggleMicrophone();
```

## 주요 기능

### WebSocket 통신
- 실시간 양방향 통신
- 자동 재연결 (구현 필요시)
- JSON 메시지 직렬화/역직렬화

### 오디오 처리
- 실시간 마이크 입력 캡처
- PCM 오디오 데이터 전송
- TTS 오디오 스트리밍 재생
- 오디오 레벨 시각화

### 사용자 인터페이스
- 실시간 채팅 표시
- 타이핑 인디케이터
- 연결 상태 표시
- 오디오 설정 조절

### 상태 관리
- 연결 상태 추적
- 오디오 재생 상태 관리
- 채팅 히스토리 관리
- 사용자 설정 저장

## 트러블슈팅

### 일반적인 문제들

1. **WebSocket 연결 실패**
   - 서버 URL 확인
   - 방화벽 설정 확인
   - CORS 정책 확인

2. **마이크 접근 권한 없음**
   - 플랫폼별 권한 설정 확인
   - 브라우저/앱 권한 허용

3. **오디오 재생 안됨**
   - AudioSource 설정 확인
   - 볼륨 설정 확인
   - 오디오 출력 장치 확인

4. **성능 문제**
   - 오디오 버퍼 크기 조정
   - 메시지 큐 처리 최적화
   - UI 업데이트 빈도 조절

### 로그 확인

Unity Console에서 다음과 같은 로그들을 확인할 수 있습니다:

```
VoiceChatManager initialized successfully
WebSocket connected
Microphone recording started
TTS playback started
```

## 확장 가능성

### 추가 기능 구현

1. **오디오 이펙트**
   - 실시간 오디오 필터링
   - 노이즈 캔슬레이션
   - 음성 변조

2. **UI 개선**
   - 애니메이션 효과
   - 테마 시스템
   - 다국어 지원

3. **네트워크 최적화**
   - 압축 알고리즘
   - 적응형 품질 조절
   - 네트워크 상태 감지

4. **플랫폼 특화**
   - 모바일 최적화
   - VR/AR 지원
   - 플랫폼별 오디오 API 활용

## 개발 팁

1. **디버깅**: Inspector에서 실시간으로 상태 변화 확인
2. **테스팅**: 에디터와 빌드 환경에서 모두 테스트
3. **최적화**: Profiler를 사용해 성능 병목 지점 확인
4. **버전 관리**: 서버와 클라이언트 API 버전 호환성 유지