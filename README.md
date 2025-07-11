# TALKi

실시간 AI 챗봇 토키입니다.
이 프로젝트는 메타버스 아카데미 4기 융합 프로젝트의 일환으로 제작되었습니다.

## Features

- AI 두 명과 대화하며 도중에 말 끊기가 가능합니다.
- TTS 엔진을 선택 가능합니다. (ElevenLabs, Coqui, Orpheus, Kokoro)
- LLM 엔진을 선택 가능합니다. (OpenAI, Ollama, LMStudio)
- AI-to-AI 대화 기능이 구현되었습니다.
- 웹소켓 베이스로 실시간 커뮤니케이션이 가능합니다.
- GPU 가속을 사용합니다.

## Quick Start

- 도커 사용을 권장합니다.

### Docker (Recommended)
```bash
docker-compose up --build
```

### Local Development
```bash
pip install -r requirements.txt
python code/server.py
```

## Configuration

Create a `.env` file:
```bash
OPENAI_API_KEY=your_openai_key
ELEVENLABS_API_KEY=your_elevenlabs_key
LANGSMITH_API_KEY=your_langsmith_key
```

## Requirements

- Python 3.8+
- NVIDIA GPU (optional, for acceleration)
- Docker & Docker Compose (for containerized deployment)