import requests
import json
import asyncio
import aiohttp
import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class BackendClient:
    """
    TALKi 백엔드 서버와 통신하기 위한 HTTP 클라이언트
    대화 내용과 발화자 정보를 데이터베이스에 저장합니다.
    """
    
    def __init__(self, backend_url: str = "http://localhost:8080", session_id: str = "default_session", branch_id: str = "main_branch"):
        """
        BackendClient 초기화
        
        Args:
            backend_url: 백엔드 서버 URL
            session_id: 대화 세션 ID
            branch_id: 대화 브랜치 ID
        """
        self.backend_url = backend_url.rstrip('/')
        self.session_id = session_id
        self.branch_id = branch_id
        self.api_base = f"{self.backend_url}/api/talki"
        
        # HTTP 세션 설정
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        logger.info(f"📡 BackendClient 초기화됨 - URL: {self.backend_url}, 세션: {session_id}, 브랜치: {branch_id}")
    
    def send_message(self, character_id: str, message: str, message_type: str = "TEXT", 
                    emotion: str = None, animation: str = None) -> bool:
        """
        동기적으로 메시지를 백엔드에 전송합니다.
        
        Args:
            character_id: 발화자 ID (user, assistant, AI 캐릭터명 등)
            message: 메시지 내용
            message_type: 메시지 타입 (TEXT, SPEECH 등)
            emotion: 감정 정보 (선택사항)
            animation: 애니메이션 정보 (선택사항)
            
        Returns:
            bool: 전송 성공 여부
        """
        payload = {
            "session_id": self.session_id,
            "branch_id": self.branch_id,
            "character_id": character_id,
            "message": message,
            "message_type": message_type
        }
        
        if emotion:
            payload["emotion"] = emotion
        if animation:
            payload["animation"] = animation
        
        try:
            url = f"{self.api_base}/message/character"
            logger.info(f"📡➡️ 백엔드로 메시지 전송: {character_id} -> 세션:{self.session_id}, 브랜치:{self.branch_id}")
            logger.debug(f"📡➡️ 메시지 내용: '{message[:50]}{'...' if len(message) > 50 else ''}'")
            
            response = self.session.post(url, json=payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                # MESSAGE_SENT는 성공 상태입니다
                if result.get("success") or result.get("message") == "MESSAGE_SENT":
                    logger.info(f"📡✅ 메시지 전송 성공: {character_id}")
                    return True
                else:
                    logger.error(f"📡❌ 백엔드 응답 오류: {result.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"📡❌ HTTP 오류: {response.status_code} - {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            logger.warning(f"📡⏰ 백엔드 요청 타임아웃: {character_id}")
            return False
        except requests.exceptions.ConnectionError:
            logger.warning(f"📡🔌 백엔드 연결 실패: {character_id}")
            return False
        except Exception as e:
            logger.error(f"📡💥 메시지 전송 중 예외 발생: {e}")
            return False
    
    async def send_message_async(self, character_id: str, message: str, message_type: str = "TEXT",
                               emotion: str = None, animation: str = None) -> bool:
        """
        비동기적으로 메시지를 백엔드에 전송합니다.
        
        Args:
            character_id: 발화자 ID (user, assistant, AI 캐릭터명 등)
            message: 메시지 내용
            message_type: 메시지 타입 (TEXT, SPEECH 등)
            emotion: 감정 정보 (선택사항)
            animation: 애니메이션 정보 (선택사항)
            
        Returns:
            bool: 전송 성공 여부
        """
        payload = {
            "session_id": self.session_id,
            "branch_id": self.branch_id,
            "character_id": character_id,
            "message": message,
            "message_type": message_type
        }
        
        if emotion:
            payload["emotion"] = emotion
        if animation:
            payload["animation"] = animation
        
        try:
            url = f"{self.api_base}/message/character"
            logger.debug(f"📡➡️ 백엔드로 비동기 메시지 전송: {character_id} -> '{message[:50]}{'...' if len(message) > 50 else ''}'")
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    if response.status == 200:
                        result = await response.json()
                        # MESSAGE_SENT는 성공 상태입니다
                        if result.get("success") or result.get("message") == "MESSAGE_SENT":
                            logger.info(f"📡✅ 비동기 메시지 전송 성공: {character_id}")
                            return True
                        else:
                            logger.error(f"📡❌ 백엔드 응답 오류: {result.get('message', 'Unknown error')}")
                            return False
                    else:
                        text = await response.text()
                        logger.error(f"📡❌ HTTP 오류: {response.status} - {text}")
                        return False
                        
        except asyncio.TimeoutError:
            logger.warning(f"📡⏰ 백엔드 비동기 요청 타임아웃: {character_id}")
            return False
        except aiohttp.ClientConnectorError:
            logger.warning(f"📡🔌 백엔드 비동기 연결 실패: {character_id}")
            return False
        except Exception as e:
            logger.error(f"📡💥 비동기 메시지 전송 중 예외 발생: {e}")
            return False
    
    def send_user_message(self, message: str) -> bool:
        """
        사용자 메시지를 백엔드에 전송합니다.
        
        Args:
            message: 사용자 메시지 내용
            
        Returns:
            bool: 전송 성공 여부
        """
        return self.send_message("user", message, "USER_SPEECH")
    
    def send_assistant_message(self, message: str, speaker_role: str = "assistant") -> bool:
        """
        AI 어시스턴트 메시지를 백엔드에 전송합니다.
        
        Args:
            message: AI 메시지 내용
            speaker_role: 발화자 역할 (assistant, AI 캐릭터명 등)
            
        Returns:
            bool: 전송 성공 여부
        """
        return self.send_message(speaker_role, message, "AI_SPEECH")
    
    async def send_user_message_async(self, message: str) -> bool:
        """
        사용자 메시지를 비동기적으로 백엔드에 전송합니다.
        
        Args:
            message: 사용자 메시지 내용
            
        Returns:
            bool: 전송 성공 여부
        """
        return await self.send_message_async("user", message, "USER_SPEECH")
    
    async def send_assistant_message_async(self, message: str, speaker_role: str = "assistant") -> bool:
        """
        AI 어시스턴트 메시지를 비동기적으로 백엔드에 전송합니다.
        
        Args:
            message: AI 메시지 내용
            speaker_role: 발화자 역할 (assistant, AI 캐릭터명 등)
            
        Returns:
            bool: 전송 성공 여부
        """
        return await self.send_message_async(speaker_role, message, "AI_SPEECH")
    
    def test_connection(self) -> bool:
        """
        백엔드 서버 연결을 테스트합니다.
        
        Returns:
            bool: 연결 성공 여부
        """
        try:
            url = f"{self.api_base}/health"
            response = self.session.get(url, timeout=3)
            
            if response.status_code == 200:
                logger.info("📡✅ 백엔드 연결 테스트 성공")
                return True
            else:
                logger.warning(f"📡⚠️ 백엔드 응답 코드: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"📡❌ 백엔드 연결 테스트 실패: {e}")
            return False
    
    def initialize_session(self) -> bool:
        """
        백엔드에 세션과 브랜치를 초기화합니다.
        
        Returns:
            bool: 초기화 성공 여부
        """
        try:
            # 1. 세션 생성 (참가자 등록)
            join_payload = {
                "session_id": self.session_id,
                "participant_id": "talki_user",
                "participant_name": "TALKi User",
                "participant_type": "HUMAN"
            }
            
            url = f"{self.api_base}/session/join"
            response = self.session.post(url, json=join_payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                # SESSION_JOINED는 성공 상태입니다
                if result.get("success") or result.get("message") == "SESSION_JOINED":
                    logger.info(f"📡✅ 세션 생성 성공: {self.session_id}")
                else:
                    logger.error(f"📡❌ 세션 생성 실패: {result.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"📡❌ 세션 생성 HTTP 오류: {response.status_code} - {response.text}")
                return False
            
            # 2. 브랜치 생성
            branch_payload = {
                "session_id": self.session_id,
                "branch_name": "Main Conversation",
                "description": "Main conversation branch for TALKi",
                "branch_type": "MAIN",
                "created_by": "talki_user"
            }
            
            url = f"{self.api_base}/branch/create"
            response = self.session.post(url, json=branch_payload, timeout=5)
            
            if response.status_code == 200:
                result = response.json()
                # BRANCH_CREATED는 성공 상태입니다
                if result.get("success") or result.get("message") == "BRANCH_CREATED":
                    # 실제 생성된 브랜치 ID 업데이트
                    if isinstance(result.get("data"), dict) and result["data"].get("branch_id"):
                        self.branch_id = result["data"]["branch_id"]
                        logger.info(f"📡✅ 브랜치 생성 성공: {self.branch_id}")
                    else:
                        logger.info(f"📡✅ 브랜치 생성 성공: {self.branch_id} (기본값 사용)")
                    return True
                else:
                    logger.error(f"📡❌ 브랜치 생성 실패: {result.get('message', 'Unknown error')}")
                    return False
            else:
                logger.error(f"📡❌ 브랜치 생성 HTTP 오류: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"📡💥 세션 초기화 중 예외 발생: {e}")
            return False
    
    def set_session_info(self, session_id: str, branch_id: str):
        """
        세션 및 브랜치 정보를 업데이트합니다.
        
        Args:
            session_id: 새로운 세션 ID
            branch_id: 새로운 브랜치 ID
        """
        self.session_id = session_id
        self.branch_id = branch_id
        logger.info(f"📡🔄 세션 정보 업데이트됨 - 세션: {session_id}, 브랜치: {branch_id}")
    
    def close(self):
        """
        HTTP 세션을 정리합니다.
        """
        if hasattr(self, 'session'):
            self.session.close()
            logger.info("📡🧹 BackendClient HTTP 세션 정리됨")


# 전역 백엔드 클라이언트 인스턴스
_backend_client = None

def get_backend_client() -> BackendClient:
    """
    전역 백엔드 클라이언트 인스턴스를 가져옵니다.
    
    Returns:
        BackendClient: 백엔드 클라이언트 인스턴스
    """
    global _backend_client
    if _backend_client is None:
        _backend_client = BackendClient()
    return _backend_client

def initialize_backend_client(backend_url: str = "http://localhost:8080", 
                            session_id: str = "default_session", 
                            branch_id: str = "main_branch") -> BackendClient:
    """
    전역 백엔드 클라이언트를 초기화합니다.
    
    Args:
        backend_url: 백엔드 서버 URL
        session_id: 대화 세션 ID
        branch_id: 대화 브랜치 ID
        
    Returns:
        BackendClient: 초기화된 백엔드 클라이언트 인스턴스
    """
    global _backend_client
    _backend_client = BackendClient(backend_url, session_id, branch_id)
    return _backend_client

def cleanup_backend_client():
    """
    전역 백엔드 클라이언트를 정리합니다.
    """
    global _backend_client
    if _backend_client:
        _backend_client.close()
        _backend_client = None