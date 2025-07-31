(function() {
  const originalLog = console.log.bind(console);
  console.log = (...args) => {
    const now = new Date();
    const hh = String(now.getHours()).padStart(2, '0');
    const mm = String(now.getMinutes()).padStart(2, '0');
    const ss = String(now.getSeconds()).padStart(2, '0');
    const ms = String(now.getMilliseconds()).padStart(3, '0');
    originalLog(
      `[${hh}:${mm}:${ss}.${ms}]`,
      ...args
    );
  };
})();

// --- 아이콘 정의 ---
const micOnIcon = `
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" stroke="currentColor" stroke-width="2"/>
    <path d="M19 10v1a7 7 0 0 1-14 0v-1" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="12" y1="19" x2="12" y2="23" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  </svg>`;
const micOffIcon = `
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <line x1="1" y1="1" x2="23" y2="23" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z" stroke="currentColor" stroke-width="2"/>
    <path d="M19 10v1a7 7 0 0 1-14 0v-1" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <line x1="12" y1="19" x2="12" y2="23" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
  </svg>`;
const pauseIcon = `
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M10 4H6V20H10V4Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
    <path d="M18 4H14V20H18V4Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;
const resumeIcon = `
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
    <path d="M8 5L8 19" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    <path d="M8 5L18 12L8 19" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`;


const micToggleBtn = document.getElementById("micToggleBtn")
const aiPauseBtn = document.getElementById("aiPauseBtn");
let isMicEnabled = true;

const statusDiv = document.getElementById("status");
const messagesDiv = document.getElementById("messages");
const speedSlider = document.getElementById("speedSlider");
speedSlider.disabled = true;  // start disabled
aiPauseBtn.disabled = true; // start disabled

let socket = null;
let audioContext = null;
let mediaStream = null;
let micWorkletNode = null;
let ttsWorkletNode = null;

let isTTSPlaying = false;
let ignoreIncomingTTS = false;
let waitingForTTSComplete = false;

let chatHistory = [];
let typingUser = "";
let typingAssistant = {
    "AI1": "",
    "AI2": "",
    "assistant": "" // Fallback
};

// --- batching + fixed 8‑byte header setup ---
const BATCH_SAMPLES = 2048;
const HEADER_BYTES  = 8;
const FRAME_BYTES   = BATCH_SAMPLES * 2;
const MESSAGE_BYTES = HEADER_BYTES + FRAME_BYTES;

const bufferPool = [];
let batchBuffer = null;
let batchView = null;
let batchInt16 = null;
let batchOffset = 0;

function initBatch() {
  if (!batchBuffer) {
    batchBuffer = bufferPool.pop() || new ArrayBuffer(MESSAGE_BYTES);
    batchView   = new DataView(batchBuffer);
    batchInt16  = new Int16Array(batchBuffer, HEADER_BYTES);
    batchOffset = 0;
  }
}

function flushBatch() {
  const ts = Date.now() & 0xFFFFFFFF;
  batchView.setUint32(0, ts, false);
  const flags = isTTSPlaying ? 1 : 0;
  batchView.setUint32(4, flags, false);

  socket.send(batchBuffer);

  bufferPool.push(batchBuffer);
  batchBuffer = null;
}

function flushRemainder() {
  if (batchOffset > 0) {
    for (let i = batchOffset; i < BATCH_SAMPLES; i++) {
      batchInt16[i] = 0;
    }
    flushBatch();
  }
}

function initAudioContext() {
  if (!audioContext) {
    audioContext = new AudioContext();
  }
}

function base64ToInt16Array(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) {
    view[i] = raw.charCodeAt(i);
  }
  return new Int16Array(buf);
}

async function startRawPcmCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        sampleRate: { ideal: 24000 },
        channelCount: 1,
        echoCancellation: true,
        // autoGainControl: true,
        noiseSuppression: true
      }
    });
    mediaStream = stream;
    isMicEnabled = true;
    updateMicButton();
    
    initAudioContext();
    await audioContext.audioWorklet.addModule('/static/pcmWorkletProcessor.js');
    micWorkletNode = new AudioWorkletNode(audioContext, 'pcm-worklet-processor');

    micWorkletNode.port.onmessage = ({ data }) => {
      const incoming = new Int16Array(data);
      let read = 0;
      while (read < incoming.length) {
        initBatch();
        const toCopy = Math.min(
          incoming.length - read,
          BATCH_SAMPLES - batchOffset
        );
        batchInt16.set(
          incoming.subarray(read, read + toCopy),
          batchOffset
        );
        batchOffset += toCopy;
        read       += toCopy;
        if (batchOffset === BATCH_SAMPLES) {
          flushBatch();
        }
      }
    };

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(micWorkletNode);
    statusDiv.textContent = "Recording...";
  } catch (err) {
    statusDiv.textContent = "Mic access denied.";
    console.error(err);
  }
}

async function setupTTSPlayback() {
  await audioContext.audioWorklet.addModule('/static/ttsPlaybackProcessor.js');
  ttsWorkletNode = new AudioWorkletNode(
    audioContext,
    'tts-playback-processor'
  );

  ttsWorkletNode.port.onmessage = (event) => {
    const { type } = event.data;
    if (type === 'ttsPlaybackStarted') {
      if (!isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
        isTTSPlaying = true;
        waitingForTTSComplete = true;
        console.log(
          "TTS playback started. Reason: ttsWorkletNode Event ttsPlaybackStarted."
        );
        socket.send(JSON.stringify({ type: 'tts_start' }));
      }
    } else if (type === 'ttsPlaybackStopped') {
      // TTS 오디오 재생이 끝났을 때
      if (isTTSPlaying) {
        isTTSPlaying = false;
        if (waitingForTTSComplete) {
          // 서버 완료 신호를 아직 기다리고 있으면 대기
          console.log(
            "TTS audio playback stopped, but waiting for server completion signal."
          );
        } else {
          // 서버 완료 신호를 이미 받았으면 tts_stop 전송
          console.log(
            "TTS playback stopped after server completion signal, sending tts_stop"
          );
          socket.send(JSON.stringify({ type: 'tts_stop' }));
        }
      }
    }
  };
  ttsWorkletNode.connect(audioContext.destination);
}

function cleanupAudio() {
  if (micWorkletNode) {
    micWorkletNode.disconnect();
    micWorkletNode = null;
  }
  if (ttsWorkletNode) {
    ttsWorkletNode.disconnect();
    ttsWorkletNode = null;
  }
  if (audioContext) {
    audioContext.close();
    audioContext = null;
  }
  if (mediaStream) {
    mediaStream.getAudioTracks().forEach(track => track.stop());
    mediaStream = null;
  }
}

function renderMessages() {
    messagesDiv.innerHTML = "";
    chatHistory.forEach(msg => {
        const msgContainer = document.createElement("div");
        msgContainer.className = `message-container ${msg.role.startsWith('AI') ? 'assistant' : msg.role}`;

        const speakerName = document.createElement("div");
        speakerName.className = "speaker-name";
        
        if (msg.role === 'user') {
            speakerName.textContent = 'You';
        } else if (msg.role === 'AI1') {
            speakerName.textContent = 'Mia';
        } else if (msg.role === 'AI2') {
            speakerName.textContent = 'Leo';
        } else {
            speakerName.textContent = 'Assistant';
        }

        const bubble = document.createElement("div");
        bubble.className = `bubble`;
        bubble.textContent = msg.content;

        msgContainer.appendChild(speakerName);
        msgContainer.appendChild(bubble);
        messagesDiv.appendChild(msgContainer);
    });

    if (typingUser) {
        const msgContainer = document.createElement("div");
        msgContainer.className = "message-container user";

        const speakerName = document.createElement("div");
        speakerName.className = "speaker-name";
        speakerName.textContent = "You";

        const bubble = document.createElement("div");
        bubble.className = "bubble typing";
        bubble.innerHTML = typingUser + '<span style="opacity:.6;">✏️</span>';
        
        msgContainer.appendChild(speakerName);
        msgContainer.appendChild(bubble);
        messagesDiv.appendChild(msgContainer);
    }

    for (const role in typingAssistant) {
        if (typingAssistant[role]) {
            const msgContainer = document.createElement("div");
            msgContainer.className = "message-container assistant";

            const speakerName = document.createElement("div");
            speakerName.className = "speaker-name";
            if (role === 'AI1') {
                speakerName.textContent = 'Mia';
            } else if (role === 'AI2') {
                speakerName.textContent = 'Leo';
            } else {
                speakerName.textContent = 'Assistant';
            }

            const bubble = document.createElement("div");
            bubble.className = "bubble typing";
            bubble.innerHTML = typingAssistant[role] + '<span style="opacity:.6;">✏️</span>';

            msgContainer.appendChild(speakerName);
            msgContainer.appendChild(bubble);
            messagesDiv.appendChild(msgContainer);
        }
    }
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
}


function handleJSONMessage({ type, content, role, is_paused }) {
  if (type === "partial_user_request") {
    typingUser = content?.trim() ? escapeHtml(content) : "";
    renderMessages();
    return;
  }
  if (type === "final_user_request") {
    if (content?.trim()) {
      chatHistory.push({ role: "user", content, type: "final" });
    }
    typingUser = "";
    renderMessages();
    return;
  }
  if (type === "partial_assistant_answer") {
    const currentRole = role || "assistant";
    // 모든 다른 타이핑 상태 초기화
    for (const key in typingAssistant) {
        if (key !== currentRole) {
            typingAssistant[key] = "";
        }
    }
    typingAssistant[currentRole] = content?.trim() ? escapeHtml(content) : "";
    renderMessages();
    return;
  }
  if (type === "final_assistant_answer") {
    const finalRole = role || "assistant";
    if (content?.trim()) {
      chatHistory.push({ role: finalRole, content, type: "final" });
    }
    // 모든 타이핑 상태 초기화
    for (const key in typingAssistant) {
        typingAssistant[key] = "";
    }
    renderMessages();
    return;
  }
  if (type === "tts_chunk") {
    if (ignoreIncomingTTS) return;
    const int16Data = base64ToInt16Array(content);
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage(int16Data);
    }
    return;
  }
  if (type === "tts_interruption") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    ignoreIncomingTTS = false;
    waitingForTTSComplete = false;
    return;
  }
  if (type === "stop_tts") {
    if (ttsWorkletNode) {
      ttsWorkletNode.port.postMessage({ type: "clear" });
    }
    isTTSPlaying = false;
    ignoreIncomingTTS = true;
    waitingForTTSComplete = false;
    console.log("TTS playback stopped. Reason: tts_interruption.");
    socket.send(JSON.stringify({ type: 'tts_stop' }));
    return;
  }
  if (type === "ai_conversation_paused_status") {
    if (is_paused) {
        aiPauseBtn.innerHTML = resumeIcon;
        aiPauseBtn.title = "Resume AI-to-AI conversation";
    } else {
        aiPauseBtn.innerHTML = pauseIcon;
        aiPauseBtn.title = "Pause AI-to-AI conversation";
    }
    return;
  }
  if (type === "tts_all_complete") {
    // 서버에서 모든 TTS 처리 완료 신호를 받았을 때
    if (waitingForTTSComplete && socket && socket.readyState === WebSocket.OPEN) {
      console.log("TTS all complete signal received from server, waiting for audio playback to finish");
      waitingForTTSComplete = false;
      // 실제 오디오 재생이 완료되면 tts_stop 신호를 보내도록 플래그 설정
      if (isTTSPlaying) {
        // 오디오가 아직 재생 중이면 재생 완료를 기다림
        console.log("Audio still playing, will send tts_stop when playback actually stops");
      } else {
        // 오디오 재생이 이미 끝났으면 즉시 tts_stop 전송
        console.log("Audio playback already finished, sending tts_stop immediately");
        socket.send(JSON.stringify({ type: 'tts_stop' }));
      }
    }
    return;
  }
}

function escapeHtml(str) {
  return (str ?? '')
    .replace(/&/g, "&amp;")
    .replace(/</g, "<")
    .replace(/>/g, ">")
    .replace(/"/g, "&quot;");
}

// 마이크 버튼 업데이트
function updateMicButton() {
    if (isMicEnabled) {
        micToggleBtn.innerHTML = micOnIcon;
        micToggleBtn.title = "Mute microphone";
        statusDiv.textContent = "Recording...";
    } else {
        micToggleBtn.innerHTML = micOffIcon;
        micToggleBtn.title = "Unmute microphone";
        statusDiv.textContent = "Mic muted";
    }
}

// UI Controls

document.getElementById("clearBtn").onclick = () => {
  chatHistory = [];
  typingUser = "";
  for (const key in typingAssistant) {
    typingAssistant[key] = "";
  }
  renderMessages();
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({ type: 'clear_history' }));
  }
};

speedSlider.addEventListener("input", (e) => {
  const speedValue = parseInt(e.target.value);
  if (socket && socket.readyState === WebSocket.OPEN) {
    socket.send(JSON.stringify({
      type: 'set_speed',
      speed: speedValue
    }));
  }
  console.log("Speed setting changed to:", speedValue);
});

document.getElementById("startBtn").onclick = async () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    statusDiv.textContent = "Already recording.";
    return;
  }
  statusDiv.textContent = "Initializing connection...";

  const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
  socket = new WebSocket(`${wsProto}//${location.host}/ws`);

  socket.onopen = async () => {
    statusDiv.textContent = "Connected. Activating mic and TTS…";
    await startRawPcmCapture();
    await setupTTSPlayback();
    speedSlider.disabled = false; 
    aiPauseBtn.disabled = false;
  };

  socket.onmessage = (evt) => {
    if (typeof evt.data === "string") {
      try {
        const msg = JSON.parse(evt.data);
        handleJSONMessage(msg);
      } catch (e) {
        console.error("Error parsing message:", e);
      }
    }
  };

  socket.onclose = () => {
    statusDiv.textContent = "Connection closed.";
    flushRemainder();
    cleanupAudio();
    speedSlider.disabled = true;
    aiPauseBtn.disabled = true;
  };

  socket.onerror = (err) => {
    statusDiv.textContent = "Connection error.";
    cleanupAudio();
    console.error(err);
    speedSlider.disabled = true; 
    aiPauseBtn.disabled = true;
  };
};

document.getElementById("stopBtn").onclick = () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    flushRemainder();
    socket.close();
  }
  cleanupAudio();
  statusDiv.textContent = "Stopped.";
};

micToggleBtn.onclick = () => {
    if (!mediaStream) return;
    isMicEnabled = !isMicEnabled;
    mediaStream.getAudioTracks().forEach(track => {
        track.enabled = isMicEnabled;
    });
    updateMicButton();
    console.log(`Microphone ${isMicEnabled ? 'enabled' : 'disabled'}`);
};

document.getElementById("copyBtn").onclick = () => {
  const text = chatHistory
    .map(msg => `${msg.role.charAt(0).toUpperCase() + msg.role.slice(1)}: ${msg.content}`)
    .join('\n');
  
  navigator.clipboard.writeText(text)
    .then(() => console.log("Conversation copied to clipboard"))
    .catch(err => console.error("Copy failed:", err));
};

document.getElementById("aiTalkBtn").onclick = () => {
  if (socket && socket.readyState === WebSocket.OPEN) {
    const topic = "Let's talk about the latest technology trends.";
    socket.send(JSON.stringify({ type: 'start_ai_conversation', topic: topic }));
    console.log("Sent start_ai_conversation request with topic:", topic);
  } else {
    console.log("Cannot start AI conversation, socket is not open.");
    statusDiv.textContent = "Connect first to start AI talk.";
  }
};

aiPauseBtn.onclick = () => {
    if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'toggle_ai_pause' }));
        console.log("Sent toggle_ai_pause request.");
    } else {
        console.log("Cannot toggle AI pause, socket is not open.");
    }
};

// First render
renderMessages();