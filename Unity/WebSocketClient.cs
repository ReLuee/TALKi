using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using WebSocketSharp;
using Newtonsoft.Json;
using System.Text;

public class WebSocketClient : MonoBehaviour
{
    [Header("Connection Settings")]
    public string serverUrl = "ws://localhost:8000/ws";
    public bool useSSL = false;
    
    [Header("Audio Settings")]
    public int sampleRate = 48000; // Will be set to system sample rate on init
    public int batchSamples = 2048;
    public float audioGain = 1.0f;
    
    [Header("Debug Info")]
    public bool showDebugInfo = false;
    
    [Header("Events")]
    public UnityEngine.Events.UnityEvent OnConnected;
    public UnityEngine.Events.UnityEvent OnDisconnected;
    public UnityEngine.Events.UnityEvent<string> OnUserTranscription;
    public UnityEngine.Events.UnityEvent<string, string> OnAssistantResponse;
    public UnityEngine.Events.UnityEvent<float[]> OnTTSAudioReceived;
    
    private WebSocket webSocket;
    private bool isConnected = false;
    private bool isTTSPlaying = false;
    private bool waitingForTTSComplete = false;
    
    // Audio processing
    private const int HEADER_BYTES = 8;
    private Queue<float[]> audioQueue = new Queue<float[]>();
    private byte[] batchBuffer;
    private int batchOffset = 0;
    
    // Message handling
    private Queue<string> messageQueue = new Queue<string>();
    private readonly object messageLock = new object();
    
    void Start()
    {
        // Sync sample rate with system settings
        sampleRate = AudioSettings.outputSampleRate;
        Debug.Log($"WebSocketClient using sample rate: {sampleRate}Hz");
        
        InitializeAudioBuffer();
    }
    
    void Update()
    {
        ProcessMessages();
    }
    
    void InitializeAudioBuffer()
    {
        int messageBytes = HEADER_BYTES + (batchSamples * 2); // 16-bit PCM
        batchBuffer = new byte[messageBytes];
    }
    
    public void Connect()
    {
        if (isConnected) return;
        
        try
        {
            string url = useSSL ? serverUrl.Replace("ws://", "wss://") : serverUrl;
            webSocket = new WebSocket(url);
            
            webSocket.OnOpen += OnWebSocketOpen;
            webSocket.OnMessage += OnWebSocketMessage;
            webSocket.OnClose += OnWebSocketClose;
            webSocket.OnError += OnWebSocketError;
            
            webSocket.Connect();
        }
        catch (Exception e)
        {
            Debug.LogError($"WebSocket connection error: {e.Message}");
        }
    }
    
    public void Disconnect()
    {
        if (webSocket != null && isConnected)
        {
            FlushRemainingAudio();
            webSocket.Close();
        }
    }
    
    void OnWebSocketOpen(object sender, EventArgs e)
    {
        isConnected = true;
        Debug.Log("WebSocket connected");
        OnConnected?.Invoke();
    }
    
    void OnWebSocketMessage(object sender, MessageEventArgs e)
    {
        if (e.IsText)
        {
            lock (messageLock)
            {
                messageQueue.Enqueue(e.Data);
            }
        }
    }
    
    void OnWebSocketClose(object sender, CloseEventArgs e)
    {
        isConnected = false;
        Debug.Log($"WebSocket closed: {e.Reason}");
        OnDisconnected?.Invoke();
    }
    
    void OnWebSocketError(object sender, ErrorEventArgs e)
    {
        Debug.LogError($"WebSocket error: {e.Message}");
    }
    
    void ProcessMessages()
    {
        lock (messageLock)
        {
            while (messageQueue.Count > 0)
            {
                string message = messageQueue.Dequeue();
                HandleMessage(message);
            }
        }
    }
    
    void HandleMessage(string jsonMessage)
    {
        try
        {
            var message = JsonConvert.DeserializeObject<WebSocketMessage>(jsonMessage);
            
            switch (message.type)
            {
                case "partial_user_request":
                case "final_user_request":
                    OnUserTranscription?.Invoke(message.content ?? "");
                    break;
                    
                case "partial_assistant_answer":
                case "final_assistant_answer":
                    OnAssistantResponse?.Invoke(message.role ?? "assistant", message.content ?? "");
                    break;
                    
                case "tts_chunk":
                    HandleTTSChunk(message.content);
                    break;
                    
                case "tts_interruption":
                case "stop_tts":
                    HandleTTSStop();
                    break;
                    
                case "tts_all_complete":
                    HandleTTSComplete();
                    break;
                    
                case "ai_conversation_paused_status":
                    Debug.Log($"AI conversation paused: {message.is_paused}");
                    break;
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error handling message: {e.Message}");
        }
    }
    
    void HandleTTSChunk(string base64Audio)
    {
        if (string.IsNullOrEmpty(base64Audio)) return;
        
        try
        {
            byte[] audioBytes = Convert.FromBase64String(base64Audio);
            float[] audioSamples = ConvertBytesToFloat(audioBytes);
            
            if (showDebugInfo)
            {
                Debug.Log($"Received TTS chunk: {audioBytes.Length} bytes, {audioSamples.Length} samples");
            }
            
            OnTTSAudioReceived?.Invoke(audioSamples);
            
            if (!isTTSPlaying)
            {
                isTTSPlaying = true;
                waitingForTTSComplete = true;
                SendMessage(new { type = "tts_start" });
                if (showDebugInfo)
                {
                    Debug.Log("TTS playback started from WebSocket");
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError($"Error processing TTS chunk: {e.Message}");
        }
    }
    
    void HandleTTSStop()
    {
        if (showDebugInfo)
        {
            Debug.Log("Handling TTS stop from server");
        }
        
        isTTSPlaying = false;
        waitingForTTSComplete = false;
        SendMessage(new { type = "tts_stop" });
    }
    
    void HandleTTSComplete()
    {
        if (showDebugInfo)
        {
            Debug.Log($"TTS complete signal received. Waiting: {waitingForTTSComplete}, Playing: {isTTSPlaying}");
        }
        
        if (waitingForTTSComplete)
        {
            waitingForTTSComplete = false;
            if (!isTTSPlaying)
            {
                SendMessage(new { type = "tts_stop" });
                if (showDebugInfo)
                {
                    Debug.Log("Sent tts_stop after completion signal");
                }
            }
        }
    }
    
    float[] ConvertBytesToFloat(byte[] bytes)
    {
        if (bytes.Length % 2 != 0)
        {
            Debug.LogWarning($"Invalid audio data length: {bytes.Length} bytes (not divisible by 2)");
            // Trim to even length
            byte[] trimmedBytes = new byte[bytes.Length - 1];
            Array.Copy(bytes, trimmedBytes, trimmedBytes.Length);
            bytes = trimmedBytes;
        }
        
        float[] samples = new float[bytes.Length / 2];
        for (int i = 0; i < samples.Length; i++)
        {
            short sample = BitConverter.ToInt16(bytes, i * 2);
            samples[i] = sample / 32768f; // Convert to -1.0 to 1.0 range (matches web worklet)
        }
        return samples;
    }
    
    public void SendAudioData(float[] audioSamples)
    {
        if (!isConnected || audioSamples == null) return;
        
        int read = 0;
        while (read < audioSamples.Length)
        {
            int toCopy = Mathf.Min(audioSamples.Length - read, batchSamples - batchOffset);
            
            // Convert float samples to 16-bit PCM and add to batch (match web client logic)
            for (int i = 0; i < toCopy; i++)
            {
                float s = audioSamples[read + i] * audioGain;
                // Clamp the sample to [-1, 1] range
                s = s < -1f ? -1f : s > 1f ? 1f : s;
                // Convert to 16-bit PCM using web client's asymmetric range
                short sample = s < 0 ? (short)(s * 0x8000) : (short)(s * 0x7FFF);
                
                int bufferIndex = HEADER_BYTES + (batchOffset + i) * 2;
                batchBuffer[bufferIndex] = (byte)(sample & 0xFF);
                batchBuffer[bufferIndex + 1] = (byte)((sample >> 8) & 0xFF);
            }
            
            batchOffset += toCopy;
            read += toCopy;
            
            if (batchOffset >= batchSamples)
            {
                FlushAudioBatch();
            }
        }
    }
    
    void FlushAudioBatch()
    {
        if (!isConnected) return;
        
        // Set timestamp (first 4 bytes) - Big-endian format to match web client
        uint timestamp = (uint)(DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() & 0xFFFFFFFF);
        batchBuffer[0] = (byte)((timestamp >> 24) & 0xFF);
        batchBuffer[1] = (byte)((timestamp >> 16) & 0xFF);
        batchBuffer[2] = (byte)((timestamp >> 8) & 0xFF);
        batchBuffer[3] = (byte)(timestamp & 0xFF);
        
        // Set flags (next 4 bytes) - Big-endian format to match web client
        uint flags = isTTSPlaying ? 1u : 0u;
        batchBuffer[4] = (byte)((flags >> 24) & 0xFF);
        batchBuffer[5] = (byte)((flags >> 16) & 0xFF);
        batchBuffer[6] = (byte)((flags >> 8) & 0xFF);
        batchBuffer[7] = (byte)(flags & 0xFF);
        
        webSocket.Send(batchBuffer);
        batchOffset = 0;
    }
    
    void FlushRemainingAudio()
    {
        if (batchOffset > 0)
        {
            // Fill remaining samples with zeros
            for (int i = batchOffset; i < batchSamples; i++)
            {
                int bufferIndex = HEADER_BYTES + i * 2;
                batchBuffer[bufferIndex] = 0;
                batchBuffer[bufferIndex + 1] = 0;
            }
            FlushAudioBatch();
        }
    }
    
    public void SendMessage(object message)
    {
        if (!isConnected) return;
        
        try
        {
            string json = JsonConvert.SerializeObject(message);
            webSocket.Send(json);
        }
        catch (Exception e)
        {
            Debug.LogError($"Error sending message: {e.Message}");
        }
    }
    
    public void ClearHistory()
    {
        SendMessage(new { type = "clear_history" });
    }
    
    public void SetSpeed(int speed)
    {
        SendMessage(new { type = "set_speed", speed = speed });
    }
    
    public void StartAIConversation(string topic = "Let's have a conversation.")
    {
        SendMessage(new { type = "start_ai_conversation", topic = topic });
    }
    
    public void ToggleAIPause()
    {
        SendMessage(new { type = "toggle_ai_pause" });
    }
    
    public void OnTTSPlaybackStarted()
    {
        if (!isTTSPlaying)
        {
            isTTSPlaying = true;
            waitingForTTSComplete = true;
            SendMessage(new { type = "tts_start" });
            
            if (showDebugInfo)
            {
                Debug.Log("TTS playback started event from AudioManager");
            }
        }
    }
    
    public void OnTTSPlaybackStopped()
    {
        if (showDebugInfo)
        {
            Debug.Log($"TTS playback stopped event. Was playing: {isTTSPlaying}, Waiting for complete: {waitingForTTSComplete}");
        }
        
        if (isTTSPlaying)
        {
            isTTSPlaying = false;
            if (!waitingForTTSComplete)
            {
                SendMessage(new { type = "tts_stop" });
                if (showDebugInfo)
                {
                    Debug.Log("Sent tts_stop from playback stopped event");
                }
            }
        }
    }
    
    void OnDestroy()
    {
        Disconnect();
    }
}

[System.Serializable]
public class WebSocketMessage
{
    public string type;
    public string content;
    public string role;
    public bool is_paused;
}