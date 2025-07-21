using UnityEngine;

public class VoiceChatManager : MonoBehaviour
{
    [Header("Core Components")]
    public WebSocketClient webSocketClient;
    public AudioManager audioManager;
    public VoiceChatUI voiceChatUI;
    
    [Header("Auto-Start Settings")]
    public bool autoConnect = false;
    public string serverUrl = "ws://localhost:8000/ws";
    
    void Start()
    {
        InitializeComponents();
        
        if (autoConnect)
        {
            StartCoroutine(AutoConnectCoroutine());
        }
    }
    
    void InitializeComponents()
    {
        // Ensure all components are properly referenced
        if (webSocketClient == null)
            webSocketClient = GetComponent<WebSocketClient>();
        
        if (audioManager == null)
            audioManager = GetComponent<AudioManager>();
        
        if (voiceChatUI == null)
            voiceChatUI = GetComponent<VoiceChatUI>();
        
        // Set up component references
        if (webSocketClient != null && audioManager != null)
        {
            audioManager.webSocketClient = webSocketClient;
        }
        
        if (voiceChatUI != null)
        {
            voiceChatUI.webSocketClient = webSocketClient;
            voiceChatUI.audioManager = audioManager;
        }
        
        // Configure WebSocket URL
        if (webSocketClient != null && !string.IsNullOrEmpty(serverUrl))
        {
            webSocketClient.serverUrl = serverUrl;
        }
        
        Debug.Log("VoiceChatManager initialized successfully");
    }
    
    System.Collections.IEnumerator AutoConnectCoroutine()
    {
        yield return new WaitForSeconds(1f); // Wait for everything to initialize
        
        if (webSocketClient != null)
        {
            Debug.Log("Auto-connecting to voice chat server...");
            webSocketClient.Connect();
        }
    }
    
    void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus)
        {
            // App is being paused - disconnect gracefully
            webSocketClient?.Disconnect();
        }
        else
        {
            // App is resuming - reconnect if auto-connect is enabled
            if (autoConnect && webSocketClient != null)
            {
                StartCoroutine(AutoConnectCoroutine());
            }
        }
    }
    
    void OnApplicationFocus(bool hasFocus)
    {
        if (!hasFocus)
        {
            // Lost focus - pause audio processing
            audioManager?.StopMicrophoneRecording();
        }
        else
        {
            // Gained focus - resume if connected
            if (webSocketClient != null && webSocketClient.isConnected)
            {
                audioManager?.StartMicrophoneRecording();
            }
        }
    }
    
    void OnDestroy()
    {
        // Clean shutdown
        webSocketClient?.Disconnect();
    }
}