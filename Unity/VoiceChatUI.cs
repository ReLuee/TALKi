using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using TMPro;

public class VoiceChatUI : MonoBehaviour
{
    [Header("UI References")]
    public Button connectButton;
    public Button disconnectButton;
    public Button micToggleButton;
    public Button clearHistoryButton;
    public Button aiTalkButton;
    public Button aiPauseButton;
    public Button copyButton;
    
    [Header("Status UI")]
    public TextMeshProUGUI statusText;
    public Image connectionStatusIndicator;
    public Image micStatusIndicator;
    public Image ttsStatusIndicator;
    
    [Header("Chat UI")]
    public ScrollRect chatScrollRect;
    public Transform chatContent;
    public GameObject userMessagePrefab;
    public GameObject assistantMessagePrefab;
    
    [Header("Settings UI")]
    public Slider speedSlider;
    public Slider micGainSlider;
    public Slider ttsVolumeSlider;
    public TMP_Dropdown microphoneDropdown;
    
    [Header("Audio Visualization")]
    public Image audioLevelBar;
    public Color[] audioLevelColors = { Color.green, Color.yellow, Color.red };
    
    [Header("References")]
    public WebSocketClient webSocketClient;
    public AudioManager audioManager;
    
    // Chat history
    private List<ChatMessage> chatHistory = new List<ChatMessage>();
    private string currentUserTyping = "";
    private Dictionary<string, string> currentAssistantTyping = new Dictionary<string, string>();
    
    // Status colors
    private Color connectedColor = Color.green;
    private Color disconnectedColor = Color.red;
    private Color micActiveColor = Color.green;
    private Color micInactiveColor = Color.gray;
    private Color ttsActiveColor = Color.blue;
    private Color ttsInactiveColor = Color.gray;
    
    void Start()
    {
        InitializeUI();
        SetupEventListeners();
        UpdateUIState();
    }
    
    void InitializeUI()
    {
        // Initialize status
        statusText.text = "Ready to connect";
        connectionStatusIndicator.color = disconnectedColor;
        micStatusIndicator.color = micInactiveColor;
        ttsStatusIndicator.color = ttsInactiveColor;
        
        // Initialize sliders
        speedSlider.value = 5; // Default speed
        micGainSlider.value = audioManager ? audioManager.micGain : 1.0f;
        ttsVolumeSlider.value = audioManager ? audioManager.ttsVolume : 1.0f;
        
        // Initialize microphone dropdown
        PopulateMicrophoneDropdown();
        
        // Initialize button states
        disconnectButton.interactable = false;
        aiTalkButton.interactable = false;
        aiPauseButton.interactable = false;
        clearHistoryButton.interactable = false;
        copyButton.interactable = false;
    }
    
    void SetupEventListeners()
    {
        // Button events
        connectButton.onClick.AddListener(OnConnectClicked);
        disconnectButton.onClick.AddListener(OnDisconnectClicked);
        micToggleButton.onClick.AddListener(OnMicToggleClicked);
        clearHistoryButton.onClick.AddListener(OnClearHistoryClicked);
        aiTalkButton.onClick.AddListener(OnAITalkClicked);
        aiPauseButton.onClick.AddListener(OnAIPauseClicked);
        copyButton.onClick.AddListener(OnCopyClicked);
        
        // Slider events
        speedSlider.onValueChanged.AddListener(OnSpeedChanged);
        micGainSlider.onValueChanged.AddListener(OnMicGainChanged);
        ttsVolumeSlider.onValueChanged.AddListener(OnTTSVolumeChanged);
        microphoneDropdown.onValueChanged.AddListener(OnMicrophoneChanged);
        
        // WebSocket events
        if (webSocketClient != null)
        {
            webSocketClient.OnConnected.AddListener(OnWebSocketConnected);
            webSocketClient.OnDisconnected.AddListener(OnWebSocketDisconnected);
            webSocketClient.OnUserTranscription.AddListener(OnUserTranscription);
            webSocketClient.OnAssistantResponse.AddListener(OnAssistantResponse);
        }
    }
    
    void Update()
    {
        UpdateAudioVisualization();
        UpdateTypingIndicators();
    }
    
    void UpdateUIState()
    {
        bool isConnected = webSocketClient != null && webSocketClient.isConnected;
        bool isMicActive = audioManager != null && audioManager.IsMicrophoneActive();
        bool isTTSActive = audioManager != null && audioManager.IsTTSPlaying();
        
        // Update button states
        connectButton.interactable = !isConnected;
        disconnectButton.interactable = isConnected;
        micToggleButton.interactable = isConnected;
        clearHistoryButton.interactable = isConnected;
        aiTalkButton.interactable = isConnected;
        aiPauseButton.interactable = isConnected;
        copyButton.interactable = chatHistory.Count > 0;
        
        // Update status indicators
        connectionStatusIndicator.color = isConnected ? connectedColor : disconnectedColor;
        micStatusIndicator.color = isMicActive ? micActiveColor : micInactiveColor;
        ttsStatusIndicator.color = isTTSActive ? ttsActiveColor : ttsInactiveColor;
        
        // Update mic button text
        Text micButtonText = micToggleButton.GetComponentInChildren<Text>();
        if (micButtonText != null)
        {
            micButtonText.text = isMicActive ? "🎤" : "🔇";
        }
    }
    
    void UpdateAudioVisualization()
    {
        // This would typically use actual audio level data
        // For now, just show activity when microphone is active
        if (audioManager != null && audioManager.IsMicrophoneActive())
        {
            float level = Random.Range(0.1f, 0.8f); // Placeholder
            audioLevelBar.fillAmount = level;
            
            if (level < 0.3f)
                audioLevelBar.color = audioLevelColors[0];
            else if (level < 0.7f)
                audioLevelBar.color = audioLevelColors[1];
            else
                audioLevelBar.color = audioLevelColors[2];
        }
        else
        {
            audioLevelBar.fillAmount = 0;
            audioLevelBar.color = audioLevelColors[0];
        }
    }
    
    void UpdateTypingIndicators()
    {
        // Remove any existing typing indicators
        for (int i = chatContent.childCount - 1; i >= 0; i--)
        {
            GameObject child = chatContent.GetChild(i).gameObject;
            if (child.name.Contains("Typing"))
            {
                DestroyImmediate(child);
            }
        }
        
        // Add current typing indicators
        if (!string.IsNullOrEmpty(currentUserTyping))
        {
            CreateTypingMessage("You", currentUserTyping, true);
        }
        
        foreach (var typing in currentAssistantTyping)
        {
            if (!string.IsNullOrEmpty(typing.Value))
            {
                string displayName = GetDisplayName(typing.Key);
                CreateTypingMessage(displayName, typing.Value, false);
            }
        }
        
        // Auto-scroll to bottom
        Canvas.ForceUpdateCanvases();
        chatScrollRect.verticalNormalizedPosition = 0f;
    }
    
    void CreateTypingMessage(string speaker, string content, bool isUser)
    {
        GameObject prefab = isUser ? userMessagePrefab : assistantMessagePrefab;
        GameObject messageObj = Instantiate(prefab, chatContent);
        messageObj.name = $"Typing_{speaker}";
        
        // Set speaker name
        TextMeshProUGUI speakerText = messageObj.transform.Find("SpeakerName")?.GetComponent<TextMeshProUGUI>();
        if (speakerText != null)
        {
            speakerText.text = speaker;
        }
        
        // Set content with typing indicator
        TextMeshProUGUI contentText = messageObj.transform.Find("MessageText")?.GetComponent<TextMeshProUGUI>();
        if (contentText != null)
        {
            contentText.text = content + " ✏️";
            contentText.fontStyle = FontStyles.Italic;
        }
    }
    
    void PopulateMicrophoneDropdown()
    {
        microphoneDropdown.ClearOptions();
        
        if (audioManager != null)
        {
            string[] microphones = audioManager.GetAvailableMicrophones();
            List<string> options = new List<string>(microphones);
            microphoneDropdown.AddOptions(options);
        }
    }
    
    string GetDisplayName(string role)
    {
        switch (role)
        {
            case "AI1_Mia": return "Mia";
            case "AI1_Leo": return "Leo";
            case "assistant": return "Assistant";
            default: return role;
        }
    }
    
    // Event Handlers
    void OnConnectClicked()
    {
        statusText.text = "Connecting...";
        webSocketClient?.Connect();
    }
    
    void OnDisconnectClicked()
    {
        statusText.text = "Disconnecting...";
        webSocketClient?.Disconnect();
    }
    
    void OnMicToggleClicked()
    {
        audioManager?.ToggleMicrophone();
        UpdateUIState();
    }
    
    void OnClearHistoryClicked()
    {
        chatHistory.Clear();
        currentUserTyping = "";
        currentAssistantTyping.Clear();
        
        // Clear UI
        for (int i = chatContent.childCount - 1; i >= 0; i--)
        {
            DestroyImmediate(chatContent.GetChild(i).gameObject);
        }
        
        webSocketClient?.ClearHistory();
        UpdateUIState();
    }
    
    void OnAITalkClicked()
    {
        webSocketClient?.StartAIConversation("Let's have an interesting conversation about technology.");
    }
    
    void OnAIPauseClicked()
    {
        webSocketClient?.ToggleAIPause();
    }
    
    void OnCopyClicked()
    {
        string conversation = "";
        foreach (var message in chatHistory)
        {
            conversation += $"{GetDisplayName(message.role)}: {message.content}\n";
        }
        
        GUIUtility.systemCopyBuffer = conversation;
        statusText.text = "Conversation copied to clipboard";
    }
    
    void OnSpeedChanged(float value)
    {
        webSocketClient?.SetSpeed((int)value);
    }
    
    void OnMicGainChanged(float value)
    {
        audioManager?.SetMicrophoneGain(value);
    }
    
    void OnTTSVolumeChanged(float value)
    {
        audioManager?.SetTTSVolume(value);
    }
    
    void OnMicrophoneChanged(int index)
    {
        if (audioManager != null && index >= 0)
        {
            string[] microphones = audioManager.GetAvailableMicrophones();
            if (index < microphones.Length)
            {
                audioManager.SelectMicrophone(microphones[index]);
            }
        }
    }
    
    // WebSocket Event Handlers
    void OnWebSocketConnected()
    {
        statusText.text = "Connected - Ready to chat";
        UpdateUIState();
    }
    
    void OnWebSocketDisconnected()
    {
        statusText.text = "Disconnected";
        UpdateUIState();
    }
    
    void OnUserTranscription(string content)
    {
        if (content.Contains("final"))
        {
            // Final transcription
            if (!string.IsNullOrEmpty(currentUserTyping))
            {
                AddMessageToHistory("user", currentUserTyping);
                currentUserTyping = "";
            }
        }
        else
        {
            // Partial transcription
            currentUserTyping = content;
        }
    }
    
    void OnAssistantResponse(string role, string content)
    {
        if (content.Contains("final"))
        {
            // Final response
            if (currentAssistantTyping.ContainsKey(role) && !string.IsNullOrEmpty(currentAssistantTyping[role]))
            {
                AddMessageToHistory(role, currentAssistantTyping[role]);
                currentAssistantTyping[role] = "";
            }
        }
        else
        {
            // Partial response
            currentAssistantTyping[role] = content;
        }
    }
    
    void AddMessageToHistory(string role, string content)
    {
        ChatMessage message = new ChatMessage
        {
            role = role,
            content = content,
            timestamp = System.DateTime.Now
        };
        
        chatHistory.Add(message);
        CreateChatMessage(message);
        UpdateUIState();
    }
    
    void CreateChatMessage(ChatMessage message)
    {
        bool isUser = message.role == "user";
        GameObject prefab = isUser ? userMessagePrefab : assistantMessagePrefab;
        GameObject messageObj = Instantiate(prefab, chatContent);
        
        // Set speaker name
        TextMeshProUGUI speakerText = messageObj.transform.Find("SpeakerName")?.GetComponent<TextMeshProUGUI>();
        if (speakerText != null)
        {
            speakerText.text = isUser ? "You" : GetDisplayName(message.role);
        }
        
        // Set content
        TextMeshProUGUI contentText = messageObj.transform.Find("MessageText")?.GetComponent<TextMeshProUGUI>();
        if (contentText != null)
        {
            contentText.text = message.content;
        }
        
        // Set timestamp
        TextMeshProUGUI timestampText = messageObj.transform.Find("Timestamp")?.GetComponent<TextMeshProUGUI>();
        if (timestampText != null)
        {
            timestampText.text = message.timestamp.ToString("HH:mm");
        }
        
        // Auto-scroll to bottom
        Canvas.ForceUpdateCanvases();
        chatScrollRect.verticalNormalizedPosition = 0f;
    }
}

[System.Serializable]
public class ChatMessage
{
    public string role;
    public string content;
    public System.DateTime timestamp;
}