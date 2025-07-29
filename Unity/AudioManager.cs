using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(AudioSource))]
public class AudioManager : MonoBehaviour
{
    [Header("Audio Settings")]
    public int sampleRate = 48000; // Use system sample rate for better compatibility
    public int microphoneBufferLength = 1; // seconds
    public bool enableMicrophone = true;
    
    [Header("Audio Processing")]
    public float micGain = 1.0f;
    public float ttsVolume = 1.0f;
    public bool enableEchoCancellation = true;
    
    [Header("References")]
    public WebSocketClient webSocketClient;
    
    private AudioSource audioSource;
    private AudioClip microphoneClip;
    private string selectedMicrophone;
    private bool isRecording = false;
    private int lastMicPosition = 0;
    
    // TTS playback
    private Queue<float[]> ttsAudioQueue = new Queue<float[]>();
    private AudioClip ttsClip;
    private int ttsWritePosition = 0;
    private bool isTTSPlaying = false;
    
    // Audio buffer management
    private const int TTS_BUFFER_SIZE = 48000 * 20; // 20 seconds buffer for smoother playback
    private float[] ttsBuffer;
    private int ttsBufferPosition = 0;
    private bool hasReceivedAnyData = false;
    private float silenceTimer = 0f;
    private const float MAX_SILENCE_TIME = 2.0f; // Max silence before stopping
    
    void Start()
    {
        audioSource = GetComponent<AudioSource>();
        InitializeAudio();
        
        if (webSocketClient != null)
        {
            webSocketClient.OnTTSAudioReceived.AddListener(OnTTSAudioReceived);
        }
    }
    
    void InitializeAudio()
    {
        // Use system's preferred sample rate for better compatibility
        sampleRate = AudioSettings.outputSampleRate;
        Debug.Log($"Using system sample rate: {sampleRate}Hz");
        
        // Initialize TTS buffer
        ttsBuffer = new float[TTS_BUFFER_SIZE];
        
        // Create TTS playback clip with system sample rate
        ttsClip = AudioClip.Create("TTSPlayback", TTS_BUFFER_SIZE, 1, sampleRate, true, OnTTSAudioRead);
        audioSource.clip = ttsClip;
        audioSource.volume = ttsVolume;
        audioSource.loop = true;
        
        // Initialize microphone
        if (enableMicrophone)
        {
            InitializeMicrophone();
        }
    }
    
    void InitializeMicrophone()
    {
        if (Microphone.devices.Length > 0)
        {
            selectedMicrophone = Microphone.devices[0];
            Debug.Log($"Selected microphone: {selectedMicrophone}");
            StartMicrophoneRecording();
        }
        else
        {
            Debug.LogWarning("No microphone devices found");
        }
    }
    
    public void StartMicrophoneRecording()
    {
        if (string.IsNullOrEmpty(selectedMicrophone)) return;
        
        try
        {
            microphoneClip = Microphone.Start(selectedMicrophone, true, microphoneBufferLength, sampleRate);
            isRecording = true;
            lastMicPosition = 0;
            Debug.Log($"Microphone recording started at {sampleRate}Hz");
        }
        catch (Exception e)
        {
            Debug.LogError($"Failed to start microphone: {e.Message}");
        }
    }
    
    public void StopMicrophoneRecording()
    {
        if (isRecording)
        {
            Microphone.End(selectedMicrophone);
            isRecording = false;
            Debug.Log("Microphone recording stopped");
        }
    }
    
    public void ToggleMicrophone()
    {
        if (isRecording)
        {
            StopMicrophoneRecording();
        }
        else
        {
            StartMicrophoneRecording();
        }
    }
    
    void Update()
    {
        if (isRecording && microphoneClip != null)
        {
            ProcessMicrophoneInput();
        }
        
        ProcessTTSQueue();
    }
    
    void ProcessMicrophoneInput()
    {
        int currentPosition = Microphone.GetPosition(selectedMicrophone);
        if (currentPosition < 0) return;
        
        // Handle wrap-around
        if (currentPosition < lastMicPosition)
        {
            // Process from lastMicPosition to end
            int samplesToEnd = microphoneClip.samples - lastMicPosition;
            if (samplesToEnd > 0)
            {
                float[] endSamples = new float[samplesToEnd];
                microphoneClip.GetData(endSamples, lastMicPosition);
                ProcessAndSendAudioData(endSamples);
            }
            
            // Process from start to current position
            if (currentPosition > 0)
            {
                float[] startSamples = new float[currentPosition];
                microphoneClip.GetData(startSamples, 0);
                ProcessAndSendAudioData(startSamples);
            }
        }
        else if (currentPosition > lastMicPosition)
        {
            // Normal case - process new samples
            int newSamples = currentPosition - lastMicPosition;
            float[] audioData = new float[newSamples];
            microphoneClip.GetData(audioData, lastMicPosition);
            ProcessAndSendAudioData(audioData);
        }
        
        lastMicPosition = currentPosition;
    }
    
    void ProcessAndSendAudioData(float[] samples)
    {
        if (samples.Length == 0) return;
        
        // Apply gain
        if (micGain != 1.0f)
        {
            for (int i = 0; i < samples.Length; i++)
            {
                samples[i] *= micGain;
                samples[i] = Mathf.Clamp(samples[i], -1.0f, 1.0f);
            }
        }
        
        // Send to WebSocket client
        webSocketClient?.SendAudioData(samples);
    }
    
    void OnTTSAudioReceived(float[] audioSamples)
    {
        lock (ttsAudioQueue)
        {
            ttsAudioQueue.Enqueue(audioSamples);
            hasReceivedAnyData = true;
            silenceTimer = 0f; // Reset silence timer when new data arrives
        }
    }
    
    void ProcessTTSQueue()
    {
        lock (ttsAudioQueue)
        {
            while (ttsAudioQueue.Count > 0)
            {
                float[] samples = ttsAudioQueue.Dequeue();
                AddTTSAudioToBuffer(samples);
            }
        }
    }
    
    void AddTTSAudioToBuffer(float[] samples)
    {
        if (!isTTSPlaying)
        {
            StartTTSPlayback();
        }
        
        // Add samples to circular buffer with overflow protection
        for (int i = 0; i < samples.Length; i++)
        {
            // Check for buffer overflow
            int nextPos = (ttsBufferPosition + 1) % ttsBuffer.Length;
            if (nextPos == ttsWritePosition)
            {
                // Buffer is full - skip this sample to prevent overrun
                Debug.LogWarning("TTS buffer overflow - dropping samples");
                break;
            }
            
            ttsBuffer[ttsBufferPosition] = samples[i] * ttsVolume;
            ttsBufferPosition = nextPos;
        }
    }
    
    void StartTTSPlayback()
    {
        if (!isTTSPlaying)
        {
            isTTSPlaying = true;
            audioSource.Play();
            webSocketClient?.OnTTSPlaybackStarted();
            Debug.Log("TTS playback started");
        }
    }
    
    public void StopTTSPlayback()
    {
        if (isTTSPlaying)
        {
            isTTSPlaying = false;
            audioSource.Stop();
            
            // Clear TTS buffer
            Array.Clear(ttsBuffer, 0, ttsBuffer.Length);
            ttsBufferPosition = 0;
            ttsWritePosition = 0;
            hasReceivedAnyData = false;
            silenceTimer = 0f;
            
            // Clear queue
            lock (ttsAudioQueue)
            {
                ttsAudioQueue.Clear();
            }
            
            webSocketClient?.OnTTSPlaybackStopped();
            Debug.Log("TTS playback stopped");
        }
    }
    
    private System.Collections.IEnumerator DelayedStopTTS()
    {
        yield return new WaitForSeconds(0.1f); // Small delay to avoid race conditions
        if (ttsAudioQueue.Count == 0 && CalculateAvailableData() == 0)
        {
            StopTTSPlayback();
        }
    }
    
    void OnTTSAudioRead(float[] data)
    {
        if (!isTTSPlaying)
        {
            // Fill with silence if not playing
            Array.Clear(data, 0, data.Length);
            return;
        }
        
        int availableData = CalculateAvailableData();
        
        // More lenient buffer underrun handling - only require minimal data
        if (availableData >= data.Length)
        {
            // Enough data available - normal playback
            for (int i = 0; i < data.Length; i++)
            {
                data[i] = ttsBuffer[ttsWritePosition];
                ttsBuffer[ttsWritePosition] = 0; // Clear read data
                ttsWritePosition = (ttsWritePosition + 1) % ttsBuffer.Length;
            }
            silenceTimer = 0f;
        }
        else if (availableData > 0)
        {
            // Some data available - play what we have and pad with silence
            int i = 0;
            for (; i < availableData && i < data.Length; i++)
            {
                data[i] = ttsBuffer[ttsWritePosition];
                ttsBuffer[ttsWritePosition] = 0;
                ttsWritePosition = (ttsWritePosition + 1) % ttsBuffer.Length;
            }
            // Fill remaining with silence
            for (; i < data.Length; i++)
            {
                data[i] = 0f;
            }
            silenceTimer += (float)data.Length / sampleRate;
        }
        else
        {
            // No data available - output silence and track time
            Array.Clear(data, 0, data.Length);
            silenceTimer += (float)data.Length / sampleRate;
            
            // Stop only after extended silence and no more data expected
            if (silenceTimer > MAX_SILENCE_TIME && ttsAudioQueue.Count == 0 && hasReceivedAnyData)
            {
                // Delay stop to avoid race conditions
                StartCoroutine(DelayedStopTTS());
            }
        }
    }
    
    int CalculateAvailableData()
    {
        if (ttsBufferPosition >= ttsWritePosition)
        {
            return ttsBufferPosition - ttsWritePosition;
        }
        else
        {
            return (ttsBuffer.Length - ttsWritePosition) + ttsBufferPosition;
        }
    }
    
    public void SetMicrophoneGain(float gain)
    {
        micGain = Mathf.Clamp(gain, 0.0f, 5.0f);
    }
    
    public void SetTTSVolume(float volume)
    {
        ttsVolume = Mathf.Clamp01(volume);
        audioSource.volume = ttsVolume;
    }
    
    public bool IsMicrophoneActive()
    {
        return isRecording && Microphone.IsRecording(selectedMicrophone);
    }
    
    public bool IsTTSPlaying()
    {
        return isTTSPlaying;
    }
    
    public string[] GetAvailableMicrophones()
    {
        return Microphone.devices;
    }
    
    public void SelectMicrophone(string deviceName)
    {
        if (Array.Exists(Microphone.devices, device => device == deviceName))
        {
            bool wasRecording = isRecording;
            if (wasRecording)
            {
                StopMicrophoneRecording();
            }
            
            selectedMicrophone = deviceName;
            
            if (wasRecording)
            {
                StartMicrophoneRecording();
            }
        }
    }
    
    void OnDestroy()
    {
        StopMicrophoneRecording();
        StopTTSPlayback();
    }
    
    void OnApplicationPause(bool pauseStatus)
    {
        if (pauseStatus)
        {
            StopMicrophoneRecording();
            StopTTSPlayback();
        }
        else if (enableMicrophone)
        {
            StartMicrophoneRecording();
        }
    }
}