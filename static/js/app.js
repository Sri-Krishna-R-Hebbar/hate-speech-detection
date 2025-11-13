// Hate Speech Detection Web App - JavaScript
// Includes speech-to-text functionality

let recognition = null;
let isRecording = false;

// Initialize Speech Recognition
function initSpeechRecognition() {
    // Check if browser supports speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        document.getElementById('speechBtn').disabled = true;
        document.getElementById('speechBtn').innerHTML = 
            '<span class="mic-icon">🎤</span><span>Speech Not Supported</span>';
        updateSpeechStatus('Speech recognition is not supported in your browser. Please use Chrome or Edge.', false);
        return;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    recognition.onstart = function() {
        isRecording = true;
        document.getElementById('speechBtn').classList.add('recording');
        document.getElementById('speechBtnText').textContent = 'Stop Recording';
        updateSpeechStatus('🎤 Listening... Speak now!', true);
    };
    
    recognition.onresult = function(event) {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
            } else {
                interimTranscript += transcript;
            }
        }
        
        const textInput = document.getElementById('textInput');
        const currentText = textInput.value;
        
        // Update textarea with final transcript
        if (finalTranscript) {
            textInput.value = currentText + finalTranscript;
            updateSpeechStatus('✅ Text added!', true);
        } else if (interimTranscript) {
            // Show interim results (optional - can be removed if distracting)
            // textInput.value = currentText + interimTranscript;
        }
    };
    
    recognition.onerror = function(event) {
        console.error('Speech recognition error:', event.error);
        let errorMsg = 'Speech recognition error occurred.';
        
        if (event.error === 'no-speech') {
            errorMsg = 'No speech detected. Please try again.';
        } else if (event.error === 'audio-capture') {
            errorMsg = 'Microphone not found. Please check your microphone.';
        } else if (event.error === 'not-allowed') {
            errorMsg = 'Microphone permission denied. Please allow microphone access.';
        } else if (event.error === 'network') {
            errorMsg = 'Network error. Please check your connection.';
        }
        
        updateSpeechStatus(`⚠️ ${errorMsg}`, false);
        stopRecording();
    };
    
    recognition.onend = function() {
        if (isRecording) {
            // Automatically restart if still recording (for continuous mode)
            try {
                recognition.start();
            } catch (e) {
                // Recognition already started or error
                stopRecording();
            }
        }
    };
}

// Start/Stop recording
function toggleRecording() {
    if (!recognition) {
        alert('Speech recognition is not available in your browser.');
        return;
    }
    
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

function startRecording() {
    try {
        recognition.start();
    } catch (e) {
        console.error('Error starting recognition:', e);
        updateSpeechStatus('⚠️ Error starting speech recognition. Please try again.', false);
    }
}

function stopRecording() {
    isRecording = false;
    if (recognition) {
        recognition.stop();
    }
    document.getElementById('speechBtn').classList.remove('recording');
    document.getElementById('speechBtnText').textContent = 'Start Recording';
    updateSpeechStatus('⏹️ Recording stopped.', false);
}

function updateSpeechStatus(message, isActive) {
    const statusDiv = document.getElementById('speechStatus');
    statusDiv.textContent = message;
    if (isActive) {
        statusDiv.classList.add('active');
    } else {
        statusDiv.classList.remove('active');
    }
}

// Analyze text function
async function analyzeText() {
    const textInput = document.getElementById('textInput');
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze.');
        return;
    }
    
    // Hide previous results and errors
    hideAllSections();
    showLoading();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        hideLoading();
        
        if (data.error) {
            showError(data.error);
            return;
        }
        
        displayResults(data);
        
    } catch (error) {
        hideLoading();
        showError('Network error: ' + error.message);
        console.error('Error:', error);
    }
}

function displayResults(data) {
    const resultSection = document.getElementById('resultSection');
    const predictionLabel = document.getElementById('predictionLabel');
    const predictionConfidence = document.getElementById('predictionConfidence');
    const probabilityBars = document.getElementById('probabilityBars');
    
    // Set prediction label
    predictionLabel.textContent = data.class;
    predictionLabel.className = 'prediction-label ' + data.class.toLowerCase();
    
    // Set confidence
    predictionConfidence.textContent = `Confidence: ${(data.confidence * 100).toFixed(2)}%`;
    
    // Display probability bars
    probabilityBars.innerHTML = '';
    
    const classes = ['None', 'Racist', 'Sexist'];
    classes.forEach(className => {
        const probability = data.probabilities[className];
        const percentage = (probability * 100).toFixed(2);
        
        const item = document.createElement('div');
        item.className = 'probability-item';
        
        const label = document.createElement('div');
        label.className = 'probability-label';
        label.textContent = className + ':';
        
        const barContainer = document.createElement('div');
        barContainer.className = 'probability-bar-container';
        
        const bar = document.createElement('div');
        bar.className = 'probability-bar ' + className.toLowerCase();
        bar.style.width = percentage + '%';
        bar.textContent = percentage + '%';
        
        barContainer.appendChild(bar);
        
        const value = document.createElement('div');
        value.className = 'probability-value';
        value.textContent = percentage + '%';
        
        item.appendChild(label);
        item.appendChild(barContainer);
        item.appendChild(value);
        
        probabilityBars.appendChild(item);
    });
    
    resultSection.classList.remove('hidden');
    
    // Scroll to results
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

function showError(message) {
    const errorSection = document.getElementById('errorSection');
    const errorText = document.getElementById('errorText');
    errorText.textContent = message;
    errorSection.classList.remove('hidden');
}

function showLoading() {
    document.getElementById('loadingSection').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loadingSection').classList.add('hidden');
}

function hideAllSections() {
    document.getElementById('resultSection').classList.add('hidden');
    document.getElementById('errorSection').classList.add('hidden');
    document.getElementById('loadingSection').classList.add('hidden');
}

function clearText() {
    document.getElementById('textInput').value = '';
    hideAllSections();
    if (isRecording) {
        stopRecording();
    }
}

// Event Listeners
document.addEventListener('DOMContentLoaded', function() {
    // Initialize speech recognition
    initSpeechRecognition();
    
    // Button event listeners
    document.getElementById('speechBtn').addEventListener('click', toggleRecording);
    document.getElementById('analyzeBtn').addEventListener('click', analyzeText);
    document.getElementById('clearBtn').addEventListener('click', clearText);
    
    // Allow Enter key to analyze (Ctrl+Enter or Cmd+Enter)
    document.getElementById('textInput').addEventListener('keydown', function(e) {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            analyzeText();
        }
    });
    
    // Stop recording when page is unloaded
    window.addEventListener('beforeunload', function() {
        if (isRecording) {
            stopRecording();
        }
    });
});

