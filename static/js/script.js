// Global variables
let recognition = null;
let isRecording = false;
let finalTranscript = '';

// Switch between tabs
function switchTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(`${tabName}-tab`).classList.add('active');
    event.target.closest('.tab-btn').classList.add('active');
    
    // Clear previous results when switching tabs
    clearResults();
}

// Analyze text input
async function analyzeText() {
    const textInput = document.getElementById('text-input');
    const text = textInput.value.trim();
    
    if (!text) {
        showError('Please enter some text to analyze');
        return;
    }
    
    // Show loading
    showLoading();
    hideError();
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ text: text })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            showError(data.error || 'An error occurred during analysis');
        }
    } catch (error) {
        showError('Failed to connect to the server. Please try again.');
        console.error('Error:', error);
    } finally {
        hideLoading();
    }
}

// Initialize Speech Recognition
function initSpeechRecognition() {
    // Check if browser supports speech recognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        const recordBtn = document.getElementById('record-btn');
        recordBtn.disabled = true;
        recordBtn.innerHTML = '<i class="fas fa-microphone-slash"></i> <span>Speech Not Supported</span>';
        showError('Speech recognition is not supported in your browser. Please use Chrome, Edge, or Safari.');
        return false;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = 'en-US';
    
    recognition.onstart = function() {
        isRecording = true;
        const recordBtn = document.getElementById('record-btn');
        const recordText = document.getElementById('record-text');
        recordBtn.classList.add('recording');
        recordText.textContent = 'Click to Stop Recording';
        document.getElementById('recording-indicator').classList.remove('hidden');
    };
    
    recognition.onresult = function(event) {
        let interimTranscript = '';
        finalTranscript = ''; // Reset and rebuild
        
        for (let i = 0; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript + ' ';
            } else {
                interimTranscript += transcript;
            }
        }
        
        // Show transcribed text (both final and interim)
        const transcribedContainer = document.getElementById('transcribed-text-container');
        const transcribedText = document.getElementById('transcribed-text');
        
        const displayText = (finalTranscript + interimTranscript).trim();
        if (displayText) {
            transcribedText.textContent = displayText;
            transcribedContainer.classList.remove('hidden');
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
        
        showError(errorMsg);
        stopRecording();
    };
    
    recognition.onend = function() {
        if (isRecording) {
            // Automatically restart if still recording (for continuous mode)
            try {
                recognition.start();
            } catch (e) {
                // Recognition already started or error - stop recording
                isRecording = false;
                const recordBtn = document.getElementById('record-btn');
                const recordText = document.getElementById('record-text');
                recordBtn.classList.remove('recording');
                recordText.textContent = 'Click to Start Recording';
                document.getElementById('recording-indicator').classList.add('hidden');
            }
        }
        // Note: processFinalTranscript will be called from stopRecording
    };
    
    return true;
}

// Toggle recording
function toggleRecording() {
    if (!recognition) {
        if (!initSpeechRecognition()) {
            return;
        }
    }
    
    if (isRecording) {
        stopRecording();
    } else {
        startRecording();
    }
}

// Start recording
function startRecording() {
    try {
        recognition.start();
    } catch (e) {
        console.error('Error starting recognition:', e);
        showError('Error starting speech recognition. Please try again.');
    }
}

// Process final transcript after recording stops
function processFinalTranscript() {
    const transcribedText = document.getElementById('transcribed-text').textContent.trim();
    if (transcribedText) {
        // Put text in the text input and analyze
        document.getElementById('text-input').value = transcribedText;
        analyzeText();
    } else {
        showError('No speech was detected. Please try speaking again.');
    }
}

// Stop recording
function stopRecording() {
    if (!isRecording) return; // Already stopped
    
    isRecording = false;
    if (recognition) {
        recognition.stop();
    }
    
    // Update UI immediately
    const recordBtn = document.getElementById('record-btn');
    const recordText = document.getElementById('record-text');
    recordBtn.classList.remove('recording');
    recordText.textContent = 'Click to Start Recording';
    document.getElementById('recording-indicator').classList.add('hidden');
    
    // Wait a moment for final transcript to be processed, then analyze
    setTimeout(processFinalTranscript, 800);
}

// Display results
function displayResults(data) {
    // Hide no results message
    document.getElementById('no-results').classList.add('hidden');
    
    // Show results container
    const resultsContainer = document.getElementById('results-container');
    resultsContainer.classList.remove('hidden');
    
    // Display input text
    const inputTextDisplay = document.getElementById('input-text-display');
    inputTextDisplay.textContent = data.original_text;
    
    // Display prediction
    const predictionLabel = document.getElementById('prediction-label');
    const confidenceBadge = document.getElementById('confidence-badge');
    
    predictionLabel.textContent = data.predicted_class;
    confidenceBadge.textContent = `${(data.confidence * 100).toFixed(2)}% Confidence`;
    
    // Update prediction badge color based on class
    const predictionResult = document.querySelector('.prediction-result');
    if (data.predicted_class.includes('Safe') || data.predicted_class.includes('None')) {
        predictionResult.style.background = 'linear-gradient(135deg, #10b981 0%, #34d399 100%)';
    } else if (data.predicted_class.includes('Racist')) {
        predictionResult.style.background = 'linear-gradient(135deg, #ef4444 0%, #f87171 100%)';
    } else if (data.predicted_class.includes('Sexist')) {
        predictionResult.style.background = 'linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%)';
    }
    
    // Display probabilities
    const probabilities = data.probabilities;
    
    // None (Safe)
    const probNone = probabilities['None (Safe)'] || 0;
    document.getElementById('prob-none').textContent = `${(probNone * 100).toFixed(2)}%`;
    document.getElementById('bar-none').style.width = `${probNone * 100}%`;
    
    // Racist
    const probRacist = probabilities['Racist'] || 0;
    document.getElementById('prob-racist').textContent = `${(probRacist * 100).toFixed(2)}%`;
    document.getElementById('bar-racist').style.width = `${probRacist * 100}%`;
    
    // Sexist
    const probSexist = probabilities['Sexist'] || 0;
    document.getElementById('prob-sexist').textContent = `${(probSexist * 100).toFixed(2)}%`;
    document.getElementById('bar-sexist').style.width = `${probSexist * 100}%`;
    
    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Clear results
function clearResults() {
    // Clear text input
    document.getElementById('text-input').value = '';
    
    // Hide results
    document.getElementById('results-container').classList.add('hidden');
    document.getElementById('no-results').classList.remove('hidden');
    
    // Hide transcribed text
    document.getElementById('transcribed-text-container').classList.add('hidden');
    
    // Hide error
    hideError();
}

// Show loading
function showLoading() {
    document.getElementById('loading').classList.remove('hidden');
}

// Hide loading
function hideLoading() {
    document.getElementById('loading').classList.add('hidden');
}

// Show error
function showError(message) {
    const errorMessage = document.getElementById('error-message');
    const errorText = document.getElementById('error-text');
    errorText.textContent = message;
    errorMessage.classList.remove('hidden');
}

// Hide error
function hideError() {
    document.getElementById('error-message').classList.add('hidden');
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Initialize speech recognition
    initSpeechRecognition();
    
    // Allow Enter key to submit (Ctrl+Enter for new line)
    const textInput = document.getElementById('text-input');
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
            e.preventDefault();
            analyzeText();
        }
    });
    
    // Stop recording when page is unloaded
    window.addEventListener('beforeunload', function() {
        if (isRecording && recognition) {
            recognition.stop();
        }
    });
});