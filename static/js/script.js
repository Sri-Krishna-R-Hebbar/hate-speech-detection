// Global variables
let mediaRecorder;
let audioChunks = [];
let isRecording = false;

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

// Toggle recording
async function toggleRecording() {
    if (!isRecording) {
        await startRecording();
    } else {
        await stopRecording();
    }
}

// Start recording
async function startRecording() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        mediaRecorder = new MediaRecorder(stream);
        audioChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            audioChunks.push(event.data);
        };
        
        mediaRecorder.onstop = async () => {
            const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
            await analyzeSpeech(audioBlob);
            
            // Stop all tracks
            stream.getTracks().forEach(track => track.stop());
        };
        
        mediaRecorder.start();
        isRecording = true;
        
        // Update UI
        const recordBtn = document.getElementById('record-btn');
        const recordText = document.getElementById('record-text');
        recordBtn.classList.add('recording');
        recordText.textContent = 'Click to Stop Recording';
        document.getElementById('recording-indicator').classList.remove('hidden');
        
    } catch (error) {
        showError('Microphone access denied or not available');
        console.error('Error accessing microphone:', error);
    }
}

// Stop recording
async function stopRecording() {
    if (mediaRecorder && isRecording) {
        mediaRecorder.stop();
        isRecording = false;
        
        // Update UI
        const recordBtn = document.getElementById('record-btn');
        const recordText = document.getElementById('record-text');
        recordBtn.classList.remove('recording');
        recordText.textContent = 'Click to Start Recording';
        document.getElementById('recording-indicator').classList.add('hidden');
        
        // Show loading while processing
        showLoading();
    }
}

// Analyze speech
async function analyzeSpeech(audioBlob) {
    hideError();
    
    try {
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.wav');
        
        const response = await fetch('/predict_speech', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // Show transcribed text
            if (data.transcribed_text) {
                const transcribedContainer = document.getElementById('transcribed-text-container');
                const transcribedText = document.getElementById('transcribed-text');
                transcribedText.textContent = data.transcribed_text;
                transcribedContainer.classList.remove('hidden');
            }
            
            displayResults(data);
        } else {
            showError(data.error || 'Failed to process speech');
        }
    } catch (error) {
        showError('Failed to analyze speech. Please try again.');
        console.error('Error:', error);
    } finally {
        hideLoading();
    }
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

// Allow Enter key to submit (Ctrl+Enter for new line)
document.addEventListener('DOMContentLoaded', function() {
    const textInput = document.getElementById('text-input');
    
    textInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey && !e.ctrlKey) {
            e.preventDefault();
            analyzeText();
        }
    });
});