# Flask Web Interface for Hate Speech Detection

A web-based interface for the Bi-MinLSTM hate speech detection model with speech-to-text functionality.

## Features

- 🎤 **Speech-to-Text Input**: Use your microphone to input text directly
- 🔍 **Real-time Analysis**: Get instant predictions with confidence scores
- 📊 **Visual Results**: See probability distributions for all classes
- 🎨 **Modern UI**: Beautiful, responsive web interface

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure the model files exist:
   - `models/bi_min_lstm_best.keras` or `models/bi_min_lstm_final.keras`
   - The preprocessor will be created automatically on first run

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

### Text Input
1. Type or paste text into the text area
2. Click "🔍 Analyze Text" to get predictions

### Speech-to-Text
1. Click "🎤 Start Recording" button
2. Allow microphone access when prompted
3. Speak your text clearly
4. The transcribed text will appear in the text area
5. Click "⏹️ Stop Recording" when done
6. Click "🔍 Analyze Text" to analyze

### Keyboard Shortcuts
- `Ctrl+Enter` (or `Cmd+Enter` on Mac): Analyze text

## API Endpoints

### POST `/predict`
Predict hate speech category for given text.

**Request:**
```json
{
  "text": "Your text here"
}
```

**Response:**
```json
{
  "class": "None",
  "confidence": 0.95,
  "probabilities": {
    "None": 0.95,
    "Racist": 0.03,
    "Sexist": 0.02
  },
  "error": null
}
```

### GET `/health`
Check if the model and preprocessor are loaded.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

## Browser Compatibility

- **Speech-to-Text**: Works best in Chrome, Edge, or Safari
- **Firefox**: Limited speech recognition support
- **HTTPS Required**: For production, use HTTPS (speech recognition requires secure context)

## Troubleshooting

### Model Not Loading
- Ensure model files exist in `models/` directory
- Check console for error messages
- Verify TensorFlow is installed correctly

### Speech Recognition Not Working
- Grant microphone permissions in browser settings
- Use Chrome or Edge for best compatibility
- Ensure you're using HTTPS in production (or localhost for development)

### Preprocessor Issues
- The preprocessor will be created automatically on first run
- It will be saved to `models/preprocessor.pkl` for faster subsequent loads
- If issues occur, delete `models/preprocessor.pkl` to regenerate

## Development

The application structure:
- `app.py`: Flask application and API endpoints
- `templates/index.html`: Main web interface
- `static/css/style.css`: Styling
- `static/js/app.js`: JavaScript for speech recognition and UI interactions

## Notes

- The model classifies text into three categories: **None**, **Racist**, or **Sexist**
- Confidence scores show the model's certainty in its prediction
- All probabilities are displayed for transparency

