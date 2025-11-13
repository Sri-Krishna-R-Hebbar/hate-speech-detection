from flask import Flask, render_template, request, jsonify
import os
import sys
import numpy as np
from tensorflow import keras
import speech_recognition as sr
from io import BytesIO

# Add src to path
sys.path.append('src')
from data_preprocessing import DataPreprocessor
from bi_min_lstm import get_custom_objects

app = Flask(__name__)

# Global variables for model and preprocessor
model = None
preprocessor = None
CLASS_NAMES = ['None (Safe)', 'Racist', 'Sexist']

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    
    print("Loading model and preprocessor...")
    
    # Load the trained model
    model_path = 'models/bi_min_lstm_best.keras'
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first!")
    
    custom_objects = get_custom_objects()
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"✓ Model loaded from {model_path}")
    
    # Create preprocessor (we'll need the tokenizer from training)
    preprocessor = DataPreprocessor(max_words=10000, max_len=100)
    
    # Load tokenizer from training
    # Note: In production, you should save the tokenizer during training
    # For now, we'll create a dummy one and load from saved model
    # You need to retrain and save tokenizer separately or use pickle
    try:
        import pickle
        with open('models/tokenizer.pkl', 'rb') as f:
            preprocessor.tokenizer = pickle.load(f)
        print("✓ Tokenizer loaded from models/tokenizer.pkl")
    except FileNotFoundError:
        print("⚠ Warning: Tokenizer not found. You need to save it during training.")
        print("For now, the preprocessor will work but predictions may be less accurate.")
        # Try to recreate tokenizer from training data
        try:
            from train import train_model
            # This is a workaround - in production, always save the tokenizer!
            preprocessor.prepare_data(
                'data/twitter_racism_parsed_dataset.csv',
                'data/twitter_sexism_parsed_dataset.csv',
                test_size=0.2,
                val_size=0.1,
                random_state=42
            )
            # Save tokenizer for future use
            with open('models/tokenizer.pkl', 'wb') as f:
                pickle.dump(preprocessor.tokenizer, f)
            print("✓ Tokenizer recreated and saved")
        except Exception as e:
            print(f"✗ Error recreating tokenizer: {e}")
            raise

def predict_text(text):
    """
    Predict hate speech class for given text
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with prediction results
    """
    if model is None or preprocessor is None:
        return {"error": "Model not loaded"}
    
    # Clean the text
    cleaned = preprocessor.clean_text(text)
    
    # Tokenize and pad
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    sequence = preprocessor.tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(
        sequence, 
        maxlen=preprocessor.max_len, 
        padding='post', 
        truncating='post'
    )
    
    # Predict
    probabilities = model.predict(padded, verbose=0)[0]
    predicted_class_idx = np.argmax(probabilities)
    confidence = float(probabilities[predicted_class_idx])
    
    return {
        'original_text': text,
        'predicted_class': CLASS_NAMES[predicted_class_idx],
        'predicted_class_index': int(predicted_class_idx),
        'confidence': confidence,
        'probabilities': {
            CLASS_NAMES[i]: float(probabilities[i]) 
            for i in range(len(CLASS_NAMES))
        }
    }

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle text prediction requests"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get prediction
        result = predict_text(text)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_speech', methods=['POST'])
def predict_speech():
    """Handle speech-to-text and prediction"""
    try:
        # Check if audio file is present
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        
        # Initialize recognizer
        recognizer = sr.Recognizer()
        
        # Convert audio to text
        try:
            # Save audio temporarily
            audio_data = audio_file.read()
            audio_file_like = BytesIO(audio_data)
            
            # Try to recognize using Google Speech Recognition
            with sr.AudioFile(audio_file_like) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                
        except sr.UnknownValueError:
            return jsonify({'error': 'Could not understand audio'}), 400
        except sr.RequestError as e:
            return jsonify({'error': f'Speech recognition error: {str(e)}'}), 500
        except Exception as e:
            # If direct conversion fails, try alternative method
            # Save to temporary wav file
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                tmp_file.write(audio_data)
                tmp_file_path = tmp_file.name
            
            try:
                with sr.AudioFile(tmp_file_path) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)
            finally:
                os.unlink(tmp_file_path)
        
        if not text:
            return jsonify({'error': 'No speech detected'}), 400
        
        # Get prediction
        result = predict_text(text)
        result['transcribed_text'] = text
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    })

if __name__ == '__main__':
    # Load model and preprocessor on startup
    try:
        load_model_and_preprocessor()
        print("\n" + "="*60)
        print("🚀 Hate Speech Detection Server Starting...")
        print("="*60)
        print("✓ Model loaded successfully")
        print("✓ Preprocessor ready")
        print("\n🌐 Open your browser and go to: http://127.0.0.1:5000")
        print("="*60 + "\n")
        
        # Start Flask app
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    except Exception as e:
        print(f"\n✗ Error starting server: {e}")
        print("\nMake sure you have:")
        print("  1. Trained model at: models/bi_min_lstm_best.keras")
        print("  2. Tokenizer saved at: models/tokenizer.pkl")
        print("\nRun 'python main.py' first to train the model!")
        sys.exit(1)