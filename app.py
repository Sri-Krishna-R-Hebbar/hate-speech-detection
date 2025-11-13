#!/usr/bin/env python3
"""
Flask Web Application for Hate Speech Detection
Includes speech-to-text functionality
"""

import os
import sys
import warnings
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow import keras

warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('src')

from data_preprocessing import DataPreprocessor
from bi_min_lstm import get_custom_objects

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global variables for model and preprocessor
model = None
preprocessor = None
CLASS_NAMES = ['None', 'Racist', 'Sexist']

def load_model_and_preprocessor():
    """Load the trained model and preprocessor"""
    global model, preprocessor
    
    try:
        # Load model
        model_path = 'models/bi_min_lstm_best.keras'
        if not os.path.exists(model_path):
            model_path = 'models/bi_min_lstm_final.keras'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        custom_objects = get_custom_objects()
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print(f"Model loaded successfully from {model_path}")
        
        # Load or create preprocessor
        preprocessor_path = 'models/preprocessor.pkl'
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, 'rb') as f:
                preprocessor = pickle.load(f)
            print(f"Preprocessor loaded from {preprocessor_path}")
        else:
            # Create preprocessor and fit it (we'll need to load data once)
            print("Preprocessor not found. Creating new one...")
            temp_preprocessor = DataPreprocessor(max_words=10000, max_len=100)
            _, _, _, _, _, _ = temp_preprocessor.prepare_data(
                racist_path='data/twitter_racism_parsed_dataset.csv',
                sexist_path='data/twitter_sexism_parsed_dataset.csv',
                test_size=0.2,
                val_size=0.1,
                random_state=42
            )
            preprocessor = temp_preprocessor
            
            # Save preprocessor
            os.makedirs('models', exist_ok=True)
            with open(preprocessor_path, 'wb') as f:
                pickle.dump(preprocessor, f)
            print(f"Preprocessor saved to {preprocessor_path}")
        
    except Exception as e:
        print(f"Error loading model/preprocessor: {str(e)}")
        raise

def predict_hate_speech(text):
    """
    Predict hate speech category for given text
    
    Args:
        text: Input text string
        
    Returns:
        Dictionary with prediction results
    """
    if model is None or preprocessor is None:
        return {
            'error': 'Model or preprocessor not loaded',
            'class': None,
            'confidence': None,
            'probabilities': None
        }
    
    try:
        # Clean and preprocess text
        cleaned_text = preprocessor.clean_text(text)
        
        if len(cleaned_text.strip()) == 0:
            return {
                'error': 'Text is empty after cleaning',
                'class': None,
                'confidence': None,
                'probabilities': None
            }
        
        # Tokenize and pad
        padded = preprocessor.tokenize_and_pad([cleaned_text], fit=False)
        
        # Predict
        predictions = model.predict(padded, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get probabilities for all classes
        probabilities = {
            CLASS_NAMES[i]: float(predictions[0][i])
            for i in range(len(CLASS_NAMES))
        }
        
        return {
            'class': CLASS_NAMES[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities,
            'error': None
        }
    
    except Exception as e:
        return {
            'error': f'Prediction error: {str(e)}',
            'class': None,
            'confidence': None,
            'probabilities': None
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'error': 'No text provided',
                'class': None,
                'confidence': None,
                'probabilities': None
            }), 400
        
        text = data['text']
        
        if not text or len(text.strip()) == 0:
            return jsonify({
                'error': 'Text is empty',
                'class': None,
                'confidence': None,
                'probabilities': None
            }), 400
        
        # Make prediction
        result = predict_hate_speech(text)
        
        if result['error']:
            return jsonify(result), 400
        
        return jsonify(result), 200
    
    except Exception as e:
        return jsonify({
            'error': f'Server error: {str(e)}',
            'class': None,
            'confidence': None,
            'probabilities': None
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'preprocessor_loaded': preprocessor is not None
    }), 200

if __name__ == '__main__':
    print("="*60)
    print("Loading Hate Speech Detection Model...")
    print("="*60)
    
    try:
        load_model_and_preprocessor()
        print("\n" + "="*60)
        print("Model loaded successfully!")
        print("Starting Flask server...")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    
    except Exception as e:
        print(f"\nError: Failed to start application: {str(e)}")
        sys.exit(1)

