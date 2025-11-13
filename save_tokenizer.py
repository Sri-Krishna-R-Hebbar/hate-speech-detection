#!/usr/bin/env python3
"""
Save Tokenizer for Web Application
This script loads the training data, creates the tokenizer, and saves it
so the web application can use it for predictions.
"""

import sys
import pickle
sys.path.append('src')

from data_preprocessing import DataPreprocessor

def save_tokenizer():
    """Save the tokenizer for use in the web application"""
    print("="*60)
    print("Saving Tokenizer for Web Application")
    print("="*60)
    
    # Create preprocessor
    preprocessor = DataPreprocessor(max_words=10000, max_len=100)
    
    # Load and prepare data (this fits the tokenizer)
    print("\nLoading training data to fit tokenizer...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
        racist_path='data/twitter_racism_parsed_dataset.csv',
        sexist_path='data/twitter_sexism_parsed_dataset.csv',
        test_size=0.2,
        val_size=0.1,
        random_state=42
    )
    
    # Save tokenizer
    print("\nSaving tokenizer...")
    with open('models/tokenizer.pkl', 'wb') as f:
        pickle.dump(preprocessor.tokenizer, f)
    
    print("✓ Tokenizer saved to: models/tokenizer.pkl")
    print("\n" + "="*60)
    print("SUCCESS! You can now run the web application.")
    print("="*60)
    print("\nTo start the web app, run:")
    print("  python app.py")

if __name__ == "__main__":
    try:
        save_tokenizer()
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure your data files are in the 'data/' directory:")
        print("  - data/twitter_racism_parsed_dataset.csv")
        print("  - data/twitter_sexism_parsed_dataset.csv")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)