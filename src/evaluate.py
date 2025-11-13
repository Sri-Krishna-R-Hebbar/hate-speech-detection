import numpy as np
from tensorflow import keras
import sys
sys.path.append('src')
from bi_min_lstm import get_custom_objects

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate the model on test data
    
    Args:
        model: Trained Keras model
        X_test: Test data
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Tuple of (loss, accuracy, predictions)
    """
    print(f"\n{'='*60}")
    print(f"Evaluating {model_name}")
    print(f"{'='*60}\n")
    
    # Evaluate on test set
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    return test_loss, test_accuracy, y_pred

def load_and_evaluate(model_path, X_test, y_test, model_name):
    """
    Load a saved model and evaluate it
    
    Args:
        model_path: Path to saved model
        X_test: Test data
        y_test: Test labels
        model_name: Name of the model
        
    Returns:
        Tuple of (loss, accuracy, predictions)
    """
    print(f"\nLoading model from: {model_path}")
    
    # Load model with custom objects
    custom_objects = get_custom_objects()
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    
    return evaluate_model(model, X_test, y_test, model_name)