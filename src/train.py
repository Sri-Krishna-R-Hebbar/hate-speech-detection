import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def train_model(model, X_train, y_train, X_val, y_val, model_name, 
                epochs=5, batch_size=64):
    """
    Train the model with callbacks
    
    Args:
        model: Keras model to train
        X_train: Training data
        y_train: Training labels
        X_val: Validation data
        y_val: Validation labels
        model_name: Name for saving the model
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Training history
    """
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}\n")
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Define callbacks
    callbacks = [
        # Early stopping
        EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Model checkpoint
        ModelCheckpoint(
            filepath=f'models/{model_name}_best.keras',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        
        # Reduce learning rate
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    print(f"\n{'='*60}")
    print(f"Training completed for {model_name}")
    print(f"{'='*60}\n")
    
    # Save final model
    model.save(f'models/{model_name}_final.keras')
    print(f"Model saved: models/{model_name}_final.keras")
    
    return history