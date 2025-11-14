#!/usr/bin/env python3
"""
Hate Speech Detection using Bi-MinLSTM
Main execution script
"""

import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add src to path

from src.data_preprocessing import DataPreprocessor
from src.bi_min_lstm import create_bi_min_lstm_model, test_model_predictions
from src.train import train_model
from src.evaluate import evaluate_model
from src.utils import (create_directories, plot_training_history,
                       plot_confusion_matrix, save_metrics, print_metrics_summary)


def main():
    """Main execution function"""

    print("=" * 60)
    print("Hate Speech Detection using Bi-MinLSTM")
    print("=" * 60)

    # Create necessary directories
    create_directories()

    # Configuration
    CONFIG = {
        'racist_dataset': 'data/twitter_racism_parsed_dataset.csv',
        'sexist_dataset': 'data/twitter_sexism_parsed_dataset.csv',
        'max_words': 13000,
        'max_len': 125,
        'embedding_dim': 200,
        'lstm_units': 128,
        'num_classes': 3,
        'dropout_rate': 0.2,
        'epochs': 10,
        'batch_size': 64,
        'test_size': 0.2,
        'val_size': 0.1,
        'random_state': 42
    }

    # Class names
    CLASS_NAMES = ['None', 'Racist', 'Sexist']

    print("\nConfiguration:")
    print("-" * 60)
    for key, value in CONFIG.items():
        print(f"{key:.<25} {value}")
    print("-" * 60)

    # Step 1: Data Preprocessing
    print("\n" + "=" * 60)
    print("STEP 1: DATA PREPROCESSING")
    print("=" * 60)

    preprocessor = DataPreprocessor(
        max_words=CONFIG['max_words'],
        max_len=CONFIG['max_len']
    )

    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_data(
        racist_path=CONFIG['racist_dataset'],
        sexist_path=CONFIG['sexist_dataset'],
        test_size=CONFIG['test_size'],
        val_size=CONFIG['val_size'],
        random_state=CONFIG['random_state']
    )

    vocab_size = preprocessor.get_vocab_size()
    print(f"\nVocabulary size for model: {vocab_size}")

    # Step 2: Model Creation
    print("\n" + "=" * 60)
    print("STEP 2: MODEL CREATION")
    print("=" * 60)

    model = create_bi_min_lstm_model(
        vocab_size=vocab_size,
        embedding_dim=CONFIG['embedding_dim'],
        lstm_units=CONFIG['lstm_units'],
        num_classes=CONFIG['num_classes'],
        max_len=CONFIG['max_len'],
        dropout_rate=CONFIG['dropout_rate']
    )

    # Step 3: Model Training
    print("\n" + "=" * 60)
    print("STEP 3: MODEL TRAINING")
    print("=" * 60)

    history = train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model_name='bi_min_lstm',
        epochs=CONFIG['epochs'],
        batch_size=CONFIG['batch_size']
    )

    # Step 4: Model Evaluation
    print("\n" + "=" * 60)
    print("STEP 4: MODEL EVALUATION")
    print("=" * 60)

    test_loss, test_accuracy, y_pred = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        model_name='Bi-MinLSTM'
    )

    # Step 5: Generate Visualizations and Metrics
    print("\n" + "=" * 60)
    print("STEP 5: GENERATING VISUALIZATIONS AND METRICS")
    print("=" * 60)

    print("\nGenerating plots and metrics...")

    # Plot training history
    plot_training_history(history, 'bi_min_lstm')

    # Plot confusion matrix
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, 'bi_min_lstm')

    # Save and print metrics
    metrics = save_metrics(y_test, y_pred, CLASS_NAMES, 'bi_min_lstm')
    print_metrics_summary(metrics, 'Bi-MinLSTM')

    # Step 6: Test with Sample Texts
    print("\n" + "=" * 60)
    print("STEP 6: TESTING WITH SAMPLE TEXTS")
    print("=" * 60)

    test_model_predictions(model, preprocessor, CLASS_NAMES)

    # Final Summary
    print("\n" + "=" * 60)
    print("PROJECT COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("\nGenerated files:")
    print("  Models:")
    print("    - models/bi_min_lstm_best_1.keras")
    print("    - models/bi_min_lstm_final_1.keras")
    print("  Plots:")
    print("    - plots/bi_min_lstm_training_history_1.png")
    print("    - plots/bi_min_lstm_confusion_matrix_1.png")
    print("  Metrics:")
    print("    - plots/bi_min_lstm_metrics_1.txt")
    print("=" * 60)


if __name__ == "__main__":
    main()
