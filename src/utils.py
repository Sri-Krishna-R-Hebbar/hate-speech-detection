import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

def create_directories():
    """Create necessary directories for the project"""
    directories = ['models', 'plots', 'data']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def plot_training_history(history, model_name):
    """Plot training and validation accuracy/loss"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[0].set_title(f'{model_name} - Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Validation Loss')
    axes[1].set_title(f'{model_name} - Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_training_history.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved: plots/{model_name}_training_history.png")

def plot_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(f'plots/{model_name}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved: plots/{model_name}_confusion_matrix.png")

def save_metrics(y_true, y_pred, class_names, model_name):
    """Calculate and save evaluation metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)
    
    # Overall metrics
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    
    with open(f'plots/{model_name}_metrics.txt', 'w') as f:
        f.write(f"{'='*60}\n")
        f.write(f"{model_name} - Evaluation Metrics\n")
        f.write(f"{'='*60}\n\n")
        
        f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
        
        f.write(f"Macro Averaged Metrics:\n")
        f.write(f"  Precision: {precision_macro:.4f}\n")
        f.write(f"  Recall: {recall_macro:.4f}\n")
        f.write(f"  F1-Score: {f1_macro:.4f}\n\n")
        
        f.write(f"Weighted Averaged Metrics:\n")
        f.write(f"  Precision: {precision_weighted:.4f}\n")
        f.write(f"  Recall: {recall_weighted:.4f}\n")
        f.write(f"  F1-Score: {f1_weighted:.4f}\n\n")
        
        f.write(f"{'='*60}\n")
        f.write(f"Per-Class Metrics:\n")
        f.write(f"{'='*60}\n\n")
        
        for i, class_name in enumerate(class_names):
            f.write(f"{class_name}:\n")
            f.write(f"  Precision: {precision[i]:.4f}\n")
            f.write(f"  Recall: {recall[i]:.4f}\n")
            f.write(f"  F1-Score: {f1[i]:.4f}\n")
            f.write(f"  Support: {support[i]}\n\n")
        
        f.write(f"{'='*60}\n")
        f.write("Classification Report:\n")
        f.write(f"{'='*60}\n\n")
        f.write(classification_report(y_true, y_pred, target_names=class_names))
    
    print(f"Metrics saved: plots/{model_name}_metrics.txt")
    
    return {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro
    }

def print_metrics_summary(metrics, model_name):
    """Print metrics summary to console"""
    print(f"\n{'='*60}")
    print(f"{model_name} - Results Summary")
    print(f"{'='*60}")
    print(f"Accuracy:       {metrics['accuracy']:.4f}")
    print(f"Precision:      {metrics['precision_macro']:.4f}")
    print(f"Recall:         {metrics['recall_macro']:.4f}")
    print(f"F1-Score:       {metrics['f1_macro']:.4f}")
    print(f"{'='*60}\n")