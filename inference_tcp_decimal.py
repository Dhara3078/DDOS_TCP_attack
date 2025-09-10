#!/usr/bin/env python3
"""
Inference script for CNN-GRU DDoS detection model using TensorFlow
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import load_model
import os

def load_tf_model(model_path):
    """Load TensorFlow model"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = load_model(model_path)
    print(f"Model loaded successfully from {model_path}")
    return model

def load_test_data(x_path, y_path):
    """Load test data and labels"""
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Test data file not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Test labels file not found: {y_path}")
    
    x_test = np.load(x_path)
    y_test = np.load(y_path)
    
    print(f"Test data loaded successfully")
    print(f"X_test shape: {x_test.shape}")
    print(f"Y_test shape: {y_test.shape}")
    print(f"X_test dtype: {x_test.dtype}")
    print(f"Y_test dtype: {y_test.dtype}")
    print(f"Class distribution: {np.bincount(y_test)}")
    
    return x_test, y_test

def evaluate_model(y_true, y_pred):
    """Evaluate model performance"""
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    
    print(f"Accuracy:  {accuracy:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 0      1")
    print(f"Actual    0    {cm[0,0]:6d} {cm[0,1]:6d}")
    print(f"          1    {cm[1,0]:6d} {cm[1,1]:6d}")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
    }

def main():
    """Main inference function"""
    print("="*60)
    print("CNN-GRU DDoS Detection Model Inference")
    print("TCP Traffic with Decimal Scaling")
    print("="*60)
    
    # File paths
    model_path = "models/cnn_gru_model_PSO_TCP_12_aug_decimal_scaling.h5"
    x_test_path = "test_data/x_test_decimal_tcp_data.npy"
    y_test_path = "test_data/y_test_decimal_tcp_labels.npy"
    
    try:
        # Load model
        print("\n1. Loading TensorFlow model...")
        model = load_model(model_path)

        # Load test data
        print("\n2. Loading test data...")
        x_test, y_test = load_test_data(x_test_path, y_test_path)
                
        # Run inference
        print("\n3. Running inference...")
        y_pred = model.predict(x_test)
        y_pred_classes = (y_pred > 0.5).astype("int32")

        # Evaluate model
        print("\n4. Evaluating model performance...")
        results = evaluate_model(y_test, y_pred_classes)
        
        # Save results
        results_file = "inference_results_tcp_decimal.txt"
        with open(results_file, 'w') as f:
            f.write("CNN-GRU DDoS Detection Model - Inference Results\n")
            f.write("="*50 + "\n")
            f.write(f"Model: {model_path}\n")
            f.write(f"Test Data: {x_test_path}\n")
            f.write(f"Test Labels: {y_test_path}\n")
            f.write(f"Samples: {len(x_test)}\n\n")
            
            f.write("Performance Metrics:\n")
            f.write(f"Accuracy: {results['accuracy']:.4f}\n")
            
            f.write("\nConfusion Matrix:\n")
            f.write(f"{results['confusion_matrix']}\n")
        
        print(f"\nResults saved to: {results_file}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        raise

if __name__ == "__main__":
    main()
