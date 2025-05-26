#!/usr/bin/env python
# training_pipeline.py
from microfs.core import FeatureStore
from microfs.utils import simple_logger, get_project_root, setup_project_dirs
# For standalone testing and model training
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import argparse
from pathlib import Path

def run_training_pipeline(fs: FeatureStore, fv_name: str):
    """
    Run the training pipeline for a given feature view.
    
    Args:
        fs: FeatureStore instance
        fv_name: Name of the feature view to train on
    """
    simple_logger("info", f"Starting Training Pipeline for '{fv_name}'...")
    
    try:
        # Get the feature view
        fv = fs.get_feature_view(fv_name)
        
        # Get training data and compute transformation parameters
        simple_logger("info", f"Getting training data for '{fv_name}'...")
        X_train, y_train = fv.get_training_data(compute_params=True)
        
        if X_train.empty or len(y_train) == 0:
            simple_logger("warning", "No training data available")
            return
        
        simple_logger("info", f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
        
        # Train a simple logistic regression model
        simple_logger("info", "Training logistic regression model...")
        
        # Use only numeric columns for training
        X_train_numeric = X_train.select_dtypes(include=['number'])
        
        if X_train_numeric.empty:
            simple_logger("error", "No numeric features available for training")
            return
            
        simple_logger("info", f"Using {X_train_numeric.shape[1]} numeric features")
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_numeric, y_train)
        
        # Save model
        models_dir = Path(get_project_root()) / "models"
        models_dir.mkdir(exist_ok=True)
        
        model_path = models_dir / f"{fv_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        # Calculate accuracy
        accuracy = accuracy_score(y_train, model.predict(X_train_numeric))
        
        simple_logger("info", f"Model saved to {model_path}")
        simple_logger("info", f"Training accuracy: {accuracy:.3f}")
        
    except Exception as e:
        simple_logger("error", f"Error in training pipeline: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the training pipeline')
    parser.add_argument('--feature_view', type=str, default='recommendation_clicks_v1',
                      help='Name of the feature view to use')
    args = parser.parse_args()
    
    setup_project_dirs()
    fs = FeatureStore()
    
    simple_logger("info", f"Running training pipeline for '{args.feature_view}'")
    run_training_pipeline(fs, args.feature_view) 