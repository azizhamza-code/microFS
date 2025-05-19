#!/usr/bin/env python
# training_pipeline.py
from microfs.core_api import FeatureStore
from microfs.utils import simple_logger, get_project_root, setup_project_dirs
# For standalone testing and model training
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle
import argparse
from pathlib import Path
import traceback

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the training pipeline')
    parser.add_argument('--feature_view', '--fv_name', type=str, default='recommendation_clicks_v1',
                      help='Name of the feature view to use')
    parser.add_argument('--compute_params', action='store_true',
                      help='Whether to compute transformation parameters')
    return parser.parse_args()

def run():
    """Main function to run the training pipeline."""
    args = parse_args()
    fs = FeatureStore()

    fv_name = args.feature_view if args.feature_view else "recommendation_clicks_v1"
    simple_logger("info", f"Running training_pipeline.py standalone for FV: {fv_name}")
    
    # Check if FV exists
    fv_list = fs.list_feature_views()
    
    simple_logger("info", f"Starting Training Pipeline for FV '{fv_name}'...")
    
    if fv_name not in fv_list:
        simple_logger("info", f"FV '{fv_name}' not found, creating...")
        fv = fs.create_feature_view(
            fv_name,
            "user_activity",
            "conversion",
            "timestamp",
            [
                {'name': 'user_profile', 'on': ['user_id'], 'prefix': 'user_profile__'},
                {'name': 'item_feature', 'on': ['item_id'], 'prefix': 'item__'}
            ],
            [
                {'feature_name': 'duration_sec', 'transform_type': 'scale'},
                {'feature_name': 'price', 'transform_type': 'scale'},
                {'feature_name': 'user_level', 'transform_type': 'one_hot_encode'},
                {'feature_name': 'item_category', 'transform_type': 'one_hot_encode'}
            ]
        )
    else:
        fv = fs.get_feature_view(fv_name)
    
    # Get training data
    simple_logger("info", f"Getting training data for FV '{fv_name}', compute_params={args.compute_params}...")
    X_train, y_train = fv.get_training_data(compute_params=args.compute_params)
    
    # Print info about features and parameters
    simple_logger("info", f"Got features shape: {X_train.shape}, labels shape: {len(y_train)}")
    simple_logger("info", f"Features include: {list(X_train.columns)}")
    
    transform_params = fv.get_transform_params()
    simple_logger("info", f"Transformation parameters are stored for {len(transform_params)} features")
    
    # Train a model
    simple_logger("info", "Training a simple logistic regression model...")
    try:
        # Filter to only use numeric columns
        X_train_numeric = X_train.select_dtypes(include=['number'])
        
        # Ensure we have some features to work with
        if X_train_numeric.empty or X_train_numeric.shape[1] == 0:
            simple_logger("error", "No numeric features available for model training")
            return
            
        simple_logger("info", f"Training model with numeric features: {list(X_train_numeric.columns)}")
        
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train_numeric, y_train)
        
        # Create models directory
        models_dir = Path(get_project_root()) / "models"
        models_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = models_dir / f"{fv_name}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
            
        simple_logger("info", f"Model trained and saved to {model_path}")
        simple_logger("info", f"Model coefficients: {model.coef_}")
        simple_logger("info", f"Model intercept: {model.intercept_}")
    except Exception as e:
        simple_logger("error", f"Error training model: {e}")
        traceback.print_exc()

if __name__ == '__main__':
    # This is for testing the pipeline script directly
    setup_project_dirs()
    run() 