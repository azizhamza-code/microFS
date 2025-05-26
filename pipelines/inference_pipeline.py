#!/usr/bin/env python
# inference_pipeline.py
from microfs.core import FeatureStore
from microfs.utils import simple_logger, get_project_root, setup_project_dirs
from typing import Dict, Any
import os
import pickle
import pandas as pd
import argparse
from pathlib import Path

def run_inference_pipeline(fs: FeatureStore, fv_name: str, entity_keys: Dict[str, Any]):
    """
    Run the inference pipeline for a given feature view and entity keys.
    
    This pipeline:
    1. Gets the feature vector for the specified entity keys
    2. Loads the trained model
    3. Makes a prediction using the model
    
    Args:
        fs: Feature store instance
        fv_name: Name of the feature view to use
        entity_keys: Dictionary of entity keys to get features for
    
    Returns:
        Prediction result (or None if an error occurred)
    """
    simple_logger("info", f"Starting Inference Pipeline for '{fv_name}'...")
    
    try:
        # Get the feature view
        fv = fs.get_feature_view(fv_name)
        simple_logger("info", f"Retrieved feature view: {fv.name}")
        
        # Get feature vector for prediction
        simple_logger("info", f"Getting feature vector for entity keys: {entity_keys}")
        feature_vector = fv.get_inference_vector(entity_keys)
        
        if feature_vector.empty:
            simple_logger("warning", f"Empty feature vector for entity keys: {entity_keys}")
            return None
        
        # Convert to DataFrame for model input
        feature_df = pd.DataFrame([feature_vector])
        simple_logger("info", f"Feature vector shape: {feature_df.shape}, columns: {list(feature_df.columns)}")
        
        # Load trained model
        models_dir = Path(get_project_root()) / "models"
        model_path = models_dir / f"{fv_name}_model.pkl"
        
        if not model_path.exists():
            simple_logger("error", f"Model file not found at {model_path}")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        simple_logger("info", f"Loaded model from {model_path}")
        
        # Handle missing features
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
            missing_features = [feat for feat in expected_features if feat not in feature_df.columns]
            
            if missing_features:
                simple_logger("warning", f"Missing features in the vector: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    feature_df[feat] = 0
            
            # Ensure correct column order
            feature_df = feature_df[expected_features]
        
        # Fill any NaN values with 0
        feature_df = feature_df.fillna(0)
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        prediction_proba = None
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            prediction_proba = model.predict_proba(feature_df)[0]
        
        # Log results
        if prediction_proba is not None:
            simple_logger("info", f"Prediction: {prediction}, Probability: {prediction_proba}")
        else:
            simple_logger("info", f"Prediction: {prediction}")
        
        return prediction
        
    except Exception as e:
        simple_logger("error", f"Error in inference pipeline: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the inference pipeline')
    parser.add_argument('--feature_view', type=str, default='recommendation_clicks_v1',
                        help='Name of the feature view to use')
    parser.add_argument('--user_id', type=int, default=1,
                        help='User ID for prediction')
    parser.add_argument('--item_id', type=int, default=102,
                        help='Item ID for prediction')
    args = parser.parse_args()
    
    setup_project_dirs()
    fs = FeatureStore()
    
    entity_keys = {'user_id': args.user_id, 'item_id': args.item_id}
    simple_logger("info", f"Running inference pipeline for '{args.feature_view}' with {entity_keys}")
    
    result = run_inference_pipeline(fs, args.feature_view, entity_keys)
    if result is not None:
        print(f"\nPrediction result: {result}")
    else:
        print("\nNo prediction was made due to errors") 