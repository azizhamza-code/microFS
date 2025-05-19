#!/usr/bin/env python
# inference_pipeline.py
from microfs.core_api import FeatureStore
from microfs.utils import simple_logger, get_project_root, setup_project_dirs
from typing import Dict, Any
import os
import pickle
import pandas as pd
import argparse
from pathlib import Path
import traceback

def run_inference_pipeline(fs: FeatureStore, fv_name: str, entity_keys_for_prediction: Dict[str, Any]):
    """
    Run the inference pipeline for a given feature view and entity keys.
    
    This pipeline:
    1. Gets the feature vector for the specified entity keys
    2. Loads the trained model
    3. Makes a prediction using the model
    
    Args:
        fs: Feature store instance
        fv_name: Name of the feature view to use
        entity_keys_for_prediction: Dictionary of entity keys to get features for
    
    Returns:
        Prediction result (or None if an error occurred)
    """
    simple_logger("info", f"Starting Inference Pipeline for FV '{fv_name}'...")
    
    # --- Get the Feature View ---
    try:
        fv = fs.get_feature_view(fv_name)
        simple_logger("info", f"Retrieved Feature View: {fv.name}")
        
        # Check for transformation parameters
        transform_params = fv.get_transform_params()
        if not transform_params:
            simple_logger("warning", "No transformation parameters found. Feature transformations may not be applied correctly.")
        else:
            simple_logger("info", f"Found transformation parameters for {len(transform_params)} features")
    except Exception as e:
        simple_logger("error", f"Error retrieving Feature View '{fv_name}': {e}")
        return None
    
    # --- Get Feature Vector for Prediction ---
    try:
        simple_logger("info", f"Getting feature vector for entity keys: {entity_keys_for_prediction}")
        feature_vector = fv.get_inference_vector(entity_keys_for_prediction)
        
        if feature_vector.empty:
            simple_logger("warning", f"Empty feature vector returned for entity keys: {entity_keys_for_prediction}")
            return None
        
        # Convert to DataFrame for model input
        feature_df = pd.DataFrame([feature_vector])
        simple_logger("info", f"Feature vector shape: {feature_df.shape}, columns: {list(feature_df.columns)}")
    except Exception as e:
        simple_logger("error", f"Error getting feature vector: {e}")
        return None
    
    # --- Load Trained Model ---
    try:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "models")
        model_path = os.path.join(model_dir, f"{fv_name}_model.pkl")
        
        if not os.path.exists(model_path):
            simple_logger("error", f"Model file not found at {model_path}. Run training pipeline first.")
            return None
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        simple_logger("info", f"Loaded model from {model_path}")
    except Exception as e:
        simple_logger("error", f"Error loading model: {e}")
        return None
    
    # --- Make Prediction ---
    try:
        # Handle potential missing features in the vector compared to training data
        expected_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else None
        
        if expected_features is not None:
            # Check for missing features
            missing_features = [feat for feat in expected_features if feat not in feature_df.columns]
            
            if missing_features:
                simple_logger("warning", f"Missing features in the vector: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    feature_df[feat] = 0
            
            # Ensure correct column order
            feature_df = feature_df[expected_features]
        
        # Impute any remaining NaNs in the final feature set with 0.0
        # This handles cases where a feature was present but had a NaN value (e.g., from online store or transformation)
        cols_with_nan = feature_df.columns[feature_df.isnull().any()].tolist()
        if cols_with_nan:
            simple_logger("warning", f"Features {cols_with_nan} contain NaN values. Imputing with 0.0.")
            for col in cols_with_nan:
                # Ensure the column is numeric before filling, though select_dtypes should handle this
                if pd.api.types.is_numeric_dtype(feature_df[col]):
                    feature_df[col] = feature_df[col].fillna(0.0)
                else:
                    simple_logger("warning", f"Feature '{col}' is not numeric and contains NaN. Skipping imputation for this column.")
        
        # Make prediction
        prediction = model.predict(feature_df)[0]
        prediction_proba = None
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            try:
                prediction_proba = model.predict_proba(feature_df)[0]
            except Exception:
                pass
        
        # Log results
        if prediction_proba is not None:
            simple_logger("info", f"Prediction: {prediction}, Probability: {prediction_proba}")
        else:
            simple_logger("info", f"Prediction: {prediction}")
        
        return prediction
    except Exception as e:
        simple_logger("error", f"Error making prediction: {e}")
        return None

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run the inference pipeline')
    parser.add_argument('--feature_view', '--fv_name', type=str, default='recommendation_clicks_v1',
                        help='Name of the feature view to use')
    parser.add_argument('--user_id', type=int, default=1,
                        help='User ID for prediction')
    parser.add_argument('--item_id', type=int, default=102,
                        help='Item ID for prediction')
    return parser.parse_args()

def run():
    """Main function to run the inference pipeline."""
    args = parse_args()
    fs = FeatureStore()

    fv_name = args.feature_view if args.feature_view else "recommendation_clicks_v1"
    simple_logger("info", f"Running inference_pipeline.py standalone for FV: {fv_name}")
    
    simple_logger("info", f"Starting Inference Pipeline for FV '{fv_name}'...")
    
    # First, check if the feature view exists
    if fv_name not in fs.list_feature_views():
        simple_logger("error", f"Feature View '{fv_name}' not found. Please run the training pipeline first.")
        print("\nNo prediction was made due to errors")
        return
    
    try:
        fv = fs.get_feature_view(fv_name)
    except Exception as e:
        simple_logger("error", f"Error retrieving Feature View '{fv_name}': {e}")
        print("\nNo prediction was made due to errors")
        return
    
    # Check if model exists
    models_dir = Path(get_project_root()) / "models"
    model_path = models_dir / f"{fv_name}_model.pkl"
    
    if not model_path.exists():
        simple_logger("error", f"Model file not found at {model_path}. Please run the training pipeline first.")
        print("\nNo prediction was made due to errors")
        return
    
    # Load the model
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        simple_logger("info", f"Model loaded from {model_path}")
    except Exception as e:
        simple_logger("error", f"Error loading model: {e}")
        print("\nNo prediction was made due to errors")
        return
    
    # Get entity keys from args or use defaults
    user_id = args.user_id if args.user_id else 1
    item_id = args.item_id if args.item_id else 102
    
    simple_logger("info", f"Making a prediction for user_id={user_id}, item_id={item_id}")
    
    # Get feature vector
    try:
        entity_keys = {'user_id': user_id, 'item_id': item_id}
        feature_vector = fv.get_inference_vector(entity_keys)
        
        # Ensure the feature vector has the required model features
        required_features = model.feature_names_in_
        
        # Convert to DataFrame for easier manipulation
        vector_df = pd.DataFrame([feature_vector])
        
        # Select only numeric columns for prediction
        vector_df_numeric = vector_df.select_dtypes(include=['number'])
        
        # Ensure we have the right columns in the right order for the model
        missing_features = set(required_features) - set(vector_df_numeric.columns)
        if missing_features:
            simple_logger("warning", f"Missing features for model: {missing_features}. Adding zeros.")
            for feat in missing_features:
                vector_df_numeric[feat] = 0.0
                
        # Reorder columns to match model expectation
        prediction_features = vector_df_numeric[required_features].copy()
        
        # Impute any remaining NaNs in the final feature set with 0.0
        # This handles cases where a feature was present but had a NaN value (e.g., from online store or transformation)
        cols_with_nan = prediction_features.columns[prediction_features.isnull().any()].tolist()
        if cols_with_nan:
            simple_logger("warning", f"Features {cols_with_nan} contain NaN values. Imputing with 0.0.")
            for col in cols_with_nan:
                # Ensure the column is numeric before filling, though select_dtypes should handle this
                if pd.api.types.is_numeric_dtype(prediction_features[col]):
                    prediction_features[col] = prediction_features[col].fillna(0.0)
                else:
                    simple_logger("warning", f"Feature '{col}' is not numeric and contains NaN. Skipping imputation for this column.")
        
        # Make prediction
        prediction = model.predict(prediction_features)[0]
        prediction_proba = model.predict_proba(prediction_features)[0]
        
        # Generate output
        print("\n" + "="*50)
        print(f"Prediction for user_id={user_id}, item_id={item_id}:")
        print(f"Class: {prediction} (Probability: {prediction_proba[1]:.4f})")
        print("="*50 + "\n")
        
        # Feature importance
        if hasattr(model, 'coef_'):
            importance = model.coef_[0]
            feature_importance = pd.DataFrame({
                'Feature': required_features,
                'Importance': importance,
                'Value': prediction_features.iloc[0].values
            })
            feature_importance['AbsImportance'] = abs(feature_importance['Importance'])
            feature_importance = feature_importance.sort_values('AbsImportance', ascending=False)
            
            print("Feature Importance:")
            print(feature_importance[['Feature', 'Importance', 'Value']].to_string(index=False))
            print("\n")
        
    except Exception as e:
        simple_logger("error", f"Error making prediction: {e}")
        traceback.print_exc()
        print("\nNo prediction was made due to errors")

if __name__ == '__main__':
    # This is for testing the pipeline script directly
    from microfs.utils import setup_project_dirs, get_project_root, simple_logger
    from microfs.core_api import FeatureStore
    from typing import Dict, Any
    
    # Setup project directories
    setup_project_dirs()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the inference pipeline')
    parser.add_argument('--feature_view', '--fv_name', type=str, default='recommendation_clicks_v1',
                        help='Name of the feature view to use')
    parser.add_argument('--user_id', type=int, default=1,
                        help='User ID for prediction')
    parser.add_argument('--item_id', type=int, default=102,
                        help='Item ID for prediction')
    
    # Run the pipeline
    run() 