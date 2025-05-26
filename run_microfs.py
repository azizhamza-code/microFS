#!/usr/bin/env python3
"""
MicroFS Demo - Educational Feature Store Implementation

This script demonstrates how a feature store works by running:
1. Feature Pipeline - Creates feature groups and ingests data
2. Training Pipeline - Creates feature views and trains ML models  
3. Inference Pipeline - Serves features for real-time predictions

This is an educational MVP to show core feature store concepts.
"""

import os
from microfs.core import FeatureStore
from microfs.utils import setup_project_dirs, get_project_root, simple_logger

# Import pipeline functions
from pipelines.feature_pipeline import run_feature_pipeline
from pipelines.training_pipeline import run_training_pipeline
from pipelines.inference_pipeline import run_inference_pipeline

def main():
    """Run the complete MicroFS demo"""
    
    # Setup
    setup_project_dirs()
    raw_data_dir = get_project_root() / "data" / "raw_data"
    
    simple_logger("info", f"Using raw data from: {raw_data_dir}")
    
    # Initialize feature store
    fs = FeatureStore()
    
    # Reset state for clean demo
    fs.reset_all_state()
    
    # 1. Feature Pipeline - Create feature groups and ingest data
    run_feature_pipeline(fs, str(raw_data_dir))
    
    print("\n" + "="*60 + "\n")
    
    # 2. Training Pipeline - Create feature views and train models
    run_training_pipeline(fs, "recommendation_clicks_v1")
    
    print("\n" + "="*60 + "\n")
    
    # 3. Inference Pipeline - Serve features for predictions
    test_entities = [
        {'user_id': 1, 'item_id': 102},
        {'user_id': 2, 'item_id': 103}, 
        {'user_id': 1, 'item_id': 104},  # Missing some features
        {'user_id': 999, 'item_id': 101}  # Missing user data
    ]
    
    for entity_keys in test_entities:
        run_inference_pipeline(fs, "recommendation_clicks_v1", entity_keys)
    
    print("\n" + "="*60 + "\n")
    
    simple_logger("info", "MicroFS Demo Complete.")
    simple_logger("info", f"Feature Store state persisted in: {get_project_root() / 'data' / 'fs_state'}")

if __name__ == "__main__":
    main() 