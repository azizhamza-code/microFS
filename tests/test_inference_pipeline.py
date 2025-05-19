# test_inference_pipeline.py 
import pytest
import pandas as pd
import os
import pickle
from microfs.core_api import FeatureStore
from sklearn.linear_model import LogisticRegression

# Sample data is now provided by conftest.py

def test_inference_workflow(fs_instance, sample_data, tmp_path):
    """Test the end-to-end workflow of creating a model and using it for inference."""
    # Create feature groups
    fg_ua = fs_instance.create_feature_group(
        "user_activity",
        ["user_id", "item_id"],
        "timestamp",
        {
            'user_id': 'int64',
            'item_id': 'int64',
            'timestamp': 'datetime64[ns, utc]',
            'duration_sec': 'float64',
            'activity_type': 'object',
            'conversion': 'int64'
        },
        ["user_id", "item_id"]
    )
    fg_ua.insert(sample_data['user_activity'])
    
    # Create Feature View - just use duration_sec and no joins
    feature_view_name = "conversion_prediction"
    
    fv = fs_instance.create_feature_view(
        feature_view_name,
        "user_activity",  # label_fg
        "conversion",     # label_column
        "timestamp",      # label_event_time_column
        [],               # no joins for this simple test
        [
            {'feature_name': 'duration_sec', 'transform_type': 'scale'}
        ]
    )
    
    # Get training data with computed parameters
    X_train, y_train = fv.get_training_data(compute_params=True)
    
    # Verify we have training data
    assert not X_train.empty
    assert 'duration_sec' in X_train.columns
    
    # Verify we've computed parameters
    transform_params = fv.get_transform_params()
    assert 'duration_sec' in transform_params 