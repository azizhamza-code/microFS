# test_inference_pipeline.py 
import pytest
import pandas as pd
import os
import pickle
from microfs.core import FeatureStore
from sklearn.linear_model import LogisticRegression

# Sample data is now provided by conftest.py

def test_inference_vector_generation(fs_instance, sample_data):
    """Test generating inference vectors from a feature view."""
    # Create feature groups and insert data
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
    
    fg_up = fs_instance.create_feature_group(
        "user_profile",
        ["user_id", "timestamp"],
        "timestamp",
        {
            'user_id': 'int64',
            'timestamp': 'datetime64[ns, utc]',
            'user_level': 'object',
            'has_premium_badge': 'bool'
        },
        ["user_id"]
    )
    
    fg_ua.insert(sample_data['user_activity'])
    fg_up.insert(sample_data['user_profile'])
    
    # Create feature view
    fv = fs_instance.create_feature_view(
        "test_fv",
        "user_activity",
        "conversion",
        [
            {'fg_name': 'user_profile', 'on': ['user_id']}
        ],
        [
            {'type': 'scale', 'column': 'duration_sec'}
        ]
    )
    
    # Compute transformation parameters first
    X_train, y_train = fv.get_training_data(compute_params=True)
    
    # Get inference vector
    entity_keys = {'user_id': 1, 'item_id': 102}
    inference_vector = fv.get_inference_vector(entity_keys)
    
    # Check that we got a vector
    assert not inference_vector.empty
    
    # Check that the vector doesn't include the label
    assert "conversion" not in inference_vector.index
    
    # Check that we have some expected features
    assert "user_id" in inference_vector.index
    assert "item_id" in inference_vector.index 