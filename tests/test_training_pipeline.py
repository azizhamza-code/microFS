# test_training_pipeline.py 
import pytest
import pandas as pd
import os
from microfs.core_api import FeatureStore
from microfs.transform_functions import TRANSFORMATION_FUNCTIONS, PARAMETER_COMPUTATION_FUNCTIONS

# Sample data is now provided by conftest.py

def test_feature_view_creation(fs_instance, sample_data):
    """Test creating a feature view."""
    # Create feature groups
    # User Activity
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
    
    # Create Feature View
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
    
    # Check feature view was created
    assert feature_view_name in fs_instance.list_feature_views()
    
    # Verify metadata 
    assert fv.name == feature_view_name
    assert fv.label_fg_name == "user_activity"
    assert fv.label_column == "conversion"

def test_get_training_data(fs_instance, sample_data):
    """Test getting training data from a feature view."""
    # Create feature groups
    # User Activity
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
    
    # Create Feature View
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
    
    # Test with computing parameters
    X_train, y_train = fv.get_training_data(compute_params=True)
    
    # Verify basic properties of training data
    assert not X_train.empty
    assert not y_train.empty
    assert 'duration_sec' in X_train.columns  # Transformed feature is included
    assert X_train.shape[0] == y_train.shape[0]  # Same number of samples
