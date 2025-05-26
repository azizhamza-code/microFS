# test_training_pipeline.py 
import pytest
import pandas as pd
import os
from microfs.core import FeatureStore

# Sample data is now provided by conftest.py

def test_feature_view_creation(fs_instance, sample_data):
    """Test creating a feature view."""
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
    
    # Insert data
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
    
    # Check if the feature view is in the list
    assert "test_fv" in fs_instance.list_feature_views()
    
    # Check feature view properties
    assert fv.name == "test_fv"
    assert fv.label_fg_name == "user_activity"
    assert fv.label_column == "conversion"

def test_training_data_generation(fs_instance, sample_data):
    """Test generating training data from a feature view."""
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
    
    # Get training data
    X_train, y_train = fv.get_training_data(compute_params=True)
    
    # Check that we got some data
    assert not X_train.empty
    assert len(y_train) > 0
    
    # Check that labels are correct
    assert y_train.name == "conversion"
    
    # Check that features don't include the label
    assert "conversion" not in X_train.columns
