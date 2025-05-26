# test_feature_pipeline.py 
import pytest
import pandas as pd
import os
from microfs.core import FeatureStore
from microfs.utils import get_project_root

# Sample data is now provided by conftest.py

def test_feature_group_creation(fs_instance):
    """Test creating feature groups."""
    # Create a feature group
    fg = fs_instance.create_feature_group(
        "user_activity", 
        ["user_id", "item_id"], 
        "timestamp", 
        {
            'user_id': 'int64',
            'item_id': 'int64',
            'timestamp': 'datetime64[ns, utc]',
            'activity_type': 'object'
        },
        ["user_id", "item_id"]
    )
    
    # Check if the feature group is in the list
    assert "user_activity" in fs_instance.list_feature_groups()
    
    # Check if the feature group has the right properties
    assert fg.name == "user_activity"
    assert fg.primary_key_columns == ["user_id", "item_id"]
    assert fg.event_time_column == "timestamp"

def test_insert_and_retrieve(fs_instance, sample_data):
    """Test inserting data into feature groups and retrieving it."""
    # Create user activity feature group
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
    
    # Insert data
    fg_ua.insert(sample_data['user_activity'])
    
    # Retrieve data
    ua_data = fg_ua.get_offline_data()
    
    # Check if data was inserted correctly
    assert len(ua_data) == 4
    assert ua_data['user_id'].tolist() == [1, 1, 2, 2]
    
    # Check online store
    online_data = fg_ua.get_online_features({"user_id": 1, "item_id": 102})
    assert online_data["conversion"] == 1
    assert online_data["activity_type"] == "click"
    
    # Verify timestamp
    timestamp_from_store = pd.Timestamp(online_data["timestamp"])
    expected_timestamp = pd.Timestamp('2023-01-02 11:00:00+00:00')
    assert timestamp_from_store == expected_timestamp 