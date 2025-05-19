# test_feature_pipeline.py 
import pytest
import pandas as pd
import os
from microfs.core_api import FeatureStore
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
    fg_meta = fg.get_metadata()
    assert fg_meta['name'] == "user_activity"
    assert fg_meta['primary_key_columns'] == ["user_id", "item_id"]
    assert fg_meta['event_time_column'] == "timestamp"

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
    online_store = fs_instance._state.online_store
    
    # Verify data in online store
    assert (1, 102) in online_store["user_activity"]
    assert online_store["user_activity"][(1, 102)]["conversion"] == 1
    
    # The latest data should have timestamp 2023-01-04
    timestamp_from_store = pd.Timestamp(online_store["user_activity"][(1, 102)]["timestamp"])
    expected_timestamp = pd.Timestamp('2023-01-02 11:00:00+00:00')
    
    # Compare the timestamps (including time parts)
    assert timestamp_from_store == expected_timestamp
    
    # Verify features can be selected
    online_data = fg_ua.get_online_features({"user_id": 1, "item_id": 102})
    assert online_data["conversion"] == 1
    assert online_data["activity_type"] == "click" 