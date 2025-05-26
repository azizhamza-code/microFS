# conftest.py
# Pytest fixtures 
import pytest
import shutil
import os
from microfs.core import FeatureStore, reset_all_state
from microfs.utils import setup_project_dirs, get_state_dir
import pandas as pd

@pytest.fixture(scope="function")
def fs_instance():
    """
    Provide a clean FeatureStore instance for each test.
    
    This fixture:
    1. Clears any previous feature store state (both in-memory and on disk)
    2. Sets up the project directories
    3. Initializes a fresh FeatureStore
    
    Returns:
        A clean FeatureStore instance
    """
    # Clear any previous state (disk and in-memory)
    state_dir = get_state_dir()
    if state_dir.exists():
        # Remove disk state
        for item in os.listdir(state_dir):
            item_path = state_dir / item
            if os.path.isdir(item_path):
                shutil.rmtree(item_path)
            else:
                os.remove(item_path)
    
    # Recreate clean directories
    setup_project_dirs()
    
    # Clear in-memory global state
    reset_all_state()
    
    # Initialize fresh FeatureStore
    fs = FeatureStore()
    return fs 

@pytest.fixture
def sample_data():
    """Create sample DataFrames for testing."""
    # User Activity
    ua_data = pd.DataFrame({
        'user_id': [1, 1, 2, 2],
        'item_id': [101, 102, 101, 103],
        'timestamp': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 11:00:00', 
                                     '2023-01-01 09:00:00', '2023-01-03 14:00:00'], utc=True),
        'duration_sec': [120.5, 85.2, 45.0, 200.1],
        'activity_type': ['view', 'click', 'view', 'purchase'],
        'conversion': [0, 1, 0, 1]
    })
    
    # User Profile
    up_data = pd.DataFrame({
        'user_id': [1, 2],
        'timestamp': pd.to_datetime(['2023-01-01 09:30:00', '2023-01-01 09:45:00'], utc=True),
        'user_level': ['silver', 'gold'],
        'has_premium_badge': [False, True]
    })
    
    # Item Feature
    if_data = pd.DataFrame({
        'item_id': [101, 102, 103],
        'timestamp': pd.to_datetime(['2023-01-01 09:00:00', '2023-01-01 09:00:00', 
                                    '2023-01-01 09:00:00'], utc=True),
        'item_category': ['electronics', 'clothing', 'books'],
        'price': [149.99, 29.99, 12.99]
    })
    
    return {
        'user_activity': ua_data,
        'user_profile': up_data,
        'item_feature': if_data
    } 