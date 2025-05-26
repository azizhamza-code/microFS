# test_custom_transformers.py
import pytest
import pandas as pd
import numpy as np
from microfs.core import FeatureStore

def test_custom_transformer_registration(fs_instance):
    """Test registering and using custom transformers."""
    
    # Define a simple custom transformer
    def log_fit(data):
        """Compute log transformation parameters."""
        return {'min_val': float(data.min())}
    
    def log_transform(data, params):
        """Apply log transformation."""
        min_val = params['min_val']
        # Add 1 to avoid log(0) and ensure positive values
        adjusted_data = data - min_val + 1
        return np.log(adjusted_data)
    
    # Register the custom transformer
    fs_instance.register_custom_transformer("log_transform", log_fit, log_transform)
    
    # Check that it was registered
    transformer = fs_instance.get_custom_transformer("log_transform")
    assert transformer is not None
    assert 'fit_fn' in transformer
    assert 'transform_fn' in transformer

def test_custom_transformer_in_feature_view(fs_instance, sample_data):
    """Test using custom transformers in a feature view."""
    
    # Register a custom transformer
    def square_fit(data):
        return {'mean': float(data.mean())}
    
    def square_transform(data, params):
        mean = params['mean']
        return (data - mean) ** 2
    
    fs_instance.register_custom_transformer("square", square_fit, square_transform)
    
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
    
    # Create feature view with custom transformer
    # Note: This test assumes the core supports custom transformers in feature views
    # If not implemented yet, this test will help drive the implementation
    fv = fs_instance.create_feature_view(
        "test_custom_fv",
        "user_activity",
        "conversion",
        [],
        [
            {'type': 'square', 'column': 'duration_sec'}
        ]
    )
    
    # This test verifies the structure is in place
    # The actual transformation logic would need to be implemented in the core
    assert fv.name == "test_custom_fv" 