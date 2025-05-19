# test_custom_transformers.py
import pytest
import pandas as pd
import numpy as np
from typing import Dict, Any
from microfs.core_api import FeatureStore, CustomTransformer

# Test custom transformer class
def test_custom_transformer_creation():
    """Test creating a custom transformer."""
    # Define simple transformer functions
    def min_max_fit(data: pd.Series) -> Dict[str, Any]:
        return {'min': float(data.min()), 'max': float(data.max())}
        
    def min_max_transform(data: pd.Series, params: Dict[str, Any]) -> pd.Series:
        min_val, max_val = params['min'], params['max']
        if max_val == min_val:
            return pd.Series(0.5, index=data.index)
        return (data - min_val) / (max_val - min_val)
    
    # Create transformer
    custom_transformer = CustomTransformer(
        name="test_minmax",
        fit_fn=min_max_fit,
        transform_fn=min_max_transform
    )
    
    # Test properties
    assert custom_transformer.name == "test_minmax"
    assert callable(custom_transformer.fit_fn)
    assert callable(custom_transformer.transform_fn)
    
    # Test fit and transform
    test_data = pd.Series([1, 2, 3, 4, 5])
    params = custom_transformer.fit(test_data)
    
    assert params == {'min': 1.0, 'max': 5.0}
    
    transformed = custom_transformer.transform(test_data)
    expected = pd.Series([0.0, 0.25, 0.5, 0.75, 1.0])
    pd.testing.assert_series_equal(transformed, expected)
    
    # Test fit_transform
    transformed = custom_transformer.fit_transform(test_data)
    pd.testing.assert_series_equal(transformed, expected)

# Test registration with feature store
def test_transformer_registration(fs_instance):
    """Test registering custom transformers with the feature store."""
    # Define a simple transformer
    def log_fit(data: pd.Series) -> Dict[str, Any]:
        min_val = float(data.min())
        offset = abs(min_val) + 1.0 if min_val <= 0 else 0.0
        return {'offset': offset}
        
    def log_transform(data: pd.Series, params: Dict[str, Any]) -> pd.Series:
        offset = params['offset']
        return np.log(data + offset)
    
    # Create and register transformer
    log_transformer = CustomTransformer(
        name="test_log",
        fit_fn=log_fit,
        transform_fn=log_transform
    )
    
    fs_instance.register_custom_transformer(log_transformer)
    
    # Test registration
    transformers = fs_instance.list_custom_transformers()
    assert "test_log" in transformers
    
    # Test retrieval
    retrieved = fs_instance.get_custom_transformer("test_log")
    assert retrieved.name == "test_log"
    assert retrieved == log_transformer

# Test custom transformers in feature views
def test_transformers_in_feature_view(fs_instance, sample_data):
    """Test using custom transformers in a feature view."""
    # Define and register transformers
    def log_fit(data: pd.Series) -> Dict[str, Any]:
        min_val = float(data.min())
        offset = abs(min_val) + 1.0 if min_val <= 0 else 0.0
        return {'offset': offset}
        
    def log_transform(data: pd.Series, params: Dict[str, Any]) -> pd.Series:
        offset = params['offset']
        return np.log(data + offset)
    
    log_transformer = CustomTransformer(
        name="custom_log",
        fit_fn=log_fit,
        transform_fn=log_transform
    )
    
    fs_instance.register_custom_transformer(log_transformer)
    
    # Create feature group
    fg = fs_instance.create_feature_group(
        "test_features",
        ["id"],
        "timestamp",
        {
            'id': 'int64',
            'timestamp': 'datetime64[ns, utc]',
            'value1': 'float64',
            'value2': 'float64',
            'target': 'int64'
        },
        ["id"]
    )
    
    # Create test data with a range of values
    test_df = pd.DataFrame({
        'id': list(range(1, 6)),
        'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']),
        'value1': [10.0, 20.0, 30.0, 40.0, 50.0],
        'value2': [1.0, 2.0, 3.0, 4.0, 5.0],
        'target': [0, 1, 0, 1, 0]
    })
    
    fg.insert(test_df)
    
    # Create feature view with custom transformer
    fv = fs_instance.create_feature_view(
        "test_view",
        "test_features",
        "target",
        "timestamp",
        [],
        [
            {'feature_name': 'value1', 'transform_type': 'custom:custom_log'},
            {'feature_name': 'value2', 'transform_type': 'custom:custom_log'}
        ]
    )
    
    # Get training data and compute parameters
    X_train, y_train = fv.get_training_data(compute_params=True)
    
    # Verify we have data and the transformed columns
    assert not X_train.empty
    assert 'value1' in X_train.columns
    assert 'value2' in X_train.columns
    
    # Check transformation parameters were computed
    transform_params = fv.get_transform_params()
    assert 'value1' in transform_params
    assert 'value2' in transform_params
    assert transform_params['value1']['type'] == 'custom:custom_log'
    assert transform_params['value2']['type'] == 'custom:custom_log'
    
    # Verify transformations were applied correctly
    expected_value1 = np.log(test_df['value1'])
    pd.testing.assert_series_equal(X_train['value1'], expected_value1, check_names=False)
    
    # Test inference
    vector = fv.get_inference_vector({"id": 3})
    assert 'value1' in vector
    assert 'value2' in vector
    
    # Verify transformations in inference vector
    assert vector['value1'] == np.log(30.0)
    assert vector['value2'] == np.log(3.0)

# Test handling of edge cases
def test_transformer_edge_cases(fs_instance):
    """Test handling of edge cases with custom transformers."""
    # Test with empty data
    def edge_fit(data: pd.Series) -> Dict[str, Any]:
        if data.empty:
            return {'default': 0.0}
        return {'value': float(data.mean())}
        
    def edge_transform(data: pd.Series, params: Dict[str, Any]) -> pd.Series:
        if 'default' in params:
            return pd.Series(params['default'], index=data.index)
        return data * params['value']
    
    edge_transformer = CustomTransformer(
        name="edge_case",
        fit_fn=edge_fit,
        transform_fn=edge_transform
    )
    
    # Test with empty series
    empty_series = pd.Series(dtype='float64')
    params = edge_transformer.fit(empty_series)
    assert params == {'default': 0.0}
    
    # Test with constant values - should handle division by zero
    def minmax_fit(data: pd.Series) -> Dict[str, Any]:
        return {'min': float(data.min()), 'max': float(data.max())}
        
    def minmax_transform(data: pd.Series, params: Dict[str, Any]) -> pd.Series:
        min_val, max_val = params['min'], params['max']
        if max_val == min_val:
            return pd.Series(0.5, index=data.index)
        return (data - min_val) / (max_val - min_val)
    
    constant_transformer = CustomTransformer(
        name="constant_handler",
        fit_fn=minmax_fit,
        transform_fn=minmax_transform
    )
    
    constant_series = pd.Series([5, 5, 5, 5])
    params = constant_transformer.fit(constant_series)
    assert params == {'min': 5.0, 'max': 5.0}
    
    transformed = constant_transformer.transform(constant_series)
    assert all(transformed == 0.5) 