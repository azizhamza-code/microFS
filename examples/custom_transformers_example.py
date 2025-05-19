# custom_transformers_example.py
# Example of defining and using client-side custom transformers with microFS

import pandas as pd
import numpy as np
from typing import Dict, Any
import sys
import os

# Add the parent directory to the path so we can import microfs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from microfs.core_api import FeatureStore, FeatureGroup, FeatureView, CustomTransformer

# Initialize the feature store
fs = FeatureStore()

# Optional: Reset the state for the demo
fs.reset_all_state_FOR_DEMO_ONLY()

# Step 1: Define custom transformers

# Example 1: Custom Min-Max Scaler
def minmax_fit(data: pd.Series) -> Dict[str, Any]:
    """Compute min and max values for scaling."""
    return {
        'min': float(data.min()),
        'max': float(data.max())
    }

def minmax_transform(data: pd.Series, params: Dict[str, Any]) -> pd.Series:
    """Apply min-max scaling using precomputed parameters."""
    min_val = params['min']
    max_val = params['max']
    if max_val == min_val:  # Handle constant features
        return pd.Series(0.5, index=data.index)
    return (data - min_val) / (max_val - min_val)

# Create the transformer
custom_minmax_scaler = CustomTransformer(
    name="custom_minmax",
    fit_fn=minmax_fit,
    transform_fn=minmax_transform
)

# Example 2: Custom Log Transform with handling for zeros and negatives
def log_fit(data: pd.Series) -> Dict[str, Any]:
    """Compute offset for log transform to handle zeros/negatives."""
    min_val = float(data.min())
    offset = abs(min_val) + 1.0 if min_val <= 0 else 0.0
    return {'offset': offset}

def log_transform(data: pd.Series, params: Dict[str, Any]) -> pd.Series:
    """Apply log transform with offset."""
    offset = params['offset']
    return np.log(data + offset)

# Create the transformer
custom_log_transformer = CustomTransformer(
    name="custom_log",
    fit_fn=log_fit,
    transform_fn=log_transform
)

# Step 2: Register the custom transformers with the feature store
fs.register_custom_transformer(custom_minmax_scaler)
fs.register_custom_transformer(custom_log_transformer)

# Step 3: Create feature groups with example data

# User Features
user_data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'age': [25, 40, 30, 55, 22],
    'income': [50000, 90000, 70000, 120000, 45000],
    'credit_score': [680, 720, 700, 800, 650],
    'timestamp': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
})

user_fg = fs.create_feature_group(
    name="user_features",
    primary_key_columns=["user_id"],
    event_time_column="timestamp",
    schema={
        "user_id": "int64",
        "age": "int64",
        "income": "float64",
        "credit_score": "float64",
        "timestamp": "datetime64[ns]"
    },
    online_key_columns=["user_id"]
)

user_fg.insert(user_data)

# Transaction Features (label source)
transaction_data = pd.DataFrame({
    'user_id': [1, 2, 3, 4, 5],
    'transaction_amount': [100, 500, 250, 1000, 50],
    'is_fraud': [0, 1, 0, 0, 1],  # Our label
    'timestamp': pd.to_datetime(['2023-01-10', '2023-01-11', '2023-01-12', '2023-01-13', '2023-01-14'])
})

transaction_fg = fs.create_feature_group(
    name="transaction_features",
    primary_key_columns=["user_id"],
    event_time_column="timestamp",
    schema={
        "user_id": "int64",
        "transaction_amount": "float64",
        "is_fraud": "int64",
        "timestamp": "datetime64[ns]"
    },
    online_key_columns=["user_id"]
)

transaction_fg.insert(transaction_data)

# Step 4: Create a feature view using custom transformers
fraud_detection_view = fs.create_feature_view(
    name="fraud_detection",
    label_fg_name="transaction_features",
    label_column="is_fraud",
    label_event_time_column="timestamp",
    feature_group_joins=[
        {
            "name": "user_features",
            "on": ["user_id"],
            "prefix": "user_"
        }
    ],
    declared_transforms=[
        # Use custom transformers with the "custom:" prefix
        {"feature_name": "user_income", "transform_type": "custom:custom_log"},
        {"feature_name": "user_age", "transform_type": "custom:custom_minmax"},
        {"feature_name": "user_credit_score", "transform_type": "custom:custom_minmax"},
        {"feature_name": "transaction_amount", "transform_type": "custom:custom_log"}
    ]
)

# Step 5: Retrieve training data (this will compute transformation parameters)
print("Getting training data with computed parameters...")
X_train, y_train = fraud_detection_view.get_training_data(compute_params=True)
print("\nTraining features:")
print(X_train)
print("\nTraining labels:")
print(y_train)

# Step 6: Inspect computed transform parameters
transform_params = fraud_detection_view.get_transform_params()
print("\nComputed transform parameters:")
for feature, params in transform_params.items():
    print(f"{feature}: {params}")

# Step 7: Get an inference vector for a real-time prediction
print("\nGetting inference vector for user_id=3...")
inference_vector = fraud_detection_view.get_inference_vector({"user_id": 3})
print(inference_vector)

# List all available custom transformers
print("\nAvailable custom transformers:")
print(fs.list_custom_transformers()) 