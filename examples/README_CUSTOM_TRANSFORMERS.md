# Client-Side Custom Transformers in microFS

This document explains how to implement and use custom model-dependent transformations in the microFS feature store.

## Overview

Custom transformers allow data scientists to define their own model-dependent transformations that can be:

1. Defined on the client side (by data scientists)
2. Registered with the feature store
3. Used consistently across training and inference pipelines

## Implementation Details

The custom transformer functionality is implemented with the following components:

1. **CustomTransformer class**: A wrapper for user-defined transformation functions
2. **Registration mechanism**: Methods to register transformers with the feature store
3. **Integration with existing transformation pipeline**: Ensuring custom transformers work with the existing feature store infrastructure

## How to Use Custom Transformers

### 1. Define a Custom Transformer

Create a new `CustomTransformer` by defining two functions:

- `fit_fn`: Computes transformation parameters from training data
- `transform_fn`: Applies transformation using computed parameters

```python
from microfs.core_api import CustomTransformer

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
```

### 2. Register the Transformer with the Feature Store

```python
from microfs.core_api import FeatureStore

fs = FeatureStore()
fs.register_custom_transformer(custom_minmax_scaler)
```

### 3. Use the Custom Transformer in a Feature View

When creating a feature view, reference the custom transformer with the `custom:` prefix:

```python
feature_view = fs.create_feature_view(
    name="my_feature_view",
    label_fg_name="label_fg",
    label_column="target",
    label_event_time_column="timestamp",
    feature_group_joins=[...],
    declared_transforms=[
        # Use custom transformer
        {"feature_name": "income", "transform_type": "custom:custom_minmax"},
        # Use built-in transformer
        {"feature_name": "age", "transform_type": "standard_scaler"}
    ]
)
```

### 4. Access Transformation Parameters

After training, you can access the computed parameters:

```python
# Get training data (computes parameters)
X_train, y_train = feature_view.get_training_data(compute_params=True)

# Get parameters
transform_params = feature_view.get_transform_params()
```

## Benefits

1. **Flexibility**: Data scientists can implement any transformation logic they need
2. **Consistency**: The same transformations are applied during training and inference
3. **Integration**: Custom transformers work alongside built-in transformations
4. **Persistence**: Transformation parameters are stored in the feature store

## Example

See the complete working example in `examples/custom_transformers_example.py`.

## Best Practices

1. Make sure your transformers are deterministic and stateless (except for the computed parameters)
2. Ensure your transform functions handle edge cases (nulls, zeros, etc.)
3. Give your transformers meaningful names to make feature view definitions clear
4. Document your transformers' behavior for team knowledge sharing
5. Test your transformers with a variety of input data before using in production 