# MicroFS 

A simple, educational implementation of a feature store to demonstrate core concepts.

## What is a Feature Store?

A feature store is a centralized repository for storing, managing, and serving machine learning features. It provides:

- **Feature Groups**: Collections of related features with consistent schemas
- **Feature Views**: Logical views that join multiple feature groups for ML training
- **Online Store**: Fast feature serving for real-time inference
- **Offline Store**: Historical feature data for training and batch processing
- **Feature Transformations**: Data preprocessing and feature engineering

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Feature Pipeline│    │Training Pipeline│    │Inference Pipeline│
│                 │    │                 │    │                 │
│ • Create FGs    │    │ • Create FVs    │    │ • Get Features  │
│ • Ingest Data   │    │ • Train Models  │    │ • Make Predictions│
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   FeatureStore  │
                    │                 │
                    │ • FeatureGroups │
                    │ • FeatureViews  │
                    │ • Online Store  │
                    │ • Offline Store │
                    └─────────────────┘
```

## Quick Start

1. **Run the complete demo:**
   ```bash
   python run_microfs.py
   ```

2. **Run individual pipelines:**
   ```bash
   # Feature pipeline
   python pipelines/feature_pipeline.py
   
   # Training pipeline  
   python pipelines/training_pipeline.py
   
   # Inference pipeline
   python pipelines/inference_pipeline.py --user_id 1 --item_id 102
   ```

3. **Run tests:**
   ```bash
   python -m pytest tests/ -v
   ```

## Core Components

### Feature Groups
Store raw features with schemas:
```python
fs = FeatureStore()
fg = fs.create_feature_group(
    name="user_activity",
    primary_keys=["user_id", "item_id"],
    event_time="timestamp",
    schema={
        'user_id': 'int64',
        'item_id': 'int64', 
        'timestamp': 'datetime64[ns, utc]',
        'duration_sec': 'float64',
        'conversion': 'int64'
    },
    online_keys=["user_id", "item_id"]
)
```

### Feature Views
Join feature groups and apply transformations:
```python
fv = fs.create_feature_view(
    name="recommendation_model_v1",
    label_fg="user_activity",
    label_col="conversion",
    joins=[
        {'fg_name': 'user_profile', 'on': ['user_id']},
        {'fg_name': 'item_feature', 'on': ['item_id']}
    ],
    transforms=[
        {'type': 'scale', 'column': 'duration_sec'},
        {'type': 'one_hot_encode', 'column': 'user_level'}
    ]
)
```

### Training & Inference
```python
# Training
X_train, y_train = fv.get_training_data(compute_params=True)

# Inference  
features = fv.get_inference_vector({'user_id': 1, 'item_id': 102})
```

## Project Structure

```
microfs/
├── microfs/
│   ├── core.py          # Core feature store logic
│   └── utils.py         # Utilities and state management
├── pipelines/
│   ├── feature_pipeline.py   # Data ingestion
│   ├── training_pipeline.py  # Model training
│   └── inference_pipeline.py # Real-time serving
├── tests/               # Unit tests
├── data/
│   ├── raw_data/       # Input CSV files
│   └── fs_state/       # Feature store state
├── models/             # Trained ML models
└── run_microfs.py      # Main demo script
```

## Educational Goals

This implementation demonstrates:

1. **Separation of Concerns**: Clear separation between data engineering (feature pipeline), data science (training pipeline), and ML engineering (inference pipeline)

2. **Feature Reusability**: Features defined once can be used across multiple models and use cases

3. **Online/Offline Consistency**: Same feature definitions used for both training (offline) and serving (online)

4. **Schema Management**: Enforced schemas ensure data quality and consistency

5. **Feature Transformations**: Centralized feature engineering with parameter persistence

## Limitations

This is an educational MVP with intentional simplifications:

- No distributed computing or scalability features
- Simple file-based storage (not production databases)
- Basic transformation types only
- No feature versioning or lineage tracking
- No advanced monitoring or data quality checks

For production use cases, consider mature solutions like Hopsworks.