# microFS: A Simplified Feature Store

microFS is a minimalist, educational implementation of a feature store designed to demonstrate core concepts like data management, transformations, and point-in-time correctness.

## Core Idea of a Feature Store

A feature store is a specialized data system that:

1. **Stores and manages features** for machine learning models in a centralized repository
2. **Ensures consistent feature transformations** across training and inference
3. **Maintains point-in-time correctness** to prevent data leakage
4. **Serves features** for both offline training and online inference

microFS implements these core ideas in a simplified way, making it ideal for learning and experimentation.

## Project Structure

```
microfs/
├── microfs/            # Core library code
│   ├── core_api.py     # Public API classes
│   ├── internal_logic.py # Low-level implementation details
│   ├── transform_functions.py # Feature transformations
│   └── utils.py        # Utility functions
├── pipelines/          # Example ML workflows
│   ├── feature_pipeline.py   # Data engineering pipeline
│   ├── training_pipeline.py  # Model training pipeline
│   └── inference_pipeline.py # Model inference pipeline
├── tests/              # Automated tests
├── data/               # Data directory
│   ├── raw_data/       # Input CSV files
│   ├── fs_state/       # Feature store state storage
│   └── models/         # Trained model storage
├── pyproject.toml      # Poetry dependency management
├── README.md           # This file
└── Makefile            # Convenience commands
```

## Setup

### Requirements

- Python 3.8 or newer
- Poetry (dependency management)

### Installation

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
make setup
# or
poetry install
```

## Using microFS

### 1. Feature Pipeline (Data Engineering)

Ingest data from CSV files into feature groups:

```bash
make run-fp
# or
python pipelines/feature_pipeline.py
```

This pipeline:
- Creates feature groups for user activity, user profiles, and item features
- Ingests data from CSV files in batches
- Updates both offline (Parquet) and online (in-memory) stores

### 2. Training Pipeline (Data Science)

Create feature views, prepare training data, and train a model:

```bash
make run-train
# or
python pipelines/training_pipeline.py
```

This pipeline:
- Creates a feature view that joins multiple feature groups
- Computes and stores transformation parameters
- Applies transformations to features
- Trains a simple logistic regression model
- Saves the model for later use

### 3. Inference Pipeline (Serving)

Get features for inference and make predictions:

```bash
make run-infer
# or
python pipelines/inference_pipeline.py
```

This pipeline:
- Loads a trained model
- Gets feature values for a specific entity from the online store
- Applies the same transformations used during training
- Makes a prediction using the model

### Run All Pipelines

```bash
make run-all-pipelines
```

## Running Tests

```bash
make test
# or
pytest tests/
```

## Cleaning Up

```bash
make clean
```

## Key Concepts Demonstrated

1. **Feature Group**: Collection of related features with a common schema and keys
   - Example: `user_activity`, `user_profile`, `item_feature`

2. **Feature View**: Definition of features joined from multiple feature groups, with transformations
   - Example: `recommendation_clicks_v1`

3. **Point-in-Time Correctness**: Ensuring that training data doesn't have "future" information
   - Implemented via timestamp-based joins

4. **Feature Transformations**: Consistent application of transformations in training and inference
   - Scaling, one-hot encoding, etc.

5. **Online/Offline Storage**: Different storage for training vs. serving
   - Offline: Parquet files for large historical data (training)
   - Online: In-memory key-value store for latest values (inference)

## Development Notes

This project is intended for educational purposes and is not production-ready. Key simplifications include:

- In-memory online store (rather than Redis/DynamoDB/etc.)
- Local file-based offline store (rather than a data lake/warehouse)
- Limited transformation options
- No distributed computing support