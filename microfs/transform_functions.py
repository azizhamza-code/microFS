# transform_functions.py
# Functions for data transformations and parameter computations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union


# --- Transformation Functions ---

def _scale_numerical_impl(df: pd.DataFrame, column_name: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Scale a numerical column using the provided parameters.
    
    Args:
        df: DataFrame containing the column to scale
        column_name: Name of the column to scale
        params: Dictionary containing 'mean' and 'std' parameters
    
    Returns:
        DataFrame with the scaled column
    """
    if column_name not in df.columns:
        return df
    
    mean = params.get('mean')
    std = params.get('std')
    
    if mean is None or std is None:
        return df
    
    if std == 0:  # Handle zero standard deviation
        df[column_name] = 0.0
    else:
        df[column_name] = (df[column_name] - mean) / std
    
    return df


def _encode_categorical_impl(df: pd.DataFrame, column_name: str, params: Dict[str, Any]) -> pd.DataFrame:
    """
    One-hot encode a categorical column using the provided parameters.
    
    Args:
        df: DataFrame containing the column to encode
        column_name: Name of the column to encode
        params: Dictionary containing 'categories' parameter
    
    Returns:
        DataFrame with the one-hot encoded columns
    """
    if column_name not in df.columns:
        return df
    
    categories = params.get('categories', [])
    if not categories:
        return df
    
    # For each category, create a binary column
    for category in categories:
        col_name = f"{column_name}_{category}"
        df[col_name] = (df[column_name] == category).astype(int)
    
    # Keep the original column as well
    return df


# --- Parameter Computation Functions ---

def _compute_scale_params_impl(series: pd.Series) -> Dict[str, Any]:
    """
    Compute scaling parameters (mean and standard deviation) for a numerical series.
    
    Args:
        series: Series to compute parameters for
    
    Returns:
        Dictionary containing 'mean' and 'std' parameters
    """
    if series.empty:
        return {'mean': 0.0, 'std': 1.0}
    
    # Handle non-numeric data gracefully
    try:
        series_numeric = pd.to_numeric(series, errors='coerce')
        series_clean = series_numeric.dropna()
        
        if series_clean.empty:
            return {'mean': 0.0, 'std': 1.0}
        
        mean = float(series_clean.mean())
        std = float(series_clean.std())
        
        # Handle zero std with a small epsilon
        if std == 0:
            std = 1.0
        
        return {'mean': mean, 'std': std}
    except Exception as e:
        # Return default parameters if computation fails
        return {'mean': 0.0, 'std': 1.0}


def _compute_one_hot_encode_params_impl(series: pd.Series) -> Dict[str, Any]:
    """
    Compute one-hot encoding parameters (list of categories) for a categorical series.
    
    Args:
        series: Series to compute parameters for
    
    Returns:
        Dictionary containing 'categories' parameter
    """
    if series.empty:
        return {'categories': []}
    
    try:
        # Get unique values, handling nulls
        unique_values = series.dropna().unique().tolist()
        
        # Convert numpy types to native Python types for JSON serialization
        categories = []
        for val in unique_values:
            if isinstance(val, (np.int64, np.int32, np.int16, np.int8)):
                categories.append(int(val))
            elif isinstance(val, (np.float64, np.float32)):
                categories.append(float(val))
            else:
                categories.append(str(val))
        
        return {'categories': categories}
    except Exception as e:
        # Return empty list if computation fails
        return {'categories': []}


# --- Function Registry ---

# Dictionary mapping transformation types to their implementation functions
TRANSFORMATION_FUNCTIONS = {
    "scale": _scale_numerical_impl,
    "one_hot_encode": _encode_categorical_impl,
}

# Dictionary mapping parameter computation types to their implementation functions
PARAMETER_COMPUTATION_FUNCTIONS = {
    "scale": _compute_scale_params_impl,
    "one_hot_encode": _compute_one_hot_encode_params_impl,
}


# --- Registration Function ---

def register_transformation_functions(state_manager):
    """
    Register all transformation functions with the state manager.
    
    Args:
        state_manager: State manager instance to register functions with
    """
    for name, func in TRANSFORMATION_FUNCTIONS.items():
        state_manager.register_transformation(name, func)
    
    for name, func in PARAMETER_COMPUTATION_FUNCTIONS.items():
        state_manager.register_parameter_computation(name, func) 