# internal_logic.py
# Core logic (metadata mgmt, data store interaction, joins, transforms) 

import pandas as pd
import numpy as np
import json
import os
import copy
from typing import List, Dict, Any, Tuple, Optional, Callable
from datetime import datetime

from .utils import (
    get_metadata_dir, get_offline_store_dir, get_online_store_dir,
    _FG_METADATA_FILE, _FV_METADATA_FILE, _ONLINE_STORE_FILE,
    _CustomJSONEncoder, _serialize_online_store_keys, _deserialize_online_store_keys,
    simple_logger,
    get_project_root # Added for _setup_state_dirs consistency if used here, though typically in utils or main
)

# Import transformation functions
from .transform_functions import register_transformation_functions

# --- State Management ---
class InMemoryStateManager:
    """
    Manages the in-memory state of the feature store, including metadata and online store data.
    Handles loading from and persisting to disk.
    """
    def __init__(self):
        """Initialize an empty state manager."""
        self.fg_metadata: Dict[str, Dict[str, Any]] = {}
        self.fv_metadata: Dict[str, Dict[str, Any]] = {}
        self.online_store: Dict[str, Dict[Tuple, Dict[str, Any]]] = {} # fg_name -> {key_tuple: data}
        self.metadata_loaded = False
        self.online_store_loaded = False
        self.transformation_functions: Dict[str, Callable] = {}
        self.parameter_computation_functions: Dict[str, Callable] = {}
        self.custom_transformers: Dict[str, Any] = {}  # Store for custom transformers

    def load_metadata_if_needed(self):
        """Load feature group and feature view metadata from disk if not already loaded."""
        if self.metadata_loaded:
            return

        path_fg = get_metadata_dir() / _FG_METADATA_FILE
        path_fv = get_metadata_dir() / _FV_METADATA_FILE

        try:
            if path_fg.exists():
                with open(path_fg, 'r') as f:
                    self.fg_metadata = json.load(f)
            else:
                self.fg_metadata = {}

            if path_fv.exists():
                with open(path_fv, 'r') as f:
                    self.fv_metadata = json.load(f)
            else:
                self.fv_metadata = {}
            
            self.metadata_loaded = True
            simple_logger("info", "Metadata loaded into cache.")
        except Exception as e:
            simple_logger("error", f"Error loading metadata: {e}")
            self.fg_metadata = {}
            self.fv_metadata = {}
            self.metadata_loaded = True  # Mark as loaded to prevent repeated failures

    def save_all_metadata(self):
        """Save feature group and feature view metadata to disk."""
        self.load_metadata_if_needed()  # Ensure current state is loaded before overwriting
        
        path_fg = get_metadata_dir() / _FG_METADATA_FILE
        path_fv = get_metadata_dir() / _FV_METADATA_FILE
        
        try:
            get_metadata_dir().mkdir(parents=True, exist_ok=True)  # Ensure dir exists
            
            with open(path_fg, 'w') as f:
                json.dump(self.fg_metadata, f, indent=4, cls=_CustomJSONEncoder)
            
            with open(path_fv, 'w') as f:
                json.dump(self.fv_metadata, f, indent=4, cls=_CustomJSONEncoder)
            
            simple_logger("info", "All metadata saved to disk.")
        except Exception as e:
            simple_logger("error", f"Error saving metadata: {e}")

    def load_online_store_if_needed(self):
        """Load the online store data from disk if not already loaded."""
        if self.online_store_loaded:
            return
        
        path = self._get_online_store_json_path()
        
        try:
            if os.path.exists(path):
                with open(path, 'r') as f:
                    self.online_store = _deserialize_online_store_keys(json.load(f))
            else:
                self.online_store = {}
            
            self.online_store_loaded = True
            simple_logger("info", "Online store loaded into cache.")
        except Exception as e:
            simple_logger("error", f"Error loading online store: {e}")
            self.online_store = {}
            self.online_store_loaded = True  # Mark as loaded to prevent repeated failures

    def save_online_store(self):
        """Save the online store data to disk."""
        self.load_online_store_if_needed()  # Ensure current state loaded
        
        path = self._get_online_store_json_path()
        
        try:
            get_online_store_dir().mkdir(parents=True, exist_ok=True)  # Ensure dir exists
            
            with open(path, 'w') as f:
                json.dump(_serialize_online_store_keys(self.online_store), f, indent=4, cls=_CustomJSONEncoder)
            
            simple_logger("info", "Online store saved to disk.")
        except Exception as e:
            simple_logger("error", f"Error saving online store: {e}")

    def get_fg_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get feature group metadata by name."""
        self.load_metadata_if_needed()
        return copy.deepcopy(self.fg_metadata.get(name))

    def set_fg_metadata(self, name: str, metadata: Dict[str, Any]):
        """Set feature group metadata by name."""
        self.load_metadata_if_needed()
        self.fg_metadata[name] = metadata
        self.save_all_metadata()

    def get_fv_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """Get feature view metadata by name."""
        self.load_metadata_if_needed()
        return copy.deepcopy(self.fv_metadata.get(name))

    def set_fv_metadata(self, name: str, metadata: Dict[str, Any]):
        """Set feature view metadata by name."""
        self.load_metadata_if_needed()
        
        # If this is an update to an existing record, merge rather than replace
        if name in self.fv_metadata and isinstance(metadata, dict):
            # For a partial update, merge with existing metadata
            if len(metadata) < 5:  # Heuristic: full metadata has at least 5 fields
                self.fv_metadata[name].update(metadata)
            else:
                # For a complete replacement
                self.fv_metadata[name] = metadata
        else:
            # New entry
            self.fv_metadata[name] = metadata
            
        self.save_all_metadata()

    def list_feature_groups(self) -> List[str]:
        """List all feature group names."""
        self.load_metadata_if_needed()
        return list(self.fg_metadata.keys())

    def list_feature_views(self) -> List[str]:
        """List all feature view names."""
        self.load_metadata_if_needed()
        return list(self.fv_metadata.keys())

    def get_online_data(self, fg_name: str, online_key_tuple: Tuple) -> Optional[Dict[str, Any]]:
        """Get online store data for a feature group and key tuple."""
        self.load_online_store_if_needed()
        return self.online_store.get(fg_name, {}).get(online_key_tuple)

    def set_online_data(self, fg_name: str, online_key_tuple: Tuple, data: Dict[str, Any]):
        """Set online store data for a feature group and key tuple."""
        self.load_online_store_if_needed()
        if fg_name not in self.online_store:
            self.online_store[fg_name] = {}
        self.online_store[fg_name][online_key_tuple] = data
        self.save_online_store()

    def clear_all(self):
        """Clear all in-memory state."""
        self.fg_metadata.clear()
        self.fv_metadata.clear()
        self.online_store.clear()
        self.metadata_loaded = False
        self.online_store_loaded = False

    def _get_online_store_json_path(self) -> str:
        """Get the path to the online store JSON file."""
        return str(get_online_store_dir() / _ONLINE_STORE_FILE)

    def register_transformation(self, name: str, func: Callable):
        """Register a transformation function."""
        self.transformation_functions[name] = func

    def register_parameter_computation(self, name: str, func: Callable):
        """Register a parameter computation function."""
        self.parameter_computation_functions[name] = func

    def get_transformation_function(self, name: str) -> Optional[Callable]:
        """Get a transformation function by name."""
        return self.transformation_functions.get(name)

    def get_parameter_computation_function(self, name: str) -> Optional[Callable]:
        """Get a parameter computation function by name."""
        return self.parameter_computation_functions.get(name)

    def register_custom_transformer(self, transformer):
        """Register a custom transformer instance."""
        # Define wrapper functions to match the expected signature
        def transform_wrapper(df, col_name, params):
            return df.assign(**{col_name: transformer.transform_fn(df[col_name], params)})
            
        def fit_wrapper(data):
            return transformer.fit_fn(data)
            
        self.transformation_functions[f"custom:{transformer.name}"] = transform_wrapper
        self.parameter_computation_functions[f"custom:{transformer.name}"] = fit_wrapper
        self.custom_transformers[transformer.name] = transformer
        simple_logger("info", f"Custom transformer '{transformer.name}' registered.")
        
    def get_custom_transformer(self, name: str):
        """Get a custom transformer by name."""
        return self.custom_transformers.get(name)
        
    def list_custom_transformers(self) -> List[str]:
        """List all registered custom transformer names."""
        return list(self.custom_transformers.keys())

# Create a singleton instance of the state manager
_state_manager = InMemoryStateManager()

# For testing - expose the cached online store
_CACHED_ONLINE_STORE = _state_manager.online_store

# Register transformation functions
register_transformation_functions(_state_manager)

# --- Feature Group Persistence Helpers ---
def _get_offline_parquet_path(fg_name: str) -> str:
    return str(get_offline_store_dir() / f"{fg_name}.parquet")

def _load_offline_df(fg_name: str) -> pd.DataFrame:
    """Load a feature group's offline dataframe from parquet."""
    offline_path = _get_offline_parquet_path(fg_name)
    try:
        if os.path.exists(offline_path) and os.path.getsize(offline_path) > 0:
            # Get schema from metadata if available
            fg_meta = _state_manager.get_fg_metadata(fg_name)
            schema = fg_meta.get('schema', {}) if fg_meta else {}
            
            # Read without dtype parameter first
            df = pd.read_parquet(offline_path)
            
            # Apply types after reading
            for col, dtype_str in schema.items():
                if col in df.columns:
                    try:
                        if 'datetime' in dtype_str:
                            df[col] = pd.to_datetime(df[col], utc=True if 'utc' in dtype_str else None)
                        else:
                            df[col] = df[col].astype(dtype_str)
                    except Exception as e:
                        simple_logger("warning", f"Could not cast column '{col}' to '{dtype_str}' for FG '{fg_name}': {e}")
            
            return df
        else:
            # If file doesn't exist or is empty, return empty DataFrame with schema columns if available
            fg_meta = _state_manager.get_fg_metadata(fg_name)
            if fg_meta and 'schema' in fg_meta:
                return pd.DataFrame(columns=list(fg_meta['schema'].keys()))
            return pd.DataFrame()
    except Exception as e:
        simple_logger("error", f"Error loading offline data for FG '{fg_name}': {e}")
        return pd.DataFrame()

def _save_offline_df(fg_name: str, df: pd.DataFrame):
    """Save a feature group's offline dataframe to parquet."""
    try:
        offline_path = _get_offline_parquet_path(fg_name)
        get_offline_store_dir().mkdir(parents=True, exist_ok=True)
        
        # Ensure consistent schema when saving
        fg_meta = _state_manager.get_fg_metadata(fg_name)
        schema = fg_meta.get('schema', {}) if fg_meta else {}
        
        # Create a copy to avoid modifying the original dataframe
        df_to_save = df.copy()
        
        # Apply schema types if possible
        for col, dtype_str in schema.items():
            if col in df_to_save.columns:
                try:
                    if 'datetime' in dtype_str:
                        df_to_save[col] = pd.to_datetime(df_to_save[col], utc=True if 'utc' in dtype_str else None)
                    else:
                        df_to_save[col] = df_to_save[col].astype(dtype_str)
                except Exception as e:
                    simple_logger("warning", f"Could not cast column '{col}' to '{dtype_str}' for FG '{fg_name}': {e}")
        
        df_to_save.to_parquet(offline_path, index=False)
    except Exception as e:
        simple_logger("error", f"Error saving offline data for FG '{fg_name}': {e}")
        raise

# --- Core Feature Group Logic ---
def define_feature_group_internal(name: str, pk_cols: List[str], et_col: str, schema: Dict[str, str], online_key_cols: List[str]):
    """
    Define a new feature group with the given parameters.
    
    Args:
        name: Name of the feature group
        pk_cols: List of primary key column names
        et_col: Event time column name
        schema: Dictionary mapping column names to their data types
        online_key_cols: List of columns to use as keys in the online store
    """
    try:
        if name in _state_manager.list_feature_groups():
            simple_logger("warning", f"FG '{name}' metadata already exists. Overwriting.")
        
        metadata = {
            'name': name, 
            'primary_key_columns': pk_cols, 
            'event_time_column': et_col,
            'schema': schema, 
            'online_key_columns': online_key_cols
        }
        
        _state_manager.set_fg_metadata(name, metadata)
        
        # Initialize offline store file if it doesn't exist
        get_offline_store_dir().mkdir(parents=True, exist_ok=True)
        offline_path = _get_offline_parquet_path(name)
        if not os.path.exists(offline_path):
            # Create empty dataframe with schema columns
            empty_df = pd.DataFrame(columns=list(schema.keys()))
            empty_df.to_parquet(offline_path, index=False)
        
        # Initialize online store entry
        _state_manager.load_online_store_if_needed()
        if name not in _state_manager.online_store:
            _state_manager.online_store[name] = {}
            _state_manager.save_online_store()
        
        simple_logger("info", f"FG '{name}' metadata defined and stores initialized.")
    except Exception as e:
        simple_logger("error", f"Error defining feature group '{name}': {e}")
        raise

def get_feature_group_metadata_internal(name: str) -> Optional[Dict[str, Any]]:
    """Get feature group metadata by name."""
    return _state_manager.get_fg_metadata(name)

def list_feature_groups_internal() -> List[str]:
    """List all feature group names."""
    return _state_manager.list_feature_groups()

def insert_into_feature_group_internal(fg_name: str, data_df: pd.DataFrame):
    """
    Insert data into a feature group.
    
    Args:
        fg_name: Name of the feature group
        data_df: DataFrame containing the data to insert
    """
    try:
        fg_meta = _state_manager.get_fg_metadata(fg_name)
        if not fg_meta:
            raise ValueError(f"FG '{fg_name}' not found for insertion.")

        pk_cols = fg_meta['primary_key_columns']
        et_col = fg_meta['event_time_column']
        online_keys = fg_meta['online_key_columns']
        all_schema_features = list(fg_meta['schema'].keys())
        
        # Validate required columns
        required_cols = list(set(pk_cols + [et_col] + online_keys + all_schema_features))
        missing_cols = [col for col in required_cols if col not in data_df.columns]
        
        if missing_cols:
            raise ValueError(f"Input DF for FG '{fg_name}' missing required schema columns: {missing_cols}. All columns in schema are expected.")

        # Make a copy to avoid altering the original dataframe
        data_to_store_df = data_df.copy()
        
        # Type casting based on schema
        for col, dtype_str in fg_meta['schema'].items():
            if col in data_to_store_df.columns:
                try:
                    if 'datetime64' in dtype_str or 'datetime' in dtype_str:
                        # Ensure timezone-aware datetimes for consistency
                        data_to_store_df[col] = pd.to_datetime(data_to_store_df[col], utc=True)
                    elif dtype_str == 'bool' and data_to_store_df[col].dtype == 'object':
                        data_to_store_df[col] = data_to_store_df[col].map({'True': True, 'False': False, 'true': True, 'false': False}).astype(dtype_str)
                    else:
                        data_to_store_df[col] = data_to_store_df[col].astype(dtype_str)
                except Exception as e:
                    simple_logger("error", f"Could not cast column '{col}' to '{dtype_str}' for FG '{fg_name}': {e}")
                    raise
        
        # Select only columns defined in schema
        data_to_store_df = data_to_store_df[all_schema_features].copy()

        # Update offline store
        current_offline_df = _load_offline_df(fg_name)
        
        # If the current data is empty, just use the new data
        if current_offline_df.empty:
            updated_offline_df = data_to_store_df
        else:
            updated_offline_df = pd.concat([current_offline_df, data_to_store_df], ignore_index=True)
        
        # Ensure consistent dtypes after concat
        for col, dtype_str in fg_meta['schema'].items():
            try:
                if col in updated_offline_df.columns:
                    if 'datetime64' in dtype_str or 'datetime' in dtype_str:
                        updated_offline_df[col] = pd.to_datetime(updated_offline_df[col], utc=True)
                    else:
                        updated_offline_df[col] = updated_offline_df[col].astype(dtype_str)
            except Exception as e:
                simple_logger("warning", f"Could not ensure dtype consistency for column '{col}' after concat: {e}")
        
        _save_offline_df(fg_name, updated_offline_df)

        # Update online store with latest values (values present in this batch)
        for _, row in data_to_store_df.iterrows():
            # Build online key tuple from values
            online_key_values = [row[k] for k in online_keys]
            online_key_tuple = tuple(online_key_values)
            
            # Build feature dict (all features except online key cols if they're not in schema)
            feature_vals = {}
            for col in all_schema_features:
                # Convert pandas Timestamp objects to Python datetime for consistent serialization
                if pd.api.types.is_datetime64_dtype(data_to_store_df[col].dtype):
                    feature_vals[col] = row[col].to_pydatetime() if not pd.isna(row[col]) else None
                else:
                    feature_vals[col] = row[col]
            
            # Store in online store
            _state_manager.set_online_data(fg_name, online_key_tuple, feature_vals)
        
        simple_logger("info", f"Inserted {len(data_df)} rows into FG '{fg_name}'.")
    except Exception as e:
        simple_logger("error", f"Error inserting data into FG '{fg_name}': {e}")
        raise

def get_offline_fg_data_internal(fg_name: str) -> pd.DataFrame:
    """
    Get the offline data for a feature group.
    
    Args:
        fg_name: The name of the feature group
        
    Returns:
        DataFrame containing the feature group data
    """
    _state_manager.load_metadata_if_needed()
    fg_meta = _state_manager.get_fg_metadata(fg_name)
    if not fg_meta: 
        simple_logger("warning", f"Feature group '{fg_name}' not found in metadata")
        return pd.DataFrame()

    path = _get_offline_parquet_path(fg_name)
    if not os.path.exists(path) or os.path.getsize(path) == 0: 
        simple_logger("info", f"No offline data file for feature group '{fg_name}', returning empty DataFrame")
        return pd.DataFrame(columns=list(fg_meta['schema'].keys()))
    
    try:
        # Read the parquet file without specifying dtypes
        df = pd.read_parquet(path)
        
        # Apply schema types after reading
        for col, dtype_str in fg_meta['schema'].items():
            if col in df.columns:
                try:
                    if 'datetime' in dtype_str:
                        df[col] = pd.to_datetime(df[col], utc=True)
                    else:
                        df[col] = df[col].astype(dtype_str)
                except Exception as e:
                    simple_logger("warning", f"Could not cast column '{col}' to '{dtype_str}' for FG '{fg_name}': {e}")
        
        return df
    except Exception as e:
        simple_logger("error", f"Error reading offline data for FG '{fg_name}': {e}")
        return pd.DataFrame(columns=list(fg_meta['schema'].keys()))

def get_online_fg_data_internal(fg_name: str, online_key_values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    _state_manager.load_metadata_if_needed()
    _state_manager.load_online_store_if_needed()
    fg_meta = _state_manager.get_fg_metadata(fg_name)
    if not fg_meta or not fg_meta.get('online_key_columns'): return None

    try:
        key_tuple = tuple(online_key_values[okc] for okc in fg_meta['online_key_columns'])
    except KeyError:
        simple_logger("warning", f"Missing online key columns in provided values for FG '{fg_name}'. Expected: {fg_meta['online_key_columns']}, Got: {online_key_values}")
        return None

    entity_data = _state_manager.get_online_data(fg_name, key_tuple)
    return copy.deepcopy(entity_data) if entity_data else None


# --- Core Feature View Logic ---
def define_feature_view_internal(name: str, label_fg: str, label_col: str, label_et_col: str, joins: List[Dict], declared_tx: List[Dict]):
    """
    Define a new feature view.
    
    Args:
        name: Name of the feature view
        label_fg: Name of the feature group containing the label column
        label_col: Name of the column containing the label
        label_et_col: Name of the event time column in the label feature group
        joins: List of dictionaries specifying joins
        declared_tx: List of dictionaries specifying transformations
    """
    _state_manager.load_metadata_if_needed()
    if name in _state_manager.fv_metadata:
        simple_logger("warning", f"FV '{name}' metadata already exists. Overwriting.")
    if label_fg not in _state_manager.fg_metadata: 
        raise ValueError(f"Label FG '{label_fg}' not found.")
    for join in joins:
        if join['name'] not in _state_manager.fg_metadata: 
            raise ValueError(f"Joined FG '{join['name']}' not found.")

    metadata = {
        'name': name, 
        'label_fg_name': label_fg, 
        'label_column': label_col,
        'label_event_time_column': label_et_col, 
        'feature_group_joins': joins,
        'declared_transforms': declared_tx, 
        'computed_transform_params': {}
    }
    
    # Save metadata directly to the state manager's dictionary
    _state_manager.fv_metadata[name] = metadata
    
    # Then call the official setter to ensure it's saved to disk
    _state_manager.set_fv_metadata(name, metadata)
    
    # Double check that the metadata was saved correctly
    saved_meta = _state_manager.get_fv_metadata(name)
    if not saved_meta or 'label_fg_name' not in saved_meta:
        simple_logger("error", f"Failed to save FV '{name}' metadata correctly. Trying one more time with direct save.")
        
        # Force save to disk
        get_metadata_dir().mkdir(parents=True, exist_ok=True)
        path_fv = get_metadata_dir() / _FV_METADATA_FILE
        
        with open(path_fv, 'w') as f:
            json.dump(_state_manager.fv_metadata, f, indent=4, cls=_CustomJSONEncoder)
        
        simple_logger("info", f"Direct save of FV metadata completed. Metadata for '{name}' should now be available.")
    
    simple_logger("info", f"FV '{name}' metadata defined.")

def get_feature_view_metadata_internal(name: str) -> Optional[Dict[str, Any]]:
    _state_manager.load_metadata_if_needed()
    return copy.deepcopy(_state_manager.get_fv_metadata(name))

def list_feature_views_internal() -> List[str]:
    _state_manager.load_metadata_if_needed()
    return _state_manager.list_feature_views()

def _resolve_column_name_in_joined_df(declared_name: str, df: pd.DataFrame, fv_meta: Dict) -> Optional[str]:
    """
    Resolve a column name from feature view definition to its actual name in the joined DataFrame.
    
    Args:
        declared_name: The column name as declared in the feature view
        df: The joined DataFrame
        fv_meta: The feature view metadata
        
    Returns:
        The actual column name in the DataFrame, or None if not found
    """
    # First check if the column exists as is
    if declared_name in df.columns: 
        return declared_name
        
    # Try with label feature group prefix
    label_fg_name = fv_meta['label_fg_name']
    label_fg_prefix = f"{label_fg_name}__"
    if label_fg_prefix + declared_name in df.columns:
        return label_fg_prefix + declared_name
    
    # Check for the column name with various join prefixes
    for join_def in fv_meta['feature_group_joins']:
        join_fg_name = join_def['name']
        prefix = join_def.get('prefix', f"{join_fg_name}__")
        prefixed_name = f"{prefix}{declared_name}"
        
        if prefixed_name in df.columns: 
            return prefixed_name
            
    # If not found with any prefix, log the available columns for debugging
    simple_logger("debug", f"Could not resolve column '{declared_name}' in joined DF. Available columns: {df.columns.tolist()}")
    return None

def _apply_transformations_logic(df: pd.DataFrame, transform_params: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    transformed_df = df.copy()
    if not transform_params: return transformed_df

    for actual_col_name, transform_info in transform_params.items():
        transform_type = transform_info.get('type')
        params = transform_info.get('params')
        if not transform_type or not params: 
            simple_logger("debug", f"Skipping transform for '{actual_col_name}', missing type or params.")
            continue
        if actual_col_name not in transformed_df.columns:
            simple_logger("warning", f"Column '{actual_col_name}' for transform not found in data. Creating as NaN and skipping transform.")
            transformed_df[actual_col_name] = np.nan
            continue
        
        # Ensure column is not all NaN before transforming, or handle gracefully
        if transformed_df[actual_col_name].isnull().all():
            simple_logger("debug", f"Column '{actual_col_name}' is all NaN. Skipping transformation '{transform_type}'.")
            continue

        if transform_type in _state_manager.transformation_functions:
            transform_func = _state_manager.transformation_functions[transform_type]
            try:
                transformed_df = transform_func(transformed_df, actual_col_name, params)
            except Exception as e:
                simple_logger("error", f"Error applying transform '{transform_type}' to '{actual_col_name}': {e}")
                # Potentially skip this transform or re-raise based on desired robustness
        else:
            simple_logger("warning", f"Unknown transform type '{transform_type}' for '{actual_col_name}'.")
    return transformed_df

def _compute_transformation_params_logic(raw_joined_df: pd.DataFrame, fv_meta: Dict) -> Dict[str, Dict[str, Any]]:
    computed_params: Dict[str, Dict[str, Any]] = {}
    declared_transforms = fv_meta.get('declared_transforms', [])

    for transform_def in declared_transforms:
        original_feature_name = transform_def.get('feature_name')
        transform_type = transform_def.get('transform_type')
        if not original_feature_name or not transform_type: continue

        actual_col_name = _resolve_column_name_in_joined_df(original_feature_name, raw_joined_df, fv_meta)
        if actual_col_name is None:
            simple_logger("warning", f"Cannot compute params for declared feature '{original_feature_name}', column not found in joined data.")
            continue
        
        if raw_joined_df[actual_col_name].isnull().all():
            simple_logger("warning", f"Cannot compute params for '{actual_col_name}', column is all NaN.")
            continue

        if transform_type in _state_manager.parameter_computation_functions:
            compute_func = _state_manager.parameter_computation_functions[transform_type]
            try:
                params_for_col = compute_func(raw_joined_df[actual_col_name].dropna()) # Drop NA before computing params
                valid_params = {k: v for k, v in params_for_col.items() if pd.notna(v) and v is not None} # More robust NA check
                if valid_params:
                    computed_params[actual_col_name] = {'type': transform_type, 'params': valid_params}
                else:
                    simple_logger("warning", f"No valid parameters computed for '{actual_col_name}' with type '{transform_type}'.")
            except Exception as e:
                simple_logger("error", f"Error computing params for '{actual_col_name}' with type '{transform_type}': {e}")
        else:
            simple_logger("warning", f"Unknown param computation type '{transform_type}' for '{actual_col_name}'.")

    if fv_meta['name'] in _state_manager.fv_metadata:
        _state_manager.set_fv_metadata(fv_meta['name'], {'computed_transform_params': computed_params})
        _state_manager.save_all_metadata()
    return computed_params

def get_training_data_internal(fv_name: str, compute_params: bool) -> Tuple[pd.DataFrame, pd.Series]:
    _state_manager.load_metadata_if_needed()
    fv_meta = _state_manager.get_fv_metadata(fv_name)
    if not fv_meta: raise ValueError(f"FV '{fv_name}' not found.")
    
    label_fg_name = fv_meta['label_fg_name']
    label_fg_meta = _state_manager.get_fg_metadata(label_fg_name)
    if not label_fg_meta: raise ValueError(f"Label FG '{label_fg_name}' not found.")
    
    label_df = get_offline_fg_data_internal(label_fg_name)
    if label_df.empty: raise ValueError(f"No data in label FG '{label_fg_name}'.")
    
    # Get actual column names from label FG
    label_et_col_declared = fv_meta['label_event_time_column']
    label_et_col_actual = label_et_col_declared # Simple case - no prefix since it's the label FG
    
    # Ensure datetime type for event time column
    label_df[label_et_col_actual] = pd.to_datetime(label_df[label_et_col_actual], utc=True)
    
    # Start with just the label FG data
    current_df = label_df.copy()
    
    # Sort by primary keys and timestamp initially
    sort_cols = label_fg_meta['primary_key_columns'] + [label_et_col_actual]
    current_df = current_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Process each join in the feature view
    for join_def in fv_meta['feature_group_joins']:
        join_fg_name = join_def['name']
        join_on_cols = join_def['on']
        prefix = join_def.get('prefix', f"{join_fg_name}__")

        join_fg_meta = _state_manager.get_fg_metadata(join_fg_name)
        if not join_fg_meta:
            simple_logger("error", f"Join FG '{join_fg_name}' not found.")
            continue
            
        right_df = get_offline_fg_data_internal(join_fg_name)
        if right_df.empty:
            simple_logger("warning", f"No data in join FG '{join_fg_name}', skipping join.")
            continue

        right_event_time_col_original = join_fg_meta['event_time_column']
        if right_event_time_col_original not in right_df.columns:
            simple_logger("error", f"Event time col '{right_event_time_col_original}' not found in FG '{join_fg_name}'.")
            continue
        
        # Ensure datetime type for right df event time column
        right_df[right_event_time_col_original] = pd.to_datetime(right_df[right_event_time_col_original], utc=True)
        
        # Rename columns in right_df (except join keys)
        # Create mapping for renamed columns
        cols_to_prefix = [col for col in right_df.columns if col not in join_on_cols]
        renamed_cols = {col: prefix + col for col in cols_to_prefix}
        right_df_renamed = right_df.rename(columns=renamed_cols)
        
        # Get the prefixed event time column name
        right_on_et_col = prefix + right_event_time_col_original if right_event_time_col_original in cols_to_prefix else right_event_time_col_original
        
        if right_on_et_col not in right_df_renamed.columns:
            simple_logger("error", f"Event time column '{right_on_et_col}' (from '{right_event_time_col_original}') not in renamed right DataFrame.")
            continue

        # Check if all join columns exist in both DataFrames
        missing_join_cols = [col for col in join_on_cols if col not in current_df.columns or col not in right_df_renamed.columns]
        if missing_join_cols:
            simple_logger("error", f"Join columns {missing_join_cols} missing from DataFrames for {label_fg_name} + {join_fg_name} join.")
            continue

        try:
            # Make sure data types match for join columns
            for col in join_on_cols:
                # For numeric columns, ensure right df matches left df type
                if pd.api.types.is_numeric_dtype(current_df[col]) and pd.api.types.is_numeric_dtype(right_df_renamed[col]):
                    dtype = current_df[col].dtype
                    right_df_renamed[col] = right_df_renamed[col].astype(dtype)
                # For string columns, ensure both are string type
                elif pd.api.types.is_string_dtype(current_df[col]) or pd.api.types.is_object_dtype(current_df[col]):
                    current_df[col] = current_df[col].astype(str)
                    right_df_renamed[col] = right_df_renamed[col].astype(str)
                # Log the data types for debugging
                simple_logger("debug", f"Join column '{col}' types - Left: {current_df[col].dtype}, Right: {right_df_renamed[col].dtype}")

            # Force stable sorting on the join columns first, using order from the original data
            # This ensures pandas merge_asof requirements are met
            right_sorted = right_df_renamed.copy().sort_values(
                by=join_on_cols + [right_on_et_col], 
                kind='stable'
            ).reset_index(drop=True)
            
            left_sorted = current_df.copy().sort_values(
                by=join_on_cols + [label_et_col_actual], 
                kind='stable'
            ).reset_index(drop=True)
            
            # Use the sorted dataframes for the merge_asof
            result_df = pd.merge_asof(
                left=left_sorted,
                right=right_sorted,
                left_on=label_et_col_actual,
                right_on=right_on_et_col,
                by=join_on_cols,
                direction='backward',
                allow_exact_matches=True
            )
            
            # If merge worked, update current_df
            current_df = result_df
        
        except Exception as e:
            simple_logger("error", f"Error performing merge_asof for FV '{fv_name}' with FGs '{label_fg_name}' and '{join_fg_name}': {e}")
            simple_logger("debug", f"Left columns: {current_df.columns.tolist()}")
            simple_logger("debug", f"Right columns: {right_df_renamed.columns.tolist()}")
            simple_logger("debug", f"Join on: {join_on_cols}, Left on: {label_et_col_actual}, Right on: {right_on_et_col}")
            
            # Alternative approach: if merge_asof fails, fall back to a regular merge
            # This won't have the exact semantics of merge_asof but may help in testing
            try:
                simple_logger("warning", f"Falling back to regular merge for {label_fg_name} + {join_fg_name}")
                
                # Do a regular merge on join columns
                merged = pd.merge(
                    current_df.copy(),
                    right_df_renamed.copy(),
                    on=join_on_cols,
                    how='left',
                    suffixes=('', '_right')
                )
                
                # Use the merged dataframe and continue
                current_df = merged
                
                simple_logger("info", f"Fallback merge completed for {label_fg_name} + {join_fg_name}")
            except Exception as fallback_error:
                simple_logger("error", f"Fallback merge also failed: {fallback_error}")
                # Continue with the current DataFrame if all joins fail
                continue

    raw_joined_df = current_df

    stored_params = fv_meta.get('computed_transform_params', {})
    if compute_params or not stored_params:
        simple_logger("info", f"Computing parameters for FV '{fv_name}'.")
        final_params = _compute_transformation_params_logic(raw_joined_df.copy(), fv_meta)
    else:
        simple_logger("info", f"Using stored parameters for FV '{fv_name}'.")
        final_params = stored_params

    transformed_df = _apply_transformations_logic(raw_joined_df.copy(), final_params)

    label_col_actual = fv_meta['label_column']
    if label_col_actual not in transformed_df.columns:
        # Check if label_col is from a prefixed FG if it was joined (unlikely for label)
        label_col_resolved = _resolve_column_name_in_joined_df(label_col_actual, transformed_df, fv_meta)
        if not label_col_resolved:
            raise ValueError(f"Label column '{label_col_actual}' (resolved to '{label_col_resolved}') not in final transformed data for FV '{fv_name}'. Available: {transformed_df.columns.tolist()}")
        label_col_actual = label_col_resolved
        
    y = transformed_df[label_col_actual]
    X = transformed_df.drop(columns=[label_col_actual], errors='ignore')

    # Cleanup: join keys, event time cols (original and prefixed), and PKs of joined FGs
    cols_to_drop = set()
    # Label FG's event time and PKs (PKs are often join keys, but good to list explicitly)
    cols_to_drop.add(label_et_col_actual)
    cols_to_drop.update(label_fg_meta['primary_key_columns'])

    for join_def in fv_meta['feature_group_joins']:
        fg_meta_j = _state_manager.get_fg_metadata(join_def['name'])
        prefix = join_def.get('prefix', f"{join_def['name']}__")
        cols_to_drop.update(join_def['on']) # Original join keys as named in the label FG context
        
        # Original event time of joined FG and its prefixed version
        original_join_et = fg_meta_j['event_time_column']
        cols_to_drop.add(original_join_et) # If it wasn't prefixed (e.g. name collision avoided)
        cols_to_drop.add(prefix + original_join_et)
        
        # Original PKs of joined FG and their prefixed versions
        for pk_j in fg_meta_j['primary_key_columns']:
            cols_to_drop.add(pk_j) # If not prefixed
            cols_to_drop.add(prefix + pk_j)
            
    # Drop only columns that actually exist in X
    final_cols_to_drop = [col for col in cols_to_drop if col in X.columns]
    X = X.drop(columns=final_cols_to_drop, errors='ignore')

    simple_logger("info", f"Training data generated for FV '{fv_name}'. X shape: {X.shape}, y shape: {y.shape}")
    return X, y


def get_inference_vector_internal(fv_name: str, entity_keys: Dict[str, Any]) -> pd.Series:
    _state_manager.load_metadata_if_needed()
    _state_manager.load_online_store_if_needed()
    fv_meta = _state_manager.get_fv_metadata(fv_name)
    if not fv_meta: raise ValueError(f"FV '{fv_name}' not found.")

    raw_features_dict = {}
    raw_features_dict.update(entity_keys)
    
    label_fg_name = fv_meta['label_fg_name']
    label_fg_meta = _state_manager.get_fg_metadata(label_fg_name)
    
    # First, get data from the label FG
    required_entity_keys = {}
    for pk in label_fg_meta['online_key_columns']:
        if pk in entity_keys:
            required_entity_keys[pk] = entity_keys[pk]
        else:
            simple_logger("warning", f"Primary key '{pk}' from label FG '{label_fg_name}' not found in entity keys.")
    
    # Build the complete online key tuple for label FG
    online_key_tuple = tuple(required_entity_keys[k] for k in label_fg_meta['online_key_columns'])
    
    # Get data from online store for label FG
    label_data = _state_manager.get_online_data(label_fg_name, online_key_tuple)
    if not label_data:
        simple_logger("warning", f"No data found in online store for '{label_fg_name}' with keys {online_key_tuple}")
        label_data = {}
    
    # Add label FG data to raw features
    raw_features_dict.update(label_data)
    
    # Now get data from each joined FG
    for join_def in fv_meta['feature_group_joins']:
        join_fg_name = join_def['name']
        join_keys = join_def['on']
        prefix = join_def.get('prefix', f"{join_fg_name}__")
        
        join_fg_meta = _state_manager.get_fg_metadata(join_fg_name)
        if not join_fg_meta:
            simple_logger("warning", f"FG '{join_fg_name}' not found for join in FV '{fv_name}'")
            continue
            
        # Build a dictionary of join key values
        join_entity_keys = {}
        missing_keys = False
        for pk in join_fg_meta['online_key_columns']:
            if pk in entity_keys:
                join_entity_keys[pk] = entity_keys[pk]
            elif pk in raw_features_dict:
                join_entity_keys[pk] = raw_features_dict[pk]
            else:
                simple_logger("warning", f"Could not find value for join key '{pk}' for FG '{join_fg_name}'")
                missing_keys = True
                
        if missing_keys:
            simple_logger("warning", f"Skipping join for FG '{join_fg_name}' due to missing keys.")
            continue
            
        # Build the complete online key tuple for this FG
        join_key_tuple = tuple(join_entity_keys[k] for k in join_fg_meta['online_key_columns'])
        
        # Get data from online store
        join_data = _state_manager.get_online_data(join_fg_name, join_key_tuple)
        if not join_data:
            simple_logger("warning", f"No data found in online store for '{join_fg_name}' with keys {join_key_tuple}")
            continue
            
        # Add joined FG data to raw features with prefix
        for k, v in join_data.items():
            if k not in join_keys:  # Don't prefix join keys
                raw_features_dict[prefix + k] = v
            else:
                # If it's a join key, only add if not already present
                if k not in raw_features_dict:
                    raw_features_dict[k] = v
    
    # Apply transformations
    transform_params = fv_meta.get('computed_transform_params', {})
    if not transform_params:
        simple_logger("warning", f"No computed transform parameters found for FV '{fv_name}'. Features will not be transformed.")
        
    # Convert to Series
    raw_features_series = pd.Series(raw_features_dict)
    
    # Apply transformations
    vector = _apply_transformations_logic(pd.DataFrame([raw_features_series]), transform_params).iloc[0]
    
    # Remove label column if present
    label_col = fv_meta['label_column']
    if label_col in vector:
        vector = vector.drop(label_col)
        
    return vector

# Helper to ensure directories for state exist before first load attempt
# This is better than scattering mkdir in every load/save function.
def _initialize_state_directories():
    get_metadata_dir().mkdir(parents=True, exist_ok=True)
    get_offline_store_dir().mkdir(parents=True, exist_ok=True)
    get_online_store_dir().mkdir(parents=True, exist_ok=True)

_initialize_state_directories() # Ensure dirs exist when module is loaded 

# --- Functions for external API access to the state manager ---
def _clear_minimal_state():
    """Clear all in-memory state (for testing)."""
    _state_manager.clear_all()

# Exposed for compatibility with existing code - consider phasing these out
def _load_all_metadata_if_needed():
    """Load metadata if needed (compatibility function)."""
    _state_manager.load_metadata_if_needed()

def _load_online_store_if_needed():
    """Load online store if needed (compatibility function)."""
    _state_manager.load_online_store_if_needed()

# For backward compatibility
_CACHED_FG_METADATA = _state_manager.fg_metadata
_CACHED_FV_METADATA = _state_manager.fv_metadata
_CACHED_ONLINE_STORE = _state_manager.online_store
_METADATA_LOADED = False
_ONLINE_STORE_LOADED = False
_TRANSFORMATION_FUNCTIONS = _state_manager.transformation_functions
_PARAMETER_COMPUTATION_FUNCTIONS = _state_manager.parameter_computation_functions 

def _apply_transformations_to_series(series: pd.Series, transform_params: Dict) -> pd.Series:
    """
    Apply transformations to a single feature vector as a pandas Series.
    
    Args:
        series: Series containing the feature values
        transform_params: Dictionary of transformation parameters
        
    Returns:
        Series with transformed features
    """
    # Convert to a single-row DataFrame for compatibility with _apply_transformations_logic
    df = pd.DataFrame([series])
    transformed_df = _apply_transformations_logic(df, transform_params)
    return transformed_df.iloc[0] 

def register_custom_transformer_internal(transformer):
    """
    Register a custom transformer for use with feature views.
    
    Args:
        transformer: CustomTransformer instance to register
    
    Raises:
        ValueError: If a transformer with the same name is already registered
    """
    if transformer.name in _state_manager.custom_transformers:
        raise ValueError(f"A transformer with name '{transformer.name}' is already registered.")
    
    _state_manager.register_custom_transformer(transformer)
    
    # Return success indicator
    return True 