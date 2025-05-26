import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable

from .utils import (
    get_metadata_dir, get_offline_store_dir, get_online_store_dir,
    simple_logger,
    load_json, save_json, load_parquet, save_parquet,
    load_metadata, save_metadata, load_online_store, save_online_store,
    get_fg_metadata, get_fv_metadata, get_online_store, get_custom_transformers,
    set_fg_metadata, set_fv_metadata, set_online_store, set_custom_transformers,
    reset_all_state
)

# --- Feature Group Functions ---

def create_feature_group(name: str, primary_keys: List[str], event_time: str, 
                        schema: Dict[str, str], online_keys: List[str]):
    """Create a new feature group"""
    load_metadata()
    
    fg_metadata = get_fg_metadata()
    if name in fg_metadata:
        simple_logger("warning", f"Feature group '{name}' already exists, overwriting")
    
    fg_metadata[name] = {
        'primary_key_columns': primary_keys,
        'event_time_column': event_time,
        'schema': schema,
        'online_key_columns': online_keys
    }
    
    set_fg_metadata(fg_metadata)
    save_metadata()
    simple_logger("info", f"Feature group '{name}' created")

def get_feature_group_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Get feature group metadata"""
    load_metadata()
    return get_fg_metadata().get(name)

def list_feature_groups() -> List[str]:
    """List all feature groups"""
    load_metadata()
    return list(get_fg_metadata().keys())

def insert_feature_group_data(fg_name: str, data: pd.DataFrame):
    """Insert data into feature group"""
    metadata = get_feature_group_metadata(fg_name)
    if not metadata:
        raise ValueError(f"Feature group '{fg_name}' not found")
    
    # Save to offline store
    file_path = get_offline_store_dir() / f"{fg_name}.parquet"
    existing_df = load_parquet(file_path)
    
    if not existing_df.empty:
        combined_df = pd.concat([existing_df, data], ignore_index=True)
    else:
        combined_df = data.copy()
    
    save_parquet(combined_df, file_path)
    
    # Update online store
    load_online_store()
    online_store = get_online_store()
    if fg_name not in online_store:
        online_store[fg_name] = {}
    
    online_keys = metadata['online_key_columns']
    for _, row in data.iterrows():
        key_tuple = tuple(row[col] for col in online_keys)
        online_store[fg_name][key_tuple] = row.to_dict()
    
    set_online_store(online_store)
    save_online_store()
    simple_logger("info", f"Inserted {len(data)} rows into '{fg_name}'")

def get_offline_data(fg_name: str, features: Optional[List[str]] = None) -> pd.DataFrame:
    """Get offline data from feature group"""
    file_path = get_offline_store_dir() / f"{fg_name}.parquet"
    df = load_parquet(file_path)
    
    if features and not df.empty:
        metadata = get_feature_group_metadata(fg_name)
        if metadata:
            cols = set(features)
            cols.update(metadata['primary_key_columns'])
            if metadata['event_time_column']:
                cols.add(metadata['event_time_column'])
            available_cols = [col for col in cols if col in df.columns]
            df = df[available_cols]
    
    return df

def get_online_data(fg_name: str, key_values: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Get online data from feature group"""
    load_online_store()
    
    metadata = get_feature_group_metadata(fg_name)
    if not metadata:
        return None
    
    online_keys = metadata['online_key_columns']
    key_tuple = tuple(key_values.get(col) for col in online_keys)
    
    online_store = get_online_store()
    return online_store.get(fg_name, {}).get(key_tuple)

# --- Feature View Functions ---

def create_feature_view(name: str, label_fg: str, label_col: str,
                       joins: List[Dict[str, Any]], transforms: List[Dict[str, Any]]):
    """Create a new feature view"""
    load_metadata()
    
    fv_metadata = get_fv_metadata()
    fv_metadata[name] = {
        'label_fg_name': label_fg,
        'label_column': label_col,
        'joins': joins,
        'transforms': transforms,
        'transform_params': {}
    }
    
    set_fv_metadata(fv_metadata)
    save_metadata()
    simple_logger("info", f"Feature view '{name}' created")

def get_feature_view_metadata(name: str) -> Optional[Dict[str, Any]]:
    """Get feature view metadata"""
    load_metadata()
    return get_fv_metadata().get(name)

def list_feature_views() -> List[str]:
    """List all feature views"""
    load_metadata()
    return list(get_fv_metadata().keys())

def _join_feature_groups(fv_metadata: Dict[str, Any]) -> pd.DataFrame:
    """Join feature groups for a feature view"""
    label_fg = fv_metadata['label_fg_name']
    label_df = get_offline_data(label_fg)
    
    if label_df.empty:
        return pd.DataFrame()
    
    result_df = label_df.copy()
    
    # Join with other feature groups
    for join_spec in fv_metadata['joins']:
        fg_name = join_spec['fg_name']
        join_keys = join_spec['on']
        
        fg_df = get_offline_data(fg_name)
        if fg_df.empty:
            continue
        
        result_df = result_df.merge(fg_df, on=join_keys, how='inner', suffixes=('', f'_{fg_name}'))
    
    return result_df

def _apply_transforms(df: pd.DataFrame, transforms: List[Dict[str, Any]], 
                     transform_params: Dict[str, Any]) -> pd.DataFrame:
    """Apply transformations to DataFrame"""
    result_df = df.copy()
    
    for transform in transforms:
        transform_type = transform['type']
        column = transform['column']
        
        if column not in result_df.columns:
            continue
        
        if transform_type == 'scale':
            params = transform_params.get(column, {})
            mean = params.get('mean', 0)
            std = params.get('std', 1)
            if std != 0:
                result_df[column] = (result_df[column] - mean) / std
        
        elif transform_type == 'one_hot_encode':
            params = transform_params.get(column, {})
            categories = params.get('categories', [])
            for cat in categories:
                result_df[f"{column}_{cat}"] = (result_df[column] == cat).astype(int)
    
    return result_df

def _compute_transform_params(df: pd.DataFrame, transforms: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute transformation parameters"""
    params = {}
    
    for transform in transforms:
        transform_type = transform['type']
        column = transform['column']
        
        if column not in df.columns:
            continue
        
        if transform_type == 'scale':
            series = pd.to_numeric(df[column], errors='coerce').dropna()
            if not series.empty:
                params[column] = {
                    'mean': float(series.mean()),
                    'std': float(series.std()) if series.std() != 0 else 1.0
                }
        
        elif transform_type == 'one_hot_encode':
            unique_vals = df[column].dropna().unique().tolist()
            params[column] = {'categories': [str(val) for val in unique_vals]}
    
    return params

def get_training_data(fv_name: str, compute_params: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Get training data from feature view"""
    metadata = get_feature_view_metadata(fv_name)
    if not metadata:
        raise ValueError(f"Feature view '{fv_name}' not found")
    
    # Join feature groups
    joined_df = _join_feature_groups(metadata)
    if joined_df.empty:
        return pd.DataFrame(), pd.Series(dtype=object)
    
    # Compute transform parameters if requested
    if compute_params:
        transform_params = _compute_transform_params(joined_df, metadata['transforms'])
        metadata['transform_params'] = transform_params
        
        fv_metadata = get_fv_metadata()
        fv_metadata[fv_name] = metadata
        set_fv_metadata(fv_metadata)
        save_metadata()
        simple_logger("info", f"Computed transform parameters for '{fv_name}'")
    
    # Apply transformations
    transformed_df = _apply_transforms(joined_df, metadata['transforms'], 
                                     metadata.get('transform_params', {}))
    
    # Extract labels
    label_col = metadata['label_column']
    if label_col in transformed_df.columns:
        labels = transformed_df[label_col]
        features = transformed_df.drop(columns=[label_col])
    else:
        labels = pd.Series(dtype=object)
        features = transformed_df
    
    return features, labels

def get_inference_vector(fv_name: str, entity_keys: Dict[str, Any]) -> pd.Series:
    """Get inference vector from feature view"""
    metadata = get_feature_view_metadata(fv_name)
    if not metadata:
        raise ValueError(f"Feature view '{fv_name}' not found")
    
    # Get data from online store
    feature_data = {}
    
    # Get label feature group data
    label_fg = metadata['label_fg_name']
    label_data = get_online_data(label_fg, entity_keys)
    if label_data:
        feature_data.update(label_data)
    
    # Get joined feature group data
    for join_spec in metadata['joins']:
        fg_name = join_spec['fg_name']
        fg_data = get_online_data(fg_name, entity_keys)
        if fg_data:
            feature_data.update(fg_data)
    
    if not feature_data:
        return pd.Series(dtype=object)
    
    # Create DataFrame for transformation
    df = pd.DataFrame([feature_data])
    
    # Apply transformations
    transformed_df = _apply_transforms(df, metadata['transforms'], 
                                     metadata.get('transform_params', {}))
    
    # Remove label column if present
    label_col = metadata['label_column']
    if label_col in transformed_df.columns:
        transformed_df = transformed_df.drop(columns=[label_col])
    
    return transformed_df.iloc[0] if not transformed_df.empty else pd.Series(dtype=object)

# --- Custom Transformers ---

def register_custom_transformer(name: str, fit_fn: Callable, transform_fn: Callable):
    """Register a custom transformer"""
    custom_transformers = get_custom_transformers()
    custom_transformers[name] = {
        'fit_fn': fit_fn,
        'transform_fn': transform_fn,
        'params': {}
    }
    set_custom_transformers(custom_transformers)

def get_custom_transformer(name: str) -> Optional[Dict[str, Any]]:
    """Get custom transformer"""
    return get_custom_transformers().get(name)

# --- Main Classes ---

class FeatureGroup:
    """Feature group for storing and retrieving features"""
    
    def __init__(self, name: str):
        self.name = name
        metadata = get_feature_group_metadata(name)
        if not metadata:
            raise ValueError(f"Feature group '{name}' not found")
        
        self.primary_key_columns = metadata['primary_key_columns']
        self.event_time_column = metadata['event_time_column']
        self.online_key_columns = metadata['online_key_columns']
        self.schema = metadata['schema']
    
    @staticmethod
    def create(name: str, primary_keys: List[str], event_time: str, 
              schema: Dict[str, str], online_keys: List[str]) -> 'FeatureGroup':
        create_feature_group(name, primary_keys, event_time, schema, online_keys)
        return FeatureGroup(name)
    
    def insert(self, data: pd.DataFrame):
        insert_feature_group_data(self.name, data)
    
    def get_offline_data(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        return get_offline_data(self.name, features)
    
    def get_online_features(self, entity_keys: Dict[str, Any]) -> Dict[str, Any]:
        return get_online_data(self.name, entity_keys) or {}

class FeatureView:
    """Feature view for ML training and inference"""
    
    def __init__(self, name: str):
        self.name = name
        metadata = get_feature_view_metadata(name)
        if not metadata:
            raise ValueError(f"Feature view '{name}' not found")
        
        self.label_fg_name = metadata['label_fg_name']
        self.label_column = metadata['label_column']
        self.joins = metadata['joins']
        self.transforms = metadata['transforms']
    
    @staticmethod
    def create(name: str, label_fg: str, label_col: str,
              joins: List[Dict[str, Any]], transforms: List[Dict[str, Any]]) -> 'FeatureView':
        create_feature_view(name, label_fg, label_col, joins, transforms)
        return FeatureView(name)
    
    def get_training_data(self, compute_params: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        return get_training_data(self.name, compute_params)
    
    def get_inference_vector(self, entity_keys: Dict[str, Any]) -> pd.Series:
        return get_inference_vector(self.name, entity_keys)

class FeatureStore:
    """Main feature store class"""
    
    def __init__(self):
        # Initialize directories
        get_metadata_dir().mkdir(parents=True, exist_ok=True)
        get_offline_store_dir().mkdir(parents=True, exist_ok=True)
        get_online_store_dir().mkdir(parents=True, exist_ok=True)
    
    def create_feature_group(self, name: str, primary_keys: List[str], event_time: str,
                           schema: Dict[str, str], online_keys: List[str]) -> FeatureGroup:
        return FeatureGroup.create(name, primary_keys, event_time, schema, online_keys)
    
    def get_feature_group(self, name: str) -> FeatureGroup:
        return FeatureGroup(name)
    
    def list_feature_groups(self) -> List[str]:
        return list_feature_groups()
    
    def create_feature_view(self, name: str, label_fg: str, label_col: str,
                          joins: List[Dict[str, Any]], 
                          transforms: List[Dict[str, Any]]) -> FeatureView:
        return FeatureView.create(name, label_fg, label_col, joins, transforms)
    
    def get_feature_view(self, name: str) -> FeatureView:
        return FeatureView(name)
    
    def list_feature_views(self) -> List[str]:
        return list_feature_views()
    
    def reset_all_state(self):
        """Reset all feature store state - for demo/educational purposes"""
        reset_all_state()
    
    def register_custom_transformer(self, name: str, fit_fn: Callable, transform_fn: Callable):
        register_custom_transformer(name, fit_fn, transform_fn)
    
    def get_custom_transformer(self, name: str) -> Optional[Dict[str, Any]]:
        return get_custom_transformer(name) 