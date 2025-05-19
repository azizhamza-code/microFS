import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Callable, Union

# Assuming internal_logic is in the same package (microfs)
from .internal_logic import (
    define_feature_group_internal,
    get_feature_group_metadata_internal,
    list_feature_groups_internal,
    insert_into_feature_group_internal,
    get_offline_fg_data_internal,
    define_feature_view_internal,
    get_feature_view_metadata_internal,
    list_feature_views_internal,
    get_training_data_internal,
    get_inference_vector_internal,
    _CACHED_ONLINE_STORE, # For FeatureGroup __repr__
    _load_all_metadata_if_needed, # For FeatureStore init
    _load_online_store_if_needed, # For FeatureStore init
    _FG_METADATA_FILE, _FV_METADATA_FILE, _ONLINE_STORE_FILE, # For reset
    get_metadata_dir, get_online_store_dir, get_offline_store_dir, # For reset
    _state_manager, # For exposing state in tests
    get_online_fg_data_internal, # For FeatureGroup get_online_features
    register_custom_transformer_internal # Add this import
)
from .utils import simple_logger # For FeatureStore init and reset
import os # For reset

class CustomTransformer:
    """
    A class that allows data scientists to define custom model-dependent transformations.
    
    This class wraps user-defined transformation functions for use with the feature store.
    Each transformer must provide both fit and transform methods to be compatible with
    the feature store's training and inference pipelines.
    
    Attributes:
        name (str): Unique name for this transformer
        fit_fn (Callable): Function to compute transformation parameters during training
        transform_fn (Callable): Function to apply the transformation during training and inference
        params (Dict[str, Any]): Stored parameters computed during fit
    """
    
    def __init__(self, name: str, 
                 fit_fn: Callable[[pd.Series], Dict[str, Any]], 
                 transform_fn: Callable[[pd.Series, Dict[str, Any]], pd.Series]):
        """
        Initialize a custom transformer.
        
        Args:
            name: Unique name for this transformer
            fit_fn: Function that computes parameters from training data
                    Should take a pandas Series and return a dict of parameters
            transform_fn: Function that applies transformation using computed parameters
                          Should take a pandas Series and parameters dict, and return a transformed Series
        """
        self.name = name
        self.fit_fn = fit_fn
        self.transform_fn = transform_fn
        self.params = {}
        
    def fit(self, data: pd.Series) -> Dict[str, Any]:
        """
        Compute transformation parameters from training data.
        
        Args:
            data: Series containing the feature data to fit
            
        Returns:
            Dictionary of computed parameters
        """
        self.params = self.fit_fn(data)
        return self.params
        
    def transform(self, data: pd.Series) -> pd.Series:
        """
        Apply transformation using stored parameters.
        
        Args:
            data: Series containing the feature data to transform
            
        Returns:
            Transformed feature data as a Series
            
        Raises:
            ValueError: If transform is called before fit or with empty params
        """
        if not self.params:
            raise ValueError(f"Transformer '{self.name}' has no parameters. Call fit() first.")
        return self.transform_fn(data, self.params)
    
    def fit_transform(self, data: pd.Series) -> pd.Series:
        """
        Compute parameters and apply transformation in one step.
        
        Args:
            data: Series containing the feature data to fit and transform
            
        Returns:
            Transformed feature data as a Series
        """
        self.fit(data)
        return self.transform(data)

class FeatureGroup:
    """
    A Feature Group represents a logical collection of related features with a common schema.
    
    Feature Groups are the primary building blocks for storing and retrieving features in the feature store.
    They handle both offline storage (for training data) and online storage (for real-time inference).
    
    Attributes:
        name (str): Name of the feature group
        primary_key_columns (List[str]): Columns that uniquely identify a record
        event_time_column (str): Column containing the timestamp for point-in-time correctness
        online_key_columns (List[str]): Columns used for key lookup in the online store
        schema (Dict[str, str]): Schema definition mapping column names to their data types
    """
    
    def __init__(self, name: str):
        """
        Initialize a FeatureGroup instance by loading its metadata.
        
        Args:
            name: Name of the feature group to load
            
        Raises:
            ValueError: If the feature group doesn't exist
        """
        self.name = name
        # Ensure metadata is loaded if not already (e.g. direct instantiation)
        _load_all_metadata_if_needed()
        meta = get_feature_group_metadata_internal(name)
        if not meta: raise ValueError(f"FG '{name}' not found in metadata. Ensure it has been created.")
        self.primary_key_columns = meta['primary_key_columns']
        self.event_time_column = meta['event_time_column']
        self.online_key_columns = meta['online_key_columns']
        self.schema = meta['schema']

    @staticmethod
    def create(name: str, primary_key_columns: List[str], event_time_column: str, 
              schema: Dict[str, str], online_key_columns: List[str]) -> 'FeatureGroup':
        """
        Create a new feature group.
        
        Args:
            name: Name of the feature group
            primary_key_columns: Columns that uniquely identify a record
            event_time_column: Column containing the timestamp for point-in-time correctness
            schema: Schema definition mapping column names to their data types
            online_key_columns: Columns used for key lookup in the online store
            
        Returns:
            A new FeatureGroup instance
            
        Note:
            If a feature group with the same name already exists, a warning will be logged
            and the existing feature group will be overwritten.
        """
        define_feature_group_internal(name, primary_key_columns, event_time_column, schema, online_key_columns)
        return FeatureGroup(name)

    def insert(self, data_df: pd.DataFrame) -> None:
        """
        Insert data into the feature group.
        
        Args:
            data_df: DataFrame containing the data to insert
            
        Raises:
            ValueError: If any required columns are missing from the DataFrame
        """
        insert_into_feature_group_internal(self.name, data_df)

    def get_offline_data(self, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Retrieve data from the offline store.
        
        Args:
            features: Optional list of feature columns to retrieve. If None, all columns are retrieved.
                     Primary key columns and event time column are always included.
            
        Returns:
            DataFrame containing the requested features
        """
        df = get_offline_fg_data_internal(self.name)
        if features:
            # Ensure PKs and ET are always included if present, then add selected features
            cols_to_select = []
            if self.primary_key_columns: cols_to_select.extend(self.primary_key_columns)
            if self.event_time_column: cols_to_select.append(self.event_time_column)

            # Add requested features, ensuring they exist and are not already included
            for feat in features:
                if feat in df.columns and feat not in cols_to_select:
                    cols_to_select.append(feat)

            # If no specific features requested beyond keys/ET, or if df is empty, this might be empty
            # Ensure we don't try to select non-existent columns
            final_cols = [col for col in cols_to_select if col in df.columns]
            return df[final_cols].copy() if final_cols else pd.DataFrame()
        return df

    def __repr__(self):
        # Ensure online store is loaded for accurate representation
        _load_online_store_if_needed()
        df_offline = get_offline_fg_data_internal(self.name) # This might be slow if large
        online_fg_data = _CACHED_ONLINE_STORE.get(self.name, {})
        return (
            f"FeatureGroup(name='{self.name}', "
            f"pk={self.primary_key_columns}, event_time='{self.event_time_column}', "
            f"online_keys={self.online_key_columns}, schema_features={list(self.schema.keys())}, "
            f"offline_shape={df_offline.shape if not df_offline.empty else (0,0)}, "
            f"online_entities={len(online_fg_data)})"
        )

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get the metadata for this feature group.
        
        Returns:
            Dictionary containing the feature group metadata
        """
        return get_feature_group_metadata_internal(self.name)

    def get_online_features(self, entity_key_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve features for an entity from the online store.
        
        Args:
            entity_key_values: Dictionary mapping key column names to their values
            
        Returns:
            Dictionary containing the feature values for the entity
            
        Raises:
            ValueError: If the entity keys don't exist in the online store
        """
        return get_online_fg_data_internal(self.name, entity_key_values)

class FeatureView:
    """
    A Feature View defines a specific view of features for a machine learning model.
    
    Feature Views join multiple Feature Groups, apply transformations, and prepare data
    for both training and inference scenarios.
    
    Attributes:
        name (str): Name of the feature view
        label_fg_name (str): Name of the feature group containing the label
        label_column (str): Name of the column containing the label
        label_event_time_column (str): Name of the event time column in the label feature group
        feature_group_joins (List[Dict]): Specifications for joining feature groups
        declared_transforms (List[Dict]): Transformations to apply to features
        _computed_transform_params (Dict): Parameters for transformations (internal)
    """
    
    def __init__(self, name: str):
        """
        Initialize a FeatureView instance by loading its metadata.
        
        Args:
            name: Name of the feature view to load
            
        Raises:
            ValueError: If the feature view doesn't exist
        """
        self.name = name
        _load_all_metadata_if_needed() # Ensure metadata loaded
        meta = get_feature_view_metadata_internal(name)
        if not meta: raise ValueError(f"FV '{name}' not found. Ensure it has been created.")
        self.label_fg_name = meta['label_fg_name']
        self.label_column = meta['label_column']
        self.label_event_time_column = meta['label_event_time_column']
        self.feature_group_joins = meta['feature_group_joins']
        self.declared_transforms = meta['declared_transforms']
        self._computed_transform_params = meta.get('computed_transform_params', {})

    @staticmethod
    def create(name: str, label_fg_name: str, label_column: str, label_event_time_column: str, 
              feature_group_joins: List[Dict[str, Any]], declared_transforms: List[Dict[str, Any]]) -> 'FeatureView':
        """
        Create a new feature view.
        
        Args:
            name: Name of the feature view
            label_fg_name: Name of the feature group containing the label
            label_column: Name of the column containing the label
            label_event_time_column: Name of the event time column in the label feature group
            feature_group_joins: Specifications for joining feature groups
                                 Each join dict should have keys: 'name', 'on', and optional 'prefix'
            declared_transforms: Transformations to apply to features
                                 Each transform dict should have keys: 'feature_name', 'transform_type'
            
        Returns:
            A new FeatureView instance
            
        Note:
            If a feature view with the same name already exists, a warning will be logged
            and the existing feature view will be overwritten.
        """
        define_feature_view_internal(name, label_fg_name, label_column, label_event_time_column, feature_group_joins, declared_transforms)
        return FeatureView(name)

    def get_training_data(self, compute_params: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Retrieve training data for this feature view.
        
        This method performs the following operations:
        1. Joins the feature groups according to the feature view specification
        2. Applies point-in-time correctness
        3. Computes transformation parameters if requested
        4. Applies transformations using stored or computed parameters
        5. Splits the result into features (X) and label (y)
        
        Args:
            compute_params: Whether to compute transformation parameters from this dataset
                           If False, uses previously stored parameters if available
            
        Returns:
            Tuple of (features_df, label_series)
        """
        return get_training_data_internal(self.name, compute_params)

    def get_inference_vector(self, entity_key_values: Dict[str, Any]) -> pd.Series:
        """
        Retrieve a feature vector for inference for the given entity keys.
        
        This method:
        1. Looks up the latest values for the entity in the online store
        2. Joins data from multiple feature groups
        3. Applies transformations using stored parameters
        
        Args:
            entity_key_values: Dictionary mapping key column names to their values
            
        Returns:
            Series containing the transformed feature vector ready for prediction
            
        Raises:
            ValueError: If the entity keys don't exist in the online store
        """
        return get_inference_vector_internal(self.name, entity_key_values)

    def get_transform_params(self) -> Dict[str, Any]:
        """
        Get the computed transformation parameters for this feature view.
        
        Returns:
            Dictionary mapping column names to their transformation parameters
        """
        meta = get_feature_view_metadata_internal(self.name)
        return meta.get('computed_transform_params', {}) if meta else {}

    def __repr__(self):
        meta = get_feature_view_metadata_internal(self.name) # Re-fetch for latest params
        params_status = "Parameters NOT loaded/computed"
        if meta and meta.get('computed_transform_params'):
            num_params = len(meta['computed_transform_params'])
            params_status = f"Parameters stored for {num_params} feature_transforms"
        num_joins = len(self.feature_group_joins if hasattr(self, 'feature_group_joins') else [])
        return (
            f"FeatureView(name='{self.name}', label_fg='{self.label_fg_name} ({self.label_column})', "
            f"joins={num_joins}, declared_transforms={len(self.declared_transforms if hasattr(self, 'declared_transforms') else [])}, "
            f"status='{params_status}')"
        )

class FeatureStore:
    """
    The main entry point for interacting with the feature store.
    
    The FeatureStore class provides methods for creating and retrieving Feature Groups
    and Feature Views, which are the core abstractions of the feature store.
    """
    
    def __init__(self):
        """
        Initialize the feature store instance.
        """
        simple_logger("info", "FeatureStore API Facade initializing...")
        _load_all_metadata_if_needed()
        _load_online_store_if_needed()
        # Expose state manager for testing
        self._state = _state_manager
        simple_logger("info", "FeatureStore API Facade initialized.")

    def create_feature_group(self, name: str, primary_key_columns: List[str], event_time_column: str, 
                           schema: Dict[str, str], online_key_columns: List[str]) -> FeatureGroup:
        """
        Create a new feature group.
        
        Args:
            name: Name of the feature group
            primary_key_columns: Columns that uniquely identify a record
            event_time_column: Column containing the timestamp for point-in-time correctness
            schema: Schema definition mapping column names to their data types
            online_key_columns: Columns used for key lookup in the online store
            
        Returns:
            A new FeatureGroup instance
            
        Note:
            If a feature group with the same name already exists, a warning will be logged
            and the existing feature group will be overwritten.
        """
        return FeatureGroup.create(name, primary_key_columns, event_time_column, schema, online_key_columns)

    def get_feature_group(self, name: str) -> FeatureGroup:
        """
        Get an existing feature group by name.
        
        Args:
            name: Name of the feature group to get
            
        Returns:
            FeatureGroup instance
            
        Raises:
            ValueError: If the feature group doesn't exist
        """
        return FeatureGroup(name)

    def list_feature_groups(self) -> List[str]:
        """
        List all available feature groups.
        
        Returns:
            List of feature group names
        """
        return list_feature_groups_internal()

    def create_feature_view(self, name: str, label_fg_name: str, label_column: str, 
                          label_event_time_column: str, feature_group_joins: List[Dict[str, Any]], 
                          declared_transforms: List[Dict[str, Any]]) -> FeatureView:
        """
        Create a new feature view.
        
        Args:
            name: Name of the feature view
            label_fg_name: Name of the feature group containing the label
            label_column: Name of the column containing the label
            label_event_time_column: Name of the event time column in the label feature group
            feature_group_joins: Specifications for joining feature groups
                                 Each join dict should have keys: 'name', 'on', and optional 'prefix'
            declared_transforms: Transformations to apply to features
                                 Each transform dict should have keys: 'feature_name', 'transform_type'
            
        Returns:
            A new FeatureView instance
            
        Note:
            If a feature view with the same name already exists, a warning will be logged
            and the existing feature view will be overwritten.
        """
        return FeatureView.create(name, label_fg_name, label_column, label_event_time_column, feature_group_joins, declared_transforms)

    def get_feature_view(self, name: str) -> FeatureView:
        """
        Get an existing feature view by name.
        
        Args:
            name: Name of the feature view to get
            
        Returns:
            FeatureView instance
            
        Raises:
            ValueError: If the feature view doesn't exist
        """
        return FeatureView(name)

    def list_feature_views(self) -> List[str]:
        """
        List all available feature views.
        
        Returns:
            List of feature view names
        """
        return list_feature_views_internal()

    def reset_all_state_FOR_DEMO_ONLY(self):
        """
        Reset all feature store state (in-memory and on-disk).
        
        This method is for demonstration purposes only and should not be used in production.
        It will delete all feature group and feature view metadata, as well as all offline and online store data.
        """
        simple_logger("warning", "RESETTING ALL MICROFS STATE (in-memory and on-disk)...")
        global _CACHED_FG_METADATA, _CACHED_FV_METADATA, _CACHED_ONLINE_STORE, _METADATA_LOADED, _ONLINE_STORE_LOADED
        # Clear in-memory cache (imported from internal_logic)
        # This requires internal_logic._CACHED... to be accessible or have reset functions
        # For now, let's assume we can tell internal_logic to clear its caches.
        # This is a bit of a hack; ideally internal_logic would expose a reset function.

        # Resetting global variables in another module directly is tricky.
        # Instead, we re-initialize them here for the purpose of this demo function.
        # The internal_logic's own global vars will be reset if it's re-imported or has a reset func.

        # For the purpose of this function, clear what this module can see/control:
        if '_CACHED_FG_METADATA' in globals(): globals()['_CACHED_FG_METADATA'].clear()
        if '_CACHED_FV_METADATA' in globals(): globals()['_CACHED_FV_METADATA'].clear()
        if '_CACHED_ONLINE_STORE' in globals(): globals()['_CACHED_ONLINE_STORE'].clear()
        if '_METADATA_LOADED' in globals(): globals()['_METADATA_LOADED'] = False
        if '_ONLINE_STORE_LOADED' in globals(): globals()['_ONLINE_STORE_LOADED'] = False

        # And tell internal_logic to clear its actual caches. This is better.
        # Need to modify internal_logic to provide such a function.
        # For now, we can try to delete the files and re-run the load functions
        # which will repopulate the caches (as empty if files are gone).

        metadata_dir = get_metadata_dir()
        online_store_dir = get_online_store_dir()
        offline_store_dir = get_offline_store_dir()

        for filename in [_FG_METADATA_FILE, _FV_METADATA_FILE]:
            path = metadata_dir / filename
            if path.exists(): os.remove(path)

        path = online_store_dir / _ONLINE_STORE_FILE
        if path.exists(): os.remove(path)

        if offline_store_dir.exists():
            for item in os.listdir(offline_store_dir):
                if item.endswith(".parquet"):
                    os.remove(offline_store_dir / item)

        simple_logger("info", "All persisted state files removed.")
        
        # Clear the in-memory state of the singleton state manager
        if hasattr(self, '_state') and self._state is not None:
            self._state.clear_all() 
            simple_logger("info", "In-memory state manager caches cleared.")
        else:
            # Fallback or warning if _state is not available as expected
            # This might occur if called in a context where FeatureStore isn't fully initialized
            # or if _state_manager is accessed differently.
            # For now, we'll rely on the direct load calls below to refresh from (empty) disk.
            # This situation ideally shouldn't happen in normal use of reset_all_state_FOR_DEMO_ONLY
            # on an fs instance.
            simple_logger("warning", "_state_manager instance not directly available via self._state for clear_all().")


        # Re-initialize caches from (now empty or non-existent) files
        # These calls will operate on internal_logic's global caches (via _state_manager)
        # and will now properly reload because metadata_loaded flags are False.
        _load_all_metadata_if_needed()
        _load_online_store_if_needed()
        simple_logger("info", "Caches re-initialized after state reset.")

    def register_custom_transformer(self, transformer: CustomTransformer) -> None:
        """
        Register a custom transformer for use with feature views.
        
        This allows data scientists to define their own model-dependent transformations
        and make them available for use in feature views.
        
        Args:
            transformer: CustomTransformer instance to register
            
        Raises:
            ValueError: If a transformer with the same name is already registered
        """
        register_custom_transformer_internal(transformer)
        
    def get_custom_transformer(self, name: str) -> Optional[CustomTransformer]:
        """
        Retrieve a registered custom transformer by name.
        
        Args:
            name: Name of the transformer to retrieve
            
        Returns:
            CustomTransformer instance if found, None otherwise
        """
        return self._state.get_custom_transformer(name)
        
    def list_custom_transformers(self) -> List[str]:
        """
        List all registered custom transformers.
        
        Returns:
            List of transformer names
        """
        return self._state.list_custom_transformers()