"""
microfs.py - A tiny, educational feature store implementation.

Inspired by micrograd, this library aims to demystify Feature Stores.
It implements the core concepts:
1. Feature Groups (Offline Storage)
2. Feature Views (Point-in-time Correctness)
3. Online Store (Low-latency Inference)

Usage:
    fs = FeatureStore()
    fs.create_feature_group(...)
    fs.create_feature_view(...)
"""

import os
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional, Union, Tuple
from pathlib import Path

# -----------------------------------------------------------------------------
# Utils
# -----------------------------------------------------------------------------

def simple_logger(msg: str):
    """A minimal logger to show what's happening under the hood."""
    print(f"[microFS] {msg}")

class Persistence:
    """Handles saving/loading state to disk.
    
    In a real system, this would be a database (Postgres/MySQL) for metadata,
    a data lake (S3/GCS) for offline data, and a KV store (Redis/DynamoDB) 
    for online data. Here, we just use the local filesystem.
    """
    def __init__(self, base_path: str = ".microfs"):
        self.base_path = Path(base_path)
        self.metadata_path = self.base_path / "metadata.json"
        self.offline_path = self.base_path / "offline"
        self.online_path = self.base_path / "online.json"
        
        # Ensure directories exist
        self.offline_path.mkdir(parents=True, exist_ok=True)

    def save_metadata(self, data: Dict):
        with open(self.metadata_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def load_metadata(self) -> Dict:
        if not self.metadata_path.exists(): return {}
        with open(self.metadata_path, 'r') as f:
            return json.load(f)

    def save_offline_data(self, name: str, df: pd.DataFrame):
        # We use Parquet for efficient columnar storage, standard in ML.
        df.to_parquet(self.offline_path / f"{name}.parquet")

    def load_offline_data(self, name: str) -> pd.DataFrame:
        p = self.offline_path / f"{name}.parquet"
        if not p.exists(): return pd.DataFrame()
        return pd.read_parquet(p)

    def save_online_store(self, data: Dict):
        # In reality, this would be a Redis SET command.
        # We serialize tuple keys to strings for JSON.
        serialized = {k: {str(pk): v2 for pk, v2 in v.items()} for k, v in data.items()}
        with open(self.online_path, 'w') as f:
            json.dump(serialized, f, indent=2, default=str)

    def load_online_store(self) -> Dict:
        if not self.online_path.exists(): return {}
        with open(self.online_path, 'r') as f:
            data = json.load(f)
        # Deserialize keys back to simple strings (or tuples if we parsed them, 
        # but for simplicity we'll keep keys as strings in this toy impl)
        return data

    def clear(self):
        if self.base_path.exists():
            shutil.rmtree(self.base_path)
            self.offline_path.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Core Components
# -----------------------------------------------------------------------------

class FeatureGroup:
    """
    A Feature Group is a collection of related features computed together.
    Think of it as a table in your data warehouse.
    
    Key Concepts:
    - Primary Keys: Unique identifiers for entities (e.g., user_id).
    - Event Time: When the event actually happened. Critical for point-in-time correctness.
    """
    def __init__(self, name: str, schema: Dict[str, str], primary_keys: List[str], event_time: str):
        self.name = name
        self.schema = schema
        self.primary_keys = primary_keys
        self.event_time = event_time
        self.data: pd.DataFrame = pd.DataFrame() # In-memory cache of offline data

    def __repr__(self):
        return f"FeatureGroup(name='{self.name}', keys={self.primary_keys})"

class FeatureView:
    """
    A Feature View is a logical view that joins multiple Feature Groups together.
    It defines the 'interface' for your model.
    
    Key Concepts:
    - Point-in-Time Join (ASOF Join): Joining features based on timestamps to avoid data leakage.
      We want the feature value *as it was* at the time of the prediction request.
    """
    def __init__(self, name: str, label_fg: str, joins: List[Dict]):
        self.name = name
        self.label_fg = label_fg # The feature group containing the target variable
        self.joins = joins # List of FGs to join and how

    def __repr__(self):
        return f"FeatureView(name='{self.name}', label_fg='{self.label_fg}')"

class FeatureStore:
    """
    The main entry point. Orchestrates metadata, storage, and retrieval.
    """
    def __init__(self, base_path: str = ".microfs"):
        self.storage = Persistence(base_path)
        self.feature_groups: Dict[str, FeatureGroup] = {}
        self.feature_views: Dict[str, FeatureView] = {}
        self.online_store: Dict[str, Dict[str, Dict]] = {} # {fg_name: {primary_key: {features}}}
        
        self._load_state()

    def _load_state(self):
        """Rehydrate state from disk."""
        meta = self.storage.load_metadata()
        for name, data in meta.get('feature_groups', {}).items():
            fg = FeatureGroup(name, data['schema'], data['primary_keys'], data['event_time'])
            fg.data = self.storage.load_offline_data(name)
            self.feature_groups[name] = fg
            
        for name, data in meta.get('feature_views', {}).items():
            self.feature_views[name] = FeatureView(name, data['label_fg'], data['joins'])
            
        self.online_store = self.storage.load_online_store()

    def _save_state(self):
        """Persist state to disk."""
        meta = {
            'feature_groups': {
                name: {
                    'schema': fg.schema,
                    'primary_keys': fg.primary_keys,
                    'event_time': fg.event_time
                } for name, fg in self.feature_groups.items()
            },
            'feature_views': {
                name: {
                    'label_fg': fv.label_fg,
                    'joins': fv.joins
                } for name, fv in self.feature_views.items()
            }
        }
        self.storage.save_metadata(meta)
        self.storage.save_online_store(self.online_store)

    # --- API ---

    def create_feature_group(self, name: str, schema: Dict[str, str], primary_keys: List[str], event_time: str) -> FeatureGroup:
        simple_logger(f"Creating Feature Group: {name}")
        fg = FeatureGroup(name, schema, primary_keys, event_time)
        self.feature_groups[name] = fg
        self._save_state()
        return fg

    def insert(self, fg_name: str, df: pd.DataFrame):
        """Ingest data into the Offline Store."""
        if fg_name not in self.feature_groups:
            raise ValueError(f"Feature Group {fg_name} does not exist.")
        
        simple_logger(f"Ingesting {len(df)} rows into {fg_name} (Offline Store)")
        fg = self.feature_groups[fg_name]
        
        # Append to existing data
        if not fg.data.empty:
            fg.data = pd.concat([fg.data, df]).drop_duplicates()
        else:
            fg.data = df
            
        # Save to disk
        self.storage.save_offline_data(fg_name, fg.data)

    def materialize(self, fg_name: str):
        """
        Publish latest feature values to the Online Store.
        This simulates a batch job that pushes data to Redis.
        """
        simple_logger(f"Materializing {fg_name} to Online Store...")
        fg = self.feature_groups[fg_name]
        df = fg.data.sort_values(fg.event_time)
        
        # For the online store, we only keep the *latest* value for each primary key
        latest_values = df.drop_duplicates(subset=fg.primary_keys, keep='last')
        
        if fg_name not in self.online_store:
            self.online_store[fg_name] = {}
            
        for _, row in latest_values.iterrows():
            # Create a composite key string (e.g., "user_123" or "user_123|item_456")
            key = "|".join([str(row[pk]) for pk in fg.primary_keys])
            self.online_store[fg_name][key] = row.to_dict()
            
        self._save_state()

    def create_feature_view(self, name: str, label_fg: str, joins: List[Dict]) -> FeatureView:
        simple_logger(f"Creating Feature View: {name}")
        fv = FeatureView(name, label_fg, joins)
        self.feature_views[name] = fv
        self._save_state()
        return fv

    def get_training_data(self, fv_name: str) -> pd.DataFrame:
        """
        Generates point-in-time correct training data.
        This is the magic of a Feature Store: joining historical data correctly.
        """
        simple_logger(f"Generating training data for {fv_name}...")
        fv = self.feature_views[fv_name]
        
        # 1. Start with the 'spine' (the label feature group)
        # This defines the entities and timestamps we want to predict for.
        spine_fg = self.feature_groups[fv.label_fg]
        df = spine_fg.data.copy()
        
        # 2. Join other feature groups
        for join in fv.joins:
            fg_name = join['fg_name']
            join_key = join['on']
            fg = self.feature_groups[fg_name]
            
            # Perform an ASOF join (Point-in-time join)
            # We want to match the spine's timestamp with the closest PREVIOUS timestamp in the feature group
            df = pd.merge_asof(
                df.sort_values(spine_fg.event_time),
                fg.data.sort_values(fg.event_time),
                left_on=spine_fg.event_time,
                right_on=fg.event_time,
                by=join_key,
                direction='backward', # Look into the past only! No future leakage.
                suffixes=('', f'_{fg_name}')
            )
            
        return df

    def get_online_features(self, fv_name: str, entity_keys: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieves low-latency feature vector for inference.
        """
        fv = self.feature_views[fv_name]
        result = {}
        
        # Helper to fetch from online store
        def _fetch(fg_name, keys):
            fg = self.feature_groups[fg_name]
            # Construct composite key
            composite_key = "|".join([str(keys.get(pk)) for pk in fg.primary_keys])
            return self.online_store.get(fg_name, {}).get(composite_key, {})

        # 1. Fetch from label FG (if needed, though usually labels aren't needed for inference)
        # But maybe it contains context features.
        result.update(_fetch(fv.label_fg, entity_keys))
        
        # 2. Fetch from joined FGs
        for join in fv.joins:
            fg_name = join['fg_name']
            # We assume the entity_keys dict contains the necessary join keys
            result.update(_fetch(fg_name, entity_keys))
            
        return result
