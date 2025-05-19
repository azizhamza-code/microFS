# utils.py
# General utility functions (path helpers, simple logging) 

import os
import pathlib
import json
import pandas as pd
from typing import Dict, Any
import datetime # Added missing import

# --- Constants for state management ---
_BASE_STATE_DIR_NAME = "fs_state"
_METADATA_SUBDIR_NAME = "metadata"
_OFFLINE_SUBDIR_NAME = "offline_store"
_ONLINE_SUBDIR_NAME = "online_store"

_FG_METADATA_FILE = "feature_groups.json"
_FV_METADATA_FILE = "feature_views.json"
_ONLINE_STORE_FILE = "online_store.json" # Simple JSON for online dict persistence

def get_project_root() -> pathlib.Path:
    """Returns the project root directory, assuming 'microfs' is a direct subdir."""
    # Assuming this file (utils.py) is in microfs/
    return pathlib.Path(__file__).resolve().parent.parent

def get_state_dir() -> pathlib.Path:
    """Returns the base directory for storing feature store state (PROJECT_ROOT/data/fs_state)."""
    return get_project_root() / "data" / _BASE_STATE_DIR_NAME

def get_metadata_dir() -> pathlib.Path:
    return get_state_dir() / _METADATA_SUBDIR_NAME

def get_offline_store_dir() -> pathlib.Path:
    return get_state_dir() / _OFFLINE_SUBDIR_NAME

def get_online_store_dir() -> pathlib.Path:
    return get_state_dir() / _ONLINE_SUBDIR_NAME

def setup_project_dirs():
    """Creates necessary directories for the project and feature store state."""
    print("Setting up project directories...")
    # data/raw_data should be relative to project root, not utils.py's parent's parent
    (get_project_root() / "data" / "raw_data").mkdir(parents=True, exist_ok=True)
    get_metadata_dir().mkdir(parents=True, exist_ok=True)
    get_offline_store_dir().mkdir(parents=True, exist_ok=True)
    get_online_store_dir().mkdir(parents=True, exist_ok=True)
    print(f"  Data dir for raw CSVs: {get_project_root() / 'data' / 'raw_data'}")
    print(f"  Feature store state dir: {get_state_dir()}")

# JSON Serialization Helpers (for Timestamps and tuple keys in online store)
class _CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat() # Convert Timestamp to ISO string
        if isinstance(obj, (datetime.datetime, datetime.date)): # Needs datetime import
            return obj.isoformat()
        if pd.isna(obj): # Handle Pandas NA/NaN
             return None
        # For online store dictionary keys if they are tuples
        if isinstance(obj, tuple) and hasattr(self, '_is_serializing_online_store_keys') and self._is_serializing_online_store_keys:
            return f"__tuple__{json.dumps(list(obj))}"
        return json.JSONEncoder.default(self, obj)

def _serialize_online_store_keys(online_store_dict: Dict[str, Dict[tuple, Any]]) -> Dict[str, Dict[str, Any]]:
    """Converts tuple keys of the online store to strings for JSON serialization."""
    serialized = {}
    encoder = _CustomJSONEncoder()
    encoder._is_serializing_online_store_keys = True # Set flag for tuple key handling
    for fg_name, fg_data in online_store_dict.items():
        serialized[fg_name] = {encoder.encode(k): v for k, v in fg_data.items()}
    return serialized

def _deserialize_online_store_keys(json_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[tuple, Any]]:
    """Converts string keys from JSON back to tuples for the online store."""
    deserialized = {}
    for fg_name, fg_data_str_keys in json_data.items():
        deserialized_fg_data = {}
        for k_str, v in fg_data_str_keys.items():
            if k_str.startswith("__tuple__"):
                try:
                    list_key = json.loads(k_str[len("__tuple__"):])
                    deserialized_fg_data[tuple(list_key)] = v
                except json.JSONDecodeError:
                    deserialized_fg_data[k_str] = v # Fallback if not a tuple
            else:
                deserialized_fg_data[k_str] = v # Should not happen if saved correctly
        deserialized[fg_name] = deserialized_fg_data
    return deserialized

def simple_logger(level: str, message: str):
    """A very basic logger."""
    print(f"[{datetime.datetime.now().isoformat()}] [{level.upper()}] {message}") 