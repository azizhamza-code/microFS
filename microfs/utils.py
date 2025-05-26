# utils.py
# General utility functions (path helpers, simple logging, file I/O, state management) 

import os
import pathlib
import json
import pandas as pd
from typing import Dict, Any
import datetime
import shutil

# --- Constants for state management ---
_BASE_STATE_DIR_NAME = "fs_state"
_METADATA_SUBDIR_NAME = "metadata"
_OFFLINE_SUBDIR_NAME = "offline_store"
_ONLINE_SUBDIR_NAME = "online_store"

_FG_METADATA_FILE = "feature_groups.json"
_FV_METADATA_FILE = "feature_views.json"
_ONLINE_STORE_FILE = "online_store.json"

# --- Global State (moved from core.py) ---
_fg_metadata = {}
_fv_metadata = {}
_online_store = {}
_custom_transformers = {}

def get_project_root() -> pathlib.Path:
    """Returns the project root directory, assuming 'microfs' is a direct subdir."""
    return pathlib.Path(__file__).resolve().parent.parent

def get_state_dir() -> pathlib.Path:
    """Returns the base directory for storing feature store state."""
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
    (get_project_root() / "data" / "raw_data").mkdir(parents=True, exist_ok=True)
    get_metadata_dir().mkdir(parents=True, exist_ok=True)
    get_offline_store_dir().mkdir(parents=True, exist_ok=True)
    get_online_store_dir().mkdir(parents=True, exist_ok=True)
    print(f"  Data dir for raw CSVs: {get_project_root() / 'data' / 'raw_data'}")
    print(f"  Feature store state dir: {get_state_dir()}")

def simple_logger(level: str, message: str):
    """A very basic logger."""
    print(f"[{datetime.datetime.now().isoformat()}] [{level.upper()}] {message}")

# --- File I/O Operations ---

def load_json(file_path: pathlib.Path) -> dict:
    """Load JSON file or return empty dict if not exists"""
    if file_path.exists():
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

def save_json(data: dict, file_path: pathlib.Path):
    """Save data to JSON file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_parquet(file_path: pathlib.Path) -> pd.DataFrame:
    """Load parquet file or return empty DataFrame"""
    if file_path.exists():
        return pd.read_parquet(file_path)
    return pd.DataFrame()

def save_parquet(df: pd.DataFrame, file_path: pathlib.Path):
    """Save DataFrame to parquet file"""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(file_path, index=False)

# --- State Persistence Functions (moved from core.py) ---

def load_metadata():
    """Load all metadata from disk"""
    global _fg_metadata, _fv_metadata
    _fg_metadata = load_json(get_metadata_dir() / _FG_METADATA_FILE)
    _fv_metadata = load_json(get_metadata_dir() / _FV_METADATA_FILE)

def save_metadata():
    """Save all metadata to disk"""
    save_json(_fg_metadata, get_metadata_dir() / _FG_METADATA_FILE)
    save_json(_fv_metadata, get_metadata_dir() / _FV_METADATA_FILE)

def load_online_store():
    """Load online store from disk"""
    global _online_store
    data = load_json(get_online_store_dir() / _ONLINE_STORE_FILE)
    _online_store = deserialize_online_store_keys(data)

def save_online_store():
    """Save online store to disk"""
    data = serialize_online_store_keys(_online_store)
    save_json(data, get_online_store_dir() / _ONLINE_STORE_FILE)

# --- Global State Access Functions ---

def get_fg_metadata() -> Dict[str, Any]:
    """Get feature group metadata dictionary"""
    return _fg_metadata

def get_fv_metadata() -> Dict[str, Any]:
    """Get feature view metadata dictionary"""
    return _fv_metadata

def get_online_store() -> Dict[str, Any]:
    """Get online store dictionary"""
    return _online_store

def get_custom_transformers() -> Dict[str, Any]:
    """Get custom transformers dictionary"""
    return _custom_transformers

def set_fg_metadata(metadata: Dict[str, Any]):
    """Set feature group metadata dictionary"""
    global _fg_metadata
    _fg_metadata = metadata

def set_fv_metadata(metadata: Dict[str, Any]):
    """Set feature view metadata dictionary"""
    global _fv_metadata
    _fv_metadata = metadata

def set_online_store(store: Dict[str, Any]):
    """Set online store dictionary"""
    global _online_store
    _online_store = store

def set_custom_transformers(transformers: Dict[str, Any]):
    """Set custom transformers dictionary"""
    global _custom_transformers
    _custom_transformers = transformers

# --- State Management ---

def clear_all_state():
    """Clear all feature store state from disk"""
    state_dirs = [get_metadata_dir(), get_offline_store_dir(), get_online_store_dir()]
    for dir_path in state_dirs:
        if dir_path.exists():
            shutil.rmtree(dir_path)
    simple_logger("info", "All feature store state cleared from disk")

def reset_all_state():
    """Reset all feature store state (both in-memory and disk)"""
    global _fg_metadata, _fv_metadata, _online_store, _custom_transformers
    
    # Clear in-memory state
    _fg_metadata.clear()
    _fv_metadata.clear()
    _online_store.clear()
    _custom_transformers.clear()
    
    # Clear disk state
    clear_all_state()

# --- JSON Serialization Helpers ---

def serialize_online_store_keys(online_store_dict: Dict[str, Dict[tuple, Any]]) -> Dict[str, Dict[str, Any]]:
    """Converts tuple keys of the online store to strings for JSON serialization."""
    serialized = {}
    for fg_name, fg_data in online_store_dict.items():
        serialized[fg_name] = {}
        for key, value in fg_data.items():
            if isinstance(key, tuple):
                key_str = json.dumps(list(key))
            else:
                key_str = str(key)
            serialized[fg_name][key_str] = value
    return serialized

def deserialize_online_store_keys(json_data: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[tuple, Any]]:
    """Converts string keys from JSON back to tuples for the online store."""
    deserialized = {}
    for fg_name, fg_data_str_keys in json_data.items():
        deserialized_fg_data = {}
        for k_str, v in fg_data_str_keys.items():
            try:
                # Try to parse as JSON list and convert to tuple
                list_key = json.loads(k_str)
                if isinstance(list_key, list):
                    deserialized_fg_data[tuple(list_key)] = v
                else:
                    deserialized_fg_data[k_str] = v
            except (json.JSONDecodeError, ValueError):
                # If parsing fails, use string key as-is
                deserialized_fg_data[k_str] = v
        deserialized[fg_name] = deserialized_fg_data
    return deserialized 