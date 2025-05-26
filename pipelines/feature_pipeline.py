# feature_pipeline.py 
import pandas as pd
import os
import argparse
from microfs.core import FeatureStore
from microfs.utils import simple_logger

def run_feature_pipeline(fs: FeatureStore, raw_data_dir: str):
    """
    Run the feature pipeline to create feature groups and ingest data.
    
    This pipeline:
    1. Reads raw CSV data from the given directory
    2. Creates feature groups if they don't exist
    3. Ingests the data into the feature groups
    4. Creates a feature view for ML training
    
    Args:
        fs: Feature store instance
        raw_data_dir: Directory containing raw CSV files
    """
    simple_logger("info", "Starting Feature Pipeline...")

    # --- Read Raw Data from CSVs ---
    try:
        df_user_activity_b1 = pd.read_csv(os.path.join(raw_data_dir, 'user_activity_batch_1.csv'))
        df_user_activity_b1['timestamp'] = pd.to_datetime(df_user_activity_b1['timestamp'], utc=True)
        
        df_user_activity_b2 = pd.read_csv(os.path.join(raw_data_dir, 'user_activity_batch_2.csv'))
        df_user_activity_b2['timestamp'] = pd.to_datetime(df_user_activity_b2['timestamp'], utc=True)
        
        df_user_profile_b1 = pd.read_csv(os.path.join(raw_data_dir, 'user_profile_batch_1.csv'))
        df_user_profile_b1['timestamp'] = pd.to_datetime(df_user_profile_b1['timestamp'], utc=True)
        
        df_user_profile_b2 = pd.read_csv(os.path.join(raw_data_dir, 'user_profile_batch_2.csv'))
        df_user_profile_b2['timestamp'] = pd.to_datetime(df_user_profile_b2['timestamp'], utc=True)
        
        df_item_feature_b1 = pd.read_csv(os.path.join(raw_data_dir, 'item_feature_batch_1.csv'))
        df_item_feature_b1['timestamp'] = pd.to_datetime(df_item_feature_b1['timestamp'], utc=True)
        
        df_item_feature_b2 = pd.read_csv(os.path.join(raw_data_dir, 'item_feature_batch_2.csv'))
        df_item_feature_b2['timestamp'] = pd.to_datetime(df_item_feature_b2['timestamp'], utc=True)
        
        simple_logger("info", f"Successfully read raw data CSVs from {raw_data_dir}")
        simple_logger("info", f"User Activity: {df_user_activity_b1.shape[0] + df_user_activity_b2.shape[0]} total rows")
        simple_logger("info", f"User Profile: {df_user_profile_b1.shape[0] + df_user_profile_b2.shape[0]} total rows")
        simple_logger("info", f"Item Feature: {df_item_feature_b1.shape[0] + df_item_feature_b2.shape[0]} total rows")
    
    except FileNotFoundError as e:
        simple_logger("error", f"Raw data CSV not found: {e}. Ensure CSVs are in '{raw_data_dir}'.")
        return
    except Exception as e:
        simple_logger("error", f"Error reading CSVs: {e}")
        return

    # --- Create Feature Groups ---
    try:
        # User Activity Feature Group
        if "user_activity" not in fs.list_feature_groups():
            fg_ua = fs.create_feature_group(
                "user_activity", 
                ["user_id", "item_id"], 
                "timestamp", 
                {
                    'user_id': 'int64', 
                    'item_id': 'int64', 
                    'timestamp': 'datetime64[ns, utc]', 
                    'duration_sec': 'float64', 
                    'activity_type': 'object', 
                    'conversion': 'int64'
                }, 
                ["user_id", "item_id"]
            )
            simple_logger("info", "Created feature group 'user_activity'")
        else:
            fg_ua = fs.get_feature_group("user_activity")
            simple_logger("info", "Feature group 'user_activity' already exists")

        # User Profile Feature Group
        if "user_profile" not in fs.list_feature_groups():
            fg_up = fs.create_feature_group(
                "user_profile", 
                ["user_id", "timestamp"], 
                "timestamp", 
                {
                    'user_id': 'int64', 
                    'timestamp': 'datetime64[ns, utc]', 
                    'user_level': 'object', 
                    'has_premium_badge': 'bool'
                }, 
                ["user_id"]
            )
            simple_logger("info", "Created feature group 'user_profile'")
        else:
            fg_up = fs.get_feature_group("user_profile")
            simple_logger("info", "Feature group 'user_profile' already exists")

        # Item Feature Feature Group
        if "item_feature" not in fs.list_feature_groups():
            fg_if = fs.create_feature_group(
                "item_feature", 
                ["item_id", "timestamp"], 
                "timestamp", 
                {
                    'item_id': 'int64', 
                    'timestamp': 'datetime64[ns, utc]', 
                    'item_category': 'object', 
                    'price': 'float64'
                }, 
                ["item_id"]
            )
            simple_logger("info", "Created feature group 'item_feature'")
        else:
            fg_if = fs.get_feature_group("item_feature")
            simple_logger("info", "Feature group 'item_feature' already exists")
            
    except Exception as e:
        simple_logger("error", f"Error creating feature groups: {e}")
        return

    # --- Ingest Data ---
    try:
        simple_logger("info", "Ingesting data...")
        
        # Combine batches and ingest
        user_activity_data = pd.concat([df_user_activity_b1, df_user_activity_b2], ignore_index=True)
        fg_ua.insert(user_activity_data)
        
        user_profile_data = pd.concat([df_user_profile_b1, df_user_profile_b2], ignore_index=True)
        fg_up.insert(user_profile_data)
        
        item_feature_data = pd.concat([df_item_feature_b1, df_item_feature_b2], ignore_index=True)
        fg_if.insert(item_feature_data)
        
        simple_logger("info", "Data ingestion complete")
        
    except Exception as e:
        simple_logger("error", f"Error during data ingestion: {e}")
        return

    # --- Create Feature View for ML Training ---
    try:
        fv_name = "recommendation_clicks_v1"
        if fv_name not in fs.list_feature_views():
            fv = fs.create_feature_view(
                fv_name,
                "user_activity",  # label feature group
                "conversion",     # label column
                [
                    {'fg_name': 'user_profile', 'on': ['user_id']},
                    {'fg_name': 'item_feature', 'on': ['item_id']}
                ],
                [
                    {'type': 'scale', 'column': 'duration_sec'},
                    {'type': 'scale', 'column': 'price'},
                    {'type': 'one_hot_encode', 'column': 'user_level'},
                    {'type': 'one_hot_encode', 'column': 'item_category'}
                ]
            )
            simple_logger("info", f"Created feature view '{fv_name}'")
        else:
            simple_logger("info", f"Feature view '{fv_name}' already exists")
            
    except Exception as e:
        simple_logger("error", f"Error creating feature view: {e}")

    simple_logger("info", "Feature Pipeline complete")

if __name__ == '__main__':
    from microfs.utils import get_project_root, setup_project_dirs
    
    parser = argparse.ArgumentParser(description='Run the feature pipeline')
    parser.add_argument('--raw_data_dir', type=str, default=None,
                        help='Directory containing raw CSV files')
    parser.add_argument('--reset', action='store_true',
                        help='Reset all feature store state before running the pipeline')
    args = parser.parse_args()
    
    setup_project_dirs()
    
    if args.raw_data_dir:
        raw_data_d = args.raw_data_dir
    else:
        project_r = get_project_root()
        raw_data_d = str(project_r / "data" / "raw_data")
    
    fs_instance = FeatureStore()
    
    if args.reset:
        simple_logger("warning", "Resetting feature store state...")
        fs_instance.reset_all_state()
    
    simple_logger("info", f"Running feature pipeline with raw data from: {raw_data_d}")
    run_feature_pipeline(fs_instance, raw_data_d)
    
    simple_logger("info", f"Available Feature Groups: {fs_instance.list_feature_groups()}")
    simple_logger("info", f"Available Feature Views: {fs_instance.list_feature_views()}") 