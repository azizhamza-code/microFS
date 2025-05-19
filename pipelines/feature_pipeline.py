# feature_pipeline.py 
import pandas as pd
import os
import argparse
from microfs.core_api import FeatureStore
from microfs.utils import simple_logger

def run_feature_pipeline(fs: FeatureStore, raw_data_dir: str):
    """
    Run the feature pipeline to create feature groups and ingest data.
    
    This pipeline:
    1. Reads raw CSV data from the given directory
    2. Creates feature groups if they don't exist
    3. Ingests the data into the feature groups
    
    Args:
        fs: Feature store instance
        raw_data_dir: Directory containing raw CSV files
    """
    simple_logger("info", "Starting Feature Pipeline simulation...")

    # --- Read Raw Data from CSVs ---
    try:
        df_user_activity_b1 = pd.read_csv(os.path.join(raw_data_dir, 'user_activity_batch_1.csv'))
        df_user_activity_b2 = pd.read_csv(os.path.join(raw_data_dir, 'user_activity_batch_2.csv'))
        df_user_profile_b1 = pd.read_csv(os.path.join(raw_data_dir, 'user_profile_batch_1.csv'))
        df_user_profile_b2 = pd.read_csv(os.path.join(raw_data_dir, 'user_profile_batch_2.csv'))
        df_item_feature_b1 = pd.read_csv(os.path.join(raw_data_dir, 'item_feature_batch_1.csv'))
        df_item_feature_b2 = pd.read_csv(os.path.join(raw_data_dir, 'item_feature_batch_2.csv'))
        
        # Log data summary
        simple_logger("info", f"Successfully read raw data CSVs from {raw_data_dir}")
        simple_logger("info", f"User Activity Batch 1: {df_user_activity_b1.shape[0]} rows")
        simple_logger("info", f"User Activity Batch 2: {df_user_activity_b2.shape[0]} rows")
        simple_logger("info", f"User Profile Batch 1: {df_user_profile_b1.shape[0]} rows")
        simple_logger("info", f"User Profile Batch 2: {df_user_profile_b2.shape[0]} rows")
        simple_logger("info", f"Item Feature Batch 1: {df_item_feature_b1.shape[0]} rows")
        simple_logger("info", f"Item Feature Batch 2: {df_item_feature_b2.shape[0]} rows")
    
    except FileNotFoundError as e:
        simple_logger("error", f"Raw data CSV not found: {e}. Ensure CSVs are in '{raw_data_dir}'.")
        return

    except Exception as e:
        simple_logger("error", f"Error reading CSVs: {e}")
        return

    # --- Define & Create Feature Groups (Idempotent) ---
    try:
        # User Activity Feature Group
        fg_ua_schema = {
            'user_id': 'int64', 
            'item_id': 'int64', 
            'timestamp': 'datetime64[ns, utc]', 
            'duration_sec': 'float64', 
            'activity_type': 'object', 
            'conversion': 'int64'
        }
        
        try: 
            fg_ua = fs.get_feature_group("user_activity")
            simple_logger("info", "FG 'user_activity' already exists.")
        except ValueError: 
            fg_ua = fs.create_feature_group(
                "user_activity", 
                ["user_id", "item_id"], 
                "timestamp", 
                fg_ua_schema, 
                ["user_id", "item_id"]
            )
            simple_logger("info", "FG 'user_activity' created.")

        # User Profile Feature Group
        fg_up_schema = {
            'user_id': 'int64', 
            'timestamp': 'datetime64[ns, utc]', 
            'user_level': 'object', 
            'has_premium_badge': 'bool'
        }
        
        try: 
            fg_up = fs.get_feature_group("user_profile")
            simple_logger("info", "FG 'user_profile' already exists.")
        except ValueError: 
            fg_up = fs.create_feature_group(
                "user_profile", 
                ["user_id", "timestamp"], 
                "timestamp", 
                fg_up_schema, 
                ["user_id"]
            )
            simple_logger("info", "FG 'user_profile' created.")

        # Item Feature Feature Group
        fg_if_schema = {
            'item_id': 'int64', 
            'timestamp': 'datetime64[ns, utc]', 
            'item_category': 'object', 
            'price': 'float64'
        }
        
        try: 
            fg_if = fs.get_feature_group("item_feature")
            simple_logger("info", "FG 'item_feature' already exists.")
        except ValueError: 
            fg_if = fs.create_feature_group(
                "item_feature", 
                ["item_id", "timestamp"], 
                "timestamp", 
                fg_if_schema, 
                ["item_id"]
            )
            simple_logger("info", "FG 'item_feature' created.")
    except Exception as e:
        simple_logger("error", f"Error defining/creating FGs: {e}")
        return

    # --- Ingest Data (Simulating batches from Feature Engineering jobs) ---
    try:
        simple_logger("info", "Ingesting Batch 1 data...")
        
        # Ingest User Activity Batch 1
        simple_logger("info", "Ingesting User Activity Batch 1...")
        fg_ua.insert(df_user_activity_b1.copy())
        
        # Ingest User Profile Batch 1
        simple_logger("info", "Ingesting User Profile Batch 1...")
        fg_up.insert(df_user_profile_b1.copy())
        
        # Ingest Item Feature Batch 1
        simple_logger("info", "Ingesting Item Feature Batch 1...")
        fg_if.insert(df_item_feature_b1.copy())

        simple_logger("info", "Ingesting Batch 2 data...")
        
        # Ingest User Activity Batch 2
        simple_logger("info", "Ingesting User Activity Batch 2...")
        fg_ua.insert(df_user_activity_b2.copy())
        
        # Ingest User Profile Batch 2
        simple_logger("info", "Ingesting User Profile Batch 2...")
        fg_up.insert(df_user_profile_b2.copy())
        
        # Ingest Item Feature Batch 2
        simple_logger("info", "Ingesting Item Feature Batch 2...")
        fg_if.insert(df_item_feature_b2.copy())
    except Exception as e:
        simple_logger("error", f"Error during data ingestion: {e}")
        return

    # Log summary of data in feature groups
    try:
        ua_data = fs.get_feature_group("user_activity").get_offline_data()
        up_data = fs.get_feature_group("user_profile").get_offline_data()
        if_data = fs.get_feature_group("item_feature").get_offline_data()
        
        simple_logger("info", "Feature Group Data Summary:")
        simple_logger("info", f"User Activity: {ua_data.shape[0]} rows, {ua_data.shape[1]} columns")
        simple_logger("info", f"User Profile: {up_data.shape[0]} rows, {up_data.shape[1]} columns")
        simple_logger("info", f"Item Feature: {if_data.shape[0]} rows, {if_data.shape[1]} columns")
    except Exception as e:
        simple_logger("warning", f"Error getting data summary: {e}")

    simple_logger("info", "Feature Pipeline simulation complete.")

if __name__ == '__main__':
    # This is for testing the pipeline script directly
    from microfs.utils import get_project_root, setup_project_dirs
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the feature pipeline')
    parser.add_argument('--raw_data_dir', type=str, default=None,
                        help='Directory containing raw CSV files')
    parser.add_argument('--reset', action='store_true',
                        help='Reset all feature store state before running the pipeline')
    args = parser.parse_args()
    
    # Setup directories
    setup_project_dirs()
    
    # Determine raw data directory
    if args.raw_data_dir:
        raw_data_d = args.raw_data_dir
    else:
        project_r = get_project_root()
        raw_data_d = str(project_r / "data" / "raw_data")
    
    # Initialize feature store
    fs_instance = FeatureStore()
    
    # Reset if requested
    if args.reset:
        simple_logger("warning", "Resetting feature store state...")
        fs_instance.reset_all_state_FOR_DEMO_ONLY()
    
    simple_logger("info", f"Running feature_pipeline.py standalone with raw data from: {raw_data_d}")
    run_feature_pipeline(fs_instance, raw_data_d)
    
    # Verify by listing FGs
    simple_logger("info", f"Available Feature Groups: {fs_instance.list_feature_groups()}") 