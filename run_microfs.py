import os

# Assuming microfs is in PYTHONPATH or installed
from microfs.core_api import FeatureStore
from microfs.utils import simple_logger, get_project_root, setup_project_dirs

# Assuming pipelines are also importable relative to the project structure
from pipelines.feature_pipeline import run_feature_pipeline
from pipelines.training_pipeline import run_training_pipeline
from pipelines.inference_pipeline import run_inference_pipeline

if __name__ == "__main__":
    # Setup project directories first. 
    # get_project_root() in utils.py assumes utils.py is in microfs/ which is a child of project root.
    # So, calling setup_project_dirs() should work as expected if microfs package is structured correctly.
    # This will create data/fs_state and its subdirectories if they don't exist.
    setup_project_dirs()

    # Initialize the Feature Store API
    fs_api = FeatureStore()

    # --- OPTIONAL: Reset all state for a clean demo run ---
    # fs_api.reset_all_state_FOR_DEMO_ONLY()
    # print("\n" + "="*60 + "\n")

    # --- Determine Raw Data Directory ---
    # RAW_DATA_INPUT_DIR should point to microfs_project/data/raw_data/
    # get_project_root() returns the path to microfs_project/
    RAW_DATA_INPUT_DIR = get_project_root() / "data" / "raw_data"
    simple_logger("info", f"Using raw data from: {RAW_DATA_INPUT_DIR}")
    if not RAW_DATA_INPUT_DIR.exists():
        simple_logger("error", f"Raw data directory does not exist: {RAW_DATA_INPUT_DIR}")
        simple_logger("error", "Please create it and populate with CSV files (e.g., user_activity_batch_1.csv).")
        exit(1)

    # --- Run Feature Pipeline (Data Engineer) ---
    run_feature_pipeline(fs_api, str(RAW_DATA_INPUT_DIR)) # Pass as string
    print("\n" + "="*60 + "\n")

    # --- Run Training Pipeline (Data Scientist) ---
    DEMO_FV_NAME = "recommendation_clicks_v1"
    run_training_pipeline(fs_api, fv_name=DEMO_FV_NAME, compute_params_flag=True) # True to force param computation
    print("\n" + "="*60 + "\n")

    # --- Run Inference Pipeline (Serving System) ---
    sample_entity_keys = {'user_id': 1, 'item_id': 102} 
    run_inference_pipeline(fs_api, fv_name=DEMO_FV_NAME, entity_keys_for_prediction=sample_entity_keys)

    sample_entity_keys_2 = {'user_id': 2, 'item_id': 103}
    run_inference_pipeline(fs_api, fv_name=DEMO_FV_NAME, entity_keys_for_prediction=sample_entity_keys_2)

    sample_entity_keys_new_item = {'user_id': 1, 'item_id': 104} # Item 104 from batch 2
    run_inference_pipeline(fs_api, fv_name=DEMO_FV_NAME, entity_keys_for_prediction=sample_entity_keys_new_item)
    
    # Example of trying to get features for an entity that might not exist fully
    sample_entity_keys_partial = {'user_id': 999, 'item_id': 101} # User 999 might not exist
    run_inference_pipeline(fs_api, fv_name=DEMO_FV_NAME, entity_keys_for_prediction=sample_entity_keys_partial)


    print("\n" + "="*60 + "\n")
    simple_logger("info", "MicroFS Demo Complete.")
    simple_logger("info", f"Feature Store state persisted in: {get_project_root() / 'data' / 'fs_state'}") 