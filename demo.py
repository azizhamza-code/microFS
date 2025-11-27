"""
demo.py - A complete walkthrough of MicroFS.

This script demonstrates the end-to-end lifecycle of a feature store:
1. Setup: Define Feature Groups and Feature Views.
2. Ingestion: Load historical data into the Offline Store.
3. Training: Generate point-in-time correct training data.
4. Deployment: Materialize features to the Online Store.
5. Inference: Fetch low-latency features for real-time prediction.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from microfs import FeatureStore

def main():
    # -------------------------------------------------------------------------
    # 0. Initialize Feature Store
    # -------------------------------------------------------------------------
    print("\n=== 0. Initializing MicroFS ===")
    fs = FeatureStore()
    fs.storage.clear() # Start fresh for the demo
    fs = FeatureStore() # Re-init to create dirs

    # -------------------------------------------------------------------------
    # 1. Define Feature Groups
    # -------------------------------------------------------------------------
    print("\n=== 1. Defining Feature Groups ===")
    
    # User Activity: Events like clicks, views
    fs.create_feature_group(
        name="user_activity",
        schema={"user_id": "int", "activity_count": "int", "timestamp": "datetime"},
        primary_keys=["user_id"],
        event_time="timestamp"
    )

    # User Payments: Transaction history
    fs.create_feature_group(
        name="user_payments",
        schema={"user_id": "int", "total_spend": "float", "timestamp": "datetime"},
        primary_keys=["user_id"],
        event_time="timestamp"
    )

    # -------------------------------------------------------------------------
    # 2. Ingest Data (Offline Store)
    # -------------------------------------------------------------------------
    print("\n=== 2. Ingesting Historical Data ===")

    # Generate some dummy data
    now = datetime.now()
    
    # User Activity Data
    activity_data = pd.DataFrame({
        "user_id": [1, 1, 2, 2],
        "activity_count": [10, 20, 5, 15],
        "timestamp": [
            now - timedelta(days=5),
            now - timedelta(days=1), # User 1 became more active recently
            now - timedelta(days=5),
            now - timedelta(days=2)
        ]
    })
    fs.insert("user_activity", activity_data)
    print("Ingested 'user_activity' data:")
    print(activity_data)

    # User Payments Data
    payments_data = pd.DataFrame({
        "user_id": [1, 2],
        "total_spend": [100.50, 50.00],
        "timestamp": [
            now - timedelta(days=3), # Payment happened between the two activity events
            now - timedelta(days=3)
        ]
    })
    fs.insert("user_payments", payments_data)
    print("\nIngested 'user_payments' data:")
    print(payments_data)

    # -------------------------------------------------------------------------
    # 3. Create Feature View & Generate Training Data
    # -------------------------------------------------------------------------
    print("\n=== 3. Training: Point-in-Time Correctness ===")
    
    # We want to predict something based on user activity AND payments.
    # We join 'user_payments' onto 'user_activity'.
    
    fs.create_feature_view(
        name="activity_payment_fv",
        label_fg="user_activity", # The 'spine' of our dataset
        joins=[
            {"fg_name": "user_payments", "on": "user_id"}
        ]
    )
    
    training_df = fs.get_training_data("activity_payment_fv")
    
    print("Training Data (Point-in-Time Join):")
    print(training_df[["user_id", "timestamp", "activity_count", "total_spend"]])
    
    print("\nNotice for User 1 at T-1 day (row 1):")
    print("- Activity Count is 20 (from T-1)")
    print("- Total Spend is 100.50 (from T-3)")
    print("-> The join correctly picked up the payment that happened BEFORE the activity.")

    # -------------------------------------------------------------------------
    # 4. Materialize to Online Store
    # -------------------------------------------------------------------------
    print("\n=== 4. Materializing to Online Store ===")
    # Push the LATEST values to the online store for real-time serving
    fs.materialize("user_activity")
    fs.materialize("user_payments")
    print("Done.")

    # -------------------------------------------------------------------------
    # 5. Online Inference
    # -------------------------------------------------------------------------
    print("\n=== 5. Online Inference ===")
    
    # Let's get features for User 1
    features = fs.get_online_features(
        fv_name="activity_payment_fv",
        entity_keys={"user_id": 1}
    )
    
    print(f"Features for User 1: {features}")
    print("These are the latest values available, ready for the model!")

if __name__ == "__main__":
    main()
