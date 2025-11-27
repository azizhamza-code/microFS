# microFS

> "What I cannot create, I do not understand." - Richard Feynman

**microFS** is a tiny, educational implementation of a Feature Store, written in ~150 lines of Python. It is designed to be simple, readable, and hackable, 
It implements the three core pillars of a Feature Store:

1.  **Offline Store**: Managing historical data for training.
2.  **Point-in-Time Correctness**: Joining features without future leakage (ASOF join).
3.  **Online Store**: Serving low-latency features for real-time inference.

## Installation

```bash
pip install pandas pyarrow
```

## Quick Start

The entire logic is in [microfs.py](microfs.py). The best way to learn is to read the code!

To see it in action, run the demo:

```bash
python demo.py
```

## Core Concepts

### 1. Feature Groups
Think of these as tables in your data warehouse. They hold the raw feature data.
```python
fs.create_feature_group(
    name="user_clicks",
    primary_keys=["user_id"],
    event_time="timestamp",
    ...
)
```

### 2. Feature Views
These define how you want to *view* your data for a model. They handle the complex logic of joining different feature groups together in a point-in-time correct way.
```python
fs.create_feature_view(
    name="click_prediction_v1",
    label_fg="user_clicks",
    joins=[...]
)
```

### 3. Point-in-Time Correctness
When training a model, you need to know what the features were *at the moment the event happened*. If you use today's feature values to predict yesterday's event, you have **data leakage**.

`microFS` solves this using `pd.merge_asof`, ensuring that for every training example, we only join feature values that existed *before* the event timestamp.

### 4. Online Store
For real-time inference (e.g., when a user loads a page), you can't query a data warehouse. You need a fast key-value store (like Redis).

`microFS` simulates this by "materializing" the latest feature values into a JSON file, which acts as our KV store.

## Project Structure

-   `microfs.py`: The core library. Read this!
-   `demo.py`: A walkthrough script demonstrating the full lifecycle.