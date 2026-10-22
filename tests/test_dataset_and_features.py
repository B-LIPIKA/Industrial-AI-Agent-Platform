"""
Unit tests for NASA C-MAPSS dataset loading, schema verification,
RUL target construction, and data leakage safeguards.
"""

import pytest
import numpy as np
import pandas as pd

from data.loader import (
    load_processed_data,
    check_dataset_availability,
    DEFAULT_PROCESSED_DIR,
    DEFAULT_RAW_DIR,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    INDEX_COLUMNS
)
from models.feature_engineering import (
    compute_engine_window_features,
    CMAPSSFeaturePipeline,
    INFORMATIVE_SENSORS
)


@pytest.fixture(scope="module")
def dataset_frames():
    return load_processed_data(DEFAULT_PROCESSED_DIR, subset="FD001")


def test_dataset_availability():
    status = check_dataset_availability(DEFAULT_RAW_DIR, subset="FD001")
    assert status["is_ready"] is True
    assert status["files"]["train"]["exists"] is True
    assert status["files"]["test"]["exists"] is True
    assert status["files"]["rul"]["exists"] is True
    assert status["files"]["train"]["size_bytes"] > 1_000_000


def test_dataset_schema(dataset_frames):
    train_df, test_df, truth_df = dataset_frames
    
    # Check expected column sets
    for col in INDEX_COLUMNS + SETTING_COLUMNS + SENSOR_COLUMNS:
        assert col in train_df.columns
        assert col in test_df.columns

    assert "rul" in train_df.columns
    assert "rul_raw" in train_df.columns
    assert "ground_truth_rul" in truth_df.columns

    # Verify engine counts and shapes
    assert train_df["unit_number"].nunique() == 100
    assert test_df["unit_number"].nunique() == 100
    assert len(truth_df) == 100
    assert len(train_df) == 20631
    assert len(test_df) == 13096


def test_rul_target_construction(dataset_frames):
    train_df, _, _ = dataset_frames
    
    # Verify RUL strictly decreases to 0 at the last cycle of each training engine
    for unit_id in [1, 2, 50, 100]:
        unit_df = train_df[train_df["unit_number"] == unit_id].sort_values("time_cycles")
        last_raw_rul = unit_df["rul_raw"].iloc[-1]
        assert last_raw_rul == 0, f"Engine {unit_id} raw RUL did not terminate at 0"

        # Verify piecewise clipping upper bound
        assert unit_df["rul"].max() <= 125


def test_data_leakage_prevention(dataset_frames):
    train_df, _, _ = dataset_frames
    sample_engine = train_df[train_df["unit_number"] == 1].head(30)
    
    window_df = compute_engine_window_features(sample_engine, window_size=15)
    
    # Feature at cycle t must only depend on cycles <= t
    # For cycle 1, rolling mean must equal the cycle 1 value exactly
    val_c1 = sample_engine["s_2"].iloc[0]
    roll_c1 = window_df["s_2_roll_mean"].iloc[0]
    assert np.isclose(val_c1, roll_c1), "Cycle 1 rolling mean should equal initial reading"

    # Delta initial must be 0 at cycle 1
    delta_c1 = window_df["s_2_delta_initial"].iloc[0]
    assert np.isclose(delta_c1, 0.0), "Initial delta must be 0"


def test_cross_engine_isolation(dataset_frames):
    train_df, _, _ = dataset_frames
    two_engines = train_df[train_df["unit_number"].isin([1, 2])]
    
    window_df = compute_engine_window_features(two_engines, window_size=15)
    
    # Ensure cycle 1 of engine 2 has delta_initial == 0 (not referencing engine 1)
    eng2_row1 = window_df[window_df["unit_number"] == 2].iloc[0]
    assert np.isclose(eng2_row1["s_2_delta_initial"], 0.0)


def test_feature_pipeline_fit_transform(dataset_frames):
    train_df, test_df, _ = dataset_frames
    pipeline = CMAPSSFeaturePipeline(window_size=15)
    
    # Fit strictly on train
    pipeline.fit(train_df)
    assert pipeline.is_fitted is True
    assert len(pipeline.feature_names) == 58  # 16 raw + 14 mean + 14 std + 14 delta

    # Transform test engine last cycle
    eng_test = test_df[test_df["unit_number"] == 24]
    X_last = pipeline.extract_features_for_last_cycle(eng_test)
    assert X_last.shape == (1, 58)
    assert not np.isnan(X_last).any()
