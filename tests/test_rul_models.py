"""
Unit tests for RUL baseline and machine learning prognostics models.
"""

import pytest
import os
import numpy as np

from models.baseline import BaselineRULPredictor
from models.inference import RULInferenceEngine
from data.loader import load_processed_data, DEFAULT_PROCESSED_DIR
from tools.engine_tools import predict_remaining_useful_life, get_model_evaluation


@pytest.fixture(scope="module")
def inference_engine():
    engine = RULInferenceEngine.get_instance()
    assert engine.load() is True
    return engine


def test_baseline_predictor():
    train_df, test_df, truth_df = load_processed_data(DEFAULT_PROCESSED_DIR, subset="FD001")
    baseline = BaselineRULPredictor()
    baseline.fit(train_df)
    preds = baseline.predict(test_df)
    
    assert len(preds) == 100
    assert (preds >= 0).all()
    assert baseline.avg_trajectory_length > 150


def test_rul_prediction_valid_engine(inference_engine):
    res = predict_remaining_useful_life(24)
    assert res["status"] == "success"
    assert res["engine_id"] == 24
    assert res["latest_observed_cycle"] == 186
    assert isinstance(res["predicted_rul"], float)
    assert res["predicted_rul"] > 0
    assert res["ground_truth_rul"] == 20.0
    assert res["unit"] == "operating cycles"
    assert "benchmark_metrics" in res


def test_rul_prediction_invalid_engine():
    res = predict_remaining_useful_life(999)
    assert res["status"] == "error"
    assert "Invalid engine ID" in res["message"]

    res_neg = predict_remaining_useful_life(-5)
    assert res_neg["status"] == "error"


def test_model_evaluation_metrics():
    eval_res = get_model_evaluation()
    assert eval_res["status"] == "success"
    eval_data = eval_res["evaluation_results"]
    
    assert "baseline" in eval_data
    assert "models" in eval_data
    assert "random_forest" in eval_data["models"]
    
    rf = eval_data["models"]["random_forest"]
    base = eval_data["baseline"]

    # Model must outperform the simple baseline
    assert rf["test_mae"] < base["test_mae"]
    assert rf["test_rmse"] < base["test_rmse"]
    assert rf["improvement_over_baseline_mae"] > 0
