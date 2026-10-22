"""
Machine Learning models for Turbofan Remaining Useful Life (RUL) estimation.
Supports Random Forest and HistGradientBoosting regressors with rigorous evaluation.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from models.feature_engineering import CMAPSSFeaturePipeline
from models.baseline import BaselineRULPredictor

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved")


def compute_nasa_scoring(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes standard NASA C-MAPSS asymmetric evaluation score:
    Penalizes late predictions (overestimating remaining life) more heavily than early predictions.
    d = y_pred - y_true
    s = sum(exp(-d/13) - 1) for d < 0 (late predictions)
    s = sum(exp(d/10) - 1) for d >= 0 (early predictions)
    """
    d = y_pred - y_true
    score = 0.0
    for diff in d:
        if diff < 0:
            score += np.exp(-diff / 13.0) - 1.0
        else:
            score += np.exp(diff / 10.0) - 1.0
    return float(score)


def train_and_evaluate_rul_pipeline(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    truth_df: pd.DataFrame,
    save_dir: str = DEFAULT_MODELS_DIR,
    model_type: str = "random_forest"
) -> Dict[str, Any]:
    """
    Trains feature pipeline, baseline, and ML regressor.
    Evaluates both on the test set engines at their latest observed cycle.
    Persists trained artifacts and evaluation metrics.
    """
    os.makedirs(save_dir, exist_ok=True)

    print("1. Fitting CMAPSS Feature Pipeline strictly on training data...")
    pipeline = CMAPSSFeaturePipeline(window_size=15)
    pipeline.fit(train_df)

    # Transform training data
    X_train, engineered_train = pipeline.transform(train_df)
    y_train = engineered_train["rul"].values

    # Train baseline
    print("2. Fitting Baseline RUL predictor...")
    baseline = BaselineRULPredictor()
    baseline.fit(train_df)
    y_pred_baseline = baseline.predict(test_df)
    y_test_true = truth_df["ground_truth_rul"].values

    base_mae = float(mean_absolute_error(y_test_true, y_pred_baseline))
    base_rmse = float(root_mean_squared_error(y_test_true, y_pred_baseline))
    base_nasa = compute_nasa_scoring(y_test_true, y_pred_baseline)

    print(f"   Baseline -> MAE: {base_mae:.2f} cycles, RMSE: {base_rmse:.2f} cycles, NASA Score: {base_nasa:.1f}")

    # Prepare test set features: extract features for the LAST observed cycle of each test engine
    print("3. Extracting features for test engines at their last observed cycle...")
    X_test_list = []
    for unit_id in truth_df["unit_number"].values:
        unit_history = test_df[test_df["unit_number"] == unit_id]
        feat_vector = pipeline.extract_features_for_last_cycle(unit_history)
        X_test_list.append(feat_vector[0])
    X_test = np.array(X_test_list)

    # Train selected model
    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=4, random_state=42, n_jobs=-1
        ),
        "gradient_boosting": HistGradientBoostingRegressor(
            max_iter=100, max_depth=8, min_samples_leaf=10, random_state=42
        )
    }

    eval_results = {
        "baseline": {
            "name": "Historical Trajectory Average Baseline",
            "test_mae": round(base_mae, 2),
            "test_rmse": round(base_rmse, 2),
            "nasa_score": round(base_nasa, 1)
        },
        "models": {}
    }

    best_model = None
    best_name = None
    best_mae = float("inf")

    for name, model in models.items():
        print(f"4. Training {name} regressor...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Clip negative predictions at 0
        y_pred = np.maximum(0.0, y_pred)

        mae = float(mean_absolute_error(y_test_true, y_pred))
        rmse = float(root_mean_squared_error(y_test_true, y_pred))
        r2 = float(r2_score(y_test_true, y_pred))
        nasa_score = compute_nasa_scoring(y_test_true, y_pred)

        print(f"   {name} -> MAE: {mae:.2f} cycles, RMSE: {rmse:.2f} cycles, R2: {r2:.3f}, NASA Score: {nasa_score:.1f}")

        eval_results["models"][name] = {
            "name": name,
            "test_mae": round(mae, 2),
            "test_rmse": round(rmse, 2),
            "r2_score": round(r2, 3),
            "nasa_score": round(nasa_score, 1),
            "improvement_over_baseline_mae": round(base_mae - mae, 2)
        }

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

    eval_results["selected_best_model"] = best_name
    eval_results["evaluation_metadata"] = {
        "dataset": "NASA C-MAPSS FD001",
        "train_engines": int(train_df["unit_number"].nunique()),
        "test_engines": int(truth_df["unit_number"].nunique()),
        "features_count": len(pipeline.feature_names),
        "split_protocol": "Official benchmark split (100 run-to-failure training engines, 100 truncated test engines evaluated on ground truth RUL)",
        "leakage_safeguards": "Rolling window stats computed per-engine strictly within past observed cycles; StandardScaler fitted exclusively on training set"
    }

    # Persist artifacts
    print(f"5. Persisting artifacts (Best Model: {best_name}) to {save_dir}...")
    joblib.dump(best_model, os.path.join(save_dir, "rul_model.joblib"))
    joblib.dump(pipeline, os.path.join(save_dir, "feature_pipeline.joblib"))
    joblib.dump(baseline, os.path.join(save_dir, "baseline_model.joblib"))

    with open(os.path.join(save_dir, "feature_names.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline.feature_names, f, indent=2)

    with open(os.path.join(save_dir, "evaluation_results.json"), "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2)

    print("[OK] Artifacts and evaluation metrics saved successfully.")
    return eval_results
