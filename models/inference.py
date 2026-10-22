"""
Inference engine for Turbofan Remaining Useful Life (RUL) estimation.
Loads persisted models and runs deterministic prediction on observed engine sensor histories.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

from data.loader import DEFAULT_PROCESSED_DIR, load_processed_data

DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "saved")


class RULInferenceEngine:
    """
    Production-grade inference engine for RUL prediction.
    Ensures model artifacts are loaded once, cached, and executed deterministically.
    """
    _instance: Optional["RULInferenceEngine"] = None

    def __init__(self, models_dir: str = DEFAULT_MODELS_DIR, processed_dir: str = DEFAULT_PROCESSED_DIR):
        self.models_dir = models_dir
        self.processed_dir = processed_dir
        self.model = None
        self.pipeline = None
        self.evaluation_results: Dict[str, Any] = {}
        self.test_df: Optional[pd.DataFrame] = None
        self.truth_df: Optional[pd.DataFrame] = None
        self.is_loaded = False

    @classmethod
    def get_instance(cls, models_dir: str = DEFAULT_MODELS_DIR) -> "RULInferenceEngine":
        """Singleton accessor for efficient reuse across API requests."""
        if cls._instance is None:
            cls._instance = cls(models_dir=models_dir)
            cls._instance.load()
        return cls._instance

    def load(self) -> bool:
        """Loads model artifacts and evaluation results from disk."""
        model_path = os.path.join(self.models_dir, "rul_model.joblib")
        pipeline_path = os.path.join(self.models_dir, "feature_pipeline.joblib")
        eval_path = os.path.join(self.models_dir, "evaluation_results.json")

        if not (os.path.isfile(model_path) and os.path.isfile(pipeline_path)):
            self.is_loaded = False
            return False

        self.model = joblib.load(model_path)
        self.pipeline = joblib.load(pipeline_path)

        if os.path.isfile(eval_path):
            with open(eval_path, "r", encoding="utf-8") as f:
                self.evaluation_results = json.load(f)

        # Pre-load test and truth data if available
        try:
            _, test_df, truth_df = load_processed_data(self.processed_dir, subset="FD001")
            self.test_df = test_df
            self.truth_df = truth_df
        except Exception:
            self.test_df = None
            self.truth_df = None

        self.is_loaded = True
        return True

    def predict_for_engine(
        self,
        engine_id: int,
        engine_history: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Executes RUL prediction for a specific engine using its observed history.
        """
        if not self.is_loaded:
            success = self.load()
            if not success:
                return {
                    "status": "error",
                    "error": "Trained RUL model artifacts not found. Please run 'python scripts/train_rul.py' first.",
                    "engine_id": engine_id
                }

        history = engine_history
        ground_truth = None

        # Look up from test set if history not provided directly
        if history is None:
            if self.test_df is None:
                return {
                    "status": "error",
                    "error": "Test dataset not loaded.",
                    "engine_id": engine_id
                }
            history = self.test_df[self.test_df["unit_number"] == engine_id]
            if history.empty:
                return {
                    "status": "error",
                    "error": f"Engine ID {engine_id} not found in available test trajectories (valid IDs: 1 to 100).",
                    "engine_id": engine_id
                }

            if self.truth_df is not None:
                truth_match = self.truth_df[self.truth_df["unit_number"] == engine_id]
                if not truth_match.empty:
                    ground_truth = float(truth_match["ground_truth_rul"].iloc[0])

        if len(history) == 0:
            return {
                "status": "error",
                "error": f"No telemetry records available for engine {engine_id}.",
                "engine_id": engine_id
            }

        history_sorted = history.sort_values("time_cycles")
        latest_cycle = int(history_sorted["time_cycles"].iloc[-1])
        initial_cycle = int(history_sorted["time_cycles"].iloc[0])
        total_observed_cycles = len(history_sorted)

        # Extract features for latest cycle without future leakage
        X_vec = self.pipeline.extract_features_for_last_cycle(history_sorted)
        pred_rul = float(self.model.predict(X_vec)[0])
        pred_rul = max(0.0, pred_rul)

        selected_model_name = self.evaluation_results.get("selected_best_model", "Random Forest Regressor")
        best_metrics = self.evaluation_results.get("models", {}).get(selected_model_name, {})

        prediction_error = None
        if ground_truth is not None:
            prediction_error = round(pred_rul - ground_truth, 1)

        return {
            "status": "success",
            "engine_id": int(engine_id),
            "latest_observed_cycle": latest_cycle,
            "initial_cycle": initial_cycle,
            "total_observed_cycles": total_observed_cycles,
            "predicted_rul": round(pred_rul, 1),
            "unit": "operating cycles",
            "ground_truth_rul": ground_truth,
            "prediction_error_cycles": prediction_error,
            "model_architecture": selected_model_name,
            "benchmark_metrics": {
                "test_mae": best_metrics.get("test_mae"),
                "test_rmse": best_metrics.get("test_rmse"),
                "baseline_mae": self.evaluation_results.get("baseline", {}).get("test_mae")
            },
            "limitations_note": (
                "RUL is estimated strictly in simulated discrete operational flight cycles. "
                "Translating cycles to calendar days requires operational flight schedules, "
                "environmental derate factors, and takeoff thrust profiles."
            )
        }
