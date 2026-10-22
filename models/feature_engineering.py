"""
Feature engineering and leakage-free transformations for NASA C-MAPSS dataset.
Ensures that all statistics, windows, and scalers are calculated only using observed past cycles.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler

# Active, degradation-sensitive sensors in FD001 (excluding constant variance channels)
INFORMATIVE_SENSORS = [
    "s_2", "s_3", "s_4", "s_7", "s_8", "s_9",
    "s_11", "s_12", "s_13", "s_14", "s_15", "s_17", "s_20", "s_21"
]
SETTING_COLS = ["setting_1", "setting_2"]
BASE_FEATURE_COLS = SETTING_COLS + INFORMATIVE_SENSORS


def compute_engine_window_features(
    df: pd.DataFrame,
    window_size: int = 15,
    is_training: bool = True
) -> pd.DataFrame:
    """
    Computes leakage-safe temporal features for each engine unit.
    For each cycle t:
    - Current sensor values
    - Rolling window mean over past window_size cycles
    - Rolling window std over past window_size cycles
    - Delta from first observed cycle (initial healthy state baseline)
    """
    feature_dfs = []

    # Process per engine unit to strictly prevent leakage across engine boundaries
    for unit_id, unit_df in df.groupby("unit_number"):
        unit_sorted = unit_df.sort_values("time_cycles").copy()
        
        # Base features
        features = unit_sorted[["unit_number", "time_cycles"]].copy()
        if "rul" in unit_sorted.columns:
            features["rul"] = unit_sorted["rul"].values
        if "rul_raw" in unit_sorted.columns:
            features["rul_raw"] = unit_sorted["rul_raw"].values

        # 1. Raw measurements
        for col in BASE_FEATURE_COLS:
            features[f"{col}_raw"] = unit_sorted[col].values

        # 2. Rolling window mean & std (using min_periods=1 to support initial cycles)
        for col in INFORMATIVE_SENSORS:
            roll = unit_sorted[col].rolling(window=window_size, min_periods=1)
            features[f"{col}_roll_mean"] = roll.mean().values
            features[f"{col}_roll_std"] = roll.std().fillna(0.0).values
            
            # Delta from initial healthy cycle (first reading of that engine)
            initial_val = unit_sorted[col].iloc[0]
            features[f"{col}_delta_initial"] = (unit_sorted[col] - initial_val).values

        feature_dfs.append(features)

    result_df = pd.concat(feature_dfs, ignore_index=True)
    return result_df


class CMAPSSFeaturePipeline:
    """
    Scikit-learn compatible preprocessor fitted strictly on training data.
    """
    def __init__(self, window_size: int = 15):
        self.window_size = window_size
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "CMAPSSFeaturePipeline":
        """Fits feature extraction and standard scaler strictly on training data."""
        engineered = compute_engine_window_features(train_df, self.window_size, is_training=True)
        # Exclude metadata and target columns from feature matrix
        exclude_cols = {"unit_number", "time_cycles", "rul", "rul_raw"}
        self.feature_names = [c for c in engineered.columns if c not in exclude_cols]

        X_train = engineered[self.feature_names].values
        self.scaler.fit(X_train)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
        """Transforms a dataframe into normalized feature matrix."""
        if not self.is_fitted:
            raise RuntimeError("CMAPSSFeaturePipeline must be fitted on training data before transforming.")
        engineered = compute_engine_window_features(df, self.window_size, is_training=False)
        X = engineered[self.feature_names].values
        X_scaled = self.scaler.transform(X)
        return X_scaled, engineered

    def extract_features_for_last_cycle(self, engine_history: pd.DataFrame) -> np.ndarray:
        """
        Extracts and scales features specifically for the latest cycle of a given engine.
        Ensures inference uses only observed history up to the latest cycle.
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline not fitted.")
        engineered = compute_engine_window_features(engine_history, self.window_size, is_training=False)
        # Take the row corresponding to the maximum cycle
        last_row = engineered.sort_values("time_cycles").iloc[[-1]]
        X = last_row[self.feature_names].values
        return self.scaler.transform(X)
