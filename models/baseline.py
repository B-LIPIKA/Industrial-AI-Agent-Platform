"""
Baseline remaining useful life models for NASA C-MAPSS dataset.
Provides a simple historical baseline for rigorous performance comparison.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any


class BaselineRULPredictor:
    """
    Simple baseline predictor that estimates remaining useful life
    using historical training trajectory statistics.
    """
    def __init__(self, strategy: str = "median"):
        self.strategy = strategy
        self.baseline_rul: float = 0.0
        self.avg_trajectory_length: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "BaselineRULPredictor":
        """Computes baseline statistics from training engines."""
        trajectory_lengths = train_df.groupby("unit_number")["time_cycles"].max()
        self.avg_trajectory_length = float(trajectory_lengths.mean())

        if self.strategy == "mean":
            self.baseline_rul = float(train_df["rul"].mean())
        else:
            self.baseline_rul = float(train_df["rul"].median())
        return self

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        """
        Baseline prediction: estimate RUL as (average historical engine life - current cycle),
        clipped at minimum 0.
        """
        # For each test engine's latest cycle
        last_cycles = test_df.groupby("unit_number")["time_cycles"].max().values
        preds = np.maximum(0.0, self.avg_trajectory_length - last_cycles)
        return preds
