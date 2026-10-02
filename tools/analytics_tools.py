"""
Deterministic Time-Series Analytical Tools
Performs mathematical computations independently from LLM reasoning to ensure 100% numerical accuracy.
"""

import numpy as np
from typing import List, Dict, Any


def compute_series_statistics(values: List[float]) -> Dict[str, float]:
    """Computes deterministic summary statistics for a sensor time-series."""
    if not values:
        return {"error": "Empty values sequence"}
    arr = np.array(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "mean": round(float(np.mean(arr)), 3),
        "std_dev": round(float(np.std(arr)), 3),
        "min": round(float(np.min(arr)), 3),
        "max": round(float(np.max(arr)), 3),
        "median": round(float(np.median(arr)), 3)
    }


def evaluate_anomaly_thresholds(
    values: List[float],
    warning_threshold: float,
    critical_threshold: float
) -> Dict[str, Any]:
    """Evaluates values against operating safety thresholds."""
    arr = np.array(values, dtype=np.float64)
    warning_violations = np.sum((arr >= warning_threshold) & (arr < critical_threshold))
    critical_violations = np.sum(arr >= critical_threshold)

    return {
        "total_readings": len(arr),
        "warning_threshold": warning_threshold,
        "critical_threshold": critical_threshold,
        "warning_violation_count": int(warning_violations),
        "critical_violation_count": int(critical_violations),
        "max_observed": round(float(np.max(arr)), 3) if len(arr) else None,
        "is_safe": bool(critical_violations == 0)
    }


def analyze_trend_and_slope(values: List[float]) -> Dict[str, Any]:
    """Computes deterministic linear slope to identify sensor drift or degradation."""
    if len(values) < 2:
        return {"trend": "Insufficient data"}
    x = np.arange(len(values))
    y = np.array(values)
    slope, intercept = np.polyfit(x, y, 1)

    direction = "STABLE"
    if slope > 0.05:
        direction = "INCREASING (DRIFT UP)"
    elif slope < -0.05:
        direction = "DECREASING (DRIFT DOWN)"

    return {
        "slope_per_step": round(float(slope), 4),
        "direction": direction,
        "initial_value": round(float(y[0]), 2),
        "final_value": round(float(y[-1]), 2)
    }
