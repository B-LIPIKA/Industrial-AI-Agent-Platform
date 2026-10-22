"""
Deterministic Tool Registry for EngineMind AI Copilot.
All numerical calculations, anomaly detection, time-series aggregations,
and ML inference are strictly executed in deterministic Python functions.
No LLM hallucinations or invented metrics.
"""

import os
import json
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from data.loader import (
    load_processed_data,
    DEFAULT_PROCESSED_DIR,
    SENSOR_METADATA,
    SENSOR_COLUMNS,
    SETTING_COLUMNS,
    INDEX_COLUMNS,
    check_dataset_availability,
    DEFAULT_RAW_DIR
)
from models.inference import RULInferenceEngine
from rag.rag_engine import TechnicalDocumentRAG

# In-memory cached dataset access
_CACHED_DATA: Dict[str, pd.DataFrame] = {}


def _get_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    global _CACHED_DATA
    if "train" not in _CACHED_DATA:
        train_df, test_df, truth_df = load_processed_data(DEFAULT_PROCESSED_DIR, subset="FD001")
        _CACHED_DATA["train"] = train_df
        _CACHED_DATA["test"] = test_df
        _CACHED_DATA["truth"] = truth_df
    return _CACHED_DATA["train"], _CACHED_DATA["test"], _CACHED_DATA["truth"]


def get_dataset_summary() -> Dict[str, Any]:
    """
    Returns verified data provenance, dimensions, available engines, and validation status.
    """
    raw_status = check_dataset_availability(DEFAULT_RAW_DIR, subset="FD001")
    prov_file = os.path.join(DEFAULT_RAW_DIR, "provenance_FD001.json")
    provenance = {}
    if os.path.isfile(prov_file):
        with open(prov_file, "r", encoding="utf-8") as f:
            provenance = json.load(f)

    meta_file = os.path.join(DEFAULT_PROCESSED_DIR, "metadata_FD001.json")
    meta = {}
    if os.path.isfile(meta_file):
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)

    train_df, test_df, truth_df = _get_data()

    return {
        "status": "success",
        "dataset_name": "NASA C-MAPSS Turbofan Engine Degradation Simulation",
        "subset": "FD001",
        "is_ready": raw_status.get("is_ready", False),
        "train_engines_count": int(train_df["unit_number"].nunique()),
        "train_rows_count": int(len(train_df)),
        "test_engines_count": int(test_df["unit_number"].nunique()),
        "test_rows_count": int(len(test_df)),
        "sensor_channels_count": len(SENSOR_COLUMNS),
        "operating_settings_count": len(SETTING_COLUMNS),
        "available_engine_ids_range": [1, 100],
        "provenance": {
            "retrieval_timestamp_utc": provenance.get("retrieval_timestamp_utc"),
            "sources": provenance.get("sources"),
            "kaggle_competition_url": provenance.get("kaggle_competition_url"),
            "nasa_pcoe_url": provenance.get("nasa_pcoe_url"),
            "license_notes": provenance.get("license_notes")
        }
    }


def get_engine_history(
    engine_id: int,
    split: str = "test",
    max_cycles: Optional[int] = None
) -> Dict[str, Any]:
    """
    Returns actual observed telemetry history and available sensor channels for a specific engine.
    """
    if not isinstance(engine_id, int) or engine_id < 1 or engine_id > 100:
        return {
            "status": "error",
            "message": f"Invalid engine ID '{engine_id}'. Valid engine IDs in FD001 are integers 1 through 100.",
            "engine_id": engine_id
        }

    train_df, test_df, _ = _get_data()
    source_df = test_df if split == "test" else train_df
    unit_data = source_df[source_df["unit_number"] == engine_id].sort_values("time_cycles")

    if unit_data.empty:
        return {
            "status": "error",
            "message": f"No telemetry observations found for engine {engine_id} in {split} set.",
            "engine_id": engine_id
        }

    if max_cycles is not None and max_cycles > 0:
        unit_data = unit_data.tail(max_cycles)

    records = unit_data[["time_cycles"] + SETTING_COLUMNS + SENSOR_COLUMNS].to_dict(orient="records")

    return {
        "status": "success",
        "engine_id": int(engine_id),
        "split": split,
        "total_observed_cycles": int(len(records)),
        "earliest_cycle": int(records[0]["time_cycles"]) if records else None,
        "latest_cycle": int(records[-1]["time_cycles"]) if records else None,
        "available_sensors": SENSOR_COLUMNS,
        "sensor_metadata": SENSOR_METADATA,
        "records": records
    }


def get_latest_engine_status(
    engine_id: int,
    split: str = "test"
) -> Dict[str, Any]:
    """
    Summarizes the latest observed cycle, operating settings, and sensor readings for an engine.
    """
    history_res = get_engine_history(engine_id=engine_id, split=split, max_cycles=1)
    if history_res.get("status") != "success":
        return history_res

    latest_record = history_res["records"][-1]
    cycle = latest_record["time_cycles"]

    # Format key sensor values with engineering names and units
    sensor_summary = {}
    for sc in SENSOR_COLUMNS:
        meta = SENSOR_METADATA.get(sc, {})
        sensor_summary[sc] = {
            "name": meta.get("name", sc),
            "description": meta.get("description", ""),
            "value": round(float(latest_record[sc]), 3),
            "unit": meta.get("unit", "")
        }

    return {
        "status": "success",
        "engine_id": int(engine_id),
        "split": split,
        "latest_observed_cycle": int(cycle),
        "operational_settings": {
            k: round(float(latest_record[k]), 4) for k in SETTING_COLUMNS
        },
        "sensor_readings": sensor_summary
    }


def predict_remaining_useful_life(engine_id: int) -> Dict[str, Any]:
    """
    Runs the persisted ML model on the selected test engine's actual observed history.
    Returns numerical RUL in operating cycles, baseline comparison, and ground-truth comparison.
    """
    if not isinstance(engine_id, int) or engine_id < 1 or engine_id > 100:
        return {
            "status": "error",
            "message": f"Invalid engine ID '{engine_id}'. Valid engine IDs in FD001 are integers 1 through 100.",
            "engine_id": engine_id
        }

    inference = RULInferenceEngine.get_instance()
    pred_res = inference.predict_for_engine(engine_id)
    return pred_res


def analyze_sensor_trend(engine_id: int, sensor_name: str, split: str = "test") -> Dict[str, Any]:
    """
    Calculates descriptive statistics, linear slope, and drift rate for a selected sensor.
    """
    sensor_clean = sensor_name.strip().lower()
    if sensor_clean not in SENSOR_COLUMNS and sensor_clean not in SETTING_COLUMNS:
        return {
            "status": "error",
            "message": f"Invalid sensor name '{sensor_name}'. Valid sensors are s_1 through s_21.",
            "available_sensors": SENSOR_COLUMNS
        }

    train_df, test_df, _ = _get_data()
    source_df = test_df if split == "test" else train_df
    unit_data = source_df[source_df["unit_number"] == engine_id].sort_values("time_cycles")

    if unit_data.empty:
        return {
            "status": "error",
            "message": f"No records found for engine {engine_id}.",
            "engine_id": engine_id
        }

    values = unit_data[sensor_clean].values.astype(np.float64)
    cycles = unit_data["time_cycles"].values.astype(np.float64)
    n = len(values)

    # Basic stats
    mean_val = float(np.mean(values))
    std_val = float(np.std(values))
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    median_val = float(np.median(values))

    # Linear slope across entire history
    slope, intercept = np.polyfit(cycles, values, 1) if n >= 2 else (0.0, values[0])

    # Recent 30 cycles trend
    recent_k = min(30, n)
    recent_values = values[-recent_k:]
    recent_cycles = cycles[-recent_k:]
    recent_slope, _ = np.polyfit(recent_cycles, recent_values, 1) if recent_k >= 2 else (0.0, recent_values[0])
    recent_delta = float(recent_values[-1] - recent_values[0])

    meta = SENSOR_METADATA.get(sensor_clean, {"name": sensor_clean, "unit": ""})

    direction = "STABLE"
    if abs(recent_slope) > 0.005:
        direction = "INCREASING (UPWARD DRIFT)" if recent_slope > 0 else "DECREASING (DOWNWARD DRIFT)"

    return {
        "status": "success",
        "engine_id": int(engine_id),
        "sensor_name": sensor_clean,
        "sensor_symbol": meta.get("name"),
        "sensor_description": meta.get("description"),
        "unit": meta.get("unit"),
        "observed_cycles_count": n,
        "statistics": {
            "mean": round(mean_val, 4),
            "std_dev": round(std_val, 4),
            "min": round(min_val, 4),
            "max": round(max_val, 4),
            "median": round(median_val, 4),
            "first_cycle_value": round(float(values[0]), 4),
            "latest_cycle_value": round(float(values[-1]), 4),
            "overall_delta": round(float(values[-1] - values[0]), 4)
        },
        "trend_analysis": {
            "overall_slope_per_cycle": round(float(slope), 5),
            "recent_30_cycles_slope": round(float(recent_slope), 5),
            "recent_30_cycles_delta": round(recent_delta, 4),
            "direction": direction
        }
    }


def detect_sensor_anomalies(
    engine_id: int,
    sensor_name: str,
    method: str = "isolation_forest",
    contamination: float = 0.05,
    split: str = "test"
) -> Dict[str, Any]:
    """
    Detects sensor anomalies using Isolation Forest combined with rolling Z-score bounds.
    Distinguishes statistical anomalies from confirmed mechanical faults.
    """
    sensor_clean = sensor_name.strip().lower()
    if sensor_clean not in SENSOR_COLUMNS:
        return {
            "status": "error",
            "message": f"Invalid sensor name '{sensor_name}'. Valid sensor channels are s_1 to s_21.",
            "available_sensors": SENSOR_COLUMNS
        }

    train_df, test_df, _ = _get_data()
    source_df = test_df if split == "test" else train_df
    unit_data = source_df[source_df["unit_number"] == engine_id].sort_values("time_cycles").copy()

    if unit_data.empty:
        return {
            "status": "error",
            "message": f"No telemetry for engine {engine_id}.",
            "engine_id": engine_id
        }

    values = unit_data[sensor_clean].values.astype(np.float64)
    cycles = unit_data["time_cycles"].values.astype(int)
    n = len(values)

    if n < 10:
        return {
            "status": "error",
            "message": f"Insufficient trajectory length ({n} cycles) to compute reliable anomaly statistics.",
            "engine_id": engine_id
        }

    # 1. Isolation Forest detector
    features = values.reshape(-1, 1)
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso_preds = iso.fit_predict(features)  # -1 for anomaly, 1 for normal
    iso_scores = iso.score_samples(features)

    # 2. Rolling Z-Score detector (window=15)
    roll_mean = unit_data[sensor_clean].rolling(window=15, min_periods=3).mean().values
    roll_std = unit_data[sensor_clean].rolling(window=15, min_periods=3).std().fillna(1e-5).values
    z_scores = np.abs((values - roll_mean) / np.maximum(roll_std, 1e-5))

    anomalous_points = []
    for i in range(n):
        is_iso_anomaly = bool(iso_preds[i] == -1)
        is_z_anomaly = bool(z_scores[i] >= 2.5)

        if is_iso_anomaly or is_z_anomaly:
            anomalous_points.append({
                "cycle": int(cycles[i]),
                "value": round(float(values[i]), 3),
                "z_score": round(float(z_scores[i]), 2),
                "isolation_forest_score": round(float(iso_scores[i]), 3),
                "rolling_window_mean": round(float(roll_mean[i]), 3),
                "reason": (
                    f"Sensor reading {round(float(values[i]), 3)} deviated by "
                    f"{round(float(z_scores[i]), 1)}σ from the 15-cycle baseline "
                    f"(rolling mean: {round(float(roll_mean[i]), 3)})."
                )
            })

    meta = SENSOR_METADATA.get(sensor_clean, {"name": sensor_clean, "unit": ""})

    # Engineering interpretation note
    is_mechanical_fault = len(anomalous_points) >= 3 and (anomalous_points[-1]["cycle"] >= cycles[-5])
    interpretation = (
        "Statistical sensor deviations observed. Readings deviate from historical rolling expectations, "
        "which may stem from transducer variance or early stage gas-path drift. "
        "A confirmed mechanical fault requires simultaneous multi-sensor corroboration."
    )
    if is_mechanical_fault:
        interpretation = (
            f"Persistent anomalous deviations detected near the latest observed cycle ({cycles[-1]}). "
            f"This trend aligns with structural degradation in sensor {sensor_clean} ({meta.get('name')})."
        )

    return {
        "status": "success",
        "engine_id": int(engine_id),
        "sensor_name": sensor_clean,
        "sensor_symbol": meta.get("name"),
        "unit": meta.get("unit"),
        "method": method,
        "contamination_threshold": contamination,
        "total_observed_cycles": n,
        "detected_anomalies_count": len(anomalous_points),
        "anomaly_rate_percent": round((len(anomalous_points) / n) * 100, 2),
        "anomalous_cycles": anomalous_points,
        "classification": "Mechanical Fault Indicator" if is_mechanical_fault else "Statistical Telemetry Anomaly",
        "interpretation": interpretation
    }


def compare_engines(engine_ids: List[int], split: str = "test") -> Dict[str, Any]:
    """
    Compares telemetry records, latest operating status, and predicted RUL across multiple engines.
    """
    if not engine_ids:
        return {"status": "error", "message": "engine_ids list cannot be empty."}

    valid_ids = []
    invalid_ids = []
    for eid in engine_ids:
        if isinstance(eid, int) and 1 <= eid <= 100:
            valid_ids.append(eid)
        else:
            invalid_ids.append(eid)

    if not valid_ids:
        return {
            "status": "error",
            "message": f"None of the provided IDs are valid. Valid IDs: 1 to 100. Invalid: {invalid_ids}"
        }

    comparison_results = []
    for eid in valid_ids:
        stat_res = get_latest_engine_status(eid, split=split)
        rul_res = predict_remaining_useful_life(eid)

        latest_cycle = stat_res.get("latest_observed_cycle")
        predicted_rul = rul_res.get("predicted_rul")
        ground_truth = rul_res.get("ground_truth_rul")

        # Urgency tier
        tier = "NOMINAL"
        if predicted_rul is not None:
            if predicted_rul <= 20:
                tier = "CRITICAL / AOG"
            elif predicted_rul <= 45:
                tier = "HIGH RISK"
            elif predicted_rul <= 80:
                tier = "MODERATE RISK"

        comparison_results.append({
            "engine_id": eid,
            "latest_observed_cycle": latest_cycle,
            "predicted_rul_cycles": predicted_rul,
            "ground_truth_rul_cycles": ground_truth,
            "risk_tier": tier,
            "key_sensors": {
                "T24 (s_2)": stat_res.get("sensor_readings", {}).get("s_2", {}).get("value"),
                "T30 (s_3)": stat_res.get("sensor_readings", {}).get("s_3", {}).get("value"),
                "T50 (s_4)": stat_res.get("sensor_readings", {}).get("s_4", {}).get("value"),
                "Ps30 (s_11)": stat_res.get("sensor_readings", {}).get("s_11", {}).get("value"),
                "BPR (s_15)": stat_res.get("sensor_readings", {}).get("s_15", {}).get("value"),
                "htBleed (s_17)": stat_res.get("sensor_readings", {}).get("s_17", {}).get("value"),
            }
        })

    # Sort by predicted RUL ascending (most urgent first)
    comparison_results.sort(
        key=lambda x: x["predicted_rul_cycles"] if x["predicted_rul_cycles"] is not None else 9999
    )

    return {
        "status": "success",
        "compared_engines_count": len(comparison_results),
        "invalid_engine_ids": invalid_ids,
        "fleet_comparison": comparison_results
    }


def get_model_evaluation() -> Dict[str, Any]:
    """
    Returns metrics from the saved, verified RUL evaluation run.
    """
    inference = RULInferenceEngine.get_instance()
    if not inference.is_loaded:
        inference.load()

    eval_data = inference.evaluation_results
    if not eval_data:
        return {
            "status": "error",
            "message": "Evaluation results not found. Please run 'python scripts/train_model.py'."
        }

    return {
        "status": "success",
        "evaluation_results": eval_data
    }


def retrieve_technical_documents(question: str, top_k: int = 3) -> Dict[str, Any]:
    """
    Retrieves grounded passages and citations from the technical documentation RAG subsystem.
    """
    rag = TechnicalDocumentRAG.get_instance()
    return rag.retrieve(question, top_k=top_k)
