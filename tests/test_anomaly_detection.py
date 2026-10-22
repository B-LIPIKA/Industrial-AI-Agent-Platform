"""
Unit tests for sensor time-series trend analysis and anomaly detection.
"""

import pytest
from tools.engine_tools import (
    analyze_sensor_trend,
    detect_sensor_anomalies,
    compare_engines
)


def test_sensor_trend_valid():
    res = analyze_sensor_trend(engine_id=24, sensor_name="s_11")
    assert res["status"] == "success"
    assert res["sensor_name"] == "s_11"
    assert res["unit"] == "psia"
    assert res["observed_cycles_count"] == 186
    assert "mean" in res["statistics"]
    assert "overall_slope_per_cycle" in res["trend_analysis"]
    assert "direction" in res["trend_analysis"]


def test_sensor_trend_invalid_sensor():
    res = analyze_sensor_trend(engine_id=24, sensor_name="invalid_sensor_xyz")
    assert res["status"] == "error"
    assert "Invalid sensor name" in res["message"]


def test_anomaly_detection_schema():
    res = detect_sensor_anomalies(engine_id=24, sensor_name="s_11", contamination=0.05)
    assert res["status"] == "success"
    assert res["engine_id"] == 24
    assert res["sensor_name"] == "s_11"
    assert res["method"] == "isolation_forest"
    assert isinstance(res["detected_anomalies_count"], int)
    assert isinstance(res["anomalous_cycles"], list)
    assert "classification" in res
    assert "interpretation" in res

    if res["anomalous_cycles"]:
        first = res["anomalous_cycles"][0]
        assert "cycle" in first
        assert "value" in first
        assert "z_score" in first
        assert "reason" in first


def test_anomaly_detection_invalid_inputs():
    res1 = detect_sensor_anomalies(engine_id=999, sensor_name="s_11")
    assert res1["status"] == "error"

    res2 = detect_sensor_anomalies(engine_id=24, sensor_name="s_99")
    assert res2["status"] == "error"


def test_compare_engines():
    res = compare_engines(engine_ids=[12, 24, 34])
    assert res["status"] == "success"
    assert res["compared_engines_count"] == 3
    fleet = res["fleet_comparison"]
    assert len(fleet) == 3
    
    # Check that engines are ordered by predicted RUL ascending
    ruls = [e["predicted_rul_cycles"] for e in fleet if e["predicted_rul_cycles"] is not None]
    assert ruls == sorted(ruls)
