"""
Unit Tests for Industrial AI Agent Platform
"""

import pytest
import os
from fastapi.testclient import TestClient

from database.db_manager import init_database, get_schema_summary, execute_read_query
from tools.schema_tools import inspect_database_schema, discover_available_time_ranges
from tools.analytics_tools import compute_series_statistics, evaluate_anomaly_thresholds, analyze_trend_and_slope
from agents.graph import IndustrialAgentOrchestrator
from app.api import app

client = TestClient(app)
TEST_DB = "test_plant.db"


@pytest.fixture(autouse=True)
def setup_test_db():
    init_database(TEST_DB)
    yield
    if os.path.exists(TEST_DB):
        try:
            os.remove(TEST_DB)
        except PermissionError:
            pass


def test_schema_discovery():
    schema = get_schema_summary(TEST_DB)
    assert "equipment" in schema
    assert "sensor_telemetry" in schema
    assert "maintenance_logs" in schema


def test_deterministic_analytics():
    values = [2.0, 3.0, 5.0, 2.5]
    stats = compute_series_statistics(values)
    assert stats["count"] == 4
    assert stats["max"] == 5.0
    assert stats["min"] == 2.0

    eval_thresh = evaluate_anomaly_thresholds(values, warning_threshold=3.0, critical_threshold=4.5)
    assert eval_thresh["critical_violation_count"] == 1
    assert eval_thresh["is_safe"] is False

    trend = analyze_trend_and_slope([1.0, 2.0, 3.0, 4.0])
    assert trend["slope_per_step"] > 0
    assert "INCREASING" in trend["direction"]


def test_orchestrator_execution():
    orchestrator = IndustrialAgentOrchestrator(db_path=TEST_DB)
    res = orchestrator.run("Check vibration levels on PUMP-101 and flag any anomalies.")
    assert res["target_equipment"] == "PUMP-101"
    assert res["target_metric"] == "vibration_rms"
    assert len(res["retrieved_data"]) > 0
    assert "CRITICAL ANOMALY DETECTED" in res["final_response"]
    assert len(res["reasoning_trace"]) == 4


def test_fastapi_agent_endpoints():
    res = client.get("/api/v1/schema")
    assert res.status_code == 200
    assert "schema" in res.json()

    res_eq = client.get("/api/v1/equipment")
    assert res_eq.status_code == 200
    assert res_eq.json()["equipment_count"] >= 4

    res_query = client.post("/api/v1/agent/query", json={
        "query": "What is the temperature on PUMP-102?"
    })
    assert res_query.status_code == 200
    data = res_query.json()
    assert data["target_equipment"] == "PUMP-102"
    assert data["target_metric"] == "temperature"
    assert len(data["reasoning_trace"]) == 4
