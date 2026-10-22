"""
Unit and integration tests for FastAPI REST endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_root_endpoint():
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["project"] == "EngineMind AI — Agentic Predictive Maintenance Copilot"
    assert "endpoints" in data


def test_health_endpoint():
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["dataset_ready"] is True
    assert data["dataset_train_rows"] > 20000
    assert data["model_trained"] is True
    assert data["best_model"] == "random_forest"


def test_dataset_summary_endpoint():
    res = client.get("/api/dataset/summary")
    assert res.status_code == 200
    data = res.json()
    assert data["subset"] == "FD001"
    assert data["train_engines_count"] == 100
    assert data["test_engines_count"] == 100


def test_engines_list_endpoint():
    res = client.get("/api/engines?split=test")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_count"] == 100
    assert len(data["engines"]) == 100


def test_engine_history_endpoint():
    res = client.get("/api/engines/24/history?split=test&max_cycles=10")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_id"] == 24
    assert len(data["records"]) == 10

    # 404 for invalid engine ID
    res_inv = client.get("/api/engines/999/history")
    assert res_inv.status_code == 404


def test_engine_status_endpoint():
    res = client.get("/api/engines/24/status?split=test")
    assert res.status_code == 200
    data = res.json()
    assert data["engine_id"] == 24
    assert data["latest_observed_cycle"] == 186
    assert "sensor_readings" in data


def test_predict_rul_endpoint():
    res = client.post("/api/predict/rul", json={"engine_id": 24})
    assert res.status_code == 200
    data = res.json()
    assert data["engine_id"] == 24
    assert data["predicted_rul"] > 0
    assert data["unit"] == "operating cycles"

    # Validation failure for out-of-range engine
    res_inv = client.post("/api/predict/rul", json={"engine_id": 999})
    assert res_inv.status_code == 422  # Pydantic validation error


def test_analyze_trends_endpoint():
    res = client.post("/api/analyze/trends", json={"engine_id": 24, "sensor_name": "s_11"})
    assert res.status_code == 200
    data = res.json()
    assert data["sensor_name"] == "s_11"
    assert "trend_analysis" in data


def test_analyze_anomalies_endpoint():
    res = client.post("/api/analyze/anomalies", json={"engine_id": 24, "sensor_name": "s_11", "contamination": 0.05})
    assert res.status_code == 200
    data = res.json()
    assert data["detected_anomalies_count"] > 0
    assert "anomalous_cycles" in data


def test_compare_engines_endpoint():
    res = client.post("/api/engines/compare", json={"engine_ids": [12, 24]})
    assert res.status_code == 200
    data = res.json()
    assert data["compared_engines_count"] == 2


def test_evaluation_endpoint():
    res = client.get("/api/evaluation")
    assert res.status_code == 200
    data = res.json()
    assert "evaluation_results" in data


def test_agent_chat_endpoint():
    res = client.post("/api/agent/chat", json={"query": "Estimate the remaining useful life of engine 24."})
    assert res.status_code == 200
    data = res.json()
    assert data["intent"] == "remaining_useful_life"
    assert "21.5 operating cycles" in data["final_answer"]


def test_rag_query_endpoint():
    res = client.post("/api/rag/query", json={"question": "What are the RUL threshold action protocols?", "top_k": 2})
    assert res.status_code == 200
    data = res.json()
    assert data["matched_chunks"] > 0


def test_legacy_endpoints_backward_compatibility():
    res_schema = client.get("/api/v1/schema")
    assert res_schema.status_code == 200
    assert "schema" in res_schema.json()

    res_eq = client.get("/api/v1/equipment")
    assert res_eq.status_code == 200
    assert res_eq.json()["equipment_count"] >= 4

    res_query = client.post("/api/v1/agent/query", json={"query": "Check vibration levels on PUMP-101 and flag any anomalies."})
    assert res_query.status_code == 200
    assert "CRITICAL ANOMALY DETECTED" in res_query.json()["final_response"]
