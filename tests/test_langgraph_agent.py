"""
Unit tests for the LangGraph EngineMind AI orchestrator:
Intent classification, conditional routing, tool execution, and response synthesis.
"""

import pytest
from agents.graph import EngineMindOrchestrator


@pytest.fixture(scope="module")
def orchestrator():
    return EngineMindOrchestrator()


def test_agent_rul_routing(orchestrator):
    res = orchestrator.run("Estimate the remaining useful life of engine 24.")
    assert res["is_valid"] is True
    assert res["intent"] == "remaining_useful_life"
    assert res["target_engine_ids"] == [24]
    assert "predict_remaining_useful_life" in res["selected_tools"]
    assert "predict_remaining_useful_life" in res["evidence"]
    assert res["status"] in ["success", "partial"]
    assert len(res["reasoning_trace"]) >= 5
    assert "Remaining Useful Life (RUL) Prediction" in res["final_answer"]
    assert "21.5 operating cycles" in res["final_answer"]


def test_agent_trend_routing(orchestrator):
    res = orchestrator.run("Analyze sensor trend for s_11 on engine 24.")
    assert res["intent"] == "sensor_trend_analysis"
    assert "analyze_sensor_trend" in res["selected_tools"]
    assert "analyze_sensor_trend" in res["evidence"]
    assert "Sensor Trend & Drift Analysis" in res["final_answer"]
    assert "s_11" in res["final_answer"]


def test_agent_anomaly_routing(orchestrator):
    res = orchestrator.run("Find unusual readings in sensor s_11 for engine 24.")
    assert res["intent"] == "anomaly_investigation"
    assert "detect_sensor_anomalies" in res["selected_tools"]
    assert "Sensor Anomaly Detection" in res["final_answer"]


def test_agent_fleet_comparison(orchestrator):
    res = orchestrator.run("Compare the status of engines 12 and 24.")
    assert res["intent"] == "fleet_comparison"
    assert "compare_engines" in res["selected_tools"]
    assert "Fleet Telemetry & Risk Comparison" in res["final_answer"]
    assert "Unit #12" in res["final_answer"]
    assert "Unit #24" in res["final_answer"]


def test_agent_rag_documentation(orchestrator):
    res = orchestrator.run("What does the manual say about RUL threshold action protocols?")
    assert res["intent"] == "technical_documentation"
    assert "retrieve_technical_documents" in res["selected_tools"]
    assert len(res["citations"]) > 0
    assert "Technical Manual Citations" in res["final_answer"]


def test_agent_input_validation_empty(orchestrator):
    res = orchestrator.run("")
    assert res["is_valid"] is False
    assert res["status"] == "error"
    assert "Input Validation Error" in res["final_answer"]


def test_agent_unsupported_query_fallback(orchestrator):
    res = orchestrator.run("Can you write a poem about autumn leaves?")
    assert res["intent"] == "unsupported_or_ambiguous"
    assert res["selected_tools"] == []
    assert "EngineMind AI Copilot — Query Guidance" in res["final_answer"]
