"""
LangGraph Orchestration Pipeline for Industrial AI Agent
Implements multi-node routing:
[Query Understanding] -> [SQL Retrieval] -> [Deterministic Analytics] -> [Evidence Synthesis]
"""

import re
from typing import Dict, Any
from agents.state import IndustrialAgentState
from tools.schema_tools import query_sensor_database
from tools.analytics_tools import compute_series_statistics, evaluate_anomaly_thresholds, analyze_trend_and_slope
from database.db_manager import DEFAULT_DB_PATH


class IndustrialAgentOrchestrator:
    """
    State Graph Orchestration Engine.
    Executes decoupled pipeline keeping deterministic calculations separate from LLM reasoning.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path

    def parse_query_node(self, state: IndustrialAgentState) -> IndustrialAgentState:
        """Node 1: Extract equipment ID, metric, and intent from user request."""
        query = state["user_query"].upper()
        trace = list(state.get("reasoning_trace", []))
        trace.append("Step 1: Analyzed user query intent.")

        # Entity Extraction (Equipment)
        target_equipment = "PUMP-101"  # default
        for eq in ["PUMP-101", "PUMP-102", "TURBINE-01", "COMPRESSOR-04"]:
            if eq in query:
                target_equipment = eq
                break

        # Metric Extraction
        target_metric = "vibration_rms"
        if "TEMP" in query or "TEMPERATURE" in query:
            target_metric = "temperature"
        elif "POWER" in query:
            target_metric = "power_kw"

        state["target_equipment"] = target_equipment
        state["target_metric"] = target_metric
        state["reasoning_trace"] = trace
        return state

    def sql_retrieval_node(self, state: IndustrialAgentState) -> IndustrialAgentState:
        """Node 2: Generates schema-valid SQL and retrieves empirical sensor telemetry."""
        trace = list(state["reasoning_trace"])
        eq = state["target_equipment"]
        metric = state["target_metric"]

        sql = f"""
        SELECT timestamp, value, unit 
        FROM sensor_telemetry 
        WHERE equipment_id = '{eq}' AND metric_name = '{metric}'
        ORDER BY timestamp ASC;
        """
        trace.append(f"Step 2: Constructed schema-aware SQL for {eq} ({metric}).")

        result = query_sensor_database(sql, db_path=self.db_path)
        state["sql_query"] = sql
        state["retrieved_data"] = result.get("data", [])
        trace.append(f"Step 2 (cont): Retrieved {len(state['retrieved_data'])} telemetry rows.")
        state["reasoning_trace"] = trace
        return state

    def deterministic_analytics_node(self, state: IndustrialAgentState) -> IndustrialAgentState:
        """Node 3: Executes deterministic numerical computation (No LLM hallucinations)."""
        trace = list(state["reasoning_trace"])
        rows = state.get("retrieved_data", [])

        if not rows:
            state["numerical_evidence"] = {"error": "No sensor data found"}
            trace.append("Step 3: No records available for numerical analytics.")
            state["reasoning_trace"] = trace
            return state

        values = [r["value"] for r in rows]
        stats = compute_series_statistics(values)
        thresholds = evaluate_anomaly_thresholds(values, warning_threshold=3.5, critical_threshold=4.5)
        trend = analyze_trend_and_slope(values)

        state["numerical_evidence"] = {
            "statistics": stats,
            "threshold_eval": thresholds,
            "trend": trend,
            "unit": rows[0]["unit"] if rows else ""
        }
        trace.append("Step 3: Computed exact summary statistics and safety threshold bounds.")
        state["reasoning_trace"] = trace
        return state

    def evidence_synthesis_node(self, state: IndustrialAgentState) -> IndustrialAgentState:
        """Node 4: Synthesizes engineering diagnostics grounded strictly in empirical evidence."""
        trace = list(state["reasoning_trace"])
        evidence = state.get("numerical_evidence", {})
        eq = state["target_equipment"]
        metric = state["target_metric"]

        if "error" in evidence:
            state["final_response"] = f"Unable to locate sensor data for {eq}."
            return state

        stats = evidence["statistics"]
        thresh = evidence["threshold_eval"]
        trend = evidence["trend"]
        unit = evidence.get("unit", "")

        status_text = "NORMAL" if thresh["is_safe"] else "CRITICAL ANOMALY DETECTED"
        
        response = (
            f"### ⚙️ Industrial Asset Diagnostic: {eq}\n"
            f"- **Metric Analyzed:** {metric} ({unit})\n"
            f"- **Operational Status:** **{status_text}**\n"
            f"- **Empirical Summary:** Mean = {stats['mean']} {unit}, Max = {stats['max']} {unit}, StdDev = {stats['std_dev']}.\n"
            f"- **Threshold Violations:** {thresh['critical_violation_count']} critical violation(s) (>= {thresh['critical_threshold']} {unit}).\n"
            f"- **Trend Direction:** {trend['direction']} (slope: {trend['slope_per_step']}/step).\n\n"
            f"**Recommendation:** "
            + (
                f"Immediate maintenance inspection recommended for {eq}. Vibration spike of {stats['max']} {unit} observed."
                if not thresh["is_safe"]
                else f"{eq} operates stably within specified standard baseline parameters."
            )
        )

        trace.append("Step 4: Synthesized final engineering report grounded in calculated evidence.")
        state["final_response"] = response
        state["reasoning_trace"] = trace
        return state

    def run(self, user_query: str) -> Dict[str, Any]:
        """Executes full workflow end-to-end."""
        state: IndustrialAgentState = {
            "user_query": user_query,
            "target_equipment": None,
            "target_metric": None,
            "sql_query": None,
            "retrieved_data": None,
            "numerical_evidence": None,
            "reasoning_trace": [],
            "final_response": None
        }

        s1 = self.parse_query_node(state)
        s2 = self.sql_retrieval_node(s1)
        s3 = self.deterministic_analytics_node(s2)
        s4 = self.evidence_synthesis_node(s3)
        return dict(s4)
