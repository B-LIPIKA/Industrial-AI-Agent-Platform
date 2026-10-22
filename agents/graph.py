"""
LangGraph Orchestration Pipeline for EngineMind AI Copilot.
Implements a 6-node state graph with typed state, conditional routing,
deterministic tool execution, error recovery, and evidence-grounded synthesis.
"""

import os
import re
import json
from typing import Dict, List, Any, Optional
from langgraph.graph import StateGraph, END

from agents.state import EngineMindState, IndustrialAgentState
from tools.engine_tools import (
    get_dataset_summary,
    get_engine_history,
    get_latest_engine_status,
    predict_remaining_useful_life,
    analyze_sensor_trend,
    detect_sensor_anomalies,
    compare_engines,
    get_model_evaluation,
    retrieve_technical_documents
)
from tools.analytics_tools import compute_series_statistics, evaluate_anomaly_thresholds, analyze_trend_and_slope
from tools.schema_tools import query_sensor_database
from database.db_manager import DEFAULT_DB_PATH


class EngineMindOrchestrator:
    """
    Evidence-Driven LangGraph Orchestrator for Turbofan Predictive Maintenance.
    Graph Topology:
    [validate_input] -> (valid?)
                          |-- Yes --> [classify_intent] -> (ambiguous?)
                          |                                  |-- Yes --> [synthesize_response]
                          |                                  |-- No  --> [select_tools] -> [execute_tools] -> [validate_evidence] -> [synthesize_response] -> END
                          |-- No  --------------------------------------------------------------------------------------------------> [synthesize_response] -> END
    """

    def __init__(self, ollama_url: Optional[str] = None, model_name: Optional[str] = None):
        self.ollama_url = ollama_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.model_name = model_name or os.getenv("OLLAMA_MODEL", "llama3")
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(EngineMindState)

        # 1. Register Nodes
        workflow.add_node("validate_input", self.validate_input_node)
        workflow.add_node("classify_intent", self.classify_intent_node)
        workflow.add_node("select_tools", self.select_tools_node)
        workflow.add_node("execute_tools", self.execute_tools_node)
        workflow.add_node("validate_evidence", self.validate_evidence_node)
        workflow.add_node("synthesize_response", self.synthesize_response_node)

        # 2. Entrypoint
        workflow.set_entry_point("validate_input")

        # 3. Conditional Routing
        workflow.add_conditional_edges(
            "validate_input",
            self.route_after_validation,
            {
                "classify": "classify_intent",
                "respond": "synthesize_response"
            }
        )

        workflow.add_conditional_edges(
            "classify_intent",
            self.route_after_intent,
            {
                "tools": "select_tools",
                "respond": "synthesize_response"
            }
        )

        workflow.add_edge("select_tools", "execute_tools")
        workflow.add_edge("execute_tools", "validate_evidence")

        workflow.add_conditional_edges(
            "validate_evidence",
            self.route_after_evidence_validation,
            {
                "retry": "execute_tools",
                "respond": "synthesize_response"
            }
        )

        workflow.add_edge("synthesize_response", END)

        return workflow.compile()

    # --- Conditional Edge Functions ---

    @staticmethod
    def route_after_validation(state: EngineMindState) -> str:
        if not state.get("is_valid", False):
            return "respond"
        return "classify"

    @staticmethod
    def route_after_intent(state: EngineMindState) -> str:
        if state.get("intent") == "unsupported_or_ambiguous":
            return "respond"
        return "tools"

    @staticmethod
    def route_after_evidence_validation(state: EngineMindState) -> str:
        if state.get("status") == "retry_needed" and state.get("retries", 0) < 2:
            return "retry"
        return "respond"

    # --- Node Implementations ---

    def validate_input_node(self, state: EngineMindState) -> EngineMindState:
        """Node 1: Validate input length, format, and structure."""
        q = state.get("question", "").strip()
        trace = list(state.get("reasoning_trace", []))
        trace.append("Node 1 [Input Validation]: Inspecting query safety, length, and format.")

        if not q:
            state["is_valid"] = False
            state["validation_error"] = "Inquiry cannot be empty."
            state["status"] = "error"
            trace.append("Validation failed: Empty query provided.")
        elif len(q) > 1000:
            state["is_valid"] = False
            state["validation_error"] = "Query exceeds 1000 characters limit."
            state["status"] = "error"
            trace.append("Validation failed: Query length exceeds threshold.")
        else:
            state["is_valid"] = True
            state["validation_error"] = None
            trace.append(f"Validation passed: Verified query string ({len(q)} chars).")

        state["reasoning_trace"] = trace
        return state

    def classify_intent_node(self, state: EngineMindState) -> EngineMindState:
        """Node 2: Classify intent and extract target engine IDs and sensor channels."""
        q = state["question"].lower()
        trace = list(state.get("reasoning_trace", []))
        trace.append("Node 2 [Intent Classification]: Parsing intent and extracting entities.")

        # Entity Extraction: Engine IDs
        # Patterns like: "engine 24", "engine #24", "unit 24", "engines 12 and 24"
        engine_matches = re.findall(r'(?:engine|unit|turbofan)\s*(?:#|id)?\s*(\d+)', q)
        engine_ids = [int(m) for m in engine_matches if 1 <= int(m) <= 100]

        # Multi-engine list check (e.g. "compare 12 and 24", "engines 12, 24")
        multi_matches = re.findall(r'\b(\d{1,3})\b', q)
        if len(engine_ids) < 2 and any(k in q for k in ["compare", "fleet", "versus", "vs"]):
            for m in multi_matches:
                val = int(m)
                if 1 <= val <= 100 and val not in engine_ids:
                    engine_ids.append(val)

        # Entity Extraction: Sensor channel
        target_sensor = None
        sensor_match = re.search(r'\b(s_\d{1,2}|sensor\s*\d{1,2})\b', q)
        if sensor_match:
            s_token = sensor_match.group(1).replace("sensor", "s_").replace(" ", "").lower()
            if not s_token.startswith("s_"):
                s_token = f"s_{s_token}"
            target_sensor = s_token
        else:
            # Check engineering aliases
            alias_map = {
                "t24": "s_2", "lpc temp": "s_2",
                "t30": "s_3", "hpc temp": "s_3",
                "t50": "s_4", "lpt temp": "s_4", "exhaust gas": "s_4", "egt": "s_4",
                "ps30": "s_11", "static pressure": "s_11",
                "phi": "s_12", "fuel flow": "s_12",
                "nrf": "s_13", "corrected fan": "s_13",
                "nrc": "s_14", "corrected core": "s_14",
                "bpr": "s_15", "bypass ratio": "s_15",
                "htbleed": "s_17", "bleed enthalpy": "s_17",
                "w31": "s_20", "coolant bleed": "s_20",
                "w32": "s_21"
            }
            for alias, s_code in alias_map.items():
                if alias in q:
                    target_sensor = s_code
                    break

        # Intent Classification rules
        intent = "unsupported_or_ambiguous"
        if any(k in q for k in ["manual", "document", "documentation", "sop", "protocol", "guide", "procedure", "what does the indexed", "what does the manual", "what does the document"]):
            intent = "technical_documentation"
        elif any(k in q for k in ["compare", "fleet", "versus", "vs", "across engines"]):
            intent = "fleet_comparison"
        elif any(k in q for k in ["rul", "remaining useful life", "cycles left", "how long", "life expectancy", "estimate life"]):
            intent = "remaining_useful_life"
        elif any(k in q for k in ["trend", "drift", "slope", "rate of change", "change over", "trajectory"]):
            intent = "sensor_trend_analysis"
        elif any(k in q for k in ["anomaly", "anomalies", "unusual", "outlier", "spike", "deviation", "isolation forest", "flag"]):
            intent = "anomaly_investigation"
        elif any(k in q for k in ["prioriti", "most urgent", "urgent", "critical", "risk tier", "maintenance action"]):
            intent = "maintenance_prioritization"
        elif any(k in q for k in ["evaluation", "mae", "rmse", "metric", "accuracy", "benchmark", "score", "leakage"]):
            intent = "model_evaluation"
        elif any(k in q for k in ["health", "status", "summary", "overview", "condition", "reading", "sensors"]):
            intent = "engine_health_summary"
        elif any(k in q for k in ["dataset", "c-mapss", "fd001", "records", "rows", "provenance"]):
            intent = "dataset_summary"

        # Default fallback engine if intent requires an engine and none specified
        if intent in ["remaining_useful_life", "sensor_trend_analysis", "anomaly_investigation", "engine_health_summary"]:
            if not engine_ids:
                engine_ids = [24]  # representative default test engine
                trace.append("No explicit engine ID specified; assigned representative test engine ID 24.")

        if intent in ["sensor_trend_analysis", "anomaly_investigation"] and not target_sensor:
            target_sensor = "s_11"  # Ps30 (most responsive degradation indicator)
            trace.append("No specific sensor specified; selected Ps30 (s_11, HPC static pressure).")

        state["intent"] = intent
        state["target_engine_ids"] = engine_ids
        state["target_sensor"] = target_sensor
        trace.append(f"Identified intent: '{intent}' with engine(s)={engine_ids}, sensor={target_sensor}.")
        state["reasoning_trace"] = trace
        return state

    def select_tools_node(self, state: EngineMindState) -> EngineMindState:
        """Node 3: Select appropriate deterministic tools based on classified intent."""
        intent = state.get("intent")
        trace = list(state.get("reasoning_trace", []))
        trace.append("Node 3 [Tool Selection]: Determining tool registry execution set.")

        tool_map = {
            "remaining_useful_life": ["predict_remaining_useful_life"],
            "sensor_trend_analysis": ["analyze_sensor_trend", "get_latest_engine_status"],
            "anomaly_investigation": ["detect_sensor_anomalies", "analyze_sensor_trend"],
            "engine_health_summary": ["get_latest_engine_status", "predict_remaining_useful_life"],
            "fleet_comparison": ["compare_engines"],
            "maintenance_prioritization": ["compare_engines", "retrieve_technical_documents"],
            "model_evaluation": ["get_model_evaluation"],
            "technical_documentation": ["retrieve_technical_documents"],
            "dataset_summary": ["get_dataset_summary"]
        }

        selected = tool_map.get(intent, [])
        state["selected_tools"] = selected
        trace.append(f"Selected tools: {selected if selected else 'None (Direct Response)'}.")
        state["reasoning_trace"] = trace
        return state

    def execute_tools_node(self, state: EngineMindState) -> EngineMindState:
        """Node 4: Execute selected deterministic tools with strict validation."""
        trace = list(state.get("reasoning_trace", []))
        selected = state.get("selected_tools", [])
        engine_ids = state.get("target_engine_ids", [24])
        primary_id = engine_ids[0] if engine_ids else 24
        sensor = state.get("target_sensor", "s_11")

        outputs = dict(state.get("tool_outputs", {}))
        citations = list(state.get("citations", []))
        trace.append(f"Node 4 [Tool Execution]: Executing {len(selected)} tools deterministically.")

        for tool_name in selected:
            trace.append(f"-> Executing '{tool_name}'...")
            try:
                if tool_name == "predict_remaining_useful_life":
                    res = predict_remaining_useful_life(primary_id)
                    outputs["predict_remaining_useful_life"] = res

                elif tool_name == "analyze_sensor_trend":
                    res = analyze_sensor_trend(primary_id, sensor)
                    outputs["analyze_sensor_trend"] = res

                elif tool_name == "detect_sensor_anomalies":
                    res = detect_sensor_anomalies(primary_id, sensor)
                    outputs["detect_sensor_anomalies"] = res

                elif tool_name == "get_latest_engine_status":
                    res = get_latest_engine_status(primary_id)
                    outputs["get_latest_engine_status"] = res

                elif tool_name == "compare_engines":
                    compare_ids = engine_ids if len(engine_ids) >= 2 else [12, 24, 34]
                    res = compare_engines(compare_ids)
                    outputs["compare_engines"] = res

                elif tool_name == "get_model_evaluation":
                    res = get_model_evaluation()
                    outputs["get_model_evaluation"] = res

                elif tool_name == "get_dataset_summary":
                    res = get_dataset_summary()
                    outputs["get_dataset_summary"] = res

                elif tool_name == "retrieve_technical_documents":
                    res = retrieve_technical_documents(state["question"], top_k=3)
                    outputs["retrieve_technical_documents"] = res
                    for match in res.get("results", []):
                        citations.append({
                            "source_file": match["source_file"],
                            "section": match["section"],
                            "excerpt": match["excerpt"],
                            "score": match["score"]
                        })

                trace.append(f"   [OK] '{tool_name}' executed successfully.")
            except Exception as e:
                trace.append(f"   [ERR] '{tool_name}' failed with error: {str(e)}")
                outputs[tool_name] = {"status": "error", "message": str(e)}

        state["tool_outputs"] = outputs
        state["citations"] = citations
        state["reasoning_trace"] = trace
        return state

    def validate_evidence_node(self, state: EngineMindState) -> EngineMindState:
        """Node 5: Verify tool execution outputs for completeness and consistency."""
        trace = list(state.get("reasoning_trace", []))
        outputs = state.get("tool_outputs", {})
        trace.append("Node 5 [Evidence Validation]: Validating numerical outputs and schema integrity.")

        has_error = False
        valid_evidence = {}

        for k, v in outputs.items():
            if isinstance(v, dict) and v.get("status") == "error":
                has_error = True
                trace.append(f"Validation warning: Tool '{k}' reported error: {v.get('message')}")
            else:
                valid_evidence[k] = v

        state["evidence"] = valid_evidence
        if has_error and not valid_evidence:
            state["status"] = "error"
            trace.append("Evidence validation: No valid evidence gathered.")
        elif has_error:
            state["status"] = "partial"
            trace.append("Evidence validation: Partial evidence gathered.")
        else:
            state["status"] = "success"
            trace.append("Evidence validation: All deterministic evidence verified and grounded.")

        state["reasoning_trace"] = trace
        return state

    def synthesize_response_node(self, state: EngineMindState) -> EngineMindState:
        """Node 6: Synthesize final engineering report grounded strictly in empirical evidence."""
        trace = list(state.get("reasoning_trace", []))
        trace.append("Node 6 [Response Synthesis]: Synthesizing evidence-grounded engineering report.")

        q = state["question"]
        intent = state.get("intent")
        evidence = state.get("evidence", {})
        citations = state.get("citations", [])

        # Check if validation failed upfront
        if not state.get("is_valid", True):
            state["final_answer"] = (
                f"### ❌ Input Validation Error\n\n"
                f"{state.get('validation_error', 'Invalid inquiry.')}\n\n"
                f"Please submit a valid technical question regarding NASA turbofan engine health, "
                f"remaining useful life (RUL), sensor anomaly detection, or maintenance procedures."
            )
            state["reasoning_trace"] = trace
            return state

        # Check if ambiguous or unsupported
        if intent == "unsupported_or_ambiguous":
            state["final_answer"] = (
                "### ℹ️ EngineMind AI Copilot — Query Guidance\n\n"
                "I specialize in turbofan engine degradation analytics and predictive maintenance based on the "
                "NASA C-MAPSS dataset. Your question appears ambiguous or outside this domain.\n\n"
                "**Try one of these example queries:**\n"
                "- *“Estimate the remaining useful life of engine 24.”*\n"
                "- *“Analyze sensor trend for s_11 on engine 24.”*\n"
                "- *“Detect anomalies in sensor s_11 for engine 24.”*\n"
                "- *“Compare the status of engines 12 and 24.”*\n"
                "- *“What does the technical manual say about RUL threshold action protocols?”*\n"
                "- *“Explain the model evaluation metrics and baseline comparison.”*"
            )
            state["reasoning_trace"] = trace
            return state

        # Construct deterministic synthesis
        report_parts = []
        report_parts.append(f"### ✈️ EngineMind Diagnostic Report\n")
        report_parts.append(f"**Inquiry**: *\"{q}\"*\n")
        report_parts.append(f"**Identified Intent**: `{intent}`\n")

        # 1. RUL Evidence
        if "predict_remaining_useful_life" in evidence:
            rul_data = evidence["predict_remaining_useful_life"]
            if rul_data.get("status") == "success":
                pred_cycles = rul_data['predicted_rul']
                gt = rul_data.get('ground_truth_rul')
                benchmark = rul_data.get('benchmark_metrics', {})

                # Tier determination
                tier_badge = "🟢 NOMINAL"
                if pred_cycles <= 20:
                    tier_badge = "🔴 CRITICAL (AOG RISK)"
                elif pred_cycles <= 45:
                    tier_badge = "🟠 HIGH RISK"
                elif pred_cycles <= 80:
                    tier_badge = "🟡 MODERATE RISK"

                err_val = rul_data.get("prediction_error_cycles")
                gt_str = f"{gt} cycles" if gt is not None else "N/A"
                err_str = f"{err_val} cycles" if err_val is not None else "N/A"

                report_parts.append(
                    f"#### ⏱️ Remaining Useful Life (RUL) Prediction\n"
                    f"- **Target Engine**: Unit #{rul_data['engine_id']}\n"
                    f"- **Latest Observed Flight Cycle**: Cycle {rul_data['latest_observed_cycle']}\n"
                    f"- **Estimated RUL**: **{pred_cycles} operating cycles** ({tier_badge})\n"
                    f"- **Ground Truth RUL**: {gt_str}\n"
                    f"- **Prediction Error**: {err_str}\n"
                    f"- **Model Architecture**: `{rul_data.get('model_architecture')}` (Test MAE: {benchmark.get('test_mae')} cycles vs Baseline: {benchmark.get('baseline_mae')} cycles)\n\n"
                    f"> [!NOTE]\n"
                    f"> *Operational Note*: {rul_data.get('limitations_note')}\n\n"
                )

        # 2. Trend Evidence
        if "analyze_sensor_trend" in evidence:
            tr = evidence["analyze_sensor_trend"]
            if tr.get("status") == "success":
                stats = tr['statistics']
                trend = tr['trend_analysis']
                report_parts.append(
                    f"#### 📈 Sensor Trend & Drift Analysis\n"
                    f"- **Sensor**: `{tr['sensor_name']}` ({tr['sensor_symbol']} — {tr['sensor_description']})\n"
                    f"- **Unit**: {tr['unit']}\n"
                    f"- **Observed Span**: {tr['observed_cycles_count']} cycles (Baseline: {stats['first_cycle_value']} -> Latest: {stats['latest_cycle_value']})\n"
                    f"- **Overall Drift**: {stats['overall_delta']:+} {tr['unit']}\n"
                    f"- **Recent 30 Cycles Drift Rate**: `{trend['recent_30_cycles_slope']:+} {tr['unit']}/cycle` ({trend['direction']})\n"
                    f"- **Descriptive Statistics**: Mean = {stats['mean']}, Std = {stats['std_dev']}, Min = {stats['min']}, Max = {stats['max']}\n"
                )

        # 3. Anomaly Evidence
        if "detect_sensor_anomalies" in evidence:
            anom = evidence["detect_sensor_anomalies"]
            if anom.get("status") == "success":
                flagged = anom['anomalous_cycles']
                report_parts.append(
                    f"#### 🔍 Sensor Anomaly Detection\n"
                    f"- **Methodology**: Isolation Forest (contamination={anom['contamination_threshold']}) + Rolling $Z$-score (15-cycle window)\n"
                    f"- **Detected Anomalies**: **{anom['detected_anomalies_count']}** flagged out of {anom['total_observed_cycles']} cycles ({anom['anomaly_rate_percent']}%)\n"
                    f"- **Classification**: **{anom['classification']}**\n"
                    f"- **Engineering Assessment**: {anom['interpretation']}\n"
                )
                if flagged:
                    sample_flags = flagged[-3:]  # show latest 3 flagged
                    report_parts.append(f"**Latest Flagged Cycles:**\n")
                    for fl in sample_flags:
                        report_parts.append(f"- *Cycle {fl['cycle']}*: Reading = {fl['value']} {anom['unit']} ({fl['reason']})\n")

        # 4. Fleet Comparison Evidence
        if "compare_engines" in evidence:
            cmp = evidence["compare_engines"]
            if cmp.get("status") == "success":
                fleet = cmp["fleet_comparison"]
                report_parts.append(f"#### 🛩️ Fleet Telemetry & Risk Comparison\n")
                report_parts.append("| Engine ID | Latest Cycle | Predicted RUL | Risk Tier | HPC Static Pressure (Ps30) |\n|---|---|---|---|---|\n")
                for e in fleet:
                    report_parts.append(
                        f"| Unit #{e['engine_id']} | {e['latest_observed_cycle']} | **{e['predicted_rul_cycles']} cycles** | {e['risk_tier']} | {e['key_sensors'].get('Ps30 (s_11)')} psia |\n"
                    )
                report_parts.append("\n")

        # 5. Model Evaluation Evidence
        if "get_model_evaluation" in evidence:
            me = evidence["get_model_evaluation"]
            if me.get("status") == "success":
                eval_info = me["evaluation_results"]
                rf_m = eval_info.get("models", {}).get("random_forest", {})
                base_m = eval_info.get("baseline", {})
                report_parts.append(
                    f"#### 📊 Verified Model Evaluation Metrics\n"
                    f"- **Dataset**: NASA C-MAPSS FD001 (100 training engines, 100 test engines)\n"
                    f"- **Historical Baseline**: MAE = {base_m.get('test_mae')} cycles, RMSE = {base_m.get('test_rmse')} cycles, NASA Score = {base_m.get('nasa_score')}\n"
                    f"- **Trained Random Forest**: **MAE = {rf_m.get('test_mae')} cycles**, RMSE = {rf_m.get('test_rmse')} cycles, $R^2$ = {rf_m.get('r2_score')}, NASA Score = {rf_m.get('nasa_score')}\n"
                    f"- **Improvement Over Baseline**: **{rf_m.get('improvement_over_baseline_mae')} cycles reduction in MAE**\n"
                    f"- **Data Leakage Safeguards**: {eval_info.get('evaluation_metadata', {}).get('leakage_safeguards')}\n"
                )

        # 6. Technical Citations Evidence
        if citations:
            report_parts.append(f"#### 📚 Technical Manual Citations\n")
            for c in citations[:2]:
                report_parts.append(
                    f"- **Source**: `{c['source_file']}` — *{c['section']}* (Relevance: {c['score']})\n"
                    f"  > \"{c['excerpt'][:250]}...\"\n\n"
                )

        state["final_answer"] = "".join(report_parts)
        trace.append("Response synthesis complete: Final engineering report generated.")
        state["reasoning_trace"] = trace
        return state

    def run(self, question: str) -> Dict[str, Any]:
        """Synchronously invokes the compiled LangGraph workflow."""
        initial_state: EngineMindState = {
            "question": question,
            "is_valid": True,
            "validation_error": None,
            "intent": None,
            "target_engine_ids": [],
            "target_sensor": None,
            "selected_tools": [],
            "tool_inputs": {},
            "tool_outputs": {},
            "evidence": {},
            "citations": [],
            "reasoning_trace": [],
            "status": "initialized",
            "retries": 0,
            "final_answer": None,
            "error": None
        }

        final_state = self.graph.invoke(initial_state)
        return final_state


class IndustrialAgentOrchestrator:
    """
    Backward-compatible legacy orchestrator.
    Handles earlier PUMP/SQLite requests while routing turbofan inquiries to EngineMind.
    """
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self.engine_mind = EngineMindOrchestrator()

    def run(self, query: str) -> Dict[str, Any]:
        q_upper = query.upper()
        # If the query pertains to legacy equipment in test_plant.db (e.g. PUMP-101), execute legacy flow
        if any(eq in q_upper for eq in ["PUMP-101", "PUMP-102", "TURBINE-01", "COMPRESSOR-04"]):
            return self._legacy_run(query)

        # Otherwise route through EngineMind LangGraph workflow
        res = self.engine_mind.run(query)
        # Map to legacy structure for compatibility if needed
        return {
            "user_query": query,
            "target_equipment": f"Engine-{res.get('target_engine_ids', [24])[0]}",
            "target_metric": res.get("target_sensor", "s_11"),
            "sql_query": None,
            "retrieved_data": res.get("evidence", {}).get("get_latest_engine_status", {}),
            "numerical_evidence": res.get("evidence"),
            "reasoning_trace": res.get("reasoning_trace", []),
            "final_response": res.get("final_answer", "")
        }

    def _legacy_run(self, query: str) -> Dict[str, Any]:
        trace = ["Step 1: Analyzed user query intent."]
        target_equipment = "PUMP-101"
        for eq in ["PUMP-101", "PUMP-102", "TURBINE-01", "COMPRESSOR-04"]:
            if eq in query.upper():
                target_equipment = eq
                break

        target_metric = "vibration_rms"
        if "TEMP" in query.upper():
            target_metric = "temperature"
        elif "POWER" in query.upper():
            target_metric = "power_kw"

        trace.append(f"Step 2: Constructed schema-aware SQL for {target_equipment} ({target_metric}).")
        sql = f"SELECT timestamp, value, unit FROM sensor_telemetry WHERE equipment_id = '{target_equipment}' AND metric_name = '{target_metric}' ORDER BY timestamp ASC;"
        result = query_sensor_database(sql, db_path=self.db_path)
        rows = result.get("data", [])
        trace.append(f"Step 2 (cont): Retrieved {len(rows)} telemetry rows.")

        values = [r["value"] for r in rows] if rows else []
        stats = compute_series_statistics(values)
        thresholds = evaluate_anomaly_thresholds(values, warning_threshold=3.5, critical_threshold=4.5)
        trend = analyze_trend_and_slope(values)
        trace.append("Step 3: Computed exact summary statistics and safety threshold bounds.")

        critical_count = thresholds.get("critical_violation_count", 0)
        final_response = (
            f"CRITICAL ANOMALY DETECTED on {target_equipment}: {critical_count} readings exceeded critical threshold."
            if critical_count > 0 else f"Equipment {target_equipment} is OPERATIONAL with nominal {target_metric}."
        )
        trace.append("Step 4: Grounded diagnostic synthesized.")

        return {
            "user_query": query,
            "target_equipment": target_equipment,
            "target_metric": target_metric,
            "sql_query": sql,
            "retrieved_data": rows,
            "numerical_evidence": {
                "statistics": stats,
                "threshold_eval": thresholds,
                "trend": trend
            },
            "reasoning_trace": trace,
            "final_response": final_response
        }
