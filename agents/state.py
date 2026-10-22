"""
Agent State Schema for LangGraph EngineMind AI Copilot
Defines typed state schemas for evidence-driven turbofan degradation analysis,
RUL estimation, anomaly detection, and document RAG synthesis.
"""

from typing import TypedDict, Optional, List, Dict, Any


class EngineMindState(TypedDict):
    """
    Typed State Schema for the EngineMind AI LangGraph Agent.
    Tracks user intent, entity extraction, selected tools, deterministic evidence,
    technical citations, and verifiable reasoning traces.
    """
    question: str
    is_valid: bool
    validation_error: Optional[str]
    intent: Optional[str]
    target_engine_ids: List[int]
    target_sensor: Optional[str]
    selected_tools: List[str]
    tool_inputs: Dict[str, Any]
    tool_outputs: Dict[str, Any]
    evidence: Dict[str, Any]
    citations: List[Dict[str, Any]]
    reasoning_trace: List[str]
    status: str  # "success", "partial", "fallback", "error"
    retries: int
    final_answer: Optional[str]
    error: Optional[str]


# Backward-compatible alias for existing tests
class IndustrialAgentState(TypedDict):
    user_query: str
    target_equipment: Optional[str]
    target_metric: Optional[str]
    sql_query: Optional[str]
    retrieved_data: Optional[List[Dict[str, Any]]]
    numerical_evidence: Optional[Dict[str, Any]]
    reasoning_trace: List[str]
    final_response: Optional[str]
