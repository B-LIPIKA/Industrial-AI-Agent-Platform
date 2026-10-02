"""
Agent State Schema for LangGraph Industrial Orchestration
"""

from typing import TypedDict, Optional, List, Dict, Any


class IndustrialAgentState(TypedDict):
    user_query: str
    target_equipment: Optional[str]
    target_metric: Optional[str]
    sql_query: Optional[str]
    retrieved_data: Optional[List[Dict[str, Any]]]
    numerical_evidence: Optional[Dict[str, Any]]
    reasoning_trace: List[str]
    final_response: Optional[str]
