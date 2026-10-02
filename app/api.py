"""
Industrial AI Agent Platform API (FastAPI)
Serves agent query orchestration, database schema inspection, and plant telemetry endpoints.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from database.db_manager import init_database, get_schema_summary, execute_read_query
from agents.graph import IndustrialAgentOrchestrator

app = FastAPI(
    title="Industrial AI Agent Platform",
    description="Evidence-driven agentic orchestrator for industrial plant telemetry with deterministic computation",
    version="1.0.0"
)

# Ensure database is initialized at startup
init_database()
orchestrator = IndustrialAgentOrchestrator()


class AgentQueryRequest(BaseModel):
    query: str = Field(..., example="What is the vibration status of PUMP-101? Did it exceed safety thresholds?")


class AgentQueryResponse(BaseModel):
    query: str
    target_equipment: Optional[str]
    target_metric: Optional[str]
    sql_query: Optional[str]
    numerical_evidence: Optional[Dict[str, Any]]
    reasoning_trace: List[str]
    final_response: str


@app.get("/")
def root():
    return {
        "service": "Industrial AI Agent Platform",
        "orchestration": "LangGraph StateGraph",
        "reasoning": "Evidence-Driven Decoupled Architecture",
        "endpoints": [
            "/api/v1/schema",
            "/api/v1/equipment",
            "/api/v1/agent/query"
        ]
    }


@app.get("/api/v1/schema")
def get_schema():
    """Inspects SQLite database schema catalog."""
    return {
        "status": "success",
        "schema": get_schema_summary()
    }


@app.get("/api/v1/equipment")
def list_equipment():
    """Lists all monitored equipment in the industrial facility."""
    rows = execute_read_query("SELECT * FROM equipment ORDER BY equipment_id ASC;")
    return {
        "equipment_count": len(rows),
        "equipment": rows
    }


@app.post("/api/v1/agent/query", response_model=AgentQueryResponse)
def execute_agent_query(req: AgentQueryRequest):
    """
    Submits an industrial inquiry to the LangGraph orchestration layer.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    result = orchestrator.run(req.query)
    return AgentQueryResponse(
        query=result["user_query"],
        target_equipment=result.get("target_equipment"),
        target_metric=result.get("target_metric"),
        sql_query=result.get("sql_query"),
        numerical_evidence=result.get("numerical_evidence"),
        reasoning_trace=result.get("reasoning_trace", []),
        final_response=result.get("final_response", "")
    )
