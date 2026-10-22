"""
FastAPI Application for EngineMind AI Copilot.
Provides REST API endpoints for turbofan telemetry discovery, RUL inference,
anomaly detection, time-series analytics, LangGraph agent chat, and document RAG.
Includes full backward compatibility with legacy plant endpoints.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agents.graph import EngineMindOrchestrator, IndustrialAgentOrchestrator
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
from rag.rag_engine import TechnicalDocumentRAG
from database.db_manager import init_database, get_schema_summary, execute_read_query

# Initialize FastAPI app
app = FastAPI(
    title="EngineMind AI — Turbofan Predictive Maintenance API",
    description=(
        "Production-style API for aircraft turbofan engine degradation analysis, "
        "remaining useful life (RUL) estimation, sensor anomaly detection, "
        "and evidence-grounded LangGraph agent orchestration."
    ),
    version="2.0.0"
)

# Enable CORS for local Streamlit or external frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure legacy DB is initialized for backward compatibility
init_database()

# Orchestrator instances
engine_mind = EngineMindOrchestrator()
legacy_orchestrator = IndustrialAgentOrchestrator()
rag_engine = TechnicalDocumentRAG.get_instance()


# --- Pydantic Schemas ---

class PredictRULRequest(BaseModel):
    engine_id: int = Field(..., ge=1, le=100, description="Turbofan unit identifier (1-100 in FD001)")

    model_config = {
        "json_schema_extra": {
            "example": {"engine_id": 24}
        }
    }


class TrendAnalysisRequest(BaseModel):
    engine_id: int = Field(..., ge=1, le=100, description="Turbofan unit identifier")
    sensor_name: str = Field(..., description="Sensor column (s_1 to s_21) or alias")

    model_config = {
        "json_schema_extra": {
            "example": {"engine_id": 24, "sensor_name": "s_11"}
        }
    }


class AnomalyAnalysisRequest(BaseModel):
    engine_id: int = Field(..., ge=1, le=100, description="Turbofan unit identifier")
    sensor_name: str = Field(..., description="Sensor column (s_1 to s_21)")
    contamination: float = Field(default=0.05, ge=0.01, le=0.2, description="Expected anomaly fraction")

    model_config = {
        "json_schema_extra": {
            "example": {"engine_id": 24, "sensor_name": "s_11", "contamination": 0.05}
        }
    }


class CompareEnginesRequest(BaseModel):
    engine_ids: List[int] = Field(..., min_length=2, max_length=10, description="List of engine IDs to compare")

    model_config = {
        "json_schema_extra": {
            "example": {"engine_ids": [12, 24, 34]}
        }
    }


class AgentChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000, description="Technical inquiry regarding engine fleet")

    model_config = {
        "json_schema_extra": {
            "example": {"query": "Estimate the remaining useful life of engine 24."}
        }
    }


class RAGQueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500, description="Question for technical manual RAG")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of excerpts to retrieve")

    model_config = {
        "json_schema_extra": {
            "example": {"question": "What is the critical RUL threshold for grounding?", "top_k": 3}
        }
    }


class RAGDocumentUploadRequest(BaseModel):
    filename: str = Field(..., description="Filename ending in .md or .txt")
    content: str = Field(..., description="Document text content")


# Legacy schemas for backward compatibility
class LegacyAgentQueryRequest(BaseModel):
    query: str = Field(...)


# --- Core System Endpoints ---

@app.get("/", tags=["System"])
def root():
    return {
        "project": "EngineMind AI — Agentic Predictive Maintenance Copilot",
        "status": "operational",
        "dataset": "NASA C-MAPSS Turbofan Engine Degradation Simulation (FD001)",
        "docs_url": "/docs",
        "endpoints": {
            "health": "/health",
            "dataset_summary": "/api/dataset/summary",
            "engines_list": "/api/engines",
            "engine_history": "/api/engines/{engine_id}/history",
            "engine_status": "/api/engines/{engine_id}/status",
            "predict_rul": "/api/predict/rul",
            "sensor_trends": "/api/analyze/trends",
            "sensor_anomalies": "/api/analyze/anomalies",
            "compare_engines": "/api/engines/compare",
            "agent_chat": "/api/agent/chat",
            "evaluation": "/api/evaluation",
            "rag_query": "/api/rag/query"
        }
    }


@app.get("/health", tags=["System"])
def health_check():
    """System health check and component readiness indicators."""
    summary = get_dataset_summary()
    eval_res = get_model_evaluation()
    rag_status = rag_engine.get_index_status()

    return {
        "status": "healthy",
        "dataset_ready": summary.get("is_ready", False),
        "dataset_train_rows": summary.get("train_rows_count", 0),
        "dataset_test_rows": summary.get("test_rows_count", 0),
        "model_trained": eval_res.get("status") == "success",
        "best_model": eval_res.get("evaluation_results", {}).get("selected_best_model"),
        "rag_indexed_chunks": rag_status.get("total_chunks", 0),
        "rag_is_indexed": rag_status.get("is_indexed", False)
    }


@app.get("/api/dataset/summary", tags=["Dataset"])
def api_dataset_summary():
    """Returns verified dataset provenance, row counts, and engine unit counts."""
    return get_dataset_summary()


@app.get("/api/engines", tags=["Engines"])
def list_available_engines(split: str = Query("test", enum=["test", "train"])):
    """Lists all available engine IDs (1-100) with their latest observed cycle."""
    engine_list = []
    for eid in range(1, 101):
        status = get_latest_engine_status(eid, split=split)
        if status.get("status") == "success":
            engine_list.append({
                "engine_id": eid,
                "latest_observed_cycle": status.get("latest_observed_cycle")
            })
    return {
        "status": "success",
        "split": split,
        "engine_count": len(engine_list),
        "engines": engine_list
    }


@app.get("/api/engines/{engine_id}/history", tags=["Engines"])
def api_engine_history(
    engine_id: int,
    split: str = Query("test", enum=["test", "train"]),
    max_cycles: Optional[int] = Query(None, ge=1, le=500)
):
    """Retrieves observed sensor telemetry history for a specific engine."""
    res = get_engine_history(engine_id, split=split, max_cycles=max_cycles)
    if res.get("status") != "success":
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res


@app.get("/api/engines/{engine_id}/status", tags=["Engines"])
def api_engine_status(
    engine_id: int,
    split: str = Query("test", enum=["test", "train"])
):
    """Retrieves latest cycle snapshot and sensor readings for a specific engine."""
    res = get_latest_engine_status(engine_id, split=split)
    if res.get("status") != "success":
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res


# --- Predictive Analytics Endpoints ---

@app.post("/api/predict/rul", tags=["Prognostics"])
def api_predict_rul(req: PredictRULRequest):
    """Predicts Remaining Useful Life (RUL) in operating cycles for a test engine."""
    res = predict_remaining_useful_life(req.engine_id)
    if res.get("status") == "error":
        raise HTTPException(status_code=400, detail=res.get("error", "Prediction failed."))
    return res


@app.post("/api/analyze/trends", tags=["Analytics"])
def api_analyze_trends(req: TrendAnalysisRequest):
    """Computes descriptive statistics, rolling slopes, and drift rates for a sensor."""
    res = analyze_sensor_trend(req.engine_id, req.sensor_name)
    if res.get("status") != "success":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/analyze/anomalies", tags=["Analytics"])
def api_analyze_anomalies(req: AnomalyAnalysisRequest):
    """Runs Isolation Forest and rolling Z-score anomaly detection on observed history."""
    res = detect_sensor_anomalies(
        engine_id=req.engine_id,
        sensor_name=req.sensor_name,
        contamination=req.contamination
    )
    if res.get("status") != "success":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.post("/api/engines/compare", tags=["Fleet"])
def api_compare_engines(req: CompareEnginesRequest):
    """Compares telemetry and predicted RUL urgency across multiple engines."""
    res = compare_engines(req.engine_ids)
    if res.get("status") != "success":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


@app.get("/api/evaluation", tags=["Evaluation"])
def api_evaluation():
    """Returns measured benchmark evaluation metrics (MAE, RMSE, NASA scoring, baseline comparison)."""
    res = get_model_evaluation()
    if res.get("status") != "success":
        raise HTTPException(status_code=404, detail=res.get("message"))
    return res


# --- LangGraph Agent & RAG Endpoints ---

@app.post("/api/agent/chat", tags=["Agent"])
def api_agent_chat(req: AgentChatRequest):
    """
    Submits a natural language query to the LangGraph EngineMind copilot.
    Executes input validation, intent classification, tool selection, execution, and evidence-grounded synthesis.
    """
    result = engine_mind.run(req.query)
    return {
        "query": result["question"],
        "status": result.get("status"),
        "intent": result.get("intent"),
        "target_engine_ids": result.get("target_engine_ids", []),
        "target_sensor": result.get("target_sensor"),
        "selected_tools": result.get("selected_tools", []),
        "evidence": result.get("evidence", {}),
        "citations": result.get("citations", []),
        "reasoning_trace": result.get("reasoning_trace", []),
        "final_answer": result.get("final_answer", "")
    }


@app.post("/api/rag/query", tags=["RAG"])
def api_rag_query(req: RAGQueryRequest):
    """Searches indexed technical documentation for relevant excerpts and citations."""
    res = retrieve_technical_documents(req.question, top_k=req.top_k)
    return res


@app.post("/api/rag/upload", tags=["RAG"])
def api_rag_upload(req: RAGDocumentUploadRequest):
    """Uploads and indexes a new technical document (.md or .txt)."""
    if not (req.filename.endswith(".md") or req.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only markdown (.md) and text (.txt) files are supported.")

    res = rag_engine.add_document(req.filename, req.content)
    return {
        "status": "success",
        "filename": req.filename,
        "index_status": res
    }


# --- Legacy Backward Compatibility Endpoints ---

@app.get("/api/v1/schema", tags=["Legacy"])
def legacy_get_schema():
    return {
        "status": "success",
        "schema": get_schema_summary()
    }


@app.get("/api/v1/equipment", tags=["Legacy"])
def legacy_list_equipment():
    rows = execute_read_query("SELECT * FROM equipment ORDER BY equipment_id ASC;")
    return {
        "equipment_count": len(rows),
        "equipment": rows
    }


@app.post("/api/v1/agent/query", tags=["Legacy"])
def legacy_execute_agent_query(req: LegacyAgentQueryRequest):
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    res = legacy_orchestrator.run(req.query)
    return res
