"""
Unit tests for the Technical Document RAG subsystem.
"""

import pytest
import os
from rag.rag_engine import TechnicalDocumentRAG, DEFAULT_DOCS_DIR


@pytest.fixture(scope="module")
def rag_instance():
    rag = TechnicalDocumentRAG.get_instance()
    rag.build_index()
    return rag


def test_rag_indexing(rag_instance):
    status = rag_instance.get_index_status()
    assert status["is_indexed"] is True
    assert status["total_chunks"] > 0
    assert "turbofan_maintenance_manual.md" in status["indexed_files"]
    assert "sensor_reference_guide.md" in status["indexed_files"]
    assert "predictive_maintenance_procedures.md" in status["indexed_files"]


def test_rag_retrieval_with_citations(rag_instance):
    res = rag_instance.retrieve("What is the protocol for critical AOG risk and grounding?", top_k=2)
    assert res["status"] == "success"
    assert res["has_relevant_passages"] is True
    assert len(res["results"]) > 0

    first = res["results"][0]
    assert "source_file" in first
    assert "section" in first
    assert "excerpt" in first
    assert "citation" in first
    assert first["score"] > 0.05
    assert "predictive_maintenance_procedures.md" in first["source_file"]


def test_rag_empty_query(rag_instance):
    res = rag_instance.retrieve("")
    assert res["status"] == "error"
    assert len(res["results"]) == 0


def test_rag_dynamic_document_ingestion(rag_instance, tmp_path):
    custom_rag = TechnicalDocumentRAG(docs_dir=str(tmp_path))
    # Initially empty
    empty_status = custom_rag.build_index()
    assert empty_status["status"] == "empty"
    assert custom_rag.retrieve("any query")["status"] == "empty"

    # Add a document
    doc_content = "# Emergency Shutdown SOP\n\nWhen vibration exceeds 5.0 mm/s, immediately cut fuel supply."
    add_res = custom_rag.add_document("emergency_sop.md", doc_content)
    assert add_res["status"] == "success"
    assert add_res["total_chunks"] == 1

    search_res = custom_rag.retrieve("emergency vibration fuel cut", top_k=1)
    assert search_res["has_relevant_passages"] is True
    assert search_res["results"][0]["source_file"] == "emergency_sop.md"
