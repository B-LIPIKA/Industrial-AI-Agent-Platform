"""
Document RAG (Retrieval-Augmented Generation) Subsystem for Turbofan Technical Manuals.
Provides modular text extraction, chunking, local vector indexing (TF-IDF cosine similarity),
source citation tracking, and semantic search.
"""

import os
import re
from typing import List, Dict, Any, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_DOCS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "docs", "technical_manuals"
)


class DocumentChunk:
    """Represents a text chunk with complete source provenance."""
    def __init__(
        self,
        chunk_id: str,
        source_file: str,
        section: str,
        text: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.chunk_id = chunk_id
        self.source_file = source_file
        self.section = section
        self.text = text
        self.metadata = metadata or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "source_file": self.source_file,
            "section": self.section,
            "text": self.text,
            "metadata": self.metadata
        }


class TechnicalDocumentRAG:
    """
    Local, deterministic RAG engine for aircraft engine technical documentation.
    Features:
    - Text chunking preserving section headers and document source
    - TF-IDF + Cosine Similarity semantic search
    - Citation tracking (filename, section, chunk ID)
    - Empty state handling
    - Dynamic document ingestion
    """
    _instance: Optional["TechnicalDocumentRAG"] = None

    def __init__(self, docs_dir: str = DEFAULT_DOCS_DIR):
        self.docs_dir = docs_dir
        self.chunks: List[DocumentChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.indexed_files: List[str] = []
        self.is_indexed = False

    @classmethod
    def get_instance(cls, docs_dir: str = DEFAULT_DOCS_DIR) -> "TechnicalDocumentRAG":
        """Singleton accessor for efficient reuse."""
        if cls._instance is None:
            cls._instance = cls(docs_dir=docs_dir)
            cls._instance.build_index()
        return cls._instance

    def _extract_chunks_from_markdown(self, filepath: str) -> List[DocumentChunk]:
        """Extracts chunks from a markdown document preserving sections."""
        filename = os.path.basename(filepath)
        chunks = []
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        # Split by markdown headers
        sections = re.split(r'\n(?=#{1,3}\s+)', content)
        chunk_idx = 0

        for sec in sections:
            sec_clean = sec.strip()
            if not sec_clean:
                continue

            # Extract header if present
            header_match = re.match(r'^#{1,3}\s+(.+)$', sec_clean, re.MULTILINE)
            section_title = header_match.group(1).strip() if header_match else "General"

            # If section is very large, split into sub-chunks of ~500 words
            paragraphs = sec_clean.split("\n\n")
            curr_text = []
            curr_len = 0

            for p in paragraphs:
                p_len = len(p.split())
                if curr_len + p_len > 120 and curr_text:
                    text_block = "\n\n".join(curr_text).strip()
                    if text_block:
                        chunk_idx += 1
                        chunks.append(DocumentChunk(
                            chunk_id=f"{filename}#c{chunk_idx}",
                            source_file=filename,
                            section=section_title,
                            text=text_block
                        ))
                    curr_text = [p]
                    curr_len = p_len
                else:
                    curr_text.append(p)
                    curr_len += p_len

            if curr_text:
                text_block = "\n\n".join(curr_text).strip()
                if text_block:
                    chunk_idx += 1
                    chunks.append(DocumentChunk(
                        chunk_id=f"{filename}#c{chunk_idx}",
                        source_file=filename,
                        section=section_title,
                        text=text_block
                    ))

        return chunks

    def build_index(self, force: bool = False) -> Dict[str, Any]:
        """
        Indexes all technical documentation files in docs_dir.
        """
        if self.is_indexed and not force:
            return self.get_index_status()

        self.chunks = []
        self.indexed_files = []

        if os.path.isdir(self.docs_dir):
            for fname in os.listdir(self.docs_dir):
                if fname.endswith((".md", ".txt")):
                    fpath = os.path.join(self.docs_dir, fname)
                    file_chunks = self._extract_chunks_from_markdown(fpath)
                    self.chunks.extend(file_chunks)
                    self.indexed_files.append(fname)

        if not self.chunks:
            self.is_indexed = False
            self.vectorizer = None
            self.tfidf_matrix = None
            return {
                "status": "empty",
                "message": f"No technical documents found in {self.docs_dir}. RAG operates in empty state.",
                "total_documents": 0,
                "total_chunks": 0
            }

        # Build TF-IDF vector matrix
        corpus = [chunk.text for chunk in self.chunks]
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=1.0,
            min_df=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.is_indexed = True

        return {
            "status": "success",
            "total_documents": len(self.indexed_files),
            "indexed_files": self.indexed_files,
            "total_chunks": len(self.chunks)
        }

    def add_document(self, filename: str, content: str) -> Dict[str, Any]:
        """Dynamically ingests and indexes a user-uploaded document."""
        os.makedirs(self.docs_dir, exist_ok=True)
        dest_path = os.path.join(self.docs_dir, filename)
        with open(dest_path, "w", encoding="utf-8") as f:
            f.write(content)
        return self.build_index(force=True)

    def retrieve(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes semantic retrieval for a user question.
        Returns top-k matching excerpts with citations and relevance scores.
        """
        if not self.is_indexed or not self.chunks or self.vectorizer is None:
            return {
                "status": "empty",
                "query": query,
                "message": "No technical documentation is currently indexed. Use 'add_document' or load manuals into docs/technical_manuals/.",
                "results": []
            }

        query_clean = query.strip()
        if not query_clean:
            return {
                "status": "error",
                "query": query,
                "message": "Query cannot be empty.",
                "results": []
            }

        query_vec = self.vectorizer.transform([query_clean])
        scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for idx in top_indices:
            score = float(scores[idx])
            if score > 0.05:  # Relevance threshold
                chunk = self.chunks[idx]
                results.append({
                    "chunk_id": chunk.chunk_id,
                    "source_file": chunk.source_file,
                    "section": chunk.section,
                    "score": round(score, 3),
                    "excerpt": chunk.text,
                    "citation": f"{chunk.source_file} (Section: {chunk.section})"
                })

        return {
            "status": "success",
            "query": query,
            "matched_chunks": len(results),
            "results": results,
            "has_relevant_passages": len(results) > 0
        }

    def get_index_status(self) -> Dict[str, Any]:
        """Returns metadata about the active index."""
        return {
            "is_indexed": self.is_indexed,
            "docs_dir": self.docs_dir,
            "indexed_files": self.indexed_files,
            "total_chunks": len(self.chunks)
        }
