"""API Pydantic Request and Response Schemas."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Incoming user query payload."""

    query: str = Field(..., min_length=1, description="Business or operational question to be answered.")
    thread_id: Optional[str] = Field(None, description="Optional conversation session ID for multi-turn state.")
    vector_store: str = Field("faiss", description="Target vector store: 'faiss' or 'chroma'.")
    retrieval_strategy: str = Field("hybrid", description="Strategy: 'dense', 'bm25', 'hybrid', 'hybrid_rerank'.")
    top_k: int = Field(3, ge=1, le=20, description="Number of document chunks to retrieve.")
    stream: bool = Field(False, description="Whether to stream response tokens and trace via SSE.")


class CitationItem(BaseModel):
    """Extracted citation detail from documents or structured queries."""

    source_type: str = Field("doc", description="'doc' or 'sql'")
    identifier: str = Field(..., description="Document filename + chunk ID, or SQL query snippet")
    snippet: Optional[str] = Field(None, description="Relevant text snippet or query result preview")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata such as version, department")


class QueryResponse(BaseModel):
    """Structured response from the DecisionIQ agentic RAG system."""

    query: str
    answer: str
    intent: str
    confidence: float
    plan: List[str]
    citations: List[CitationItem] = Field(default_factory=list)
    execution_trace: List[str] = Field(default_factory=list)
    grounded: bool = True
    abstention: bool = False
    latency_ms: float
    thread_id: str
    sql_executed: Optional[str] = None
    anomalies_detected: Optional[int] = None


class IngestRequest(BaseModel):
    """Payload to trigger document re-ingestion."""

    store: str = Field("faiss", description="Vector store backend to ingest into ('faiss' or 'chroma')")
    data_dir: Optional[str] = Field(None, description="Optional directory path containing source documents")
    force_recreate: bool = Field(False, description="Whether to purge existing index and rebuild from scratch")


class IngestResponse(BaseModel):
    """Response detailing ingestion metrics."""

    status: str
    store: str
    documents_loaded: int
    chunks_created: int
    duration_seconds: float


class FeedbackRequest(BaseModel):
    """User evaluation feedback payload."""

    query_id: str = Field(..., description="Unique query execution or request ID")
    rating: int = Field(..., ge=-1, le=1, description="1 for thumbs up, -1 for thumbs down, 0 for neutral")
    comment: Optional[str] = Field(None, max_length=1000, description="Optional user commentary")
    thread_id: Optional[str] = Field(None, description="Session thread ID")


class FeedbackResponse(BaseModel):
    """Feedback receipt acknowledgment."""

    status: str
    feedback_id: str
    recorded_at: str


class HealthResponse(BaseModel):
    """System health and subsystem availability check."""

    status: str
    version: str
    faiss_available: bool
    chroma_available: bool
    database_connected: bool
    ml_models_loaded: bool
