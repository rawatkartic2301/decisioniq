"""API route definitions for DecisionIQ."""

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse

from decisioniq.agent.graph import DecisionIQAgent
from decisioniq.api.schemas import (
    CitationItem,
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from decisioniq.config import settings
from decisioniq.ingestion.pipeline import run_ingestion_pipeline

logger = logging.getLogger("decisioniq.api.routes")
router = APIRouter(prefix="/api/v1")

# Global agent singleton
_agent: DecisionIQAgent | None = None


def get_agent() -> DecisionIQAgent:
    global _agent
    if _agent is None:
        _agent = DecisionIQAgent()
    return _agent


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Submit query to DecisionIQ Agentic RAG",
)
def query_endpoint(req: QueryRequest):
    """Processes user queries combining documents, SQL database, and ML tools."""
    start_time = time.time()
    thread_id = req.thread_id or f"session_{uuid.uuid4().hex[:8]}"

    # Update dynamic settings if requested
    if req.vector_store:
        settings.vector_store = req.vector_store  # type: ignore
    if req.retrieval_strategy:
        settings.retrieval_strategy = req.retrieval_strategy
    if req.top_k:
        settings.top_k = req.top_k

    agent = get_agent()

    if req.stream:
        # SSE Streaming Response
        async def event_generator():
            steps_yielded = 0
            for step in agent.stream_steps(req.query, thread_id=thread_id):
                steps_yielded += 1
                for node_name, state_update in step.items():
                    event_data = {
                        "type": "step",
                        "step_index": steps_yielded,
                        "node": node_name,
                        "summary": f"Executed node: {node_name}",
                    }
                    if "intent" in state_update:
                        event_data["intent"] = state_update["intent"]
                        event_data["confidence"] = state_update.get("intent_confidence", 0.0)
                    if "plan" in state_update:
                        event_data["plan"] = state_update["plan"]
                    if "final_answer" in state_update:
                        event_data["final_answer"] = state_update["final_answer"]

                    yield f"data: {json.dumps(event_data)}\n\n"

            # Final complete response event
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            final_data = {
                "type": "complete",
                "latency_ms": elapsed_ms,
                "thread_id": thread_id,
            }
            yield f"data: {json.dumps(final_data)}\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # Non-streaming full response
    try:
        final_state = agent.invoke(req.query, thread_id=thread_id)
    except Exception as e:
        logger.error(f"Error during agent invocation: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failure: {str(e)}",
        ) from e

    latency_ms = round((time.time() - start_time) * 1000, 2)
    answer = final_state.get("final_answer", "")
    intent = final_state.get("intent", "policy_rag")
    confidence = final_state.get("intent_confidence", 1.0)
    plan = final_state.get("plan", [])
    grounded_verdict = final_state.get("grounding_verdict", "GROUNDED")
    is_grounded = grounded_verdict == "GROUNDED"
    is_abstention = "don't have enough verified information" in answer.lower()

    # Build structured citations
    citations: list[CitationItem] = []
    for doc in final_state.get("retrieved_docs", []):
        meta = doc.metadata or {}
        doc_src = meta.get("source_doc") or Path(meta.get("source", "document")).name
        chunk_id = meta.get("chunk_id", "chunk")
        identifier = f"{doc_src}#{chunk_id}"
        snippet = doc.page_content[:200] + ("..." if len(doc.page_content) > 200 else "")
        citations.append(
            CitationItem(
                source_type="doc",
                identifier=identifier,
                snippet=snippet,
                metadata=meta,
            )
        )

    for sql_q in final_state.get("sql_queries", []):
        citations.append(
            CitationItem(
                source_type="sql",
                identifier="Database SQL Query",
                snippet=sql_q,
                metadata={"database": "decisioniq.db"},
            )
        )

    sql_executed = final_state.get("sql_queries", [None])[0] if final_state.get("sql_queries") else None
    anomalies_detected = len(final_state.get("anomalies", [])) if final_state.get("anomalies") else None

    # Derive trace steps from state
    execution_trace = ["classify_intent", "planner", "tools", "synthesize", "grounding_check"]
    if final_state.get("retry_count", 0) > 0:
        execution_trace.append(f"query_rewrite(retries={final_state['retry_count']})")
    if is_abstention:
        execution_trace.append("abstain")

    return QueryResponse(
        query=req.query,
        answer=answer,
        intent=intent,
        confidence=confidence,
        plan=plan,
        citations=citations,
        execution_trace=execution_trace,
        grounded=is_grounded,
        abstention=is_abstention,
        latency_ms=latency_ms,
        thread_id=thread_id,
        sql_executed=sql_executed,
        anomalies_detected=anomalies_detected,
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger document re-ingestion into vector store",
)
def ingest_endpoint(req: IngestRequest):
    """Ingests, chunks, embeds, and indexes documents into FAISS or Chroma."""
    start_time = time.time()
    try:
        docs_dir = Path(req.data_dir) if req.data_dir else Path("data/docs")
        chunk_count = run_ingestion_pipeline(
            store_type=req.store,  # type: ignore
            docs_dir=docs_dir,
            force_data_gen=req.force_recreate,
        )
        duration = round(time.time() - start_time, 2)
        doc_count = len(list(docs_dir.glob("*.*"))) if docs_dir.exists() else 0
        return IngestResponse(
            status="success",
            store=req.store,
            documents_loaded=doc_count,
            chunks_created=chunk_count,
            duration_seconds=duration,
        )
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}",
        ) from e


@router.post(
    "/feedback",
    response_model=FeedbackResponse,
    summary="Submit user feedback on answer quality",
)
def feedback_endpoint(req: FeedbackRequest):
    """Records user rating (thumbs up/down) and qualitative feedback into SQLite."""
    feedback_id = f"fb_{uuid.uuid4().hex[:10]}"
    timestamp = datetime.utcnow().isoformat()

    db_path = Path("data/db/decisioniq.db")
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS user_feedback (
                feedback_id TEXT PRIMARY KEY,
                query_id TEXT,
                rating INTEGER,
                comment TEXT,
                thread_id TEXT,
                created_at TEXT
            )
            """
        )
        cur.execute(
            """
            INSERT INTO user_feedback (feedback_id, query_id, rating, comment, thread_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (feedback_id, req.query_id, req.rating, req.comment or "", req.thread_id or "", timestamp),
        )
        conn.commit()
        conn.close()

        logger.info(f"Recorded feedback {feedback_id} for query {req.query_id}: rating={req.rating}")
        return FeedbackResponse(
            status="recorded",
            feedback_id=feedback_id,
            recorded_at=timestamp,
        )
    except Exception as e:
        logger.error(f"Failed to record feedback: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to record user feedback.",
        ) from e


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Check health and readiness of all subsystems",
)
def health_endpoint():
    """Validates connectivity to vector stores, SQLite, and scikit-learn models."""
    faiss_ok = (settings.faiss_index_path / "index.faiss").exists() or settings.faiss_index_path.exists()
    chroma_ok = settings.chroma_persist_dir.exists()

    db_path = Path("data/db/decisioniq.db")
    db_ok = False
    if db_path.exists():
        try:
            conn = sqlite3.connect(db_path)
            conn.execute("SELECT 1 FROM employees LIMIT 1;")
            conn.close()
            db_ok = True
        except Exception:
            db_ok = False

    models_ok = (
        Path("src/decisioniq/ml/saved_models/intent_classifier.joblib").exists()
        and Path("src/decisioniq/ml/saved_models/anomaly_detector.joblib").exists()
    )

    all_healthy = db_ok and (faiss_ok or chroma_ok)
    sys_status = "healthy" if all_healthy else "degraded"

    return HealthResponse(
        status=sys_status,
        version="1.0.0",
        faiss_available=faiss_ok,
        chroma_available=chroma_ok,
        database_connected=db_ok,
        ml_models_loaded=models_ok,
    )
