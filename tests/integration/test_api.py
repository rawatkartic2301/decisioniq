"""Integration and functional tests for DecisionIQ FastAPI endpoints and business logic."""

from decisioniq.api.app import app
from decisioniq.api.routes import (
    feedback_endpoint,
    health_endpoint,
    query_endpoint,
)
from decisioniq.api.schemas import (
    FeedbackRequest,
    FeedbackResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)


def test_root_endpoint():
    """Test root application status endpoint."""
    root_handler = None
    for route in app.routes:
        if getattr(route, "path", None) == "/" and "GET" in getattr(route, "methods", set()):
            root_handler = route.endpoint
            break

    assert root_handler is not None
    res = root_handler()
    assert res["name"] == "DecisionIQ API"
    assert res["status"] == "online"
    assert res["version"] == "1.0.0"


def test_health_endpoint():
    """Test health and readiness endpoint across all subsystems."""
    res: HealthResponse = health_endpoint()
    assert res.status == "healthy"
    assert res.database_connected is True
    assert res.faiss_available is True
    assert res.ml_models_loaded is True


def test_query_policy_rag():
    """Test standard policy RAG query through API route."""
    req = QueryRequest(
        query="What is our policy on carry-forward leave?",
        vector_store="faiss",
        retrieval_strategy="hybrid",
        top_k=3,
        stream=False,
    )
    res: QueryResponse = query_endpoint(req)
    assert isinstance(res, QueryResponse)
    assert len(res.answer) > 0
    assert res.intent in ["policy_rag", "hybrid_rag_sql", "unanswerable"]
    assert res.latency_ms > 0
    assert res.thread_id.startswith("session_")
    assert len(res.citations) > 0
    assert any(c.source_type == "doc" for c in res.citations)


def test_query_sql_analytics():
    """Test SQL analytics query through API route."""
    req = QueryRequest(
        query="How many overtime hours did the Pune team log in March?",
        vector_store="faiss",
        retrieval_strategy="hybrid",
        stream=False,
    )
    res: QueryResponse = query_endpoint(req)
    assert isinstance(res, QueryResponse)
    assert len(res.answer) > 0
    assert res.sql_executed is not None or any(c.source_type == "sql" for c in res.citations)


def test_query_unanswerable_abstention():
    """Test that an unanswerable query results in verified abstention."""
    req = QueryRequest(
        query="What is our company policy on pet insurance reimbursement?",
        stream=False,
    )
    res: QueryResponse = query_endpoint(req)
    assert isinstance(res, QueryResponse)
    assert res.abstention is True or "don't have enough verified information" in res.answer.lower()


def test_query_streaming_sse():
    """Test SSE streaming response generation through API route."""
    req = QueryRequest(
        query="What is the overtime rule for grade L3?",
        stream=True,
    )
    response = query_endpoint(req)
    assert response.media_type == "text/event-stream"

    # Consume async generator without requiring an OS socket-based event loop
    events = []
    gen = response.body_iterator
    while True:
        try:
            gen.asend(None).send(None)
        except StopIteration as e:
            val = e.value
            events.append(val if isinstance(val, str) else val.decode())
        except StopAsyncIteration:
            break

    assert len(events) > 0
    combined = "".join(events)
    assert "data: " in combined
    assert '"type":' in combined


def test_feedback_submission():
    """Test user feedback recording into the database."""
    req = FeedbackRequest(
        query_id="query_test_001",
        rating=1,
        comment="Accurate policy citation and clear breakdown.",
        thread_id="test_session_abc",
    )
    res: FeedbackResponse = feedback_endpoint(req)
    assert isinstance(res, FeedbackResponse)
    assert res.status == "recorded"
    assert res.feedback_id.startswith("fb_")
    assert len(res.recorded_at) > 0
