"""Typed state representation for the DecisionIQ LangGraph agent."""

from typing import Annotated, Any, Dict, List

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """Execution state passed through the LangGraph StateGraph nodes."""

    # Chat history
    messages: Annotated[List[BaseMessage], add_messages]

    # Query & Classification
    query: str
    intent: str
    intent_confidence: float

    # Execution Plan & Tool Output
    plan: List[str]
    tool_results: Dict[str, Any]
    retrieved_docs: List[Document]
    sql_results: List[Dict[str, Any]]
    sql_queries: List[str]
    anomalies: List[Dict[str, Any]]
    calc_results: List[str]

    # Grounding & Verification Loop
    grounding_verdict: str  # "GROUNDED" or "UNGROUNDED"
    grounding_reasoning: str
    retry_count: int

    # Final Response & Citations
    citations: List[str]
    final_answer: str
    is_interrupted: bool
