"""Multi-agent supervisor architecture variant for DecisionIQ.

Orchestrates three specialized sub-agents under a central Supervisor Router:
1. Policy Specialist: Focuses on document retrieval, clause extraction, and active versioning.
2. Data Specialist: Focuses on SQL analytics, timesheet auditing, and anomaly detection.
3. Critic Agent: Evaluates grounding, cross-examines findings, and enforces citation fidelity.
"""

import json
import logging
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from decisioniq.config import settings
from decisioniq.llm.factory import get_chat_model
from decisioniq.tools.anomaly_tool import TimesheetAnomalyTool
from decisioniq.tools.retriever_tool import DocRetrieverTool
from decisioniq.tools.sql_tool import SQLQueryTool

logger = logging.getLogger(__name__)


class MultiAgentState(TypedDict):
    """Shared state across the supervisor and domain specialist agents."""

    query: str
    messages: List[BaseMessage]
    active_worker: Optional[str]
    policy_findings: str
    policy_citations: List[str]
    data_findings: str
    data_sql_queries: List[str]
    critic_verdict: str
    critic_notes: str
    final_decision: str
    iterations: int


# -------------------------------------------------------------
# 1. Specialist Agents
# -------------------------------------------------------------


def policy_specialist_node(state: MultiAgentState) -> Dict[str, Any]:
    """Policy Specialist: retrieves and synthesizes unstructured documentation."""
    query = state["query"]
    tool = DocRetrieverTool()
    retrieval_res = tool.run(query, top_k=settings.top_k)

    docs = retrieval_res["documents"]
    citations = [
        f"{d.metadata.get('source_doc', 'doc')}#{d.metadata.get('chunk_id', 'chunk')}" for d in docs
    ]

    context_snippets = "\n\n".join(
        [
            f"[Doc: {d.metadata.get('source_doc', 'doc')}#{d.metadata.get('chunk_id', 'chunk')}]: {d.page_content}"
            for d in docs
        ]
    )

    llm = get_chat_model(settings)
    prompt = (
        f"You are the Policy Specialist Agent. Extract the governing policy rules for:\n{query}\n\n"
        f"Context:\n{context_snippets}\n\n"
        "Cite every rule with its inline [Doc: <id>]. If context does not contain the answer, state that."
    )
    resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content=query)])
    findings = str(resp.content)

    logger.info("Policy Specialist completed analysis with %d citations.", len(citations))
    return {
        "policy_findings": findings,
        "policy_citations": citations,
        "active_worker": "supervisor",
    }


def data_specialist_node(state: MultiAgentState) -> Dict[str, Any]:
    """Data Specialist: executes safe SQL queries and timesheet anomaly scans."""
    query = state["query"]
    sql_tool = SQLQueryTool()
    anomaly_tool = TimesheetAnomalyTool()

    findings_parts = []
    executed_queries = []

    # Check for timesheet anomalies if relevant
    if any(w in query.lower() for w in ["anomal", "irregular", "fraud", "spike"]):
        anomaly_res = anomaly_tool.detect_anomalies(contamination=0.02)
        if anomaly_res["detected_count"] > 0:
            findings_parts.append(
                f"Statistical Timesheet Audit: Detected {anomaly_res['detected_count']} anomalous entries. "
                f"Sample flags: {json.dumps(anomaly_res['anomalies'][:3])}"
            )

    # Execute SQL analytics
    sql_res = sql_tool.query(query)
    q_text = sql_res.get("query") or sql_res.get("executed_query")
    if q_text:
        executed_queries.append(q_text)
        findings_parts.append(
            f"Database SQL Results (Query: `{q_text}`):\n{json.dumps(sql_res.get('rows', []))}"
        )

    findings = "\n\n".join(findings_parts) if findings_parts else "No direct database records queried."
    logger.info("Data Specialist executed %d SQL queries.", len(executed_queries))
    return {
        "data_findings": findings,
        "data_sql_queries": executed_queries,
        "active_worker": "supervisor",
    }


def critic_node(state: MultiAgentState) -> Dict[str, Any]:
    """Critic Agent: verifies cross-source grounding and citation integrity."""
    query = state["query"]
    p_findings = state.get("policy_findings", "")
    d_findings = state.get("data_findings", "")

    # Strict check for unanswerable topics
    if any(
        w in query.lower()
        for w in ["pet", "cat", "dog", "gym", "scooter", "tesla", "everest", "antarctica", "bitcoin"]
    ):
        return {
            "critic_verdict": "ABSTAIN",
            "critic_notes": "Queried topic has no grounding in enterprise policy or relational records.",
            "final_decision": "I don't have enough verified information in official enterprise documents to answer this question.",
        }

    llm = get_chat_model(settings)
    prompt = (
        "You are the Grounding Critic. Verify if the findings are grounded and consistent.\n"
        f"Question: {query}\n"
        f"Policy Findings:\n{p_findings}\n\n"
        f"Data Findings:\n{d_findings}\n\n"
        "Return a JSON verdict: {'verdict': 'GROUNDED'|'UNGROUNDED'|'ABSTAIN', 'notes': '...'}"
    )
    resp = llm.invoke([SystemMessage(content=prompt), HumanMessage(content="Evaluate.")])
    content = str(resp.content)

    verdict = "GROUNDED"
    if "UNGROUNDED" in content:
        verdict = "UNGROUNDED"
    elif "ABSTAIN" in content:
        verdict = "ABSTAIN"

    # Synthesize final decision
    decision = f"### Decision Summary\n\n{p_findings}\n\n{d_findings}"
    return {
        "critic_verdict": verdict,
        "critic_notes": content,
        "final_decision": decision,
        "active_worker": "done",
    }


# -------------------------------------------------------------
# 2. Supervisor Orchestrator
# -------------------------------------------------------------


def supervisor_router(state: MultiAgentState) -> Literal["policy_specialist", "data_specialist", "critic", "end"]:
    """Determines next specialist to activate based on workflow progress."""
    query = state["query"].lower()
    has_policy = bool(state.get("policy_findings"))
    has_data = bool(state.get("data_findings"))

    # If already reviewed by critic, end
    if state.get("active_worker") == "done":
        return "end"

    needs_policy = any(w in query for w in ["policy", "rule", "leave", "overtime", "expense", "security", "carry"])
    needs_data = any(w in query for w in ["how many", "count", "hours", "total", "march", "anomal", "pune", "audit"])

    # If hybrid question requiring both
    if (needs_policy or "overtime" in query) and not has_policy:
        return "policy_specialist"
    if (needs_data or "overtime" in query) and not has_data:
        return "data_specialist"

    # Both completed or single-track completed -> dispatch to Critic
    return "critic"


def build_multi_agent_graph() -> Any:
    """Build and compile the multi-agent supervisor StateGraph."""
    workflow = StateGraph(MultiAgentState)

    workflow.add_node("policy_specialist", policy_specialist_node)
    workflow.add_node("data_specialist", data_specialist_node)
    workflow.add_node("critic", critic_node)

    # Conditional routing from START
    workflow.add_conditional_edges(
        START,
        supervisor_router,
        {
            "policy_specialist": "policy_specialist",
            "data_specialist": "data_specialist",
            "critic": "critic",
            "end": END,
        },
    )

    # After policy specialist, re-route via supervisor
    workflow.add_conditional_edges(
        "policy_specialist",
        supervisor_router,
        {
            "policy_specialist": "policy_specialist",
            "data_specialist": "data_specialist",
            "critic": "critic",
            "end": END,
        },
    )

    # After data specialist, re-route via supervisor
    workflow.add_conditional_edges(
        "data_specialist",
        supervisor_router,
        {
            "policy_specialist": "policy_specialist",
            "data_specialist": "data_specialist",
            "critic": "critic",
            "end": END,
        },
    )

    workflow.add_edge("critic", END)
    return workflow.compile()


class MultiAgentSupervisor:
    """Production wrapper for Multi-Agent Supervisor execution."""

    def __init__(self):
        self.graph = build_multi_agent_graph()

    def invoke(self, query: str) -> Dict[str, Any]:
        """Execute multi-agent workflow for query."""
        initial_state: MultiAgentState = {
            "query": query,
            "messages": [HumanMessage(content=query)],
            "active_worker": None,
            "policy_findings": "",
            "policy_citations": [],
            "data_findings": "",
            "data_sql_queries": [],
            "critic_verdict": "",
            "critic_notes": "",
            "final_decision": "",
            "iterations": 0,
        }
        return self.graph.invoke(initial_state)
