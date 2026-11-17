"""Compiled LangGraph StateGraph with grounding verification, memory, and guardrails."""

import logging
from typing import Any, Dict, Iterator, Optional

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from decisioniq.agent.edges import route_grounding, route_intent
from decisioniq.agent.nodes import (
    abstain_node,
    classify_intent_node,
    grounding_check_node,
    plan_node,
    query_rewrite_node,
    synthesize_node,
    tools_node,
)
from decisioniq.agent.state import AgentState

logger = logging.getLogger(__name__)


def build_decisioniq_graph(checkpointer: Optional[Any] = None) -> Any:
    """Construct and compile the DecisionIQ StateGraph agent."""
    workflow = StateGraph(AgentState)

    # 1. Add nodes
    workflow.add_node("classify_intent", classify_intent_node)
    workflow.add_node("planner", plan_node)
    workflow.add_node("tools", tools_node)
    workflow.add_node("synthesize", synthesize_node)
    workflow.add_node("grounding_check", grounding_check_node)
    workflow.add_node("query_rewrite", query_rewrite_node)
    workflow.add_node("abstain", abstain_node)

    # 2. Add edges
    workflow.add_edge(START, "classify_intent")

    workflow.add_conditional_edges(
        "classify_intent",
        route_intent,
        {
            "planner": "planner",
            "abstain": "abstain",
        },
    )

    workflow.add_edge("planner", "tools")
    workflow.add_edge("tools", "synthesize")
    workflow.add_edge("synthesize", "grounding_check")

    workflow.add_conditional_edges(
        "grounding_check",
        route_grounding,
        {
            "end": END,
            "rewrite": "query_rewrite",
            "abstain": "abstain",
        },
    )

    # Query rewrite re-enters planning loop
    workflow.add_edge("query_rewrite", "planner")
    workflow.add_edge("abstain", END)

    # 3. Compile with memory checkpointer
    saver = checkpointer if checkpointer is not None else MemorySaver()
    compiled = workflow.compile(checkpointer=saver)
    logger.info("Compiled DecisionIQ StateGraph with grounding loop and thread memory.")
    return compiled


class DecisionIQAgent:
    """Enterprise DecisionIQ Agent with streaming, execution tracing, and session memory."""

    def __init__(self, checkpointer: Optional[Any] = None):
        self.checkpointer = checkpointer or MemorySaver()
        self.graph = build_decisioniq_graph(checkpointer=self.checkpointer)

    def invoke(
        self,
        query: str,
        thread_id: str = "default_session",
    ) -> Dict[str, Any]:
        """Execute full stategraph and return final state with execution trace."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "intent": "",
            "intent_confidence": 0.0,
            "plan": [],
            "tool_results": {},
            "retrieved_docs": [],
            "sql_results": [],
            "sql_queries": [],
            "anomalies": [],
            "calc_results": [],
            "grounding_verdict": "",
            "grounding_reasoning": "",
            "retry_count": 0,
            "citations": [],
            "final_answer": "",
            "is_interrupted": False,
        }

        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.graph.invoke(initial_state, config=config)
        return final_state

    def stream_steps(
        self,
        query: str,
        thread_id: str = "default_session",
    ) -> Iterator[Dict[str, Any]]:
        """Yield intermediate execution steps (active node and state delta)."""
        initial_state: AgentState = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "intent": "",
            "intent_confidence": 0.0,
            "plan": [],
            "tool_results": {},
            "retrieved_docs": [],
            "sql_results": [],
            "sql_queries": [],
            "anomalies": [],
            "calc_results": [],
            "grounding_verdict": "",
            "grounding_reasoning": "",
            "retry_count": 0,
            "citations": [],
            "final_answer": "",
            "is_interrupted": False,
        }

        config = {"configurable": {"thread_id": thread_id}}
        for event in self.graph.stream(initial_state, config=config, stream_mode="updates"):
            yield event
