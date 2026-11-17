"""Conditional routing edge functions for DecisionIQ StateGraph."""

from typing import Literal

from decisioniq.agent.state import AgentState


def route_intent(state: AgentState) -> Literal["planner", "abstain"]:
    """Route after classification: unanswerable or adversarial queries abstain immediately."""
    intent = state.get("intent", "")
    if intent in ["adversarial", "unanswerable"]:
        return "abstain"
    return "planner"


def route_grounding(state: AgentState) -> Literal["end", "rewrite", "abstain"]:
    """Route after grounding verification check: accept, retry, or abstain."""
    verdict = state.get("grounding_verdict", "GROUNDED")
    retries = state.get("retry_count", 0)

    if verdict == "GROUNDED":
        return "end"

    # Retry budget max 2
    if retries < 2:
        return "rewrite"

    return "abstain"
