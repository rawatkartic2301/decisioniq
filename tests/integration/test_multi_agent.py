"""Integration tests for Multi-Agent Supervisor variant."""

from decisioniq.agent.multi_agent import MultiAgentSupervisor


def test_multi_agent_policy_query():
    """Verify PolicySpecialist and Critic activate on pure policy question."""
    supervisor = MultiAgentSupervisor()
    res = supervisor.invoke("What is our policy on carry-forward leave?")

    assert res["policy_findings"] != ""
    assert len(res["policy_citations"]) > 0
    assert res["critic_verdict"] == "GROUNDED"
    assert "Decision Summary" in res["final_decision"]


def test_multi_agent_hybrid_query():
    """Verify both Policy and Data specialists activate on hybrid questions."""
    supervisor = MultiAgentSupervisor()
    query = "What is the overtime rule for grade L3, and how many overtime hours did Pune log in March?"
    res = supervisor.invoke(query)

    assert res["policy_findings"] != ""
    assert res["data_findings"] != ""
    assert res["critic_verdict"] == "GROUNDED"
    assert len(res["data_sql_queries"]) > 0 or "Timesheet" in res["data_findings"] or "SQL" in res["data_findings"]


def test_multi_agent_unanswerable_query():
    """Verify Critic flags ungrounded unanswerable query for abstention."""
    supervisor = MultiAgentSupervisor()
    res = supervisor.invoke("What is our company policy on pet insurance reimbursement?")

    assert res["critic_verdict"] == "ABSTAIN"
    assert "don't have enough verified information" in res["final_decision"].lower()
